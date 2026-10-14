"""The replay store (src/llm/replay.py) and the bench's --llm / --answers.

A replay store answers LLM calls from completions written ahead of time, so an
eval measures the pipeline with one fixed LLM. A miss must never pass for an
answer: in replay mode it raises, in record mode it queues the prompt and raises.
Offline: no model, no provider.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.questions import question_from_dict
from eval.bench.runner import run_bench
from src.llm.replay import UNFILLED, ReplayLLM, ReplayMiss, prompt_key


def _store(path, *rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _done(system, user, text, model="claude-sonnet-5-5"):
    return {"key": prompt_key(system, user), "model": model, "completion": text}


# ---- the store ----

def test_a_hit_answers_with_the_stores_model_label(tmp_path):
    llm = ReplayLLM(_store(tmp_path / "s.jsonl", _done("sys", "u1", "draft one")))
    r = llm.complete(system="sys", user="u1", temperature=0.3, max_tokens=200)
    assert r.text == "draft one" and r.model == llm.model == "claude-sonnet-5-5"


def test_replay_miss_raises_and_records_nothing(tmp_path):
    llm = ReplayLLM(_store(tmp_path / "s.jsonl", _done("sys", "u1", "x")), mode="replay")
    with pytest.raises(ReplayMiss):
        llm.complete(system="sys", user="other")
    assert not llm.pending.exists()


def test_record_miss_queues_the_prompt_once_and_still_raises(tmp_path):
    llm = ReplayLLM(tmp_path / "s.jsonl", mode="record")
    assert llm.model == UNFILLED
    for _ in range(2):
        with pytest.raises(ReplayMiss):
            llm.complete(system="sys", user="u", temperature=0.0, max_tokens=50)
    rows = [json.loads(l) for l in llm.pending.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["key"] == prompt_key("sys", "u") and rows[0]["user"] == "u"
    assert rows[0]["max_tokens"] == 50
    # a fresh client over the same files does not queue it again
    with pytest.raises(ReplayMiss):
        ReplayLLM(tmp_path / "s.jsonl", mode="record").complete(system="sys", user="u")
    assert len(llm.pending.read_text(encoding="utf-8").splitlines()) == 1


def test_a_store_mixing_models_is_refused(tmp_path):
    with pytest.raises(ValueError, match="mixes models"):
        ReplayLLM(_store(tmp_path / "s.jsonl", _done("s", "a", "x", "m1"), _done("s", "b", "y", "m2")))


def test_a_malformed_completion_is_refused(tmp_path):
    with pytest.raises(ValueError, match="lacks completion"):
        ReplayLLM(_store(tmp_path / "s.jsonl", {"key": "k", "model": "m"}))


def test_an_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError):
        ReplayLLM(tmp_path / "s.jsonl", mode="live")


# ---- the bench ----

class _Doc:
    def __init__(self, i, f):
        self.id, self.metadata = i, {"source_file": f}


class _Rag:
    rerank_top_k = 1

    def __init__(self, hyde_status="off"):
        self.hyde = SimpleNamespace(cache=object(), llm=None)
        self.retriever = SimpleNamespace(_get_collection=lambda: SimpleNamespace(count=lambda: 2))
        self.hyde_status = hyde_status
        self.answered = []
        rag = self

        class _Gen:
            llm = None

            def generate(self, question, docs):
                rag.answered.append([d.id for d in docs])
                r = self.llm.complete(system="gen", user=question)
                return SimpleNamespace(text=r.text, confidence="HIGH",
                                       citations=[SimpleNamespace(number=1)])
        self.generator = _Gen()

    def search(self, q, **kw):
        return [_Doc("1", "a.md"), _Doc("2", "b.md")], {
            "timings": {"total": 1.0}, "lanes_run": {}, "cold": False,
            "hyde_cache": self.hyde_status}


def _q(i, text="q?"):
    return question_from_dict({
        "id": f"lex-{i:04d}", "question": text, "suite": "lexical", "tier": "T1",
        "split": "dev", "answerable": True, "gold": [{"file": "a.md"}], "nuggets": ["n"],
        "expect_course": None, "provenance": {"author": "draft", "status": "draft"}}, "t")


@pytest.fixture
def go(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.bench.runner.index_fingerprint", lambda cfg, rag: {})
    monkeypatch.setattr("eval.bench.runner.git_state", lambda repo: {"sha": "abc1234", "dirty": False})
    monkeypatch.setattr("eval.bench.runner.config_digest", lambda cfg: "d")

    def _go(rag, **kw):
        run = run_bench(None, [_q(1)], [("x", {"lanes": ["dense"]})], split="dev",
                        out_root=tmp_path / "runs", ledger=tmp_path / "l.jsonl",
                        progress=lambda *_: None, rag=rag, **kw)
        rows = [json.loads(l) for l in (run / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
        return rows, json.loads((run / "run.json").read_text(encoding="utf-8"))
    return _go


def test_answers_come_from_the_store_and_use_the_production_k(go, tmp_path):
    llm = ReplayLLM(_store(tmp_path / "s.jsonl", _done("gen", "q?", "It is A [1]. CONFIDENCE: HIGH")))
    rag = _Rag()
    rows, meta = go(rag, llm=llm, answers=True)
    assert rows[0]["answer"] == {"text": "It is A [1]. CONFIDENCE: HIGH", "confidence": "HIGH",
                                 "cited": [1], "k": 1}
    assert rag.answered == [["1"]]                       # rerank_top_k = 1 chunk, not all 10
    assert rag.hyde.llm is llm and rag.generator.llm is llm
    assert meta["llm"] == {"mode": "replay", "store": str(llm.store), "model": "claude-sonnet-5-5"}
    assert meta["answers"] == {"k": 1} and "metrics" in rows[0]


def test_a_replay_miss_in_generation_is_a_row_error_not_an_answer(go, tmp_path):
    rows, _ = go(_Rag(), llm=ReplayLLM(tmp_path / "s.jsonl"), answers=True)
    assert rows[0]["error"].startswith("answer: ReplayMiss") and "answer" not in rows[0]


def test_hyde_falling_back_under_a_replay_store_fails_the_row(go, tmp_path):
    rows, _ = go(_Rag(hyde_status="error"), llm=ReplayLLM(tmp_path / "s.jsonl"))
    assert "HyDE got no draft" in rows[0]["error"] and "metrics" not in rows[0]


def test_without_llm_a_hyde_error_is_still_scored_as_before(go):
    rows, meta = go(_Rag(hyde_status="error"))
    assert "error" not in rows[0] and "metrics" in rows[0]
    assert meta["llm"] is None and meta["answers"] is None
