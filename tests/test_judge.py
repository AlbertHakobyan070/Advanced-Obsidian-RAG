"""The answer tier's judge (eval/bench/judge.py): verdict parsing, the metrics
on hand-built rows, and a record -> fill -> replay pass over a tiny run.

A verdict that does not match what was asked (wrong nugget count, a citation the
answer never made) is an error, not a guess. Offline: no model, no provider.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.judge import judge_run, parse_verdict, summarise
from eval.bench.questions import question_from_dict
from src.llm.replay import ReplayLLM


# ---- parsing ----

def _v(**over):
    v = {"abstained": False, "stated": [True, False], "supported": [True, True],
         "citations": [{"n": 1, "supports": True}]}
    v.update(over)
    return json.dumps(v)


def test_a_well_formed_verdict_parses_even_inside_a_fence():
    v = parse_verdict("```json\n" + _v() + "\n```", 2, [1])
    assert v["stated"] == [True, False] and v["citations"] == [{"n": 1, "supports": True}]


@pytest.mark.parametrize("text,match", [
    ("no json", "no JSON"),
    (_v(stated=[True]), "'stated' must be 2"),
    (_v(supported=["yes", True]), "'supported'"),
    (_v(abstained="no"), "abstained"),
    (_v(citations=[{"n": 2, "supports": True}]), "the answer cites"),
    (_v(citations=[{"n": 1}]), "citations"),
])
def test_a_verdict_that_does_not_match_the_question_is_refused(text, match):
    with pytest.raises(ValueError, match=match):
        parse_verdict(text, 2, [1])


# ---- metrics ----

def _row(qid, cfg, answerable, verdict, confidence="HIGH", match=([0],), k=1):
    return {"qid": qid, "config": cfg, "answerable": answerable, "confidence": confidence,
            "k": k, "match": [list(m) for m in match], "verdict": verdict}


def _verdict(stated, supported=None, cites=((1, True),), abstained=False):
    return {"abstained": abstained, "stated": list(stated),
            "supported": list(supported if supported is not None else stated),
            "citations": [{"n": n, "supports": s} for n, s in cites]}


def test_the_metrics_on_hand_built_rows():
    rows = [
        _row("a", "X", True, _verdict([True, True])),                                # pass
        _row("b", "X", True, _verdict([True, False], [False, False]), confidence="LOW"),  # generation
        _row("c", "X", True, _verdict([True], cites=((1, False),))),                 # citation
        _row("d", "X", True, _verdict([False], cites=()), match=([],)),              # retrieval
        _row("e", "X", False, _verdict([], cites=(), abstained=True)),               # right abstention
        {"qid": "f", "config": "X", "answerable": True, "confidence": "HIGH", "k": 1,
         "match": [[0]], "error": "judge: unparseable"},
    ]
    s = summarise(rows)["X"]
    assert s["nugget_recall"]["mean"] == pytest.approx((1 + 0.5 + 1 + 0) / 4)
    assert s["grounded_recall"]["mean"] == pytest.approx((1 + 0 + 1 + 0) / 4)
    assert s["citation_precision"]["mean"] == pytest.approx((1 + 1 + 0) / 3)   # d cites nothing
    assert s["uncited_rate"]["mean"] == pytest.approx(1 / 4)
    assert s["abstention_accuracy"]["mean"] == 1.0 and s["false_abstention"]["mean"] == 0.0
    assert s["attribution"] == {"pass": 1, "generation": 1, "citation": 1, "retrieval": 1}
    assert s["calibration"]["LOW"]["mean"] == 0.5 and s["errors"] == 1


def test_paired_nugget_recall_follows_the_baselines():
    rows = [_row(q, "Q-off", True, _verdict([False])) for q in "ab"] + \
           [_row(q, "Q-on", True, _verdict([True])) for q in "ab"]
    p = summarise(rows)["Q-on"]["paired_nugget_recall"]
    assert p["vs"] == "Q-off" and p["mean"] == 1.0 and p["n"] == 2


# ---- record -> fill -> replay ----

def test_judge_run_records_then_replays(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "x_chunks.jsonl").write_text(
        json.dumps({"doc_id": "c1", "text": "ARIMA has p, d and q.", "metadata": {}}) + "\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    row = {"qid": "lex-0001", "config": "X", "ids": ["c1"], "match": [[0]],
           "answer": {"text": "It has p, d, q [1].", "confidence": "HIGH", "cited": [1], "k": 1}}
    (run / "per_query.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    q = question_from_dict({
        "id": "lex-0001", "question": "What orders does ARIMA have?", "suite": "lexical", "tier": "T1",
        "split": "dev", "answerable": True, "gold": [{"file": "a.md"}], "nuggets": ["p, d and q"],
        "expect_course": None, "provenance": {"author": "draft", "status": "draft"}}, "t")
    store = tmp_path / "haiku.jsonl"

    s = judge_run(run, {q.id: q}, ReplayLLM(store, mode="record"), data, progress=lambda *_: None)
    assert s["X"]["errors"] == 1                                   # recorded, not judged
    pending = [json.loads(l) for l in (tmp_path / "haiku.pending.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(pending) == 1 and "ARIMA has p, d and q." in pending[0]["user"]

    store.write_text(json.dumps({"key": pending[0]["key"], "model": "claude-haiku-5-5",
                                 "completion": _v(stated=[True], supported=[True])}) + "\n", encoding="utf-8")
    s = judge_run(run, {q.id: q}, ReplayLLM(store), data, progress=lambda *_: None)
    assert s["X"]["errors"] == 0 and s["X"]["nugget_recall"]["mean"] == 1.0
    meta = json.loads((run / "judge_summary.json").read_text(encoding="utf-8"))["meta"]
    assert meta["judge_model"] == "claude-haiku-5-5" and (run / "judge.md").exists()


def test_a_dangling_citation_is_judged_and_counted_unsupported(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "x_chunks.jsonl").write_text(
        json.dumps({"doc_id": "c1", "text": "ARIMA has p, d and q.", "metadata": {}}) + "\n", encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    # [9] against one source: the generator's resolved list keeps only [1]
    row = {"qid": "lex-0001", "config": "X", "ids": ["c1"], "match": [[0]],
           "answer": {"text": "p, d, q [1], see also [9].", "confidence": "HIGH", "cited": [1], "k": 1}}
    (run / "per_query.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    q = question_from_dict({
        "id": "lex-0001", "question": "ARIMA orders?", "suite": "lexical", "tier": "T1",
        "split": "dev", "answerable": True, "gold": [{"file": "a.md"}], "nuggets": ["p, d and q"],
        "expect_course": None, "provenance": {"author": "draft", "status": "draft"}}, "t")
    store = tmp_path / "haiku.jsonl"
    judge_run(run, {q.id: q}, ReplayLLM(store, mode="record"), data, progress=lambda *_: None)
    key = json.loads((tmp_path / "haiku.pending.jsonl").read_text(encoding="utf-8").splitlines()[0])["key"]
    # even a judge that calls [9] supporting cannot make a pointer to nothing count
    verdict = _v(stated=[True], supported=[True],
                 citations=[{"n": 1, "supports": True}, {"n": 9, "supports": True}])
    store.write_text(json.dumps({"key": key, "model": "m", "completion": verdict}) + "\n", encoding="utf-8")
    s = judge_run(run, {q.id: q}, ReplayLLM(store), data, progress=lambda *_: None)["X"]
    assert s["errors"] == 0 and s["citation_precision"]["mean"] == 0.5


def test_a_run_without_answers_is_refused(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "per_query.jsonl").write_text(json.dumps({"qid": "x", "config": "X"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="--answers"):
        judge_run(run, {}, ReplayLLM(tmp_path / "s.jsonl"), tmp_path, progress=lambda *_: None)
