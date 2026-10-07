"""The experimental Laya reranker: the inference adapter, the `laya` rerank mode and its flag.

Fully offline. The `laya` package is NOT installed in the test venv (it is
optional and deliberately not in requirements.txt), so a stand-in module is
injected through sys.modules. It is shaped like laya 0.3.28's real API, which
was read from its source, not guessed:
  laya.load(path, device=...)          -> an Agent with .device (a torch.device),
                                          .amp_enabled and .dtype
  agent.predict_batch(states, questions) -> one dict per state, in order:
      {"model": ..., "answers": {qid: {"type": "noul", "noul": P(true), ...}},
       "usage": {..., "truncated": bool}}
The scorer, the reranker, the pipeline echo and the API are the real ones.

What is pinned, in the order a request meets it:
  * nothing is imported or loaded until the first score() — a disabled feature
    costs nothing at startup;
  * the checkpoint is checked BEFORE laya.load, by absolute path: laya turns a
    missing relative path into a Hugging Face Hub lookup, so it must never see one;
  * P(true) comes back one per passage, in order, in batch_size chunks, and a
    wrong count or a non-finite score is an ERROR, never a half-ranking;
  * on a CUDA card below compute capability 8 laya's forced fp16 autocast is
    switched off before the first forward;
  * `rerank: laya` refuses — naming the flag or the path — rather than fall back
    to another reranker, and the API reports that as "Reranking failed: …".
"""
import logging
import os
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

import serve_api as S
from eval.bench.configs import resolve
from src.pipeline import RAGPipeline
from src.retrieval import laya_reranker as LR
from src.retrieval.laya_reranker import LayaScorer, LayaScorerError
from src.retrieval.relevance_gate import _SCORING_MODES, RelevanceGate
from src.retrieval.reranker import RERANK_MODES, Reranker, RerankerExecutionError
from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]

# P(true) by passage text: the fake reads it off the passage, so order and
# batching are checkable whatever the batch composition.
PROBS = {"a": 0.2, "b": 0.9, "c": 0.5, "d": 0.7, "e": 0.1}


# ---- the stand-in laya package ----

def _result(qid, p, truncated=False):
    """One predict_batch result for a `noul` question, shaped like laya 0.3.28's."""
    return {
        "model": "laya-rl-agent",
        "answers": {qid: {"type": "noul", "noul": p,
                          "confidence": round(max(p, 1 - p), 4),
                          "answer_confidence": round(max(p, 1 - p), 4),
                          "action": {"act_probability": 0.6714}}},
        "usage": {"input_tokens": 30, "output_tokens": 0, "state_tokens": 3,
                  "state_tokens_dropped": 1 if truncated else 0, "truncated": truncated,
                  "truncated_questions": [qid] if truncated else []},
    }


class FakeAgent:
    """The slice of laya.Agent the adapter touches."""

    def __init__(self, probabilities=None, device="cpu", dtype=None, truncated=(), short_by=0):
        self.probabilities = dict(PROBS if probabilities is None else probabilities)
        self.device = torch.device(device)
        # laya's own load-time policy: reduced-precision autocast on every CUDA card.
        cuda = self.device.type == "cuda"
        self.amp_enabled = cuda
        self.dtype = dtype or (torch.float16 if cuda else torch.float32)
        self.truncated = set(truncated)
        self.short_by = short_by
        self.calls = []
        self.precision_seen = []             # (amp_enabled, dtype) at each forward

    def predict_batch(self, states, questions, batch_size=None, **kwargs):
        self.calls.append({"states": list(states), "questions": questions})
        self.precision_seen.append((self.amp_enabled, self.dtype))
        [qid] = questions
        out = [_result(qid, self.probabilities[s], s in self.truncated) for s in states]
        return out[:len(out) - self.short_by]


class FakeLaya(types.ModuleType):
    """`import laya` as 0.3.28 exposes it: a module whose load() returns an Agent."""

    def __init__(self, agent, delay=0.0, error=None):
        super().__init__("laya")
        self.__version__ = "0.3.28"
        self.agent, self.delay, self.error = agent, delay, error
        self.load_calls = []

    def load(self, model_id_or_path="convaiinnovations/laya", device=None, **kwargs):
        self.load_calls.append((model_id_or_path, device))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.agent


@pytest.fixture
def install_laya(monkeypatch):
    """Inject a fake `laya` through sys.modules and hand it back."""
    def install(agent=None, **kwargs):
        fake = FakeLaya(agent if agent is not None else FakeAgent(), **kwargs)
        monkeypatch.setitem(sys.modules, "laya", fake)
        return fake
    return install


def make_checkpoint(root, missing=()):
    """The four parts a real checkpoint holds (laya/agent.py's local-dir layout)."""
    root.mkdir(parents=True, exist_ok=True)
    for name in ("rl_agent_config.json", "model.safetensors"):
        if name not in missing:
            (root / name).write_bytes(b"{}")
    for name in ("tokenizer", "encoder"):
        if name not in missing:
            (root / name).mkdir(exist_ok=True)
    return root


@pytest.fixture
def checkpoint(tmp_path):
    return make_checkpoint(tmp_path / "models" / "laya-noetrix")


def cfg_of(data, root=ROOT):
    return Config(data, root)


def docs(*texts):
    return [RetrievedDoc(id=f"d{i}", text=t, metadata={}) for i, t in enumerate(texts)]


def ids(items):
    return [d.id for d in items]


class FakeScorer:
    """What the Reranker needs from a LayaScorer: score(query, texts) -> P(true) per text."""
    model_dir = "/fake/models/laya"

    def __init__(self, *probabilities, error=None):
        self.probabilities = list(probabilities)
        self.error = error
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        if self.error:
            raise self.error
        return list(self.probabilities)


# ---- the question: ONE wording, shared with the training rows ----

def test_the_question_wording_is_the_one_the_training_rows_use():
    """The question text is part of the model's input, so the notebook's training
    rows and this adapter must say exactly the same thing. This is the pin."""
    assert LR.QUESTION_ID == "rel"
    assert LR.NOUL_QUESTION == {
        "type": "noul",
        "instructions": "Does the passage answer: {query}?",
        "criteria": {"false": "Irrelevant.", "true": "Answers the query."},
    }


def test_the_passage_is_the_state_and_the_query_goes_into_the_question(install_laya, checkpoint):
    fake = install_laya()
    LayaScorer(checkpoint).score("what is a?", ["a", "b"])
    [call] = fake.agent.calls
    assert call["states"] == ["a", "b"]                      # the passages ARE the states
    assert call["questions"] == {"rel": {
        "type": "noul",
        "instructions": "Does the passage answer: what is a??",
        "criteria": {"false": "Irrelevant.", "true": "Answers the query."},
    }}


def test_a_query_with_braces_or_percent_signs_reaches_the_question_untouched(install_laya, checkpoint):
    fake = install_laya()
    LayaScorer(checkpoint).score("what is {x} at 100% {0}", ["a"])
    [call] = fake.agent.calls
    assert call["questions"]["rel"]["instructions"] == (
        "Does the passage answer: what is {x} at 100% {0}?")


# ---- score(): order, length, batching ----

def test_score_returns_one_p_true_per_text_in_order(install_laya, checkpoint):
    install_laya()
    out = LayaScorer(checkpoint).score("q", ["c", "a", "b"])
    assert out == [0.5, 0.2, 0.9]
    assert all(type(p) is float for p in out)


def test_texts_go_through_in_batch_size_chunks_and_come_back_in_order(install_laya, checkpoint):
    fake = install_laya()
    texts = ["a", "b", "c", "d", "e"]
    out = LayaScorer(checkpoint, batch_size=2).score("q", texts)
    assert [c["states"] for c in fake.agent.calls] == [["a", "b"], ["c", "d"], ["e"]]
    assert out == [PROBS[t] for t in texts]
    assert len(out) == len(texts)


def test_one_chunk_when_batch_size_covers_every_text(install_laya, checkpoint):
    fake = install_laya()
    LayaScorer(checkpoint, batch_size=16).score("q", ["a", "b", "c"])
    assert len(fake.agent.calls) == 1


def test_empty_texts_score_to_nothing_and_load_nothing(install_laya, tmp_path):
    fake = install_laya()
    assert LayaScorer(tmp_path / "no-such-dir").score("q", []) == []   # not even a checkpoint check
    assert fake.load_calls == []


def test_a_wrong_result_count_is_an_error_not_a_misaligned_ranking(install_laya, checkpoint):
    install_laya(FakeAgent(short_by=1))
    with pytest.raises(LayaScorerError, match=r"2 result\(s\) for 3 passage\(s\)"):
        LayaScorer(checkpoint).score("q", ["a", "b", "c"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_probability_is_an_error_not_a_ranking_key(install_laya, checkpoint, bad):
    """NaN compares False against everything, so it would sort anywhere and still
    read as a ranking. A third party has reported NaN scores from laya's fused
    attention at padded positions on CUDA, so this is a live failure mode."""
    install_laya(FakeAgent({"a": 0.2, "b": bad}))
    with pytest.raises(LayaScorerError, match="non-finite"):
        LayaScorer(checkpoint).score("q", ["a", "b"])


def test_truncated_passages_are_logged_not_silent(install_laya, checkpoint, caplog):
    """Laya right-truncates the passage to the checkpoint's max_len without a word;
    its usage block is the only place that says so."""
    install_laya(FakeAgent(truncated={"a", "c"}))
    with caplog.at_level(logging.INFO):
        LayaScorer(checkpoint).score("q", ["a", "b", "c"])
    assert any("2 of 3" in r.message and "truncated" in r.message for r in caplog.records)


# ---- lazy: nothing imported or loaded until the first score() ----

def test_constructing_a_scorer_imports_and_loads_nothing(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "laya", None)       # any `import laya` now raises
    LayaScorer(tmp_path / "no-such-dir")                 # so this must not import it, nor touch the disk


def test_the_checkpoint_loads_once_and_is_reused(install_laya, checkpoint):
    fake = install_laya()
    scorer = LayaScorer(checkpoint)
    scorer.score("q", ["a"])
    scorer.score("q", ["b"])
    assert fake.load_calls == [(os.path.abspath(checkpoint), "cpu")]


def test_concurrent_first_calls_load_the_checkpoint_once(install_laya, checkpoint):
    """:8051 serves requests on a thread pool, and the model is ~2 GB of RAM."""
    fake = install_laya(delay=0.05)
    scorer = LayaScorer(checkpoint)
    gate = threading.Barrier(4)

    def first_call():
        gate.wait()
        scorer.score("q", ["a"])

    threads = [threading.Thread(target=first_call) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(fake.load_calls) == 1


def test_a_failed_load_is_not_swallowed_and_is_retried(install_laya, checkpoint):
    fake = install_laya(error=OSError("corrupt weights"))
    scorer = LayaScorer(checkpoint)
    with pytest.raises(OSError, match="corrupt weights"):
        scorer.score("q", ["a"])
    fake.error = None
    assert scorer.score("q", ["a"]) == [0.2]             # the failure was not cached as a loaded model
    assert len(fake.load_calls) == 2


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_load_gets_the_absolute_path_and_the_configured_device(
        install_laya, tmp_path, monkeypatch, device):
    """A RELATIVE path that is missing is not an error to laya.load: it becomes a
    Hugging Face Hub repo id. The adapter must hand it an absolute one."""
    make_checkpoint(tmp_path / "models" / "laya-noetrix")
    monkeypatch.chdir(tmp_path)
    fake = install_laya(FakeAgent(device=device))
    LayaScorer("models/laya-noetrix", device=device).score("q", ["a"])
    [(path, dev)] = fake.load_calls
    assert os.path.isabs(path) and path == os.path.abspath("models/laya-noetrix")
    assert dev == device


def test_the_default_device_is_cpu(install_laya, checkpoint):
    fake = install_laya()
    LayaScorer(checkpoint).score("q", ["a"])
    assert fake.load_calls[0][1] == "cpu"


# ---- the checkpoint is checked before laya.load ----

def test_a_missing_directory_names_the_absolute_path_and_never_reaches_laya(
        install_laya, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = install_laya()
    with pytest.raises(LayaScorerError) as exc:
        LayaScorer("models/laya-noetrix").score("q", ["a"])      # the plan's default, relative
    wanted = os.path.abspath("models/laya-noetrix")              # resolved against the cwd, as the adapter does
    assert wanted in str(exc.value) and "does not exist" in str(exc.value)
    assert fake.load_calls == []                                 # no Hub lookup was ever possible


@pytest.mark.parametrize("missing", ["rl_agent_config.json", "model.safetensors",
                                     "tokenizer", "encoder"])
def test_an_incomplete_checkpoint_names_the_directory_and_what_is_missing(
        install_laya, tmp_path, missing):
    """`rl_agent_config.json` missing is upstream's DDP checkpoint_latest/; tokenizer/ or
    encoder/ missing is the case laya.load answers with a Hub request, not an error."""
    root = make_checkpoint(tmp_path / "ckpt", missing=(missing,))
    fake = install_laya()
    with pytest.raises(LayaScorerError) as exc:
        LayaScorer(root).score("q", ["a"])
    assert os.path.abspath(root) in str(exc.value) and missing in str(exc.value)
    assert fake.load_calls == []


def test_a_config_only_directory_lists_every_missing_part(install_laya, tmp_path):
    root = make_checkpoint(tmp_path / "ckpt", missing=("model.safetensors", "tokenizer", "encoder"))
    fake = install_laya()
    with pytest.raises(LayaScorerError) as exc:
        LayaScorer(root).score("q", ["a"])
    for part in ("model.safetensors", "tokenizer", "encoder"):
        assert part in str(exc.value)
    assert fake.load_calls == []


def test_a_file_where_a_directory_belongs_counts_as_missing(install_laya, tmp_path):
    root = make_checkpoint(tmp_path / "ckpt", missing=("tokenizer",))
    (root / "tokenizer").write_text("not a directory")
    install_laya()
    with pytest.raises(LayaScorerError, match="tokenizer"):
        LayaScorer(root).score("q", ["a"])


def test_laya_not_installed_names_the_pinned_pip_command(monkeypatch, checkpoint):
    monkeypatch.setitem(sys.modules, "laya", None)
    with pytest.raises(LayaScorerError, match=r"pip install laya==0\.3\.28") as exc:
        LayaScorer(checkpoint).score("q", ["a"])
    assert isinstance(exc.value.__cause__, ImportError)


# ---- the forced fp16 autocast (laya/agent.py: every CUDA card below compute capability 8) ----

@pytest.mark.parametrize("capability, disabled", [
    ((6, 1), True),        # Pascal: fp16 at ~1/64 rate
    ((7, 5), True),        # Turing: the cut is upstream's own line, compute capability 8
    ((8, 0), False),
    ((8, 6), False),
])
def test_the_forced_fp16_autocast_is_disabled_below_compute_capability_8(
        install_laya, checkpoint, monkeypatch, capability, disabled):
    probed = []
    monkeypatch.setattr(torch.cuda, "get_device_capability",
                        lambda device=None: probed.append(device) or capability)
    agent = FakeAgent(device="cuda:0", dtype=torch.float16)
    install_laya(agent)
    LayaScorer(checkpoint, device="cuda:0").score("q", ["a", "b"])
    assert probed == [agent.device]                            # asked about the card laya really placed it on
    if disabled:
        assert (agent.amp_enabled, agent.dtype) == (False, torch.float32)
    else:
        assert (agent.amp_enabled, agent.dtype) == (True, torch.float16)
    assert agent.precision_seen[0] == (agent.amp_enabled, agent.dtype)    # decided BEFORE the first forward


def test_cpu_leaves_the_precision_alone_and_never_asks_cuda(install_laya, checkpoint, monkeypatch):
    def boom(device=None):
        raise AssertionError("CUDA was queried on a CPU agent")
    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    agent = FakeAgent(device="cpu")
    install_laya(agent)
    LayaScorer(checkpoint).score("q", ["a"])
    assert (agent.amp_enabled, agent.dtype) == (False, torch.float32)


def test_an_agent_without_the_autocast_attributes_is_an_error_not_a_silent_no_op(
        install_laya, checkpoint, monkeypatch):
    """The attributes are laya 0.3.28's, not a documented API. If a release drops them,
    setting them would do nothing while fp16 stayed forced — so say so instead."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (6, 1))
    agent = FakeAgent(device="cuda:0")
    del agent.amp_enabled
    install_laya(agent)
    with pytest.raises(LayaScorerError, match=r"amp_enabled.*laya==0\.3\.28"):
        LayaScorer(checkpoint, device="cuda:0").score("q", ["a"])


# ---- construction is validated: a bad config fails the build, not a request ----

@pytest.mark.parametrize("bad", [0, -4, 2.5, "16", None, True])
def test_batch_size_must_be_a_positive_integer(bad, checkpoint):
    with pytest.raises(ValueError, match=r"batch_size.*positive integer"):
        LayaScorer(checkpoint, batch_size=bad)


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_a_blank_device_is_refused_rather_than_left_to_laya_to_pick(bad, checkpoint):
    with pytest.raises(ValueError, match=r"device"):
        LayaScorer(checkpoint, device=bad)


# ---- the rerank mode ----

def test_laya_is_a_rerank_mode_and_the_gate_sees_it_as_scoring():
    assert "laya" in RERANK_MODES
    assert "laya" in _SCORING_MODES                      # derived, so the gate's error text lists it
    assert Reranker("x", mode="laya").mode == "laya"


@pytest.mark.parametrize("data", [
    {},                                                                    # no block at all
    {"retrieval": {"laya": {"enabled": False}}},
    {"retrieval": {"rerank_mode": "laya", "laya": {"enabled": False}}},    # laya as the DEFAULT mode
])
def test_with_the_flag_off_there_is_no_scorer_and_laya_is_refused_naming_the_flag(data):
    r = Reranker.from_config(cfg_of(data))
    assert r.laya_scorer is None
    pool = docs("a", "b")
    with pytest.raises(RerankerExecutionError, match=r"retrieval\.laya\.enabled"):
        r.rerank("q", pool, mode="laya")
    assert [d.rerank_score for d in pool] == [None, None]


def test_the_flag_off_default_mode_laya_refuses_every_call_too():
    r = Reranker.from_config(cfg_of({"retrieval": {"rerank_mode": "laya"}}))
    assert r.mode == "laya"
    with pytest.raises(RerankerExecutionError, match=r"retrieval\.laya\.enabled"):
        r.rerank("q", docs("a"))


def test_a_refused_laya_call_never_falls_back_to_another_reranker(monkeypatch):
    """The failure mode this exists to prevent: an answer built from a ranking nobody
    asked for while the logs still say 'reranked'."""
    r = Reranker("x", mode="cross_encoder")

    def not_this_one():
        raise AssertionError("the cross-encoder was touched")
    monkeypatch.setattr(r, "_get_model", not_this_one)
    pool = docs("alpha beta", "beta gamma")
    with pytest.raises(RerankerExecutionError, match="laya"):
        r.rerank("beta", pool, mode="laya")
    assert [d.rerank_score for d in pool] == [None, None]     # lexical did not run either


def test_enabled_without_a_checkpoint_refuses_naming_the_path(tmp_path, install_laya):
    fake = install_laya()
    r = Reranker.from_config(cfg_of({"retrieval": {"laya": {"enabled": True}}}, tmp_path))
    with pytest.raises(RerankerExecutionError) as exc:
        r.rerank("q", docs("a"), mode="laya")
    assert os.path.abspath(tmp_path / "models" / "laya-noetrix") in str(exc.value)
    assert isinstance(exc.value.__cause__, LayaScorerError)
    assert fake.load_calls == []


def test_enabled_builds_one_lazy_scorer_from_config(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", None)       # building must not import laya
    r = Reranker.from_config(cfg_of({"retrieval": {"laya": {
        "enabled": True, "model_dir": "models/mine", "device": "cuda:1", "batch_size": 4}}}, tmp_path))
    assert isinstance(r.laya_scorer, LayaScorer)
    assert r.laya_scorer.model_dir == os.path.abspath(tmp_path / "models" / "mine")
    assert (r.laya_scorer.device, r.laya_scorer.batch_size) == ("cuda:1", 4)


def test_an_absolute_model_dir_in_config_is_kept(tmp_path):
    target = tmp_path / "elsewhere" / "ckpt"
    r = Reranker.from_config(cfg_of({"retrieval": {"laya": {
        "enabled": True, "model_dir": str(target)}}}, tmp_path / "project"))
    assert r.laya_scorer.model_dir == os.path.abspath(target)


def test_a_bad_laya_block_fails_the_build_when_enabled():
    with pytest.raises(ValueError, match="batch_size"):
        Reranker.from_config(cfg_of({"retrieval": {"laya": {"enabled": True, "batch_size": 0}}}))


def test_laya_mode_sorts_by_p_true_and_sets_rerank_score_on_every_doc():
    scorer = FakeScorer(0.2, 0.9, 0.5, 0.7)
    r = Reranker("x", mode="cross_encoder", laya_scorer=scorer)
    pool = docs("a", "b", "c", "d")
    top = r.rerank("the query", pool, top_k=2, mode="laya")
    assert ids(top) == ["d1", "d3"]                       # 0.9, 0.7
    assert [d.rerank_score for d in pool] == [0.2, 0.9, 0.5, 0.7]    # all four scored, not just the kept two
    assert scorer.calls == [("the query", ["a", "b", "c", "d"])]


def test_ties_keep_the_fused_order():
    """P(true) is rounded to 4 places by laya, so ties near 0 and 1 are normal."""
    r = Reranker("x", mode="laya", laya_scorer=FakeScorer(0.5, 0.5, 0.9, 0.5))
    assert ids(r.rerank("q", docs("a", "b", "c", "d"), top_k=10)) == ["d2", "d0", "d1", "d3"]


def test_the_real_scorer_end_to_end_with_the_stand_in_package(install_laya, checkpoint):
    install_laya()
    r = Reranker("x", mode="laya", laya_scorer=LayaScorer(checkpoint, batch_size=2))
    top = r.rerank("q", docs("a", "b", "c", "d", "e"), top_k=3)
    assert [d.text for d in top] == ["b", "d", "c"]       # 0.9, 0.7, 0.5
    assert top[0].rerank_score == 0.9


def test_the_instruction_is_not_applied_to_laya_and_that_is_warned_once(caplog):
    """Its query goes into the model's own question — the wording it was trained on."""
    scorer = FakeScorer(0.1, 0.8)
    r = Reranker("x", mode="laya", laya_scorer=scorer, instruction="prefer worked procedures")
    with caplog.at_level(logging.WARNING):
        assert r.scoring_query("how do I rotate keys", "laya") == "how do I rotate keys"
        r.rerank("how do I rotate keys", docs("a", "b"))
        r.rerank("how do I rotate keys", docs("a", "b"))
    assert [q for q, _ in scorer.calls] == ["how do I rotate keys"] * 2        # never the joined text
    ignored = [m for m in (rec.message for rec in caplog.records) if "rerank instruction ignored" in m]
    assert len(ignored) == 1 and "laya" in ignored[0]


def test_a_scorer_failure_is_a_typed_reranker_error_with_its_cause():
    r = Reranker("x", mode="laya", laya_scorer=FakeScorer(error=RuntimeError("CUDA out of memory")))
    with pytest.raises(RerankerExecutionError, match=r"laya.*2 candidate\(s\).*CUDA out of memory") as exc:
        r.rerank("q", docs("a", "b"))
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_a_misaligned_score_list_is_an_error_and_leaves_the_docs_unscored():
    r = Reranker("x", mode="laya", laya_scorer=FakeScorer(0.9, 0.1))
    pool = docs("a", "b", "c")
    with pytest.raises(RerankerExecutionError, match=r"2 score\(s\) for 3 candidate\(s\)"):
        r.rerank("q", pool)
    assert [d.rerank_score for d in pool] == [None, None, None]


def test_an_empty_pool_scores_nothing_in_laya_mode():
    scorer = FakeScorer()
    assert Reranker("x", mode="laya", laya_scorer=scorer).rerank("q", []) == []
    assert scorer.calls == []


# ---- the pipeline: the echo, the cold flag, and the gate on laya's probabilities ----

class _Hyde:
    @staticmethod
    def expand_with_info(question, enabled=None):
        return question, "off"

    @staticmethod
    def code_intent_signal(question):
        return None


class _Retriever:
    """Fresh docs on every call: the reranker writes scores onto them."""
    dense_top_k = sparse_top_k = 20
    omnisearch = None
    hype_enabled = False
    lane_weights: dict = {}
    metadata_boost = True

    def __init__(self, texts):
        self.texts = texts

    def retrieve(self, query, **kwargs):
        return [RetrievedDoc(id=i, text=t, metadata={}) for i, t in self.texts.items()]

    def _get_collection(self):
        return object()


TEXTS = {"x": "a", "y": "b", "z": "c"}                  # ids -> passage text (P(true) 0.2, 0.9, 0.5)


def _pipeline(reranker, **parts):
    return RAGPipeline(retriever=_Retriever(TEXTS), reranker=reranker, hyde=_Hyde(),
                       generator=None, rerank_top_k=10, **parts)


def test_search_in_laya_mode_orders_by_p_true_and_the_echo_says_the_instruction_was_not_applied():
    reranker = Reranker("x", mode="cross_encoder", laya_scorer=FakeScorer(0.2, 0.9, 0.5),
                        instruction="prefer worked procedures")
    top, info = _pipeline(reranker).search("q", rerank="laya")
    assert ids(top) == ["y", "z", "x"]
    assert info["rerank_mode"] == "laya"
    # like lexical and none: the instruction was asked for (configured) but did not apply
    assert info["rerank_instruction"] is None and info["rerank_instruction_applied"] is False
    # reranker_model / reranker_max_length describe the CROSS-ENCODER; laya's are its checkpoint's own
    assert info["reranker_model"] is None and info["reranker_max_length"] is None


def test_the_cold_flag_counts_the_laya_checkpoint_load(install_laya, checkpoint):
    """Loading ~2 GB on the first laya call is a spike the latency numbers must flag."""
    install_laya()
    reranker = Reranker("x", mode="laya", laya_scorer=LayaScorer(checkpoint))
    rag = _pipeline(reranker)
    assert rag.search("q")[1]["cold"] is True              # the checkpoint loaded during this call
    assert rag.search("q")[1]["cold"] is False


def test_a_pipeline_whose_reranker_has_no_laya_scorer_never_reports_cold():
    assert _pipeline(Reranker("x", mode="none")).search("q")[1]["cold"] is False


def test_the_rerank_score_gate_cuts_on_laya_probabilities():
    reranker = Reranker("x", mode="laya", laya_scorer=FakeScorer(0.2, 0.9, 0.5))
    gate = RelevanceGate(enabled=True, threshold=0.5)
    top, info = _pipeline(reranker, gate=gate).search("q")
    assert ids(top) == ["y", "z"]                           # 0.9 and 0.5 clear 0.5; 0.2 does not
    assert info["gate"]["scorer"] == "rerank_score" and info["gate"]["dropped"] == 1


# ---- the API: the refusal is a readable 200, never a 500 or a silent switch ----

def _client(monkeypatch, rag):
    monkeypatch.setitem(S._STATE, "rag", rag)
    monkeypatch.setattr(S, "_HISTORY", deque(maxlen=50))
    return TestClient(S.app)               # never entered: the lifespan would build the real pipeline


def test_search_with_rerank_laya_and_the_flag_off_is_a_200_naming_the_flag(monkeypatch):
    rag = _pipeline(Reranker.from_config(cfg_of({"retrieval": {"rerank_mode": "none"}})))
    client = _client(monkeypatch, rag)
    r = client.post("/search", json={"q": "x", "rerank": "laya"})
    assert r.status_code == 200
    body = r.json()
    assert body["error"].startswith("Reranking failed:") and "retrieval.laya.enabled" in body["error"]
    assert body["results"] == [] and body["retrieval"] == {}
    # and the same pipeline still answers when asked for a mode that works
    ok = client.post("/search", json={"q": "x", "rerank": "none"}).json()
    assert "error" not in ok and len(ok["results"]) == 3


def test_query_with_rerank_laya_and_the_flag_off_is_the_same_readable_error(monkeypatch):
    rag = _pipeline(Reranker.from_config(cfg_of({})))
    client = _client(monkeypatch, rag)
    for retrieve_only in (True, False):
        r = client.post("/query", json={"q": "x", "rerank": "laya", "retrieve_only": retrieve_only})
        assert r.status_code == 200
        body = r.json()
        assert body["confidence"] == "ERROR"
        assert body["answer"].startswith("Reranking failed:") and "retrieval.laya.enabled" in body["answer"]


def test_compare_reports_a_laya_branch_as_a_reranking_failure_and_its_sibling_still_answers(monkeypatch):
    """/compare/options offers a branch per RERANK_MODES entry, laya included."""
    client = _client(monkeypatch, _pipeline(Reranker.from_config(cfg_of({}))))
    r = client.post("/compare", json={"q": "x", "branches": [
        {"id": "laya", "rerank": "laya"}, {"id": "fused", "rerank": "none"}]})
    assert r.status_code == 200
    bad, fine = r.json()["branches"]
    assert bad["retrieval_error"] is True
    assert bad["error"].startswith("Reranking failed:") and "retrieval.laya.enabled" in bad["error"]
    assert fine["error"] is None and len(fine["sources"]) == 3


def test_search_with_rerank_laya_and_an_enabled_flag_but_no_checkpoint_names_the_path(monkeypatch, tmp_path):
    reranker = Reranker.from_config(cfg_of({"retrieval": {"laya": {"enabled": True}}}, tmp_path))
    client = _client(monkeypatch, _pipeline(reranker))
    body = client.post("/search", json={"q": "x", "rerank": "laya"}).json()
    assert body["error"].startswith("Reranking failed:")
    assert os.path.abspath(tmp_path / "models" / "laya-noetrix") in body["error"]


def test_the_schema_lists_laya_and_says_it_is_experimental_and_ignores_the_instruction(monkeypatch):
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(presets={}))
    contract = S.schema()
    assert "laya" in contract["rerank_modes"]
    for endpoint in ("POST /search", "POST /query"):
        body = contract["endpoints"][endpoint]["body"]
        assert "laya" in body["rerank"] and "EXPERIMENTAL" in body["rerank"]
        assert "lexical|none|laya" in body["rerank_instruction"]


# ---- the console: the flag is a registry row and an editable setting ----

def test_laya_rerank_is_in_the_consoles_experimental_list(management_module):
    payload = management_module.settings()
    rows = {f["id"]: f for f in payload["experimental"]}
    row = rows["laya_rerank"]
    assert row["key"] == "retrieval.laya.enabled" and row["on"] is False
    # the same tone as the graph row: what it is, why it is off, what it leaves alone, where it shows
    assert "unmeasured" in row["why_off"] and "§11" in row["why_off"]
    spec = payload["editable"]["retrieval.laya.enabled"]
    assert spec["kind"] == "enum" and spec["values"] == ["true", "false"]
    assert "laya" in payload["editable"]["retrieval.rerank_mode"]["values"]


def test_the_console_writes_the_laya_flag_and_only_that_line(management_module, tmp_path):
    target = tmp_path / "probe.yaml"
    target.write_text((ROOT / "config.example.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    written = management_module._persist_section_keys(target, {"retrieval.laya.enabled": "true"})
    assert written == ["retrieval.laya.enabled"]
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data["retrieval"]["laya"]["enabled"] is True
    assert data["retrieval"]["relevance_gate"]["enabled"] is False      # a neighbouring `enabled:` untouched
    assert data["retrieval"]["hyde_cache"]["enabled"] is True
    assert data["graph"]["enabled"] is False


# ---- the shipped configs and the bench ----

@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_laya_ships_off_with_the_documented_defaults_in_both_configs(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"no {name} in this checkout")
    # Parsed directly: load_config() also reconfigures the parser's global taxonomy maps.
    cfg = cfg_of(yaml.safe_load(path.read_text(encoding="utf-8-sig")))
    assert cfg.get("retrieval.laya") == {
        "enabled": False, "model_dir": "models/laya-noetrix", "device": "cpu", "batch_size": 16}
    assert Reranker.from_config(cfg).laya_scorer is None      # off: nothing is even constructed


def test_the_laya_rerank_bench_config_resolves_to_search_keywords():
    [(name, overrides)] = resolve("laya-rerank", ROOT / "eval" / "configs.yaml")
    assert name == "laya-rerank"
    assert overrides["rerank"] == "laya" and overrides["top_k"] == 10
    import inspect
    assert set(overrides) <= set(inspect.signature(RAGPipeline.search).parameters)
