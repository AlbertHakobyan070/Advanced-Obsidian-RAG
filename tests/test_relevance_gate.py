"""The relevance gate: a cutoff on the reranked list, applied before generation.

Fully offline. The retriever, HyDE, the Laya scorer and the LLM are stand-ins;
the gate, the pipeline, the real `none`/`lexical` rerank modes, the generator's
abstain path and the API are the real ones.

What is pinned, in the order a request meets it:
  * the gate is OFF unless asked for, and off means untouched;
  * `rerank_score` keeps exactly the docs at or above the threshold;
  * per call beats config, and a per-call threshold alone is enough to turn the
    gate on (a threshold that did nothing would be a knob silently ignored);
  * a half-configured gate is an ERROR — at build for the config, at call time
    for a per-call request — never a pass-through, and never a quiet switch to
    a different scorer;
  * nothing passing means abstaining: the generator's fixed LOW answer, with no
    LLM call and the echo saying why.
"""
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

import serve_api as S
from eval.bench.configs import resolve
from src.generation.generator import Generator
from src.pipeline import RAGPipeline
from src.retrieval.relevance_gate import GATE_SCORERS, RelevanceGate
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]


# ---- helpers ----

def cfg_of(data):
    return Config(data, ROOT)


def build_gate(laya_scorer=None, **block):
    """A gate from a `retrieval.relevance_gate` block, built the way the pipeline builds it."""
    cfg = cfg_of({"retrieval": {"relevance_gate": block}})
    return RelevanceGate.from_config(cfg, reranker=None, laya_scorer=laya_scorer)


def scored(*scores):
    """Docs in reranked order, each carrying the score a reranker would have set
    (None = rerank mode `none`, which scores nothing)."""
    return [RetrievedDoc(id=f"d{i}", text=f"passage {i}", metadata={}, rerank_score=s)
            for i, s in enumerate(scores)]


def ids(docs):
    return [d.id for d in docs]


class FakeLaya:
    """P(relevant) from a fixed list, in the order the texts arrive; records every call."""

    def __init__(self, *probabilities):
        self.probabilities = list(probabilities)
        self.calls = []

    def score(self, query, texts):
        self.calls.append((query, list(texts)))
        return list(self.probabilities)


# ---- off means untouched ----

def test_an_absent_or_empty_block_is_the_shipped_default():
    for cfg in (cfg_of({}),                                              # no block at all
                cfg_of({"retrieval": {"relevance_gate": None}}),         # a block with every key commented out
                cfg_of({"retrieval": {"relevance_gate": {}}})):
        gate = RelevanceGate.from_config(cfg, None)
        assert (gate.enabled, gate.scorer, gate.threshold, gate.laya_scorer) == (
            False, "rerank_score", None, None)


@pytest.mark.parametrize("config, per_call", [
    ({}, {}),                                                      # the shipped default
    ({"enabled": False, "threshold": 5.0}, {}),                    # configured off: the threshold alone does nothing
    ({"enabled": True, "threshold": 5.0}, {"enabled": False}),     # per-call False beats a configured-on gate
])
def test_a_gate_that_is_off_returns_the_docs_unchanged(config, per_call):
    # rerank mode `none` leaves rerank_score unset; an OFF gate must not care
    docs = scored(None, None, None)
    kept, info = build_gate(**config).apply("q", docs, **per_call)
    assert kept == docs and ids(kept) == ["d0", "d1", "d2"]
    assert info == {"enabled": False}
    assert all("gate_score" not in d.debug for d in docs)             # nothing was scored


# ---- the rerank_score scorer ----

def test_rerank_score_keeps_exactly_the_docs_at_or_above_the_threshold():
    docs = scored(2.0, 0.5, 0.25, 1.0, -1.0)
    kept, info = build_gate(enabled=True, threshold=0.5).apply("q", docs)
    # d1 sits exactly AT the cutoff and stays. The input order is the reranker's
    # and stays too: d3 outscores d1, so a re-sort by score would show here.
    assert ids(kept) == ["d0", "d1", "d3"]
    assert info == {"enabled": True, "scorer": "rerank_score", "threshold": 0.5,
                    "kept": 3, "dropped": 2, "abstained": False}
    # Every scored doc carries its score, the dropped ones too, so a run can be
    # re-thresholded offline.
    assert [d.debug["gate_score"] for d in docs] == [2.0, 0.5, 0.25, 1.0, -1.0]


def test_an_integer_threshold_from_yaml_is_fine():
    assert build_gate(enabled=True, threshold=1).threshold == 1.0


def test_nothing_passing_is_an_abstention():
    kept, info = build_gate(enabled=True, threshold=10.0).apply("q", scored(2.0, 0.5))
    assert kept == []
    assert (info["kept"], info["dropped"], info["abstained"]) == (0, 2, True)


# ---- per call beats config ----

def test_a_per_call_threshold_alone_turns_a_disabled_gate_on():
    kept, info = build_gate().apply("q", scored(2.0, -1.0), threshold=0.0)   # config: off, no threshold
    assert ids(kept) == ["d0"]
    assert info["enabled"] is True and info["threshold"] == 0.0


def test_gate_true_alone_uses_the_configured_threshold():
    kept, info = build_gate(threshold=0.5).apply("q", scored(2.0, 0.25), enabled=True)
    assert ids(kept) == ["d0"] and info["threshold"] == 0.5


def test_a_per_call_threshold_beats_the_configured_one():
    docs = scored(2.0, 0.75, 0.25)
    gate = build_gate(enabled=True, threshold=0.5)
    assert ids(gate.apply("q", docs)[0]) == ["d0", "d1"]
    kept, info = gate.apply("q", docs, threshold=1.0)
    assert ids(kept) == ["d0"] and info["threshold"] == 1.0


def test_gate_on_with_no_threshold_anywhere_names_the_knob_and_the_config_key():
    with pytest.raises(ValueError) as e:
        build_gate().apply("q", scored(2.0), enabled=True)
    assert "gate_threshold" in str(e.value)
    assert "retrieval.relevance_gate.threshold" in str(e.value)


# ---- a half-configured gate is an error ----

@pytest.mark.parametrize("block", [{"enabled": True}, {"enabled": True, "threshold": None}])
def test_enabled_without_a_threshold_is_a_config_error(block):
    with pytest.raises(ValueError, match=r"retrieval\.relevance_gate\.threshold"):
        build_gate(**block)


def test_an_unknown_scorer_is_rejected_naming_the_real_ones():
    assert GATE_SCORERS == ("rerank_score", "laya")
    for enabled in (False, True):                  # a typo in a gate that is off is still a typo
        with pytest.raises(ValueError) as e:
            build_gate(enabled=enabled, scorer="cross_encoder", threshold=0.0)
        message = str(e.value)
        assert "cross_encoder" in message and all(s in message for s in GATE_SCORERS)


def test_a_block_that_is_not_a_mapping_is_a_readable_error():
    # `relevance_gate: true` is the natural mistake for a flag-shaped feature
    with pytest.raises(ValueError, match="mapping"):
        RelevanceGate.from_config(cfg_of({"retrieval": {"relevance_gate": True}}), None)


@pytest.mark.parametrize("bad", ["0.5", True, [0.5], float("nan"), float("inf"), float("-inf")])
def test_a_threshold_that_is_not_a_finite_number_is_rejected(bad):
    # NaN or +inf would drop every doc: a gate that abstains on everything and
    # looks like it works.
    with pytest.raises(ValueError, match=r"retrieval\.relevance_gate\.threshold"):
        build_gate(enabled=True, threshold=bad)
    with pytest.raises(ValueError, match="gate_threshold"):
        build_gate().apply("q", scored(1.0), threshold=bad)


# ---- rerank mode `none` ----

def test_the_rerank_score_scorer_refuses_unscored_docs_instead_of_passing_them():
    docs = scored(2.0, None)             # `none` truncates the fused order and scores nothing
    with pytest.raises(ValueError) as e:
        build_gate(enabled=True, threshold=0.0).apply("q", docs)
    message = str(e.value)
    assert "rerank_score" in message
    assert all(mode in message for mode in ("cross_encoder", "http", "lexical"))


# ---- the laya scorer ----

def test_an_enabled_laya_gate_without_a_laya_scorer_fails_the_build():
    with pytest.raises(ValueError, match="Laya"):
        build_gate(enabled=True, scorer="laya", threshold=0.5)


def test_laya_without_a_scorer_is_a_readable_error_never_a_switch_to_rerank_score():
    gate = build_gate(scorer="laya")                  # off in config, so it builds; a per-call threshold turns it on
    docs = scored(9.0, 8.0)                           # would sail through the rerank_score scorer at 0.5
    with pytest.raises(ValueError) as e:
        gate.apply("q", docs, threshold=0.5)
    assert "Laya" in str(e.value) and "retrieval.laya.enabled" in str(e.value)
    assert all("gate_score" not in d.debug for d in docs)             # nothing scored anything


def test_the_laya_scorer_is_called_once_with_the_question_and_every_text():
    laya = FakeLaya(0.9, 0.4, 0.5, 0.1)
    docs = scored(None, None, None, None)             # P(relevant) does not read rerank scores
    gate = build_gate(enabled=True, scorer="laya", threshold=0.5, laya_scorer=laya)
    kept, info = gate.apply("the question", docs)
    assert ids(kept) == ["d0", "d2"]                  # 0.5 is AT the cutoff
    assert laya.calls == [("the question", ["passage 0", "passage 1", "passage 2", "passage 3"])]
    assert [d.debug["gate_score"] for d in docs] == [0.9, 0.4, 0.5, 0.1]
    assert (info["scorer"], info["kept"], info["dropped"]) == ("laya", 2, 2)


def test_a_laya_scorer_that_loses_a_score_is_an_error():
    gate = build_gate(enabled=True, scorer="laya", threshold=0.5, laya_scorer=FakeLaya(0.9, 0.4))
    with pytest.raises(ValueError, match=r"2 score\(s\) for 3 doc\(s\)"):
        gate.apply("q", scored(None, None, None))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_scorer_that_returns_a_non_finite_score_is_an_error_not_a_dropped_doc(bad):
    # NaN compares False against every cutoff, so it would quietly drop its doc:
    # a numeric failure in the scorer must not read as "not relevant".
    gate = build_gate(enabled=True, scorer="laya", threshold=0.5,
                      laya_scorer=FakeLaya(0.9, bad))
    docs = scored(None, None)
    with pytest.raises(ValueError, match="non-finite"):
        gate.apply("q", docs)
    assert all("gate_score" not in d.debug for d in docs)             # nothing half-applied
    with pytest.raises(ValueError, match="non-finite"):               # the rerank_score scorer too
        build_gate(enabled=True, threshold=0.0).apply("q", scored(1.0, bad))


# ---- the pipeline: stand-ins for everything the gate is not ----

class _Hyde:
    @staticmethod
    def expand_with_info(question, enabled=None):
        return question, "off"

    @staticmethod
    def code_intent_signal(question):
        return None


class _ExpandingHyde(_Hyde):
    """Writes a hypothetical answer onto the question, as HyDE does."""

    @staticmethod
    def expand_with_info(question, enabled=None):
        return question + "\n\nA hypothetical answer about something else.", "miss"


class _Retriever:
    """Fresh docs on every call: the reranker and the gate write onto them."""
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


class _TableReranker:
    """A cross-encoder stand-in: scores each doc from a table (logits, so negatives
    occur), sorts best first and truncates — the contract of Reranker.rerank."""
    mode = "cross_encoder"
    model_name = "test/reranker"
    max_length = 512
    instruction = None
    instruction_format = "prefix"

    def __init__(self, table):
        self.table = table

    def rerank(self, question, docs, top_k=None, mode=None, instruction=None):
        for d in docs:
            d.rerank_score = self.table[d.id]
        return sorted(docs, key=lambda d: d.rerank_score, reverse=True)[:top_k]

    @staticmethod
    def scoring_query(query, mode, instruction=None):
        return query


class _RecordingParent:
    enabled = True

    def __init__(self):
        self.got = None

    def apply(self, docs):
        self.got = ids(docs)
        return docs, 0, 0


class _RecordingNeighbor:
    enabled = True

    def __init__(self):
        self.got = None

    def apply(self, docs, collection):
        self.got = ids(docs)
        return docs, 0


class _CountingLLM:
    """Answers every generation call with a cited answer and counts the calls —
    the number the abstain path must keep at zero."""
    provider = backend = "openai"
    model = "fake"

    def __init__(self):
        self.calls = 0
        self.prompts = []

    def complete(self, **kwargs):
        self.calls += 1
        self.prompts.append(kwargs["user"])
        return SimpleNamespace(text="Per the notes [1].\n\nCONFIDENCE: HIGH", usage=None)


TABLE = {"a": 1.0, "b": -5.0, "c": 0.0}
TEXTS = {i: f"text of {i}" for i in TABLE}


def _pipeline(reranker=None, texts=TEXTS, hyde=None, generator=None, **parts):
    return RAGPipeline(retriever=_Retriever(texts), reranker=reranker or _TableReranker(TABLE),
                       hyde=hyde or _Hyde(), generator=generator, rerank_top_k=10, **parts)


def test_a_pipeline_without_a_gate_is_off_and_the_echo_says_so():
    top, info = _pipeline().search("q")
    assert ids(top) == ["a", "c", "b"]                  # nothing dropped
    assert info["gate"] == {"enabled": False}
    assert "gate" not in info["timings"]                # no span for a stage that did not run


def test_a_gated_search_drops_what_scores_below_the_threshold_and_reports_it():
    top, info = _pipeline(gate=build_gate(enabled=True, threshold=0.0)).search("q")
    assert ids(top) == ["a", "c"]                       # c sits exactly at the cutoff
    assert info["gate"] == {"enabled": True, "scorer": "rerank_score", "threshold": 0.0,
                            "kept": 2, "dropped": 1, "abstained": False}
    assert "gate" in info["timings"]


def test_per_call_gate_controls_beat_the_pipelines_config():
    gated = _pipeline(gate=build_gate(enabled=True, threshold=0.0))
    assert ids(gated.search("q", gate=False)[0]) == ["a", "c", "b"]
    assert ids(gated.search("q", gate_threshold=0.5)[0]) == ["a"]
    bare = _pipeline()                                  # built with no gate object at all
    assert ids(bare.search("q", gate_threshold=0.5)[0]) == ["a"]
    with pytest.raises(ValueError, match="gate_threshold"):
        bare.search("q", gate=True)


def test_the_gate_runs_before_the_small_to_big_expansion():
    parent, neighbor = _RecordingParent(), _RecordingNeighbor()
    rag = _pipeline(gate=build_gate(enabled=True, threshold=0.0),
                    parent_ctx=parent, neighbor_ctx=neighbor)
    top, _ = rag.search("q")
    # Both expansions only ever see the gated list — and so, by design, what
    # they add (adjacent pages) is never judged by the gate.
    assert parent.got == neighbor.got == ["a", "c"] and ids(top) == ["a", "c"]
    rag.search("q", gate_threshold=100.0)
    assert parent.got == neighbor.got == []


def test_the_gate_scores_the_original_question_not_the_hyde_text():
    laya = FakeLaya(0.9, 0.9, 0.9)
    gate = build_gate(enabled=True, scorer="laya", threshold=0.5, laya_scorer=laya)
    top, _ = _pipeline(hyde=_ExpandingHyde(), gate=gate).search("the question")
    assert [query for query, _ in laya.calls] == ["the question"] and len(top) == 3


def test_gating_a_search_that_scored_nothing_raises_and_the_same_call_ungated_does_not():
    rag = _pipeline(reranker=Reranker("test/reranker", mode="none"),
                    gate=build_gate(enabled=True, threshold=0.0))
    with pytest.raises(ValueError, match="lexical"):
        rag.search("q")
    assert len(rag.search("q", gate=False)[0]) == 3


def test_the_gate_cuts_on_a_real_rerankers_own_scores():
    texts = {"a": "arima forecasting with seasonal terms",
             "b": "a recipe for sourdough bread",
             "c": "forecasting demand with exponential smoothing"}
    rag = _pipeline(reranker=Reranker("test/reranker", mode="lexical"), texts=texts)
    top, info = rag.search("arima forecasting", gate_threshold=0.01)
    assert ids(top) == ["a", "c"]                       # b shares no query term: lexical score 0.0
    assert info["gate"]["dropped"] == 1


# ---- abstaining: no docs, no LLM call ----

def test_query_abstains_without_an_llm_call_when_nothing_passes():
    llm = _CountingLLM()
    rag = _pipeline(gate=build_gate(enabled=True, threshold=100.0),
                    generator=Generator(llm, verify_citations=False))
    answer = rag.query("q")
    assert llm.calls == 0
    assert answer.confidence == "LOW" and answer.citations == [] and answer.sources == []
    assert answer.text == rag.generator.generate("q", []).text       # the generator's own no-docs answer
    assert answer.retrieval["gate"]["abstained"] is True


def test_query_hands_generation_only_the_kept_docs():
    llm = _CountingLLM()
    rag = _pipeline(gate=build_gate(enabled=True, threshold=0.0),
                    generator=Generator(llm, verify_citations=False))
    answer = rag.query("q")
    assert llm.calls == 1 and ids(answer.sources) == ["a", "c"]
    assert "text of b" not in llm.prompts[0]            # the dropped doc never reaches the model
    rag.query("q", gate_threshold=100.0)                # a per-call cutoff rides through query() too
    assert llm.calls == 1


# ---- the pipeline build path ----

def _stub_the_heavy_builders(monkeypatch):
    """RAGPipeline.from_config builds an embedder, an LLM client, the Chroma-backed
    retriever and more — none of it what these tests are about. Stand-ins replace
    those (and the logging bootstrap, which clears the root handlers pytest
    captures through); the gate, reranker and context lanes are the real ones."""
    import src.pipeline as P
    stand_in = SimpleNamespace(from_config=lambda *args, **kwargs: SimpleNamespace())
    for name in ("Embedder", "LLMClient", "HybridRetriever", "HyDE", "Generator", "GraphExpander"):
        monkeypatch.setattr(P, name, stand_in)
    monkeypatch.setattr(P, "configure_logging", lambda **kwargs: None)


def test_the_pipeline_build_refuses_an_enabled_gate_without_a_threshold(monkeypatch):
    _stub_the_heavy_builders(monkeypatch)
    with pytest.raises(ValueError, match=r"retrieval\.relevance_gate\.threshold"):
        RAGPipeline.from_config(cfg_of({"retrieval": {"relevance_gate": {"enabled": True}}}))


def test_the_pipeline_build_wires_the_configured_gate(monkeypatch):
    _stub_the_heavy_builders(monkeypatch)
    on = RAGPipeline.from_config(cfg_of(
        {"retrieval": {"relevance_gate": {"enabled": True, "threshold": -1.5}}}))
    assert (on.gate.enabled, on.gate.scorer, on.gate.threshold) == (True, "rerank_score", -1.5)
    assert RAGPipeline.from_config(cfg_of({})).gate.enabled is False


# ---- the API ----

def _client(monkeypatch, rag):
    monkeypatch.setitem(S._STATE, "rag", rag)
    monkeypatch.setattr(S, "_HISTORY", deque(maxlen=50))        # /query and /search append to it
    return TestClient(S.app)               # never entered: the lifespan would build the real pipeline


def test_gate_and_gate_threshold_reach_rag_search_from_every_endpoint(monkeypatch):
    seen = []

    def record(q, **kwargs):
        seen.append(kwargs)
        return [], {"rerank_mode": "cross_encoder"}

    client = _client(monkeypatch, SimpleNamespace(search=record))
    knobs = {"gate": True, "gate_threshold": 0.25}
    assert client.post("/search", json={"q": "x", **knobs}).status_code == 200
    assert client.post("/query", json={"q": "x", "retrieve_only": True, **knobs}).status_code == 200
    assert client.post("/compare", json={"q": "x", "branches": [
        {"id": "a", **knobs}, {"id": "b"}]}).status_code == 200
    assert [(k["gate"], k["gate_threshold"]) for k in seen] == [(True, 0.25)] * 3 + [(None, None)]


def test_compare_branches_differing_only_by_the_gate_do_not_share_evidence():
    gated = S.CompareBranchIn(id="a", gate=True, gate_threshold=0.0)
    key = S._retrieval_cache_key
    assert key(gated) != key(S.CompareBranchIn(id="b"))
    assert key(gated) != key(S.CompareBranchIn(id="c", gate=True, gate_threshold=1.0))
    assert key(gated) == key(S.CompareBranchIn(id="d", gate=True, gate_threshold=0.0, provider="p"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_gate_threshold_is_finite_on_every_request_shape(bad):
    for build in (lambda: S.SearchIn(q="x", gate_threshold=bad),
                  lambda: S.QueryIn(q="x", gate_threshold=bad),
                  lambda: S.CompareBranchIn(id="a", gate_threshold=bad)):
        with pytest.raises(ValidationError):
            build()


def test_a_non_finite_threshold_over_http_is_a_validation_error(monkeypatch):
    client = _client(monkeypatch, SimpleNamespace(search=lambda q, **kw: ([], {})))
    for text in ("inf", "nan"):
        r = client.post("/search", json={"q": "x", "gate_threshold": text})
        assert r.status_code == 422 and "finite" in r.text


def test_a_gate_error_is_reported_the_way_a_bad_lanes_value_is(monkeypatch):
    """HTTP 200 carrying the text — one convention for every bad per-call knob,
    not a special case for the gate."""
    rag = _pipeline(reranker=Reranker("test/reranker", mode="none"))     # the real `none`: scores nothing
    client = _client(monkeypatch, rag)
    unscored = {"gate": True, "gate_threshold": 0.0}
    no_threshold = {"gate": True}

    for body, needle in ((unscored, "lexical"), (no_threshold, "gate_threshold")):
        r = client.post("/search", json={"q": "x", **body})
        assert r.status_code == 200
        assert needle in r.json()["error"] and r.json()["results"] == []

        for retrieve_only in (True, False):                              # both /query paths
            r = client.post("/query", json={"q": "x", "retrieve_only": retrieve_only, **body})
            assert r.status_code == 200
            assert r.json()["confidence"] == "ERROR"
            assert r.json()["answer"].startswith("Bad request:") and needle in r.json()["answer"]

        r = client.post("/compare", json={"q": "x", "branches": [{"id": "a", **body}, {"id": "b"}]})
        assert r.status_code == 200
        bad, fine = r.json()["branches"]
        assert bad["retrieval_error"] is True
        assert bad["error"].startswith("Bad branch configuration:") and needle in bad["error"]
        assert fine["error"] is None and len(fine["sources"]) == 3     # the sibling branch is unaffected


def test_query_abstains_through_the_api_without_an_llm_call(monkeypatch):
    llm = _CountingLLM()
    rag = _pipeline(generator=Generator(llm, verify_citations=False))
    client = _client(monkeypatch, rag)

    body = client.post("/query", json={"q": "x", "gate": True, "gate_threshold": 100.0}).json()
    assert llm.calls == 0
    assert body["confidence"] == "LOW" and body["citations"] == [] and body["sources"] == []
    assert body["answer"] == rag.generator.generate("x", []).text
    assert body["retrieval"]["gate"] == {"enabled": True, "scorer": "rerank_score", "threshold": 100.0,
                                         "kept": 0, "dropped": 3, "abstained": True}

    # the control: the same request with a cutoff the docs clear does ask the model, once
    body = client.post("/query", json={"q": "x", "gate": True, "gate_threshold": 0.0}).json()
    assert llm.calls == 1 and [s["id"] for s in body["sources"]] == ["a", "c"]
    assert body["retrieval"]["gate"]["abstained"] is False


def test_the_schema_documents_both_fields_wherever_it_describes_retrieval(monkeypatch):
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(presets={}))
    endpoints = S.schema()["endpoints"]
    for body in (endpoints["POST /search"]["body"], endpoints["POST /query"]["body"],
                 endpoints["POST /compare"]["body"]["branch"]):
        assert "gate" in body and "gate_threshold" in body


# ---- the shipped configs ----

@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_the_gate_ships_off_with_no_threshold_in_both_configs(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"no {name} in this checkout")
    # Parsed directly: load_config() also reconfigures the parser's global
    # taxonomy maps, a side effect this test has no business causing.
    cfg = cfg_of(yaml.safe_load(path.read_text(encoding="utf-8-sig")))
    assert cfg.get("retrieval.relevance_gate") == {
        "enabled": False, "scorer": "rerank_score", "threshold": None}
    assert RelevanceGate.from_config(cfg, None).enabled is False      # and the shipped block builds


def test_the_baseline_gate_bench_config_resolves_to_search_keywords():
    [(name, overrides)] = resolve("gate-rerank-score", ROOT / "eval" / "configs.yaml")
    assert name == "gate-rerank-score"
    assert overrides["gate"] is True and overrides["gate_threshold"] == 0.0
    assert overrides["rerank"] == "cross_encoder" and overrides["top_k"] == 10
    # the bench forwards overrides as keyword arguments, so each one must be a search() parameter
    import inspect
    assert set(overrides) <= set(inspect.signature(RAGPipeline.search).parameters)


# ---- the laya scorer, wired (Task 14 part B) ----

def test_the_gate_shares_the_rerankers_laya_scorer():
    """One checkpoint, two roles: the gate takes the reranker's LayaScorer
    instead of loading a second copy (~2 GB) of the same model."""
    shared = FakeLaya(0.9)
    reranker = SimpleNamespace(laya_scorer=shared)
    gate = RelevanceGate.from_config(
        cfg_of({"retrieval": {"relevance_gate": {"enabled": True, "scorer": "laya",
                                                 "threshold": 0.5}}}), reranker)
    assert gate.laya_scorer is shared


def test_the_pipeline_build_hands_the_gate_the_rerankers_laya_scorer(monkeypatch):
    _stub_the_heavy_builders(monkeypatch)
    rag = RAGPipeline.from_config(cfg_of({"retrieval": {
        "laya": {"enabled": True, "model_dir": "models/none-here"},
        "relevance_gate": {"enabled": True, "scorer": "laya", "threshold": 0.5}}}))
    assert rag.reranker.laya_scorer is not None
    assert rag.gate.laya_scorer is rag.reranker.laya_scorer


def test_a_laya_gate_with_laya_switched_off_fails_the_build_naming_the_flag(monkeypatch):
    _stub_the_heavy_builders(monkeypatch)
    with pytest.raises(ValueError, match=r"retrieval\.laya\.enabled"):
        RAGPipeline.from_config(cfg_of({"retrieval": {
            "relevance_gate": {"enabled": True, "scorer": "laya", "threshold": 0.5}}}))


def test_a_per_call_scorer_switches_the_scorer_for_that_call_only():
    laya = FakeLaya(0.9, 0.2)
    gate = build_gate(laya_scorer=laya, enabled=True, threshold=0.0)
    kept, info = gate.apply("q", scored(-3.0, 5.0), scorer="laya", threshold=0.5)
    assert ids(kept) == ["d0"] and info["scorer"] == "laya" and len(laya.calls) == 1
    kept, info = gate.apply("q", scored(-3.0, 5.0))           # next call: the configured scorer
    assert ids(kept) == ["d1"] and info["scorer"] == "rerank_score"


def test_a_per_call_scorer_unlike_the_configured_one_must_bring_its_own_threshold():
    """The configured threshold is in the configured scorer's units (logits for
    rerank_score, P for laya): reusing it for the other scorer would gate on a
    meaningless number."""
    gate = build_gate(laya_scorer=FakeLaya(0.9), enabled=True, threshold=0.0)
    with pytest.raises(ValueError, match="gate_threshold"):
        gate.apply("q", scored(1.0), scorer="laya")


def test_a_per_call_scorer_alone_turns_the_gate_on():
    gate = build_gate(laya_scorer=FakeLaya(0.9, 0.1), scorer="laya", threshold=0.5)
    assert gate.enabled is False
    kept, info = gate.apply("q", scored(None, None), scorer="laya")
    assert ids(kept) == ["d0"] and info["enabled"] is True


@pytest.mark.parametrize("enabled", [None, True, False])
def test_an_unknown_per_call_scorer_is_an_error_even_with_the_gate_off(enabled):
    with pytest.raises(ValueError, match="gate_scorer"):
        build_gate(enabled=True, threshold=0.0).apply("q", scored(1.0), enabled=enabled,
                                                      scorer="bm25")


def test_a_laya_scorer_failure_names_the_gate_and_keeps_the_cause():
    class Broken:
        def score(self, query, texts):
            raise RuntimeError("checkpoint missing at /abs/models/laya-noetrix")

    gate = build_gate(laya_scorer=Broken(), enabled=True, scorer="laya", threshold=0.5)
    with pytest.raises(RuntimeError, match="relevance gate") as err:
        gate.apply("q", scored(1.0))
    assert "checkpoint missing" in str(err.value) and err.value.__cause__ is not None


def test_gate_scorer_reaches_rag_search_and_is_part_of_the_compare_key(monkeypatch):
    seen = []

    def record(q, **kwargs):
        seen.append(kwargs)
        return [], {"rerank_mode": "cross_encoder"}

    client = _client(monkeypatch, SimpleNamespace(search=record))
    knobs = {"gate": True, "gate_scorer": "laya", "gate_threshold": 0.5}
    assert client.post("/search", json={"q": "x", **knobs}).status_code == 200
    assert client.post("/query", json={"q": "x", "retrieve_only": True, **knobs}).status_code == 200
    assert client.post("/compare", json={"q": "x", "branches": [
        {"id": "a", **knobs}, {"id": "b"}]}).status_code == 200
    assert [k["gate_scorer"] for k in seen] == ["laya"] * 3 + [None]
    key = S._retrieval_cache_key
    assert key(S.CompareBranchIn(id="a", gate=True, gate_threshold=0.5, gate_scorer="laya")) != \
        key(S.CompareBranchIn(id="b", gate=True, gate_threshold=0.5))


def test_a_bad_gate_scorer_over_http_is_a_200_with_an_error(monkeypatch):
    def search(q, **kw):
        RelevanceGate().apply(q, scored(1.0), enabled=kw["gate"],
                              threshold=kw["gate_threshold"], scorer=kw["gate_scorer"])
        return [], {}

    client = _client(monkeypatch, SimpleNamespace(search=search))
    r = client.post("/search", json={"q": "x", "gate": True, "gate_threshold": 0.1,
                                      "gate_scorer": "bm25"})
    error = r.json()["error"]
    assert r.status_code == 200 and "'bm25'" in error and "rerank_score" in error


def test_the_gate_laya_bench_config_only_uses_search_parameters():
    import inspect
    [(name, overrides)] = resolve("gate-laya", ROOT / "eval" / "configs.yaml")
    assert overrides["gate_scorer"] == "laya" and overrides["gate"] is True
    assert set(overrides) <= set(inspect.signature(RAGPipeline.search).parameters)
