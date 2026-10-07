"""Per-call lane selection (`lanes`) and the metadata-boost override.

Fully offline: the retriever's lane searches are stubbed, so no index, model or
Chroma client is built.

`lanes` has to RESTRICT and never force. A lane outside the list must not run
at all (an ablation that merely down-weighted it would still be measuring it),
and a conditional lane (code, scope, omnisearch, hype) still needs its own
trigger. An unknown name is an ERROR: a typo that silently ran every lane would
make an ablation measure the full pipeline and call it a baseline.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.pipeline import RAGPipeline
from src.retrieval.retriever import LANES, HybridRetriever
from src.utils.timing import StageTrace


# (copies of tests/test_lane_weights.py's helper: private test helpers are not
# imported across files)
def _retriever():
    r = HybridRetriever.__new__(HybridRetriever)
    r.embedder = None
    r.rrf_k = 60
    r.dense_top_k = r.sparse_top_k = 20
    r.metadata_boost = False
    r.code_file_types = ["py"]
    r.omnisearch = None
    r.hype_enabled = False
    r.hype_top_k = 15
    r.lane_weights = {}
    return r


def _stub_counting(r, monkeypatch, dense, sparse):
    calls = {"dense": 0, "sparse": 0, "embed": 0}
    rows = lambda ids: [(i, f"text {i}", {"file_type": "note"}) for i in ids]

    def dense_search(qvec, k, where=None):
        calls["dense"] += 1
        return rows(dense)

    def sparse_search(q, k, predicate=None):
        calls["sparse"] += 1
        return rows(sparse)

    def embed(t):
        calls["embed"] += 1
        return [0.0]

    monkeypatch.setattr(r, "_dense_search", dense_search)
    monkeypatch.setattr(r, "_sparse_search", sparse_search)
    monkeypatch.setattr(r, "embedder", type("E", (), {"embed_query": staticmethod(embed)})())
    return calls


# ---- the retriever ----

def test_sparse_only_runs_no_dense_and_no_embedding(monkeypatch):
    r = _retriever()
    calls = _stub_counting(r, monkeypatch, dense=["a"], sparse=["c", "b"])
    out = r.retrieve("q", lanes=["sparse"])
    assert [d.id for d in out] == ["c", "b"]
    assert calls == {"dense": 0, "sparse": 1, "embed": 0}


def test_none_keeps_both_always_on_lanes(monkeypatch):
    r = _retriever()
    calls = _stub_counting(r, monkeypatch, dense=["a"], sparse=["b"])
    r.retrieve("q")
    assert calls["dense"] == 1 and calls["sparse"] == 1


def test_lanes_never_force_a_conditional_lane(monkeypatch):
    r = _retriever()
    calls = _stub_counting(r, monkeypatch, dense=["a"], sparse=["b"])
    out = r.retrieve("q", lanes=["dense_code"])        # no boost_code
    assert out == [] and calls == {"dense": 0, "sparse": 0, "embed": 0}


@pytest.mark.parametrize("bad", [[], ["bogus"], ["dense", "nope"]])
def test_bad_lane_sets_raise(monkeypatch, bad):
    r = _retriever()
    _stub_counting(r, monkeypatch, dense=["a"], sparse=["b"])
    with pytest.raises(ValueError):
        r.retrieve("q", lanes=bad)


def test_the_error_names_the_offending_lane_and_the_real_ones(monkeypatch):
    r = _retriever()
    _stub_counting(r, monkeypatch, dense=["a"], sparse=["b"])
    with pytest.raises(ValueError) as exc_info:
        r.retrieve("q", lanes=["dense", "bogus"])
    message = str(exc_info.value)
    assert "bogus" in message and "sparse" in message


def test_metadata_boost_override_both_ways(monkeypatch):
    r = _retriever()
    _stub_counting(r, monkeypatch, dense=["a", "b"], sparse=["a", "b"])
    monkeypatch.setattr(r, "_dense_search", lambda qvec, k, where=None: [
        ("a", "t", {"domain": "stats"}), ("b", "t", {"domain": "time series"})])
    monkeypatch.setattr(r, "_sparse_search", lambda q, k, predicate=None: [
        ("a", "t", {"domain": "stats"}), ("b", "t", {"domain": "time series"})])
    r.metadata_boost = False
    forced = r.retrieve("arima in time series", metadata_boost=True)
    assert forced[0].id == "b" and forced[0].debug.get("metadata_boost") == "time series"
    r.metadata_boost = True
    off = r.retrieve("arima in time series", metadata_boost=False)
    assert all("metadata_boost" not in d.debug for d in off)


def _all_lanes_eligible(r, monkeypatch):
    """Stub every lane and return the kwargs that make each conditional one
    eligible to run, so a test can tell 'not selected' from 'not triggered'."""
    rows = lambda tag: [(f"{tag}1", "t", {"file_type": "py"})]
    monkeypatch.setattr(r, "_dense_search", lambda qvec, k, where=None: rows("d"))
    monkeypatch.setattr(r, "_sparse_search", lambda q, k, predicate=None: rows("s"))
    monkeypatch.setattr(r, "_dense_scope_search", lambda qvec, k, scope: rows("ds"))
    monkeypatch.setattr(r, "_hype_search", lambda qvec, k: rows("h"))
    monkeypatch.setattr(r, "embedder", type("E", (), {
        "embed_query": staticmethod(lambda t: [0.0])})())
    r.omnisearch = SimpleNamespace(enabled=False, lane=lambda q: rows("o"))
    return dict(boost_code=True, scope=SimpleNamespace(matches=lambda m: True),
                omnisearch=True, hype=True)


def test_without_lanes_every_eligible_lane_runs(monkeypatch):
    r = _retriever()
    eligible = _all_lanes_eligible(r, monkeypatch)
    tr = StageTrace()
    r.retrieve("q", trace=tr, **eligible)
    assert set(tr.lanes) == set(LANES)


@pytest.mark.parametrize("lane", LANES)
def test_each_lane_can_be_selected_alone(monkeypatch, lane):
    """Also the drift guard: a lane added to LANES (or to retrieve()) without
    its `lanes` gate would run alongside the one that was asked for."""
    r = _retriever()
    eligible = _all_lanes_eligible(r, monkeypatch)
    tr = StageTrace()
    r.retrieve("q", lanes=[lane], trace=tr, **eligible)
    assert set(tr.lanes) == {lane}
    # the query embedding is only paid for when a lane that uses it is left
    assert ("embed" in tr.ms) == (lane in {"dense", "dense_code", "dense_scope", "hype"})


# ---- the pipeline: forwarded and echoed ----

class _Reranker:
    mode = "none"
    model_name = "test/reranker"
    max_length = 512
    instruction = None

    @staticmethod
    def rerank(question, docs, top_k=None, mode=None, instruction=None):
        return docs[:top_k]

    @staticmethod
    def scoring_query(query, mode, instruction=None):
        return query


class _Hyde:
    @staticmethod
    def expand_with_info(question, enabled=None):
        return question, "off"

    @staticmethod
    def code_intent_signal(question):
        return None


class _RecordingRetriever:
    dense_top_k = sparse_top_k = 20
    omnisearch = None
    hype_enabled = False
    lane_weights: dict = {}
    metadata_boost = True            # the configured default, as on HybridRetriever

    def __init__(self):
        self.kwargs = None

    def retrieve(self, query, **kwargs):
        self.kwargs = kwargs
        return []


def _pipeline(retriever, generator=None):
    return RAGPipeline(retriever=retriever, reranker=_Reranker(), hyde=_Hyde(),
                       generator=generator, rerank_top_k=3)


def test_pipeline_forwards_both_controls_and_echoes_them():
    retr = _RecordingRetriever()
    _, info = _pipeline(retr).search("q", lanes=["sparse", "dense"], metadata_boost=False)
    assert retr.kwargs["lanes"] == ["sparse", "dense"]
    assert retr.kwargs["metadata_boost"] is False
    assert info["lanes_requested"] == ["dense", "sparse"]    # sorted: a selection, not an order
    assert info["metadata_boost"] is False


def test_unset_controls_echo_the_configured_behaviour():
    retr = _RecordingRetriever()
    _, info = _pipeline(retr).search("q")
    assert retr.kwargs["lanes"] is None and retr.kwargs["metadata_boost"] is None
    assert info["lanes_requested"] is None
    assert info["metadata_boost"] is True                    # the retriever's configured value


class _CountingHyde:
    """Counts the drafts asked for: each one is an LLM call in the real HyDE."""

    def __init__(self):
        self.drafts = 0

    def expand_with_info(self, question, enabled=None):
        self.drafts += 1
        return question, "off"

    @staticmethod
    def code_intent_signal(question):
        return None


@pytest.mark.parametrize("bad", [[], ["bogus"], ["dense", "nope"]])
def test_a_bad_lane_set_is_refused_before_hyde_spends_an_llm_call(monkeypatch, bad):
    r = _retriever()
    calls = _stub_counting(r, monkeypatch, dense=["a"], sparse=["b"])
    hyde = _CountingHyde()
    rag = RAGPipeline(retriever=r, reranker=_Reranker(), hyde=hyde, generator=None,
                      rerank_top_k=3)
    with pytest.raises(ValueError):
        rag.search("q", lanes=bad)
    assert hyde.drafts == 0 and calls == {"dense": 0, "sparse": 0, "embed": 0}
    rag.search("q", lanes=["sparse"])                     # a good set still drafts, once
    assert hyde.drafts == 1


def test_query_forwards_both_controls_to_search():
    retr = _RecordingRetriever()
    generator = SimpleNamespace(
        generate=lambda q, docs, max_tokens=None: SimpleNamespace(retrieval=None))
    _pipeline(retr, generator).query("q", lanes=["sparse"], metadata_boost=True)
    assert retr.kwargs["lanes"] == ["sparse"] and retr.kwargs["metadata_boost"] is True


# ---- the API ----

def test_search_and_retrieve_only_query_forward_lanes_and_metadata_boost(monkeypatch):
    import serve_api as S
    seen = []

    def record(q, **kwargs):
        seen.append(kwargs)
        return [], {"rerank_mode": "none"}

    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(search=record))
    S.search(S.SearchIn(q="x", lanes=["sparse"], metadata_boost=False))
    S.query(S.QueryIn(q="x", retrieve_only=True, lanes=["sparse"], metadata_boost=False))
    assert len(seen) == 2
    for kwargs in seen:
        assert kwargs["lanes"] == ["sparse"] and kwargs["metadata_boost"] is False


def test_compare_branches_differing_only_by_lanes_do_not_share_evidence():
    import serve_api as S
    a = S.CompareBranchIn(id="a", lanes=["sparse"])
    assert S._retrieval_kwargs(a)["lanes"] == ["sparse"]
    assert S._retrieval_cache_key(a) != S._retrieval_cache_key(
        S.CompareBranchIn(id="b", lanes=["dense"]))
    assert S._retrieval_cache_key(a) == S._retrieval_cache_key(
        S.CompareBranchIn(id="c", lanes=["sparse"]))


def test_a_bad_lane_set_is_reported_the_way_a_bad_lane_weight_is(monkeypatch):
    """Both reach the retriever, raise ValueError and come back as the API's
    ordinary error payload (HTTP 200 carrying `error`) — one convention, not a
    special case for lanes."""
    import serve_api as S
    r = _retriever()
    _stub_counting(r, monkeypatch, dense=["a"], sparse=["b"])
    monkeypatch.setitem(S._STATE, "rag", _pipeline(r))

    bad_lanes = S.search(S.SearchIn(q="x", lanes=["bogus"]))
    bad_weight = S.search(S.SearchIn(q="x", lane_weights={"bogus": 1.0}))
    assert "bogus" in bad_lanes["error"] and "bogus" in bad_weight["error"]
    assert bad_lanes["results"] == [] and set(bad_lanes) == set(bad_weight)

    answer = S.query(S.QueryIn(q="x", retrieve_only=True, lanes=["bogus"]))
    assert answer.confidence == "ERROR" and "bogus" in answer.answer


def test_the_schema_describes_a_bad_lane_weight_the_way_it_is_answered(monkeypatch):
    """/schema once said a bad lane weight "is a 400". Over HTTP an unknown lane or
    a negative weight is a 200 carrying `error` (the API's one convention for a bad
    per-call knob), and a non-number never reaches the pipeline: request validation
    answers 422."""
    from collections import deque
    from fastapi.testclient import TestClient
    import serve_api as S
    r = _retriever()
    _stub_counting(r, monkeypatch, dense=["a"], sparse=["b"])
    monkeypatch.setitem(S._STATE, "rag", _pipeline(r))
    monkeypatch.setattr(S, "_HISTORY", deque(maxlen=50))
    client = TestClient(S.app)          # never entered: the lifespan would build the real pipeline

    negative = client.post("/search", json={"q": "x", "lane_weights": {"dense": -1}})
    assert negative.status_code == 200 and "error" in negative.json()
    heavy = client.post("/search", json={"q": "x", "lane_weights": {"dense": "heavy"}})
    assert heavy.status_code == 422

    said = client.get("/schema").json()["endpoints"]["POST /search"]["body"]["lane_weights"]["errors"]
    assert "400" not in said and "200" in said and "422" in said


def test_schema_documents_both_fields_wherever_it_describes_retrieval(monkeypatch):
    import serve_api as S
    monkeypatch.setitem(S._STATE, "rag", SimpleNamespace(presets={}))
    endpoints = S.schema()["endpoints"]
    bodies = (endpoints["POST /search"]["body"], endpoints["POST /query"]["body"],
              endpoints["POST /compare"]["body"]["branch"])
    for body in bodies:
        assert "lanes" in body and "metadata_boost" in body
    assert set(endpoints["POST /search"]["body"]["lanes"]["lanes"]) == set(LANES)
