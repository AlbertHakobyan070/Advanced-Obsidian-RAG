"""Per-lane RRF fusion weights.

Fully offline: the retriever's lane searches are stubbed, so no index, model
or Chroma client is built.

Three things have to hold, in this order of importance:
  1. An install that sets nothing fuses EXACTLY as it did before weighting
     existed. This is a change to the ranking of every query in the corpus;
     "byte-identical by default" is the only safe way to ship it.
  2. A weight set per call actually REACHES the retriever. serve_api hand
     enumerates its per-call fields, so a new one that is accepted, echoed and
     then ignored is the precise failure that lost rerank_instruction for a
     whole release.
  3. Weights compose lane by lane — config, then preset, then call — instead
     of replacing one another wholesale.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.retrieval.retriever import (
    LANES, HybridRetriever, RetrievedDoc, _clean_lane_weights,
)


def _retriever(**kw):
    r = HybridRetriever.__new__(HybridRetriever)
    r.embedder = None
    r.rrf_k = 60
    r.dense_top_k = r.sparse_top_k = 20
    r.metadata_boost = False
    r.code_file_types = ["py"]
    r.omnisearch = None
    r.hype_enabled = False
    r.hype_top_k = 15
    r.lane_weights = _clean_lane_weights(kw.get("lane_weights"),
                                         "retrieval.lane_weights")
    return r


def _stub_lanes(r, monkeypatch, dense, sparse):
    """Give the two always-on lanes fixed, known rankings."""
    rows = lambda ids: [(i, f"text {i}", {"file_type": "note"}) for i in ids]
    monkeypatch.setattr(r, "_dense_search",
                        lambda qvec, k, where=None: rows(dense))
    monkeypatch.setattr(r, "_sparse_search",
                        lambda q, k, predicate=None: rows(sparse))
    monkeypatch.setattr(r, "embedder", type("E", (), {
        "embed_query": staticmethod(lambda t: [0.0])})())


# ---- 1. the default must not move a single ranking ----

def test_no_weights_configured_reproduces_plain_rrf_exactly(monkeypatch):
    r = _retriever()
    _stub_lanes(r, monkeypatch, dense=["a", "b", "c"], sparse=["c", "b", "a"])
    got = {d.id: d.score for d in r.retrieve("q")}
    expected = {}
    for lane in (["a", "b", "c"], ["c", "b", "a"]):
        for rank, cid in enumerate(lane):
            expected[cid] = expected.get(cid, 0.0) + 1.0 / (60 + rank)
    assert got == pytest.approx(expected)


def test_all_lanes_at_one_is_identical_to_unset(monkeypatch):
    plain = _retriever()
    _stub_lanes(plain, monkeypatch, ["a", "b"], ["b", "a"])
    a = {d.id: d.score for d in plain.retrieve("q")}

    explicit = _retriever(lane_weights={lane: 1.0 for lane in LANES})
    _stub_lanes(explicit, monkeypatch, ["a", "b"], ["b", "a"])
    b = {d.id: d.score for d in explicit.retrieve("q")}
    assert a == pytest.approx(b)


# ---- the weighting itself ----

def test_a_lane_weight_scales_only_that_lanes_contribution(monkeypatch):
    r = _retriever(lane_weights={"sparse": 2.0})
    _stub_lanes(r, monkeypatch, dense=["a"], sparse=["b"])
    by_id = {d.id: d.score for d in r.retrieve("q")}
    assert by_id["a"] == pytest.approx(1.0 / 60)
    assert by_id["b"] == pytest.approx(2.0 / 60)


def test_weighting_can_flip_the_fused_order(monkeypatch):
    """The whole point: a doc the keyword lane loves can be made to outrank
    one the vector lane loves, without touching either lane's pool."""
    # "vec" tops the vector lane; "kw" is only SECOND in the keyword lane, so
    # unweighted fusion puts vec first by a clear margin (1/60 vs 1/61).
    dense, sparse = ["vec", "filler"], ["other", "kw"]
    flat = _retriever()
    _stub_lanes(flat, monkeypatch, dense, sparse)
    assert [d.id for d in flat.retrieve("q")][0] == "vec"

    heavy = _retriever(lane_weights={"sparse": 10.0})
    _stub_lanes(heavy, monkeypatch, dense, sparse)
    order = [d.id for d in heavy.retrieve("q")]
    assert order[0] == "other" and order[1] == "kw"
    assert order.index("kw") < order.index("vec"), \
        "a weighted keyword lane failed to lift its hits above the vector lane's"


def test_zero_keeps_the_candidate_but_drops_its_fusion_score(monkeypatch):
    """0 is not "off": the chunk still reaches the reranker, which is the
    documented behaviour and the reason turning a LANE off is a different
    control."""
    r = _retriever(lane_weights={"sparse": 0.0})
    _stub_lanes(r, monkeypatch, dense=["a"], sparse=["only_sparse"])
    out = {d.id: d.score for d in r.retrieve("q")}
    assert "only_sparse" in out
    assert out["only_sparse"] == pytest.approx(0.0)


def test_per_call_weights_merge_lane_by_lane_over_the_configured_ones(monkeypatch):
    r = _retriever(lane_weights={"dense": 3.0, "sparse": 5.0})
    _stub_lanes(r, monkeypatch, dense=["a"], sparse=["b"])
    out = {d.id: d.score for d in r.retrieve("q", lane_weights={"sparse": 1.0})}
    assert out["a"] == pytest.approx(3.0 / 60), "the untouched lane moved"
    assert out["b"] == pytest.approx(1.0 / 60)


# ---- validation: never a silent no-op ----

def test_an_unknown_lane_name_is_an_error():
    with pytest.raises(ValueError) as exc_info:
        _clean_lane_weights({"spars": 2.0}, "retrieval.lane_weights")
    assert "spars" in str(exc_info.value)
    # and the message names the real lanes
    assert "sparse" in str(exc_info.value)


def test_negative_and_non_numeric_weights_are_errors():
    with pytest.raises(ValueError):
        _clean_lane_weights({"dense": -1}, "lane_weights")
    with pytest.raises(ValueError):
        _clean_lane_weights({"dense": "heavy"}, "lane_weights")
    with pytest.raises(ValueError):
        _clean_lane_weights(["dense"], "lane_weights")


def test_empty_and_missing_weights_are_simply_no_weights():
    assert _clean_lane_weights(None, "x") == {}
    assert _clean_lane_weights({}, "x") == {}


def test_every_lane_retrieve_can_build_is_a_known_lane_name():
    """LANES is the single source of truth for config, the API schema and the
    weighting. If retrieve() grows a lane and this list doesn't, that lane
    silently becomes unweightable."""
    import inspect
    from src.retrieval import retriever as mod
    src = inspect.getsource(mod.HybridRetriever.retrieve)
    built = set(re.findall(r'lanes\.append\(\s*\(\s*"([a-z_]+)"', src))
    built |= set(re.findall(r'\(\s*"([a-z_]+)",\s*self\._\w+search', src))
    assert built, "could not find any lane construction to check"
    assert built <= set(LANES), f"lanes missing from LANES: {built - set(LANES)}"


# ---- 2. the serve_api trap: accepted, echoed, then silently ignored ----

def test_lane_weights_is_listed_as_a_per_call_retrieval_field():
    """serve_api hand enumerates the per-call knobs it forwards. A field the
    models accept but this tuple omits is dropped without a word — exactly how
    rerank_instruction was lost in session 20. The lane-shaped controls added
    after lane_weights (`lanes`, `metadata_boost`) are pinned the same way, and
    so is the relevance gate's pair (`gate`, `gate_threshold`)."""
    import serve_api
    for field in ("lane_weights", "lanes", "metadata_boost", "gate", "gate_threshold"):
        assert field in serve_api._RETRIEVAL_FIELDS, field
        for model in (serve_api.QueryIn, serve_api.SearchIn,
                      serve_api.CompareBranchIn):
            assert field in model.model_fields, f"{model.__name__}.{field}"


def test_every_retrieval_field_survives_the_trip_to_search_kwargs():
    """The general form of the same bug: whatever _RETRIEVAL_FIELDS lists must
    be extractable from the request model and land in the kwargs."""
    import serve_api
    body = serve_api.SearchIn(q="x", lane_weights={"sparse": 1.5},
                              lanes=["sparse"], metadata_boost=False)
    kwargs = serve_api._retrieval_kwargs(body)
    assert kwargs["lane_weights"] == {"sparse": 1.5}
    assert kwargs["lanes"] == ["sparse"] and kwargs["metadata_boost"] is False
    assert set(kwargs) == set(serve_api._RETRIEVAL_FIELDS)


def test_pipeline_search_accepts_every_field_serve_api_forwards():
    """_retrieval_kwargs is splatted straight into RAGPipeline.search — a name
    mismatch is a TypeError at request time, in production, not here."""
    import inspect
    import serve_api
    from src.pipeline import RAGPipeline
    accepted = set(inspect.signature(RAGPipeline.search).parameters)
    missing = set(serve_api._RETRIEVAL_FIELDS) - accepted
    assert not missing, f"search() cannot accept: {missing}"
