"""Per-stage timings: the StageTrace itself, the retriever filling one per
lane, and the pipeline echoing timings / lanes_run / cold.

Fully offline: lane searches and every pipeline collaborator are stubbed.
"""
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.generator import Generator
from src.pipeline import RAGPipeline
from src.retrieval.retriever import HybridRetriever, RetrievedDoc
from src.utils.timing import StageTrace


def test_span_records_and_accumulates():
    tr = StageTrace()
    with tr.span("a"):
        time.sleep(0.01)
    with tr.span("a"):
        time.sleep(0.01)
    assert tr.ms["a"] >= 15.0          # two ~10 ms spans summed
    assert set(tr.timings()) == {"a"}
    assert tr.timings()["a"] == round(tr.ms["a"], 2)


def test_span_records_even_when_the_body_raises():
    tr = StageTrace()
    try:
        with tr.span("boom"):
            raise RuntimeError("x")
    except RuntimeError:
        pass
    assert "boom" in tr.ms


# ---- the retriever fills a trace ----
# (copies of tests/test_lane_weights.py's helpers: private test helpers are not
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


def _stub_lanes(r, monkeypatch, dense, sparse):
    """Give the two always-on lanes fixed, known rankings."""
    rows = lambda ids: [(i, f"text {i}", {"file_type": "note"}) for i in ids]
    monkeypatch.setattr(r, "_dense_search",
                        lambda qvec, k, where=None: rows(dense))
    monkeypatch.setattr(r, "_sparse_search",
                        lambda q, k, predicate=None: rows(sparse))
    monkeypatch.setattr(r, "embedder", type("E", (), {
        "embed_query": staticmethod(lambda t: [0.0])})())


def test_retrieve_fills_trace_per_lane(monkeypatch):
    r = _retriever()
    _stub_lanes(r, monkeypatch, dense=["a", "b"], sparse=["b", "c", "d"])
    tr = StageTrace()
    r.retrieve("q", trace=tr)
    assert {"embed", "lane.dense", "lane.sparse", "fuse"} <= set(tr.ms)
    assert tr.lanes == {"dense": 2, "sparse": 3}
    assert "boost" not in tr.ms        # the boost span exists only when the boost ran


def test_retrieve_without_a_trace_still_works(monkeypatch):
    r = _retriever()
    _stub_lanes(r, monkeypatch, dense=["a"], sparse=["b"])
    assert {d.id for d in r.retrieve("q")} == {"a", "b"}


# ---- the pipeline echo ----

class _BareRetriever:
    """No lazily-loaded handle at all, so it can never make a search cold."""
    dense_top_k = sparse_top_k = 20
    omnisearch = None
    hype_enabled = False
    lane_weights: dict = {}
    metadata_boost = True

    def retrieve(self, query, trace=None, **kwargs):
        return []


class _StubRetriever(_BareRetriever):
    """Records into the trace it is handed, and "loads" its collection on the
    first call the way the real lazy loader does."""
    _collection = None

    def retrieve(self, query, trace=None, **kwargs):
        with trace.span("lane.dense"):
            pass
        trace.lanes["dense"] = 2
        self._collection = object()
        return []

    def _get_collection(self):
        return self._collection


class _BareReranker:
    mode = "none"
    model_name = "test/reranker"
    max_length = 512
    instruction = None

    def rerank(self, question, docs, top_k=None, mode=None, instruction=None):
        return docs[:top_k]

    @staticmethod
    def scoring_query(query, mode, instruction=None):
        return query


class _LazyReranker(_BareReranker):
    """The cross-encoder: absent until the first rerank call."""
    _model = None

    def rerank(self, question, docs, top_k=None, mode=None, instruction=None):
        self._model = object()
        return docs[:top_k]


class _StubHyde:
    @staticmethod
    def expand_with_info(question, enabled=None):
        return question, "off"

    @staticmethod
    def code_intent_signal(question):
        return None


def _pipeline(retriever, reranker=None):
    return RAGPipeline(retriever=retriever, reranker=reranker or _BareReranker(),
                       hyde=_StubHyde(), generator=None, rerank_top_k=3)


def test_search_echo_carries_timings_lanes_and_cold():
    _, info = _pipeline(_StubRetriever()).search("q")
    assert {"preset", "scope", "hyde", "lane.dense", "rerank", "total"} <= set(info["timings"])
    assert info["timings"]["total"] >= info["timings"]["rerank"]
    assert info["lanes_run"] == {"dense": 2}       # what the stub recorded in the trace
    assert isinstance(info["cold"], bool)


def test_cold_is_true_only_for_the_call_that_loaded_something():
    rag = _pipeline(_StubRetriever())
    assert rag.search("q")[1]["cold"] is True      # the collection loaded during this call
    assert rag.search("q")[1]["cold"] is False     # already resident


def test_cold_counts_the_cross_encoder_load_too():
    rag = _pipeline(_BareRetriever(), _LazyReranker())
    assert rag.search("q")[1]["cold"] is True
    assert rag.search("q")[1]["cold"] is False


def test_a_pipeline_with_nothing_lazy_reads_warm():
    """Stand-ins without _collection / _bm25_payload / _model never report cold."""
    assert _pipeline(_BareRetriever()).search("q")[1]["cold"] is False


# ---- the generator's own stages ----

class _FakeLLM:
    """Answers the generation call, then (when it is made) the verification call."""
    provider, model, backend = "openai", "fake", "fake"

    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, **kwargs):
        return SimpleNamespace(text=self.replies.pop(0), usage=None)


_ANSWER = "Water boils at 100 C [1].\n\nCONFIDENCE: HIGH"
_VERDICT = '{"overall": "SUPPORTED", "verdicts": [{"citation": 1, "supported": true}]}'
_DOC = RetrievedDoc(id="c1", text="water boils at 100 C", metadata={"filename": "f.md"})


def test_generate_times_generate_parse_and_verify():
    ans = Generator(_FakeLLM([_ANSWER, _VERDICT]), verify_citations=True).generate("q", [_DOC])
    assert set(ans.timings) == {"generate", "parse", "verify"}


def test_generate_has_no_verify_span_when_verification_did_not_run():
    ans = Generator(_FakeLLM([_ANSWER]), verify_citations=False).generate("q", [_DOC])
    assert set(ans.timings) == {"generate", "parse"}


def test_generate_with_no_docs_has_empty_timings_and_never_calls_the_llm():
    ans = Generator(_FakeLLM([]), verify_citations=True).generate("q", [])
    assert ans.timings == {}


# ---- the API: history rows and the /query generation block ----

def test_search_history_row_carries_timings_and_a_sane_duration(monkeypatch):
    import serve_api
    monkeypatch.setattr(serve_api, "_HISTORY", deque(maxlen=50))
    monkeypatch.setitem(serve_api._STATE, "rag", SimpleNamespace(
        search=lambda q, **kw: ([], {"timings": {"total": 3.5}})))
    serve_api.search(serve_api.SearchIn(q="x"))
    row = serve_api._HISTORY[0]
    assert row["timings"] == {"total": 3.5}
    # ms is a perf_counter difference, so it is a small non-negative duration.
    # A t0 still taken from time.time() would make it hugely negative.
    assert 0 <= row["ms"] < 5000


def test_query_reports_generation_timings_and_nests_both_in_history(monkeypatch):
    import serve_api
    monkeypatch.setattr(serve_api, "_HISTORY", deque(maxlen=50))
    ans = SimpleNamespace(text="a", confidence="HIGH", citations=[], sources=[_DOC],
                          usage=None, retrieval=None, timings={"generate": 2.0})
    generator = SimpleNamespace(
        llm=SimpleNamespace(backend="b", provider="p", model="m"),
        generate=lambda q, docs, max_tokens=None: ans)
    monkeypatch.setitem(serve_api._STATE, "rag", SimpleNamespace(
        generator=generator,
        search=lambda q, **kw: ([_DOC], {"timings": {"total": 3.5}})))
    monkeypatch.setattr(serve_api, "_generator_for", lambda provider, model: generator)

    out = serve_api.query(serve_api.QueryIn(q="x"))

    assert out.generation["timings"] == {"generate": 2.0}
    assert out.retrieval["timings"] == {"total": 3.5}
    assert serve_api._HISTORY[0]["timings"] == {
        "retrieval": {"total": 3.5}, "generation": {"generate": 2.0}}
    assert 0 <= serve_api._HISTORY[0]["ms"] < 5000


def test_retrieve_only_query_records_retrieval_timings_and_no_generation(monkeypatch):
    import serve_api
    monkeypatch.setattr(serve_api, "_HISTORY", deque(maxlen=50))
    monkeypatch.setitem(serve_api._STATE, "rag", SimpleNamespace(
        search=lambda q, **kw: ([], {"timings": {"total": 3.5}})))
    serve_api.query(serve_api.QueryIn(q="x", retrieve_only=True))
    assert serve_api._HISTORY[0]["timings"] == {
        "retrieval": {"total": 3.5}, "generation": None}
