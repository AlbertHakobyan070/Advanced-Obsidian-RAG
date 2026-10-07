"""HyDE's disk cache: the key, persistence, hit/miss/error semantics, thread
safety, and that repeated searches really do retrieve with identical text.

Fully offline: the LLM is a fake and every cache lives under tmp_path (never
under data/).
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import RAGPipeline
from src.retrieval.hyde import HyDE
from src.retrieval.hyde_cache import HydeCache
from src.utils.config_loader import Config


class FakeLLM:
    model = "fake-model"

    def __init__(self, text="a hypothetical answer", fail=False):
        self.calls, self.text, self.fail = 0, text, fail

    def complete(self, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("backend down")
        return type("R", (), {"text": self.text})()


def test_key_normalises_whitespace_only():
    k = HydeCache.make_key
    assert k("what  is\nARIMA", "p", "m", 0.3) == k("what is ARIMA", "p", "m", 0.3)
    assert k("what is ARIMA", "p", "m", 0.3) != k("what is arima", "p", "m", 0.3)
    assert k("q", "p", "m", 0.3) != k("q", "p2", "m", 0.3)
    assert k("q", "p", "m", 0.3) != k("q", "p", "m2", 0.3)
    assert k("q", "p", "m", 0.3) != k("q", "p", "m", 0.0)


def test_roundtrip_and_persistence(tmp_path):
    c = HydeCache(tmp_path / "h.sqlite")
    c.put("k1", "text")
    assert c.get("k1") == "text" and c.get("nope") is None
    assert HydeCache(tmp_path / "h.sqlite").get("k1") == "text"


def test_second_identical_query_is_a_hit(tmp_path):
    llm = FakeLLM()
    h = HyDE(llm, enabled=True, cache=HydeCache(tmp_path / "h.sqlite"))
    t1, s1 = h.expand_with_info("explain arima")
    t2, s2 = h.expand_with_info("explain   arima")
    assert (s1, s2) == ("miss", "hit") and llm.calls == 1
    assert t1.endswith("a hypothetical answer") and t2.endswith("a hypothetical answer")


def test_failure_is_not_cached(tmp_path):
    cache = HydeCache(tmp_path / "h.sqlite")
    h = HyDE(FakeLLM(fail=True), enabled=True, cache=cache)
    text, status = h.expand_with_info("explain arima")
    assert text == "explain arima" and status == "error" and len(cache) == 0


def test_an_empty_draft_is_an_error_and_is_not_cached(tmp_path):
    cache = HydeCache(tmp_path / "h.sqlite")
    h = HyDE(FakeLLM(text="   "), enabled=True, cache=cache)
    assert h.expand_with_info("explain arima") == ("explain arima", "error")
    assert len(cache) == 0


def test_code_signal_bypass_never_touches_cache(tmp_path):
    cache = HydeCache(tmp_path / "h.sqlite")
    llm = FakeLLM()
    h = HyDE(llm, enabled=True, skip_signals=["python"], cache=cache)
    assert h.expand_with_info("python code for arima") == ("python code for arima", "bypass")
    assert llm.calls == 0 and len(cache) == 0


def test_expand_keeps_its_old_contract(tmp_path):
    h = HyDE(FakeLLM(), enabled=False)
    assert h.expand("q") == "q"


def test_disabled_reports_off_and_never_calls_the_llm(tmp_path):
    llm = FakeLLM()
    h = HyDE(llm, enabled=True, cache=HydeCache(tmp_path / "h.sqlite"))
    assert h.expand_with_info("q", enabled=False) == ("q", "off")
    assert llm.calls == 0


def test_without_a_cache_the_status_says_so():
    llm = FakeLLM()
    h = HyDE(llm, enabled=True)
    text, status = h.expand_with_info("explain arima")
    assert status == "nocache" and text.endswith("a hypothetical answer")
    h.expand_with_info("explain arima")
    assert llm.calls == 2                       # nothing was kept


def test_a_different_model_does_not_reuse_the_draft(tmp_path):
    llm = FakeLLM()
    h = HyDE(llm, enabled=True, cache=HydeCache(tmp_path / "h.sqlite"))
    assert h.expand_with_info("explain arima")[1] == "miss"
    llm.model = "another-model"
    assert h.expand_with_info("explain arima")[1] == "miss" and llm.calls == 2


def test_put_keeps_the_first_text_and_returns_the_stored_one(tmp_path):
    c = HydeCache(tmp_path / "h.sqlite")
    assert c.put("k", "first") == "first"
    assert c.put("k", "second") == "first"          # the loser of a race gets the winner's text
    assert c.get("k") == "first" and len(c) == 1


class _BothInsideLLM(FakeLLM):
    """Holds each caller inside complete() until the other has arrived, so both
    have already missed the cache when either one writes. Each draft is its own."""

    def __init__(self):
        super().__init__()
        self.both_arrived = threading.Barrier(2, timeout=10)

    def complete(self, **kw):
        self.both_arrived.wait()
        return type("R", (), {"text": f"draft from {threading.current_thread().name}"})()


def test_two_concurrent_misses_retrieve_with_the_same_draft(tmp_path):
    hyde = HyDE(_BothInsideLLM(), enabled=True, cache=HydeCache(tmp_path / "h.sqlite"))
    out = {}

    def ask(name):
        out[name] = hyde.expand_with_info("explain arima")

    threads = [threading.Thread(target=ask, args=(n,), name=n) for n in ("alpha", "beta")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    (text_a, status_a), (text_b, status_b) = out["alpha"], out["beta"]
    assert status_a == status_b == "miss"            # each one paid for a draft...
    assert text_a == text_b                          # ...but only one is used: the one now stored
    again, status = hyde.expand_with_info("explain arima")
    assert status == "hit" and again == text_a       # and that is what later searches replay


def test_concurrent_puts_and_gets(tmp_path):
    c = HydeCache(tmp_path / "h.sqlite")
    errors = []

    def work(i):
        try:
            for j in range(50):
                c.put(f"k{i}-{j}", "t")
                assert c.get(f"k{i}-{j}") == "t"
        except Exception as e:     # collected and asserted below, not swallowed
            errors.append(e)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errors == [] and len(c) == 200


# ---- config wiring ----

def _cfg(tmp_path, **hyde_cache):
    retrieval = {"hyde_cache": hyde_cache} if hyde_cache else {}
    return Config({"retrieval": retrieval}, tmp_path)


def test_from_config_builds_the_cache_only_when_enabled(tmp_path):
    on = HyDE.from_config(
        _cfg(tmp_path, enabled=True, path="cache/h.sqlite"), FakeLLM())
    assert isinstance(on.cache, HydeCache) and on.cache.path == tmp_path / "cache" / "h.sqlite"
    assert HyDE.from_config(_cfg(tmp_path, enabled=False), FakeLLM()).cache is None
    assert HyDE.from_config(_cfg(tmp_path), FakeLLM()).cache is None    # block absent = today's behaviour


# ---- the pipeline: the status is echoed, and repeated searches are identical ----

class _Retriever:
    """Remembers the text it was asked to retrieve with."""
    dense_top_k = sparse_top_k = 20
    omnisearch = None
    hype_enabled = False
    lane_weights: dict = {}
    metadata_boost = True

    def __init__(self):
        self.seen = []

    def retrieve(self, query, **kwargs):
        self.seen.append(query)
        return []


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


class _DriftingLLM(FakeLLM):
    """A different draft on every call, like a real temperature-0.3 model."""

    def complete(self, **kw):
        self.calls += 1
        return type("R", (), {"text": f"draft number {self.calls}"})()


def test_repeated_searches_retrieve_with_identical_text_and_echo_the_status(tmp_path):
    retriever = _Retriever()
    rag = RAGPipeline(retriever=retriever, reranker=_Reranker(), generator=None,
                      hyde=HyDE(_DriftingLLM(), enabled=True,
                                cache=HydeCache(tmp_path / "h.sqlite")))
    _, first = rag.search("explain arima")
    _, second = rag.search("explain arima")
    assert (first["hyde_cache"], second["hyde_cache"]) == ("miss", "hit")
    assert first["hyde_used"] and second["hyde_used"]
    assert retriever.seen[0] == retriever.seen[1]       # the whole point of the cache


def test_echo_reports_off_when_hyde_is_off(tmp_path):
    rag = RAGPipeline(retriever=_Retriever(), reranker=_Reranker(), generator=None,
                      hyde=HyDE(FakeLLM(), enabled=True,
                                cache=HydeCache(tmp_path / "h.sqlite")))
    _, info = rag.search("explain arima", hyde=False)
    assert info["hyde_cache"] == "off" and info["hyde_used"] is False
