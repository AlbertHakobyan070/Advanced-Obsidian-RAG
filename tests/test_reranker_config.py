"""Reranker switch tests (session 16).

Run:  python -m pytest tests/ -q

Covers the config surface for swapping the cross-encoder (model / max_length /
device). Deliberately loads NO model: bge-reranker-v2-m3 is 2.2 GB and ~145s to
load on CPU, so anything that touches a real model belongs in a benchmark, not
the test suite. The lexical and none modes are exercised for real, since they
need no model at all.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.retrieval.reranker import KNOWN_RERANKERS, Reranker
from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config

ROOT = Path(__file__).resolve().parents[1]


def cfg_of(data):
    return Config(data, ROOT)


def docs(*texts):
    return [RetrievedDoc(id=f"d{i}", text=t, metadata={})
            for i, t in enumerate(texts)]


def test_defaults_match_the_shipped_config():
    r = Reranker.from_config(cfg_of({}))
    assert r.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert r.max_length == 512
    assert r.device is None          # auto
    assert r.mode == "cross_encoder"


def test_switching_model_and_length_from_config():
    r = Reranker.from_config(cfg_of({"retrieval": {
        "cross_encoder_model": "BAAI/bge-reranker-v2-m3",
        "cross_encoder_max_length": 1024,
    }}))
    assert r.model_name == "BAAI/bge-reranker-v2-m3"
    assert r.max_length == 1024


@pytest.mark.parametrize("given,expected", [
    ("auto", None), ("AUTO", None), ("", None), (None, None),
    ("cpu", "cpu"), ("cuda", "cuda"), ("cuda:1", "cuda:1"), ("  CUDA:0 ", "cuda:0"),
])
def test_device_normalisation(given, expected):
    """'auto' must become None so sentence-transformers picks for itself."""
    r = Reranker.from_config(cfg_of({"retrieval": {"cross_encoder_device": given}}))
    assert r.device == expected


def test_known_rerankers_are_well_formed():
    """The console renders this dict; a missing label would show blank rows."""
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in KNOWN_RERANKERS
    assert "BAAI/bge-reranker-v2-m3" in KNOWN_RERANKERS
    for model_id, spec in KNOWN_RERANKERS.items():
        assert spec.get("label"), f"{model_id} has no label"
        assert isinstance(spec.get("max_length"), int)


def test_bad_mode_is_rejected():
    with pytest.raises(ValueError, match="rerank mode"):
        Reranker(model_name="x", mode="turbo")


def test_lexical_and_none_modes_need_no_model():
    """Swapping in a 2.2 GB model must not make these paths load anything."""
    d = docs("gradient clipping rescales the gradient norm",
             "a completely unrelated sentence about harpsichords",
             "clipping the gradient prevents exploding gradients")
    r = Reranker(model_name="BAAI/bge-reranker-v2-m3", mode="lexical", top_k=2)
    out = r.rerank("gradient clipping", d)
    assert len(out) == 2
    assert "clipping" in out[0].text
    assert r._model is None                      # nothing was loaded

    r2 = Reranker(model_name="BAAI/bge-reranker-v2-m3", mode="none", top_k=2)
    assert [x.id for x in r2.rerank("q", d)] == ["d0", "d1"]   # fused order kept
    assert r2._model is None


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_http_mode_scores_from_the_endpoint(monkeypatch):
    """The GPU lane: a reranker served out-of-process via /v1/rerank."""
    import httpx
    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent["url"], sent["json"] = url, json
        # deliberately out of order + not index-sorted, like llama.cpp
        return FakeResp({"results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.5},
        ]})

    monkeypatch.setattr(httpx, "post", fake_post)
    r = Reranker(model_name="bge-reranker-v2-m3", mode="http", top_k=2,
                 http_url="http://127.0.0.1:8101/v1", http_model="bge")
    out = r.rerank("q", docs("alpha", "beta", "gamma"))
    assert sent["url"] == "http://127.0.0.1:8101/v1/rerank"
    assert sent["json"]["model"] == "bge"
    assert sent["json"]["documents"] == ["alpha", "beta", "gamma"]
    assert [d.text for d in out] == ["gamma", "beta"]      # 0.9, 0.5
    assert out[0].rerank_score == 0.9
    assert r._model is None                                # nothing local loaded


def test_http_mode_without_a_url_fails_loudly():
    r = Reranker(model_name="x", mode="http")
    with pytest.raises(ValueError, match="rerank_http.base_url"):
        r.rerank("q", docs("a", "b"))


def test_http_mode_rejects_a_partial_scoring(monkeypatch):
    """A reranker that scores only some documents must not silently pass.

    llama.cpp has had rerank bugs (ggml-org/llama.cpp#16407); half-scored
    results would otherwise reorder on garbage while the log said 'reranked'.
    """
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp(
        {"results": [{"index": 0, "relevance_score": 0.7}]}))
    r = Reranker(model_name="x", mode="http", http_url="http://h/v1")
    with pytest.raises(ValueError, match="did not score every document"):
        r.rerank("q", docs("a", "b", "c"))


def test_http_mode_rejects_a_malformed_response(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "post", lambda *a, **k: FakeResp({"data": []}))
    r = Reranker(model_name="x", mode="http", http_url="http://h/v1")
    with pytest.raises(ValueError, match="unexpected /rerank response"):
        r.rerank("q", docs("a"))


def test_http_config_resolves():
    r = Reranker.from_config(cfg_of({"retrieval": {
        "rerank_mode": "http",
        "rerank_http": {"base_url": "http://127.0.0.1:8101/v1/",
                        "model": "bge-reranker-v2-m3", "timeout": 90},
    }}))
    assert r.mode == "http"
    assert r.http_url == "http://127.0.0.1:8101/v1"      # trailing / stripped
    assert r.http_model == "bge-reranker-v2-m3"
    assert r.http_timeout == 90


def test_per_call_mode_override_still_works():
    d = docs("alpha beta", "beta gamma")
    r = Reranker(model_name="BAAI/bge-reranker-v2-m3", mode="cross_encoder")
    r.rerank("beta", d, mode="lexical")          # must not load the big model
    assert r._model is None
