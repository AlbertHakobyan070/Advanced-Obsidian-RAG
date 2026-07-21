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


def test_per_call_mode_override_still_works():
    d = docs("alpha beta", "beta gamma")
    r = Reranker(model_name="BAAI/bge-reranker-v2-m3", mode="cross_encoder")
    r.rerank("beta", d, mode="lexical")          # must not load the big model
    assert r._model is None
