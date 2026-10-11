"""EmbeddingGemma 2 as a registry provider: the shipped entries, and the token cap
(max_seq_length) as part of a collection's embedding identity.

Run:  python -m pytest tests/ -q

A collection built with a 512-token cap and one built with 1024 hold different
vectors for every chunk longer than 512 tokens, yet share model, prefixes and
width. These tests pin that the fingerprint tells them apart, that the client
refuses a sidecar truncating somewhere other than where the entry says, and that
nothing changes for anyone who does not name the new providers.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.embeddings.embedder import Embedder, _OpenAIEmbedding
from src.embeddings.registry import EmbeddingSpec, resolve_embedding_spec
from src.embeddings.sidecar import fingerprint_diff, make_sidecar
from src.utils.config_loader import Config, load_config

ROOT = Path(__file__).resolve().parents[1]
EG2_QUERY = "task: search result | query: "
EG2_DOC = "title: none | text: "


def _cfg(entry, provider="eg2"):
    return Config({"embedding": {"provider": provider, "providers": {"eg2": entry}}}, ROOT)


ENTRY = {"kind": "openai", "base_url": "http://127.0.0.1:8104/v1", "model": "google/embeddinggemma-2",
         "dimensions": 768, "max_seq_length": 1024, "query_prefix": EG2_QUERY, "doc_prefix": EG2_DOC}


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_both_shipped_configs_define_the_eg2_providers_and_keep_the_active_embedder(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} is not present")
    cfg = load_config(path)
    assert cfg.get("embedding.provider") == "local"                  # nothing switches until someone does
    assert resolve_embedding_spec(cfg).model == "BAAI/bge-small-en-v1.5"

    long_, short = (resolve_embedding_spec(cfg, p) for p in ("eg2", "eg2_l512"))
    for spec, cap in ((long_, 1024), (short, 512)):
        assert (spec.kind, spec.model, spec.dimensions, spec.max_seq_length) == \
            ("openai", "google/embeddinggemma-2", 768, cap)
        assert (spec.query_prefix, spec.doc_prefix) == (EG2_QUERY, EG2_DOC)
        assert spec.api_key_env is None and spec.base_url.startswith("http://127.0.0.1:810")
    assert long_.base_url != short.base_url                          # two caps can serve at once


def test_the_cap_is_part_of_the_spec_identity_only_when_declared():
    spec = resolve_embedding_spec(_cfg(ENTRY))
    assert spec.max_seq_length == 1024 and spec.identity()["max_seq_length"] == 1024
    bge = EmbeddingSpec("local", "local", "BAAI/bge-small-en-v1.5")
    assert "max_seq_length" not in bge.identity()                    # today's identity is unchanged


@pytest.mark.parametrize("bad", [0, -1, 1.5, "1024", True])
def test_a_bad_max_seq_length_is_an_error_naming_the_key(bad):
    with pytest.raises(ValueError, match="max_seq_length"):
        resolve_embedding_spec(_cfg({**ENTRY, "max_seq_length": bad}))


def test_collections_built_at_512_and_1024_get_different_fingerprints():
    spec_1024 = resolve_embedding_spec(_cfg(ENTRY))
    spec_512 = resolve_embedding_spec(_cfg({**ENTRY, "max_seq_length": 512}))
    sidecar = make_sidecar(spec_1024, collection="c", role="chunks", status="complete",
                           dimension=768, count=1, written_by="rebuild-into")
    assert sidecar["max_seq_length"] == 1024
    assert fingerprint_diff(spec_1024, 768, sidecar, role="chunks") == []
    assert fingerprint_diff(spec_512, 768, sidecar, role="chunks") == ["max_seq_length"]


def test_a_sidecar_written_before_the_field_existed_still_matches_an_uncapped_embedder():
    spec = EmbeddingSpec("local", "local", "BAAI/bge-small-en-v1.5")
    sidecar = make_sidecar(spec, collection="c", role="chunks", status="complete",
                           dimension=384, count=1, written_by="index")
    del sidecar["max_seq_length"]                                    # an old file
    assert fingerprint_diff(spec, 384, sidecar, role="chunks") == []
    capped = EmbeddingSpec("eg2", "openai", "m", max_seq_length=512)
    assert "max_seq_length" in fingerprint_diff(capped, None, {**sidecar, "model": "m", "kind": "openai",
                                                              "normalize": False}, role="chunks")


# ---- the client checks the cap the server says it applied ----

class _Resp:
    def __init__(self, extra):
        self.data = [SimpleNamespace(embedding=[0.1, 0.2])]
        self.model_extra = extra


def _backend(monkeypatch, response, declared):
    created = []

    class FakeOpenAI:
        def __init__(self, **kw):
            self.embeddings = SimpleNamespace(create=lambda **k: created.append(k) or response)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    return _OpenAIEmbedding("m", "k", 768, "http://x/v1", max_seq_length=declared), created


def test_a_sidecar_truncating_elsewhere_than_declared_is_refused(monkeypatch):
    backend, _ = _backend(monkeypatch, _Resp({"max_seq_length": 512}), declared=1024)
    with pytest.raises(RuntimeError, match="512.*declares 1024"):
        backend.embed(["a"])


def test_a_matching_or_silent_endpoint_is_accepted(monkeypatch):
    ok, _ = _backend(monkeypatch, _Resp({"max_seq_length": 1024}), declared=1024)
    assert ok.embed(["a"]) == [[0.1, 0.2]]
    silent, _ = _backend(monkeypatch, _Resp(None), declared=1024)    # Ollama / OpenAI report nothing
    assert silent.embed(["a"]) == [[0.1, 0.2]]
    undeclared, _ = _backend(monkeypatch, _Resp({"max_seq_length": 7}), declared=None)
    assert undeclared.embed(["a"]) == [[0.1, 0.2]]


def test_from_config_hands_the_entrys_cap_to_the_backend_and_the_prefixes_to_the_embedder(monkeypatch):
    backend, created = _backend(monkeypatch, _Resp({"max_seq_length": 1024}), declared=None)
    seen = {}
    monkeypatch.setattr("src.embeddings.embedder._OpenAIEmbedding",
                        lambda **kw: seen.update(kw) or backend)
    cfg = Config({"paths": {"chunks_file": "c.jsonl", "chroma_dir": "x", "bm25_index": "b"},
                  "embedding": {"provider": "eg2", "providers": {"eg2": ENTRY}}}, ROOT)
    emb = Embedder.from_config(cfg)
    assert seen["max_seq_length"] == 1024 and seen["dimensions"] == 768 and seen["base_url"].endswith("/v1")
    emb.embed_documents(["chunk"])
    emb.embed_queries(["question"])
    assert [c["input"] for c in created] == [[EG2_DOC + "chunk"], [EG2_QUERY + "question"]]
    assert all(c["dimensions"] == 768 for c in created)
