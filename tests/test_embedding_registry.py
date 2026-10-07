"""Embedder registry, device and query/doc prefix tests (embedding-switch slice 1).

Run:  python -m pytest tests/ -q

The rule that must never break: with today's config (`provider: local`,
BAAI/bge-small-en-v1.5, no prefixes) the embedder is constructed and called
EXACTLY as it was before the registry existed. The two PIN tests below were
written against the unmodified code and pin that; everything else covers what
the registry adds (base_url, device, prefixes, named providers).

No model is ever loaded and nothing touches the network: sentence_transformers
and openai.OpenAI are replaced by recording fakes, and the only store is a
Chroma directory under tmp_path with explicit vectors.
"""
import hashlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.embeddings.embedder import Chunk, Embedder
from src.embeddings.registry import (
    KNOWN_EMBEDDERS, EmbeddingSpec, check_prefix, resolve_embedding_spec)
from src.utils.config_loader import Config, load_config

chromadb = pytest.importorskip("chromadb")

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
ENCODE_KWARGS = {"normalize_embeddings": True, "show_progress_bar": False}

# A keyless-or-keyed Ollama-style entry; tests derive variants from it.
OLLAMA = {
    "kind": "openai",
    "base_url": "http://localhost:11434/v1",
    "model": "nomic-embed-text",
    "api_key_env": "TEST_EMB_KEY",
}


def _vec(text, dim=4):
    """A deterministic unit vector for `text`. Zero-mean on purpose (sha256
    bytes - 127.5): unrelated non-negative vectors all have cosine ~0.75."""
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest()[:dim], dtype=np.uint8)
    raw = raw.astype(float) - 127.5
    return raw / np.linalg.norm(raw)


def _cfg(tmp_path, **embedding):
    """A config whose store paths all live under tmp_path."""
    return Config({
        "paths": {"chunks_file": "chunks.jsonl", "chroma_dir": "chroma_db",
                  "bm25_index": "bm25.pkl", "collection_name": "obsidian_vault"},
        "embedding": embedding,
    }, tmp_path)


def _stored(emb, ids):
    """{id: stored document} straight from Chroma. Chroma returns rows in
    storage order, not request order, so always map by id."""
    col = chromadb.PersistentClient(path=str(emb.chroma_dir)).get_collection(emb.collection_name)
    got = col.get(ids=ids)
    return dict(zip(got["ids"], got["documents"]))


def _chunks():
    return [Chunk("d1", "alpha", {"source_file": "a.md"}),
            Chunk("d2", "beta", {"source_file": "a.md"})]


@pytest.fixture
def fake_st(monkeypatch):
    """sentence_transformers.SentenceTransformer, recording how it was built
    (`inits`: (args, kwargs)) and what it was asked to encode (`encodes`:
    (texts, args, kwargs)). The import sits inside _LocalEmbedding.__init__, so
    swapping the module in sys.modules is enough."""
    rec = types.SimpleNamespace(inits=[], encodes=[], dim=384)

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            rec.inits.append((args, kwargs))

        def encode(self, texts, *args, **kwargs):
            rec.encodes.append((list(texts), args, kwargs))
            return np.stack([_vec(t) for t in texts])

        def get_embedding_dimension(self):
            return rec.dim

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return rec


@pytest.fixture
def fake_openai(monkeypatch):
    """openai.OpenAI, recording its constructor kwargs (`inits`) and every
    embeddings.create(**kwargs) call (`creates`)."""
    rec = types.SimpleNamespace(inits=[], creates=[])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            rec.inits.append(kwargs)
            self.embeddings = types.SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            rec.creates.append(kwargs)
            return types.SimpleNamespace(data=[
                types.SimpleNamespace(embedding=_vec(t).tolist()) for t in kwargs["input"]])

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    return rec


# ------------------------------------------------- characterization (PIN) ----

def test_default_local_embedder_is_built_and_called_exactly_as_before(tmp_path, fake_st):
    """PIN. Today's local path, byte for byte: SentenceTransformer gets the
    model name and NOTHING else (no device kwarg), encode gets the raw texts
    with exactly these two kwargs, and Chroma is handed the raw chunk text."""
    emb = Embedder.from_config(_cfg(
        tmp_path, provider="local", local_model=LOCAL_MODEL, batch_size=100, dimensions=384))
    assert fake_st.inits == [((LOCAL_MODEL,), {})]

    emb.embed_query("q")
    assert fake_st.encodes == [(["q"], (), ENCODE_KWARGS)]

    fake_st.encodes.clear()
    emb._append_dense(_chunks())
    assert fake_st.encodes == [(["alpha", "beta"], (), ENCODE_KWARGS)]
    assert _stored(emb, ["d1", "d2"]) == {"d1": "alpha", "d2": "beta"}


def test_legacy_openai_embedder_is_built_and_called_exactly_as_before(
        tmp_path, monkeypatch, fake_openai):
    """PIN. The legacy `openai` provider: the client gets the key and nothing
    else (no base_url kwarg), and `dimensions` rides along on every call."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    emb = Embedder.from_config(_cfg(tmp_path, provider="openai", batch_size=100, dimensions=384))
    assert fake_openai.inits == [{"api_key": "sk-test"}]

    emb.embed_query("q")
    assert fake_openai.creates == [
        {"model": "text-embedding-3-small", "input": ["q"], "dimensions": 384}]


# ------------------------------------------------------ registry resolution ----

def test_default_config_resolves_to_todays_local_embedder(tmp_path):
    spec = resolve_embedding_spec(_cfg(
        tmp_path, provider="local", local_model=LOCAL_MODEL, batch_size=100, dimensions=384))
    assert spec == EmbeddingSpec("local", "local", LOCAL_MODEL)    # every other field at its default
    assert (spec.device, spec.query_prefix, spec.doc_prefix) == ("auto", "", "")
    assert spec.normalize is True
    assert spec.label() == f"local · {LOCAL_MODEL}"
    assert spec.identity() == {"kind": "local", "model": LOCAL_MODEL, "normalize": True,
                               "query_prefix": "", "doc_prefix": ""}

    # An unset provider still means legacy openai: the default from_config always had.
    legacy = resolve_embedding_spec(_cfg(tmp_path, dimensions=384))
    assert (legacy.provider, legacy.kind, legacy.model) == ("openai", "openai", "text-embedding-3-small")
    assert (legacy.api_key_env, legacy.dimensions, legacy.base_url) == ("OPENAI_API_KEY", 384, None)
    assert legacy.normalize is False


@pytest.mark.parametrize("name", ["config.yaml", "config.example.yaml"])
def test_both_shipped_configs_resolve_to_todays_embedder(name):
    if not (ROOT / name).exists():  # a fresh clone has no config.yaml (see conftest)
        pytest.skip(f"{name} is not present")
    spec = resolve_embedding_spec(load_config(ROOT / name))
    assert spec == EmbeddingSpec("local", "local", LOCAL_MODEL)


@pytest.mark.parametrize("bad", ["gpu", "cuda:", "cuda:x", "cpu\n", "", 3, None])
def test_an_unknown_device_is_an_error_listing_the_valid_forms(tmp_path, bad):
    with pytest.raises(ValueError, match=r"embedding\.device") as exc:
        resolve_embedding_spec(_cfg(tmp_path, provider="local", device=bad))
    for form in ("auto", "cpu", "mps", "cuda"):
        assert form in str(exc.value)


def test_an_unknown_provider_is_an_error_listing_the_known_ones(tmp_path):
    cfg = _cfg(tmp_path, provider="nope", providers={"alpha": OLLAMA, "beta": OLLAMA})
    with pytest.raises(ValueError) as exc:
        resolve_embedding_spec(cfg)
    msg = str(exc.value)
    assert "embedding.provider = 'nope'" in msg
    assert "('local', 'openai')" in msg
    assert "['alpha', 'beta']" in msg

    # The explicit argument (re-embed targets, bench) resolves the same way.
    assert resolve_embedding_spec(cfg, "alpha").provider == "alpha"
    with pytest.raises(ValueError, match="'gamma'"):
        resolve_embedding_spec(cfg, "gamma")

    # Only the SELECTED entry is validated: a broken sibling does not take
    # down a service that is not using it.
    broken = _cfg(tmp_path, provider="alpha", providers={"alpha": OLLAMA, "broken": {"doc_prefx": "x"}})
    assert resolve_embedding_spec(broken).provider == "alpha"
    assert resolve_embedding_spec(_cfg(tmp_path, provider="local", providers={"broken": {"x": 1}})).kind == "local"


def test_a_registry_entry_may_not_take_a_reserved_name(tmp_path):
    # Whichever provider is selected: a `local` entry would otherwise be a
    # silent redirect of every pre-registry config.
    for reserved in ("local", "openai"):
        with pytest.raises(ValueError, match=rf"embedding\.providers\.{reserved}.*reserved"):
            resolve_embedding_spec(_cfg(tmp_path, provider="local", providers={reserved: OLLAMA}))
    with pytest.raises(ValueError, match="mapping"):
        resolve_embedding_spec(_cfg(tmp_path, provider="local", providers=["alpha"]))


@pytest.mark.parametrize("entry, needle", [
    ({**OLLAMA, "doc_prefx": "x"}, "doc_prefx"),                     # a typo must not be a silent no-op
    ({**OLLAMA, "kind": "anthropic"}, "anthropic"),
    ({k: v for k, v in OLLAMA.items() if k != "model"}, "model"),    # required
    ({**OLLAMA, "dimensions": 0}, "dimensions"),
    ({**OLLAMA, "dimensions": "512"}, "dimensions"),
    ({**OLLAMA, "api_key_env": 5}, "api_key_env"),
    ({**OLLAMA, "api_key_optional": "yes"}, "api_key_optional"),
])
def test_an_unknown_entry_key_or_kind_is_an_error_not_a_silent_no_op(tmp_path, entry, needle):
    with pytest.raises(ValueError, match=needle):
        resolve_embedding_spec(_cfg(tmp_path, provider="p", providers={"p": entry}))


def test_prefixes_must_be_one_line_without_hash_or_backslash(tmp_path):
    assert check_prefix(None, "x") == ""
    assert check_prefix("query: ", "x") == "query: "     # the trailing space is meaningful: never stripped

    for bad in ("a\nb", "a\rb", "# x", "a#b", "a\\b", 5, ["x"]):
        for key in ("query_prefix", "doc_prefix"):
            with pytest.raises(ValueError, match=rf"embedding\.{key}"):
                resolve_embedding_spec(_cfg(tmp_path, provider="local", **{key: bad}))
        with pytest.raises(ValueError, match=r"embedding\.providers\.p\.doc_prefix"):
            resolve_embedding_spec(_cfg(tmp_path, provider="p", providers={"p": {**OLLAMA, "doc_prefix": bad}}))


def test_known_embedders_catalogue_is_well_formed():
    fields = {"label", "dimension", "multilingual", "query_prefix", "doc_prefix",
              "prefixes_required", "note"}
    assert {"BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-m3",
            "intfloat/multilingual-e5-base"} <= set(KNOWN_EMBEDDERS)
    for model, row in KNOWN_EMBEDDERS.items():
        assert set(row) == fields, model
        assert isinstance(row["label"], str) and row["label"], model
        assert isinstance(row["dimension"], int) and not isinstance(row["dimension"], bool), model
        assert row["dimension"] > 0, model
        assert isinstance(row["multilingual"], bool) and isinstance(row["prefixes_required"], bool), model
        assert isinstance(row["note"], str), model
        for key in ("query_prefix", "doc_prefix"):
            assert check_prefix(row[key], f"{model}.{key}") == row[key]
        if row["prefixes_required"]:
            assert row["query_prefix"] and row["doc_prefix"], model


# ------------------------------------------------ embedder: prefixes, device ----

@pytest.mark.parametrize("write", ["_append_dense", "_build_dense"])
def test_a_configured_doc_prefix_reaches_document_embeddings_but_not_stored_text(
        tmp_path, fake_st, write):
    """REGRESSION: `doc_prefix` used to be silently ignored. The prefix belongs
    to the embedding INPUT only; the text Chroma stores (and BM25 indexes)
    stays the raw chunk text."""
    emb = Embedder.from_config(_cfg(
        tmp_path, provider="local", local_model=LOCAL_MODEL, doc_prefix="passage: "))
    getattr(emb, write)(_chunks())
    assert fake_st.encodes == [(["passage: alpha", "passage: beta"], (), ENCODE_KWARGS)]
    assert _stored(emb, ["d1", "d2"]) == {"d1": "alpha", "d2": "beta"}


def test_query_prefix_applies_to_queries_only(tmp_path, fake_st):
    emb = Embedder.from_config(_cfg(
        tmp_path, provider="local", local_model=LOCAL_MODEL,
        query_prefix="query: ", doc_prefix="passage: "))
    emb.embed_query("q")
    emb.embed_queries(["a", "b"])
    emb.embed_documents(["d"])
    assert [e[0] for e in fake_st.encodes] == [["query: q"], ["query: a", "query: b"], ["passage: d"]]

    # No doc prefix configured: documents reach the model untouched, whatever
    # the query prefix is.
    fake_st.encodes.clear()
    only_query = Embedder.from_config(_cfg(
        tmp_path, provider="local", local_model=LOCAL_MODEL, query_prefix="query: "))
    only_query.embed_documents(["d"])
    assert [e[0] for e in fake_st.encodes] == [["d"]]


@pytest.mark.parametrize("device, kwargs", [
    ("auto", {}),                                  # exactly what happened before the key existed
    ("cpu", {"device": "cpu"}),
    ("CPU", {"device": "cpu"}),                    # lower-cased
    ("cuda:1", {"device": "cuda:1"}),
])
def test_auto_device_passes_no_device_kwarg_and_explicit_ones_pass_through(
        tmp_path, fake_st, device, kwargs):
    Embedder.from_config(_cfg(tmp_path, provider="local", local_model=LOCAL_MODEL, device=device))
    assert fake_st.inits == [((LOCAL_MODEL,), kwargs)]


# -------------------------------------------------- embedder: hosted entries ----

def test_a_registry_entry_builds_an_openai_client_with_its_base_url_and_key(
        tmp_path, monkeypatch, fake_openai):
    monkeypatch.setenv("TEST_EMB_KEY", "from-the-environment")
    entry = {**OLLAMA, "query_prefix": "search_query: ", "doc_prefix": "search_document: "}
    # Top-level prefixes configure the reserved providers only; an entry
    # carries its own and must not inherit these.
    emb = Embedder.from_config(_cfg(
        tmp_path, provider="ollama_nomic", providers={"ollama_nomic": entry},
        query_prefix="TOP: ", doc_prefix="TOP: "))
    assert fake_openai.inits == [
        {"api_key": "from-the-environment", "base_url": "http://localhost:11434/v1"}]
    assert (emb.spec.provider, emb.spec.model) == ("ollama_nomic", "nomic-embed-text")
    assert emb.hype_collection_name == "hype_questions"    # carried for the sidecar writer; no behaviour yet

    emb.embed_query("q")
    emb.embed_documents(["d"])
    assert [(c["model"], c["input"]) for c in fake_openai.creates] == [
        ("nomic-embed-text", ["search_query: q"]), ("nomic-embed-text", ["search_document: d"])]


def test_dimensions_is_sent_only_when_the_entry_sets_it(tmp_path, monkeypatch, fake_openai):
    monkeypatch.setenv("TEST_EMB_KEY", "k")
    # The shipped config's top-level `dimensions: 384` belongs to the legacy
    # provider only: a registry entry never inherits it.
    plain = Embedder.from_config(_cfg(
        tmp_path, provider="p", dimensions=384, providers={"p": OLLAMA}))
    plain.embed_query("q")
    sized = Embedder.from_config(_cfg(
        tmp_path, provider="p", dimensions=384, providers={"p": {**OLLAMA, "dimensions": 512}}))
    sized.embed_query("q")
    assert "dimensions" not in fake_openai.creates[0]
    assert fake_openai.creates[1]["dimensions"] == 512


def test_a_keyless_or_optional_key_entry_uses_a_placeholder(tmp_path, monkeypatch, fake_openai):
    monkeypatch.delenv("TEST_EMB_KEY", raising=False)
    keyless = {**OLLAMA, "api_key_env": None}
    optional = {**OLLAMA, "api_key_optional": True}
    for entry in (keyless, optional):
        Embedder.from_config(_cfg(tmp_path, provider="p", providers={"p": entry}))
    assert [kw["api_key"] for kw in fake_openai.inits] == ["not-needed", "not-needed"]

    # An optional key that IS set is used.
    monkeypatch.setenv("TEST_EMB_KEY", "real-key")
    Embedder.from_config(_cfg(tmp_path, provider="p", providers={"p": optional}))
    assert fake_openai.inits[-1]["api_key"] == "real-key"


def test_a_missing_required_key_is_an_error_naming_the_variable(tmp_path, monkeypatch, fake_openai):
    monkeypatch.delenv("TEST_EMB_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEST_EMB_KEY"):
        Embedder.from_config(_cfg(tmp_path, provider="p", providers={"p": OLLAMA}))
    assert fake_openai.inits == []          # no client was built around a made-up key


def test_dimension_comes_from_the_model_for_local_and_the_entry_for_hosted(
        tmp_path, monkeypatch, fake_st, fake_openai):
    monkeypatch.setenv("TEST_EMB_KEY", "k")
    local = Embedder.from_config(_cfg(tmp_path, provider="local", local_model=LOCAL_MODEL))
    assert local.dimension() == 384
    sized = Embedder.from_config(_cfg(
        tmp_path, provider="p", providers={"p": {**OLLAMA, "dimensions": 512}}))
    assert sized.dimension() == 512
    unsized = Embedder.from_config(_cfg(tmp_path, provider="p", providers={"p": OLLAMA}))
    assert unsized.dimension() is None
