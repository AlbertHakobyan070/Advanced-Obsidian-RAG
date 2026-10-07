"""The startup guard, the HyPE lane and the write paths (embedding-switch slice 2).

Run:  python -m pytest tests/ -q

The rule: a collection's vectors are only searched, and only added to, by the
embedder that built them. Every collection carries a fingerprint sidecar; the
retriever checks it when the pipeline is built, the HyPE lane checks its own,
and `index --append` / `build_hype.py` refuse to mix a second embedder in. A
collection with NO sidecar (the live store predates fingerprints) keeps working
and only logs a warning, so the first start after this lands never takes :8051
down.

No model is ever loaded: the embedder is the real Embedder around a recording
fake backend, or around a fake sentence_transformers module. Every store is a
Chroma directory under tmp_path with explicit vectors.
"""
import hashlib
import json
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from fastapi.testclient import TestClient

import serve_api as S
from src.embeddings.embedder import Chunk, Embedder
from src.embeddings.registry import EmbeddingSpec
from src.embeddings.sidecar import (
    EmbeddingMismatchError, check_collection, ids_digest, make_sidecar, read_sidecar,
    sidecar_path, write_sidecar)
from src.retrieval.retriever import HybridRetriever
from src.utils.config_loader import Config, load_config

chromadb = pytest.importorskip("chromadb")

DIM = 16
MODEL_A, MODEL_B = "fake/model-a", "fake/model-b"
ROWS = [("d1", "alpha"), ("d2", "beta"), ("d3", "gamma")]


def _vec(text, dim=DIM):
    """A deterministic unit vector for `text`. Zero-mean on purpose (sha256
    bytes - 127.5): unrelated non-negative vectors all have cosine ~0.75."""
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest()[:dim], dtype=np.uint8)
    raw = raw.astype(float) - 127.5
    return raw / np.linalg.norm(raw)


class FakeBackend:
    """An embedder backend whose vectors depend on `salt` (standing in for the
    model) and the text, and which records every call."""

    def __init__(self, salt, dim=DIM):
        self.salt, self.dim, self.calls = salt, dim, []

    @property
    def dimension(self):
        return self.dim

    def embed(self, texts):
        self.calls.append(list(texts))
        return [_vec(self.salt + t, self.dim).tolist() for t in texts]


def _spec(model=MODEL_A, **kw):
    return EmbeddingSpec("local", "local", model, **kw)


def _cfg(tmp_path, collection="obsidian_vault", **sections):
    data = {"paths": {"chunks_file": "chunks.jsonl", "chroma_dir": "chroma_db",
                      "bm25_index": "bm25.pkl", "collection_name": collection}}
    data.update(sections)
    return Config(data, tmp_path)


def _embedder(cfg, spec=None, dim=DIM):
    spec = spec or _spec()
    return Embedder(
        backend=FakeBackend(spec.model, dim), chunks_file=cfg.path("paths.chunks_file"),
        chroma_dir=cfg.path("paths.chroma_dir"), bm25_index=cfg.path("paths.bm25_index"),
        collection_name=cfg.get("paths.collection_name"), batch_size=2, spec=spec,
        hype_collection_name=cfg.get("retrieval.hype.collection", "hype_questions"))


def _write(emb, name, **fields):
    """A sidecar for `emb`, as the code that built the collection would leave it."""
    args = dict(collection=name, role="chunks", status="complete",
                dimension=emb.dimension(), count=1, written_by="stamp")
    args.update(fields)
    return write_sidecar(emb.chroma_dir, name, make_sidecar(emb.spec, **args))


def _chunks(rows=ROWS):
    return [Chunk(i, t, {"source_file": "a.md"}) for i, t in rows]


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps({"doc_id": i, "text": t, "metadata": {"source_file": "a.md"}})
                            + "\n" for i, t in rows), encoding="utf-8")


def _count(emb, name=None):
    client = chromadb.PersistentClient(path=str(emb.chroma_dir))
    return client.get_collection(name or emb.collection_name).count()


def _hype_collection(emb, name="hype_questions"):
    """The question collection for d1, its vectors embedded query-side by `emb`."""
    questions = ["what is alpha?", "how does alpha work?"]
    col = chromadb.PersistentClient(path=str(emb.chroma_dir)).get_or_create_collection(
        name, metadata={"hnsw:space": "cosine"})
    col.add(ids=["d1::q0", "d1::q1"], embeddings=emb.embed_queries(questions), documents=questions,
            metadatas=[{"doc_id": "d1", "source_file": "a.md"}] * 2)


@pytest.fixture
def fake_st(monkeypatch):
    """sentence_transformers.SentenceTransformer as a stand-in; the import sits
    inside _LocalEmbedding.__init__, so swapping the module is enough."""
    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, *args, **kwargs):
            return np.stack([_vec(t) for t in texts])

        def get_embedding_dimension(self):
            return DIM

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


# ------------------------------------------------------- the startup guard ----

def test_the_guard_passes_a_matching_collection(tmp_path, caplog):
    cfg = _cfg(tmp_path)
    emb = _embedder(cfg)
    _write(emb, "obsidian_vault")

    with caplog.at_level(logging.INFO):
        retriever = HybridRetriever.from_config(cfg, emb)

    assert retriever.collection_name == "obsidian_vault"
    ok = [r for r in caplog.records if "Embedding fingerprint OK" in r.getMessage()]
    assert len(ok) == 1 and ok[0].levelno == logging.INFO
    assert "'obsidian_vault' was built by local · fake/model-a" in ok[0].getMessage()
    assert _warnings(caplog) == []


def test_the_guard_refuses_a_same_dimension_model_mismatch(tmp_path):
    """REGRESSION: same width, different model. Chroma compares the vectors
    without complaint and every search returns confident nonsense; before the
    guard the retriever simply built."""
    cfg = _cfg(tmp_path)
    _write(_embedder(cfg, _spec("BAAI/bge-small-en-v1.5")), "obsidian_vault")

    with pytest.raises(EmbeddingMismatchError) as exc:
        HybridRetriever.from_config(cfg, _embedder(cfg, _spec("fake/other-model")))

    msg = str(exc.value)
    assert "BAAI/bge-small-en-v1.5" in msg and "fake/other-model" in msg      # both embedders named
    assert "differ in model." in msg                                           # and only the model
    assert "paths.collection_name" in msg and "embedding.*" in msg             # the two ways out


def test_the_guard_refuses_a_dimension_mismatch(tmp_path):
    cfg = _cfg(tmp_path)
    hosted = EmbeddingSpec("hosted", "openai", "text-embedding-3-large", dimensions=512)
    emb = _embedder(cfg, hosted, dim=512)           # the entry's `dimensions`, as the backend reports it
    _write(emb, "obsidian_vault", dimension=1024)

    with pytest.raises(EmbeddingMismatchError) as exc:
        HybridRetriever.from_config(cfg, emb)

    msg = str(exc.value)
    assert "differ in dimension." in msg
    assert "1024-d" in msg and "512-d" in msg and "text-embedding-3-large" in msg


@pytest.mark.parametrize("field, value", [("doc_prefix", "passage: "), ("query_prefix", "query: ")])
def test_the_guard_refuses_a_prefix_mismatch(tmp_path, field, value):
    """Either prefix on its own: doc_prefix is baked into the stored vectors, and
    an e5-class model degrades silently without the query one."""
    cfg = _cfg(tmp_path)
    _write(_embedder(cfg), "obsidian_vault")

    with pytest.raises(EmbeddingMismatchError, match=f"differ in {field}"):
        HybridRetriever.from_config(cfg, _embedder(cfg, _spec(**{field: value})))


def test_the_guard_refuses_an_unfinished_reembed(tmp_path):
    cfg = _cfg(tmp_path)
    emb = _embedder(cfg)
    _write(emb, "obsidian_vault", status="building", count=40, expected=100)

    with pytest.raises(EmbeddingMismatchError, match=r"unfinished re-embed.*40 of 100"):
        HybridRetriever.from_config(cfg, emb)


def test_the_guard_refuses_a_hype_collection_configured_as_the_chunk_collection(tmp_path):
    cfg = _cfg(tmp_path, collection="hype_questions")
    emb = _embedder(cfg)
    _write(emb, "hype_questions", role="hype")

    with pytest.raises(EmbeddingMismatchError, match=r"'hype_questions' is recorded as a 'hype'.*'chunks'"):
        HybridRetriever.from_config(cfg, emb)


def test_a_missing_sidecar_is_a_warning_and_the_build_proceeds(tmp_path, caplog):
    cfg = _cfg(tmp_path)
    emb = _embedder(cfg)

    with caplog.at_level(logging.INFO):
        retriever = HybridRetriever.from_config(cfg, emb)

    assert retriever.collection_name == "obsidian_vault"
    (warning,) = _warnings(caplog)
    text = warning.getMessage()
    assert warning.name == "src.retrieval.retriever"
    assert "'obsidian_vault' has no embedding fingerprint" in text
    assert str(sidecar_path(emb.chroma_dir, "obsidian_vault")) in text
    assert "local · fake/model-a" in text and "`rag stamp`" in text
    assert not [r for r in caplog.records if "fingerprint OK" in r.getMessage()]


# ------------------------------------- the same guard, behind the real lifespan ----

def _boot(monkeypatch, cfg):
    """:8051's lifespan with the real Embedder and the real HybridRetriever: only
    the builders that have nothing to do with the fingerprint are stand-ins (and
    the logging bootstrap, whose handler swap would blind pytest's capture)."""
    import src.pipeline as P
    stand_in = SimpleNamespace(from_config=lambda *args, **kwargs: SimpleNamespace())
    for name in ("LLMClient", "HyDE", "Generator", "GraphExpander"):
        monkeypatch.setattr(P, name, stand_in)
    monkeypatch.setattr(P, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(S, "load_config", lambda *args, **kwargs: cfg)
    return TestClient(S.app)


def test_a_mismatch_brings_the_query_service_up_failed_with_both_models_named(
        tmp_path, monkeypatch, fake_st):
    cfg = _cfg(tmp_path, embedding={"provider": "local", "local_model": MODEL_B})
    _write(_embedder(cfg, _spec("BAAI/bge-small-en-v1.5")), "obsidian_vault")

    with _boot(monkeypatch, cfg) as client:
        body = client.get("/health").json()
        refused = client.post("/search", json={"q": "anything"})

    assert (body["ready"], body["state"]) == (False, "failed")
    assert body["error"].startswith("EmbeddingMismatchError:")
    assert "BAAI/bge-small-en-v1.5" in body["error"] and MODEL_B in body["error"]
    # The existing failed path: every retrieval endpoint answers 503 with the same reason.
    assert refused.status_code == 503
    assert refused.json()["detail"]["reason"] == body["error"]


def test_an_unstamped_collection_still_comes_up_ready(tmp_path, monkeypatch, fake_st):
    """The live desktop index has no sidecar: the first start after this lands
    must come up ready (and only warn), never failed."""
    cfg = _cfg(tmp_path, embedding={"provider": "local", "local_model": MODEL_B})

    with _boot(monkeypatch, cfg) as client:
        body = client.get("/health").json()

    assert (body["ready"], body["state"]) == (True, "ready")
    assert "error" not in body


# ------------------------------------------------------------- the HyPE lane ----

def test_the_hype_lane_refuses_a_mismatched_hype_collection_instead_of_failing_soft(tmp_path):
    """REGRESSION: the lane used to swallow every problem and return [], which is
    how a lane searching another model's questions would go unnoticed."""
    cfg = _cfg(tmp_path)
    emb = _embedder(cfg)
    emb._build_dense(_chunks())
    _hype_collection(emb)
    qvec = emb.embed_query("alpha")

    # The control: with a matching fingerprint the lane maps its hit to the parent chunk.
    _write(emb, "hype_questions", role="hype")
    assert [hit[0] for hit in HybridRetriever.from_config(cfg, emb)._hype_search(qvec, 3)] == ["d1"]

    # The questions were embedded by another model: refused, every time it is asked.
    _write(_embedder(cfg, _spec(MODEL_B)), "hype_questions", role="hype")
    retriever = HybridRetriever.from_config(cfg, emb)
    for _ in range(2):
        with pytest.raises(EmbeddingMismatchError, match=f"{MODEL_A}.*{MODEL_B}|{MODEL_B}.*{MODEL_A}"):
            retriever._hype_search(qvec, 3)


def test_an_enabled_hype_lane_checks_its_sidecar_at_build(tmp_path):
    cfg_on = _cfg(tmp_path, retrieval={"hype": {"enabled": True, "collection": "hype_questions"}})
    emb = _embedder(cfg_on)
    _write(emb, "obsidian_vault")
    HybridRetriever.from_config(cfg_on, emb)               # no hype sidecar yet: nothing to check

    _write(_embedder(cfg_on, _spec(MODEL_B)), "hype_questions", role="hype")
    with pytest.raises(EmbeddingMismatchError, match="hype_questions"):
        HybridRetriever.from_config(cfg_on, emb)           # the operator asked for the lane: refuse at build

    HybridRetriever.from_config(_cfg(tmp_path), emb)       # lane off: its sidecar is not this build's business


def test_a_hype_collection_that_does_not_exist_still_fails_soft(tmp_path):
    """PIN: only a MISMATCH raises. A lane whose collection was never built is
    skipped, as before."""
    cfg = _cfg(tmp_path, retrieval={"hype": {"enabled": True, "collection": "hype_questions"}})
    emb = _embedder(cfg)
    _write(emb, "obsidian_vault")
    retriever = HybridRetriever.from_config(cfg, emb)

    assert retriever._get_hype_collection() is None
    assert retriever._hype_search(emb.embed_query("alpha"), 3) == []


# ---------------------------------------------------------- the write paths ----

def test_append_refuses_to_mix_two_embedders_in_one_collection(tmp_path):
    """REGRESSION: appending with another embedder used to put its vectors into
    the collection next to the first one's."""
    cfg = _cfg(tmp_path)
    original, other = _embedder(cfg), _embedder(cfg, _spec(MODEL_B))
    original._build_dense(_chunks())
    _write(original, "obsidian_vault")
    extra = tmp_path / "extra.jsonl"
    _write_jsonl(extra, [("d9", "delta")])

    with pytest.raises(EmbeddingMismatchError, match=MODEL_A):
        other.append_indexes(extra)
    assert _count(other) == 3
    assert other.backend.calls == []                       # refused before anything was embedded

    original.append_indexes(extra)                         # the embedder that built it still may
    assert _count(original) == 4


def test_append_into_a_new_collection_writes_its_sidecar(tmp_path, fake_st):
    cfg = _cfg(tmp_path, embedding={"provider": "local", "local_model": MODEL_A},
               retrieval={"hype": {"collection": "my_hype"}})
    emb = Embedder.from_config(cfg)
    extra = tmp_path / "extra.jsonl"
    _write_jsonl(extra, ROWS)

    emb.append_indexes(extra)                              # the store does not exist yet

    sidecar = read_sidecar(emb.chroma_dir, "obsidian_vault")
    assert (sidecar["written_by"], sidecar["status"], sidecar["role"]) == ("append", "complete", "chunks")
    assert (sidecar["provider"], sidecar["kind"], sidecar["model"]) == ("local", "local", MODEL_A)
    assert (sidecar["count"], sidecar["dimension"]) == (3, DIM)
    assert sidecar["source_digest"] == ids_digest(i for i, _ in ROWS)
    assert sidecar["hype_collection"] == "my_hype"         # from retrieval.hype.collection
    assert check_collection(emb, emb.chroma_dir, "obsidian_vault") == sidecar

    # An existing but EMPTY collection holds nothing to protect either.
    empty_cfg = _cfg(tmp_path / "second", embedding={"provider": "local", "local_model": MODEL_A})
    empty_cfg.path("paths.chroma_dir").mkdir(parents=True)
    chromadb.PersistentClient(path=str(empty_cfg.path("paths.chroma_dir"))).create_collection(
        "obsidian_vault", metadata={"hnsw:space": "cosine"})
    second = Embedder.from_config(empty_cfg)
    second.append_indexes(extra)
    assert read_sidecar(second.chroma_dir, "obsidian_vault")["written_by"] == "append"


def test_append_into_an_unstamped_legacy_collection_proceeds_with_a_warning(tmp_path, caplog):
    cfg = _cfg(tmp_path)
    emb = _embedder(cfg)
    emb._build_dense(_chunks())                            # the live store: vectors, no sidecar
    extra = tmp_path / "extra.jsonl"
    _write_jsonl(extra, [("d9", "delta")])

    with caplog.at_level(logging.INFO):
        emb.append_indexes(extra)

    assert _count(emb) == 4                                # written as it always was
    (warning,) = _warnings(caplog)
    assert "'obsidian_vault' has no embedding fingerprint" in warning.getMessage()
    # Only `stamp`, after verifying, may vouch for a collection nobody fingerprinted.
    assert read_sidecar(emb.chroma_dir, "obsidian_vault") is None


def test_a_full_rebuild_replaces_a_stale_sidecar(tmp_path):
    cfg = _cfg(tmp_path)
    emb = _embedder(cfg)
    _write_jsonl(cfg.path("paths.chunks_file"), ROWS)
    stale = _embedder(cfg, _spec(MODEL_B))
    _write(stale, "obsidian_vault", count=999, source_digest=ids_digest(["gone"]), written_by="stamp")

    emb.build_indexes()

    sidecar = read_sidecar(emb.chroma_dir, "obsidian_vault")
    assert (sidecar["written_by"], sidecar["model"], sidecar["status"]) == ("index", MODEL_A, "complete")
    assert (sidecar["count"], sidecar["dimension"]) == (3, DIM)
    assert sidecar["source_digest"] == ids_digest(i for i, _ in ROWS)
    assert sidecar["hype_collection"] == "hype_questions"
    assert check_collection(emb, emb.chroma_dir, "obsidian_vault") == sidecar

    # The old record goes BEFORE its vectors do: a rebuild that dies half way
    # leaves an unfingerprinted store, never a fingerprint for vectors that are gone.
    class Exploding(FakeBackend):
        def embed(self, texts):
            raise RuntimeError("embedder died")

    dying = _embedder(cfg)
    dying.backend = Exploding(MODEL_A)
    with pytest.raises(RuntimeError, match="embedder died"):
        dying.build_indexes()
    assert read_sidecar(emb.chroma_dir, "obsidian_vault") is None


# ------------------------------------------------------------- build_hype.py ----

def test_build_hype_embeds_questions_on_the_query_side_and_refuses_a_mismatched_collection(
        tmp_path, monkeypatch, capsys):
    import build_hype
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "paths:\n  chunks_file: chunks.jsonl\n  chroma_dir: chroma_db\n"
        "  bm25_index: bm25.pkl\n  collection_name: obsidian_vault\n"
        "retrieval:\n  hype:\n    collection: hype_questions\n    questions_per_chunk: 2\n",
        encoding="utf-8")
    cfg = load_config(config_path)
    chunks = cfg.path("paths.chunks_file")
    _write_jsonl(chunks, [("d1", "alpha")])

    class FakeLLM:
        def complete(self, system, user):
            return SimpleNamespace(text="What does the alpha passage say?\nHow is alpha applied in practice?")

    monkeypatch.setattr("src.llm.llm_client.LLMClient.from_config",
                        classmethod(lambda cls, cfg, role=None: FakeLLM()))
    monkeypatch.setattr(build_hype, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["build_hype.py", "--config", str(config_path)])

    def run_with(emb):
        monkeypatch.setattr(build_hype, "Embedder", SimpleNamespace(from_config=lambda cfg: emb))
        build_hype.main()

    # A new collection: the questions are matched against QUERY vectors, so they
    # take the query prefix, and the collection records who embedded them.
    first = _embedder(cfg, _spec(query_prefix="q: ", doc_prefix="d: "))
    run_with(first)
    assert first.backend.calls == [["q: What does the alpha passage say?",
                                    "q: How is alpha applied in practice?"]]
    sidecar = read_sidecar(first.chroma_dir, "hype_questions")
    assert (sidecar["role"], sidecar["written_by"], sidecar["status"]) == ("hype", "build_hype", "complete")
    assert (sidecar["model"], sidecar["query_prefix"], sidecar["count"]) == (MODEL_A, "q: ", 2)
    assert sidecar["hype_collection"] is None
    assert _count(first, "hype_questions") == 2

    # Another embedder must not add its vectors to that collection.
    _write_jsonl(chunks, [("d1", "alpha"), ("d2", "beta")])
    other = _embedder(cfg, _spec(MODEL_B, query_prefix="q: ", doc_prefix="d: "))
    capsys.readouterr()
    with pytest.raises(SystemExit) as exit_info:
        run_with(other)
    assert exit_info.value.code == 3                       # the script's existing failure code
    out = capsys.readouterr().out
    assert "ERROR:" in out and MODEL_A in out and MODEL_B in out
    assert other.backend.calls == []
    assert _count(first, "hype_questions") == 2
