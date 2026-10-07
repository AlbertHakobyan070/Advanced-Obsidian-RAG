"""`main.py stamp`: record which embedder built a collection, after proving it
(embedding-switch slice 2).

Run:  python -m pytest tests/ -q

The live collection predates fingerprints, so `stamp` is the one-time way to
add one. A stamp must be a proof, not a claim: a sample of stored chunks is
re-embedded by the configured embedder and each must match its stored vector,
and nothing is written until that has passed.

No model is ever loaded: the embedder is the real Embedder around a fake
backend whose vectors depend on a salt (its "model") and the text. The vectors
are zero-mean on purpose: unrelated non-negative ones all have cosine ~0.75,
which would blur the pass/fail line this file tests.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.embeddings.embedder import Chunk, Embedder
from src.embeddings.registry import EmbeddingSpec
from src.embeddings.sidecar import (
    StampError, check_collection, delete_sidecar, ids_digest, make_sidecar, read_sidecar,
    role_for, sidecar_path, stamp_collection, write_sidecar)
from src.utils.config_loader import Config, load_config

chromadb = pytest.importorskip("chromadb")

ROOT = Path(__file__).resolve().parents[1]
DIM = 16
MODEL_A, MODEL_B = "fake/model-a", "fake/model-b"
ROWS = [(f"d{i:02d}", f"text number {i}") for i in range(40)]


def _vec(text, dim=DIM):
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest()[:dim], dtype=np.uint8)
    raw = raw.astype(float) - 127.5
    return raw / np.linalg.norm(raw)


class FakeBackend:
    def __init__(self, salt, dim=DIM):
        self.salt, self.dim, self.calls = salt, dim, []

    @property
    def dimension(self):
        return self.dim

    def embed(self, texts):
        self.calls.append(list(texts))
        return [_vec(self.salt + t, self.dim).tolist() for t in texts]


def _cfg(tmp_path, collection="obsidian_vault", **sections):
    data = {"paths": {"chunks_file": "chunks.jsonl", "chroma_dir": "chroma_db",
                      "bm25_index": "bm25.pkl", "collection_name": collection}}
    data.update(sections)
    return Config(data, tmp_path)


def _embedder(cfg, model=MODEL_A, dim=DIM, **spec_kw):
    spec = EmbeddingSpec("local", "local", model, **spec_kw)
    return Embedder(
        backend=FakeBackend(model, dim), chunks_file=cfg.path("paths.chunks_file"),
        chroma_dir=cfg.path("paths.chroma_dir"), bm25_index=cfg.path("paths.bm25_index"),
        collection_name=cfg.get("paths.collection_name"), batch_size=16, spec=spec,
        hype_collection_name=cfg.get("retrieval.hype.collection", "hype_questions"))


def _build(emb, rows=ROWS):
    """The collection `emb` would have built: its vectors, no sidecar."""
    emb._build_dense([Chunk(i, t, {"source_file": "a.md"}) for i, t in rows])


def _write(emb, name, **fields):
    args = dict(collection=name, role="chunks", status="complete",
                dimension=emb.dimension(), count=1, written_by="index")
    args.update(fields)
    return write_sidecar(emb.chroma_dir, name, make_sidecar(emb.spec, **args))


def _sidecar_files(emb):
    return sorted(p.name for p in emb.chroma_dir.glob("*.embedding.json*"))


# ------------------------------------------------------------ the happy path ----

def test_stamp_verifies_then_writes(tmp_path):
    cfg = _cfg(tmp_path)
    emb = _embedder(cfg)
    _build(emb)

    sidecar, outcome = stamp_collection(cfg, emb)

    assert outcome == "written"
    assert read_sidecar(emb.chroma_dir, "obsidian_vault") == sidecar
    assert (sidecar["written_by"], sidecar["status"], sidecar["role"]) == ("stamp", "complete", "chunks")
    assert (sidecar["provider"], sidecar["model"], sidecar["dimension"]) == ("local", MODEL_A, DIM)
    assert sidecar["count"] == 40
    assert sidecar["source_digest"] == ids_digest(i for i, _ in ROWS)
    assert sidecar["hype_collection"] == "hype_questions"            # the legacy twin's name, from config
    verified = sidecar["verified"]
    assert (verified["sample"], verified["threshold"]) == (32, 0.99)   # the documented defaults
    assert verified["min_cosine"] >= 0.99
    assert check_collection(emb, emb.chroma_dir, "obsidian_vault") == sidecar   # the guard accepts it now

    # The sample size and the bar come from embedding.verify.
    delete_sidecar(emb.chroma_dir, "obsidian_vault")
    tuned = _cfg(tmp_path, embedding={"verify": {"sample_size": 5, "min_cosine": 0.95}})
    sidecar, _ = stamp_collection(tuned, emb)
    assert (sidecar["verified"]["sample"], sidecar["verified"]["threshold"]) == (5, 0.95)

    # A collection smaller than the sample is verified whole.
    small = _cfg(tmp_path, collection="small_collection")
    small_emb = _embedder(small)
    _build(small_emb, ROWS[:3])
    assert stamp_collection(small, small_emb)[0]["verified"]["sample"] == 3

    # Both shipped configs carry the keys stamp reads, with the documented defaults.
    # A fresh clone has no config.yaml (see conftest).
    for name in ("config.yaml", "config.example.yaml"):
        if (ROOT / name).exists():
            assert load_config(ROOT / name).get("embedding.verify") == {"sample_size": 32, "min_cosine": 0.99}, name


# --------------------------------------------------------------- refusals ----

def test_stamp_refuses_a_collection_built_by_another_embedder_and_writes_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    built_by = _embedder(cfg, MODEL_A)
    _build(built_by)
    configured = _embedder(cfg, MODEL_B)             # same width, another model: Chroma would not notice

    with pytest.raises(StampError, match=rf"min cosine.*threshold 0\.99.*{MODEL_B}"):
        stamp_collection(cfg, configured)

    assert _sidecar_files(built_by) == []            # the proof comes BEFORE the write

    # A missing collection and a missing store are refused too, not created.
    with pytest.raises(StampError, match="does not exist"):
        stamp_collection(cfg, configured, "no_such_collection")
    with pytest.raises(StampError, match="no Chroma store"):
        stamp_collection(_cfg(tmp_path / "elsewhere"), configured)
    assert not (tmp_path / "elsewhere").exists()


def test_stamp_reports_a_dimension_mismatch_instead_of_crashing(tmp_path):
    cfg = _cfg(tmp_path)
    _build(_embedder(cfg))                           # 16-d vectors
    narrower = _embedder(cfg, dim=8)                 # the same model, asked for 8-d

    with pytest.raises(StampError, match=r"16-d vectors.*8-d"):
        stamp_collection(cfg, narrower)
    assert _sidecar_files(narrower) == []


def test_stamp_is_a_no_op_on_a_matching_sidecar_and_needs_force_to_overwrite_a_different_one(tmp_path):
    cfg = _cfg(tmp_path)
    emb_a, emb_b = _embedder(cfg, MODEL_A), _embedder(cfg, MODEL_B)
    _build(emb_a)
    path = sidecar_path(emb_a.chroma_dir, "obsidian_vault")

    written, outcome = stamp_collection(cfg, emb_a)
    assert outcome == "written"
    before = path.read_bytes()
    again, outcome = stamp_collection(cfg, emb_a)
    assert (again, outcome) == (written, "already")
    assert path.read_bytes() == before                           # nothing rewritten

    # A record that disagrees with the configured embedder: refused, naming both.
    _write(emb_b, "obsidian_vault")
    disagreeing = path.read_bytes()
    with pytest.raises(StampError, match=rf"{MODEL_B}.*{MODEL_A}.*--force"):
        stamp_collection(cfg, emb_a)
    assert path.read_bytes() == disagreeing

    # --force replaces it, and only because the sample passes.
    replaced, outcome = stamp_collection(cfg, emb_a, force=True)
    assert (outcome, replaced["model"]) == ("written", MODEL_A)

    # --force replaces the RECORD, never the proof: a collection that is not the
    # configured embedder's still fails its sample check.
    other = _cfg(tmp_path, collection="built_by_b")
    _build(_embedder(other, MODEL_B))
    _write(_embedder(other, MODEL_B), "built_by_b")
    recorded = sidecar_path(emb_a.chroma_dir, "built_by_b").read_bytes()
    with pytest.raises(StampError, match="min cosine"):
        stamp_collection(other, _embedder(other, MODEL_A), force=True)
    assert sidecar_path(emb_a.chroma_dir, "built_by_b").read_bytes() == recorded


def test_stamp_refuses_an_unfinished_reembed(tmp_path):
    cfg = _cfg(tmp_path)
    emb = _embedder(cfg)
    _build(emb)
    _write(emb, "obsidian_vault", status="building", count=40, expected=100)
    before = sidecar_path(emb.chroma_dir, "obsidian_vault").read_bytes()

    for force in (False, True):                      # not even --force: finish it, don't stamp it
        with pytest.raises(StampError, match=r"unfinished re-embed.*40 of 100"):
            stamp_collection(cfg, emb, force=force)
    assert sidecar_path(emb.chroma_dir, "obsidian_vault").read_bytes() == before


# ---------------------------------------------------------- the HyPE side ----

def test_stamping_the_hype_collection_uses_the_query_side(tmp_path):
    cfg = _cfg(tmp_path)
    emb = _embedder(cfg, query_prefix="q: ", doc_prefix="d: ")
    questions = [f"what is topic {i}?" for i in range(6)]
    col = chromadb.PersistentClient(path=str(emb.chroma_dir)).create_collection(
        "hype_questions", metadata={"hnsw:space": "cosine"})
    col.add(ids=[f"d{i}::q0" for i in range(6)], embeddings=emb.embed_queries(questions),
            documents=questions, metadatas=[{"doc_id": f"d{i}", "source_file": "a.md"} for i in range(6)])
    emb.backend.calls.clear()

    sidecar, outcome = stamp_collection(cfg, emb, "hype_questions")

    assert outcome == "written" and sidecar["role"] == "hype"
    assert sidecar["hype_collection"] is None                    # a HyPE collection has no twin of its own
    (sample,) = emb.backend.calls                                # embedded query-side, never document-side
    assert sample and all(text.startswith("q: ") for text in sample)

    assert role_for(cfg, "hype_questions") == role_for(cfg, "vault_e5__hype") == "hype"
    assert role_for(cfg, "obsidian_vault") == "chunks"


# ------------------------------------------------------------------ the CLI ----

def test_stamp_cli_exits_2_with_an_error_line_on_a_mismatch(tmp_path, monkeypatch, capsys):
    import main
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "paths:\n  chunks_file: chunks.jsonl\n  chroma_dir: chroma_db\n"
        "  bm25_index: bm25.pkl\n  collection_name: obsidian_vault\n", encoding="utf-8")
    cfg = load_config(config_path)
    built_by = _embedder(cfg, MODEL_A)
    _build(built_by)
    chromadb.PersistentClient(path=str(built_by.chroma_dir)).create_collection(
        "hype_questions", metadata={"hnsw:space": "cosine"})     # a legacy twin nobody stamped

    configured = {}
    monkeypatch.setattr(main, "_bootstrap_logging", lambda cfg: None)
    monkeypatch.setattr("src.embeddings.embedder.Embedder.from_config",
                        classmethod(lambda cls, cfg, spec=None: configured["emb"]))

    def stamp(*extra):
        args = main.build_parser().parse_args(["--config", str(config_path), "stamp", *extra])
        args.func(args)

    # The matching embedder first, so the failure below differs in nothing else.
    configured["emb"] = _embedder(cfg, MODEL_A)
    stamp()
    out = capsys.readouterr()
    assert "Verifying 'obsidian_vault' (40 chunks) against local · fake/model-a" in out.out
    assert "32 sampled chunks re-embedded: min cosine" in out.out and "(threshold 0.99)" in out.out
    assert "Stamped 'obsidian_vault': local · fake/model-a, 16-d, 40 chunks" in out.out
    assert str(sidecar_path(built_by.chroma_dir, "obsidian_vault")) in out.out
    assert "HyPE collection 'hype_questions' is not stamped: `rag stamp --collection hype_questions`" in out.out
    assert "ERROR" not in out.err

    stamp()                                                    # again: nothing to do
    assert "already" in capsys.readouterr().out

    delete_sidecar(built_by.chroma_dir, "obsidian_vault")
    configured["emb"] = _embedder(cfg, MODEL_B)
    with pytest.raises(SystemExit) as exit_info:
        stamp()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "ERROR: " in err and "min cosine" in err and MODEL_B in err
    assert _sidecar_files(built_by) == []

    with pytest.raises(SystemExit) as exit_info:                 # a collection that is not there
        stamp("--collection", "no_such_collection")
    assert exit_info.value.code == 2
    assert "ERROR: " in capsys.readouterr().err
