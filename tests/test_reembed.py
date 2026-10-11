"""`main.py index --rebuild-into`: re-embed the active collection's corpus into a
NEW collection, resumably, and `bench run --collection` scoring it.

Run:  python -m pytest tests/ -q

Real Chroma stores under tmp_path, a fake embedding backend (vectors depend on
a salt that stands for the model, and on the text), no model, no network, no
GPU. The vectors are zero-mean so a wrong-model comparison lands far below the
0.99 sample check instead of near 0.75.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.embeddings import reembed
from src.embeddings.embedder import Chunk, Embedder
from src.embeddings.reembed import ReembedError, config_for_collection, rebuild_into
from src.embeddings.registry import EmbeddingSpec, resolve_embedding_spec
from src.embeddings.sidecar import (
    EmbeddingMismatchError, check_collection, ids_digest, iter_ids, read_sidecar)
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import Config

pytest.importorskip("chromadb")

ROOT = Path(__file__).resolve().parents[1]
DIM = 16
N = 30
ENTRY = {"kind": "openai", "base_url": "http://127.0.0.1:8104/v1", "model": "fake/eg2",
         "dimensions": DIM, "max_seq_length": 1024, "query_prefix": "q: ", "doc_prefix": "d: "}


def _vec(text):
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest()[:DIM], dtype=np.uint8).astype(float) - 127.5
    return raw / np.linalg.norm(raw)


class FakeBackend:
    def __init__(self, salt="eg2", fail_on_call=None):
        self.salt, self.fail_on_call, self.calls = salt, fail_on_call, []
        self.dimension = DIM

    def embed(self, texts):
        if self.fail_on_call is not None and len(self.calls) + 1 == self.fail_on_call:
            raise ConnectionError("embedding sidecar went away")
        self.calls.append(list(texts))
        return [_vec(self.salt + t).tolist() for t in texts]


def _jsonl(path, rows):
    path.write_text("".join(json.dumps({"doc_id": i, "text": t, "metadata": {"source_file": "a.md"}})
                            + "\n" for i, t in rows), encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    """An active collection of N chunks, chunks.jsonl with the first 20 and
    pdf_chunks.jsonl with the rest (the union is the corpus), and a config."""
    data = tmp_path / "data"
    data.mkdir()
    rows = [(f"d{i:02d}", f"chunk text {i}") for i in range(N)]
    _jsonl(data / "chunks.jsonl", rows[:20])
    _jsonl(data / "pdf_chunks.jsonl", rows[20:])
    cfg = Config({"paths": {"chunks_file": str(data / "chunks.jsonl"), "chroma_dir": str(tmp_path / "chroma"),
                            "bm25_index": str(tmp_path / "bm25.pkl"), "collection_name": "active"},
                  "embedding": {"provider": "eg2", "batch_size": 7, "providers": {"eg2": ENTRY}}}, tmp_path)
    client = persistent_client(cfg.path("paths.chroma_dir"))
    active = client.get_or_create_collection("active", metadata={"hnsw:space": "cosine"})
    active.add(ids=[i for i, _ in rows], embeddings=[_vec("old" + t).tolist() for _, t in rows],
               documents=[t for _, t in rows],
               metadatas=[{"source_file": "a.md", "course": "ml" if n % 2 else "db"} for n, _ in enumerate(rows)])
    return SimpleNamespace(cfg=cfg, client=client, active=active, rows=rows, data=data)


def _embedder(cfg, backend=None, entry=None):
    if entry is not None:
        cfg = Config({**cfg.as_dict(), "embedding": {**cfg.as_dict()["embedding"], "providers": {"eg2": entry}}},
                     cfg.project_root)
    spec = resolve_embedding_spec(cfg)
    return Embedder(backend=backend or FakeBackend(), chunks_file=cfg.path("paths.chunks_file"),
                    chroma_dir=cfg.path("paths.chroma_dir"), bm25_index=cfg.path("paths.bm25_index"),
                    collection_name="active", batch_size=cfg.get("embedding.batch_size"), spec=spec)


def _quiet(_):
    pass


def _run(w, target="eg2_new", embedder=None, **kw):
    return rebuild_into(w.cfg, target, embedder or _embedder(w.cfg), client=w.client, progress=_quiet, **kw)


# ---------------------------------------------------------------- the build ----

def test_rebuild_holds_exactly_the_active_ids_with_their_metadata_and_a_complete_fingerprint(world):
    emb = _embedder(world.cfg)
    sidecar = _run(world, embedder=emb)

    target = world.client.get_collection("eg2_new")
    got = target.get(include=["documents", "metadatas"])
    assert sorted(got["ids"]) == [i for i, _ in world.rows]
    assert dict(zip(got["ids"], got["documents"]))["d25"] == "chunk text 25"      # raw text, no prefix
    assert dict(zip(got["ids"], got["metadatas"]))["d01"] == {"source_file": "a.md", "course": "ml"}
    assert target.metadata["hnsw:space"] == "cosine"
    assert world.active.count() == N                                              # the active one untouched

    assert (sidecar["status"], sidecar["role"], sidecar["count"], sidecar["expected"]) == ("complete", "chunks", N, N)
    assert (sidecar["model"], sidecar["dimension"], sidecar["max_seq_length"]) == ("fake/eg2", DIM, 1024)
    assert sidecar["source_digest"] == ids_digest(i for i, _ in world.rows)
    assert sidecar["source_collection"] == "active" and sidecar["written_by"] == "rebuild-into"
    assert sidecar["verified"]["min_cosine"] >= 0.99
    assert (sidecar["hype"], sidecar["hype_collection"]) == (
        {"mode": "none", "collection": "eg2_new__hype", "count": 0}, "eg2_new__hype")
    assert sidecar["build"]["embedded"] == N and sidecar["build"]["text_drift"] == 0
    assert read_sidecar(world.cfg.path("paths.chroma_dir"), "eg2_new") == sidecar
    assert read_sidecar(world.cfg.path("paths.chroma_dir"), "active") is None     # no fingerprint written there
    # the vectors are the prefixed ones, and the guard accepts the finished collection
    assert emb.backend.calls[0][0] == "d: chunk text 0"
    assert check_collection(emb, world.cfg.path("paths.chroma_dir"), "eg2_new") == sidecar


def test_text_comes_from_the_jsonls_first_occurrence_winning_and_drift_is_counted(world):
    # chunks.jsonl is read before pdf_chunks.jsonl, which also holds d25
    _jsonl(world.data / "chunks.jsonl", world.rows[:20] + [("d25", "FROM chunks.jsonl")])
    sidecar = _run(world)
    got = world.client.get_collection("eg2_new").get(ids=["d25"], include=["documents"])
    assert got["documents"] == ["FROM chunks.jsonl"]
    assert sidecar["build"]["text_drift"] == 1                                   # the active collection says otherwise


def test_ids_the_jsonls_cannot_supply_are_listed_and_nothing_is_written(world):
    _jsonl(world.data / "pdf_chunks.jsonl", world.rows[20:27])                    # d27..d29 now in no file
    backend = FakeBackend()
    with pytest.raises(ReembedError, match=r"3 of 30 ids.*d27.*d28.*d29"):
        _run(world, embedder=_embedder(world.cfg, backend))
    assert backend.calls == []
    assert "eg2_new" not in {c.name for c in world.client.list_collections()}
    assert read_sidecar(world.cfg.path("paths.chroma_dir"), "eg2_new") is None


# ------------------------------------------------------------------- resume ----

def test_a_crash_leaves_an_unservable_building_sidecar_and_a_rerun_embeds_only_the_rest(world):
    chroma_dir = world.cfg.path("paths.chroma_dir")
    with pytest.raises(ConnectionError):
        _run(world, embedder=_embedder(world.cfg, FakeBackend(fail_on_call=3)))   # 2 batches of 7 land

    sidecar = read_sidecar(chroma_dir, "eg2_new")
    assert sidecar["status"] == "building" and sidecar["expected"] == N
    assert world.client.get_collection("eg2_new").count() == 14
    with pytest.raises(EmbeddingMismatchError, match="unfinished re-embed"):      # the guard refuses it
        check_collection(_embedder(world.cfg), chroma_dir, "eg2_new")

    backend = FakeBackend()
    done = _run(world, embedder=_embedder(world.cfg, backend))
    assert sum(len(c) for c in backend.calls[:-1]) == N - 14                      # present ids skipped
    assert len(backend.calls[-1]) == N                                            # (the last call is the sample check)
    assert (done["status"], done["count"], done["build"]["skipped_present"]) == ("complete", N, 14)
    assert done["created_at"] == sidecar["created_at"]                            # the first run's start kept
    assert set(iter_ids(world.client.get_collection("eg2_new"))) == {i for i, _ in world.rows}


def test_a_top_up_prunes_ids_the_active_collection_lost_and_resyncs_changed_metadata(world):
    _run(world)
    world.active.delete(ids=["d29"])
    world.active.update(ids=["d00"], metadatas=[{"source_file": "a.md", "course": "retagged"}])
    sidecar = _run(world)
    target = world.client.get_collection("eg2_new")
    assert target.count() == N - 1 and sidecar["build"]["pruned"] == 1
    assert sidecar["build"]["metadata_synced"] == 1 and sidecar["build"]["embedded"] == 0
    assert target.get(ids=["d00"], include=["metadatas"])["metadatas"][0]["course"] == "retagged"
    assert sidecar["source_digest"] == ids_digest(i for i, _ in world.rows[:-1])


def test_limit_embeds_a_subset_verifies_it_and_leaves_it_unservable(world):
    backend = FakeBackend()
    sidecar = _run(world, embedder=_embedder(world.cfg, backend), limit=5)
    assert world.client.get_collection("eg2_new").count() == 5
    assert (sidecar["status"], sidecar["count"], sidecar["expected"]) == ("building", 5, N)
    assert sidecar["verified"]["sample"] == 5 and sidecar["build"]["limited_to"] == 5
    with pytest.raises(EmbeddingMismatchError, match="unfinished"):
        check_collection(_embedder(world.cfg), world.cfg.path("paths.chroma_dir"), "eg2_new")
    with pytest.raises(ReembedError, match="--limit is a smoke option"):          # never prunes a real target
        _run(world, limit=5)
    full = _run(world)                                                            # a full run resumes it
    assert full["status"] == "complete" and full["build"]["skipped_present"] == 5


# --------------------------------------------------------- what it refuses ----

@pytest.mark.parametrize("name, fragment", [
    ("active", "never written"),
    ("hype_questions", "never written"),
    ("x__hype", "reserved"),
    ("ab", "not a collection name"),
    ("has space", "not a collection name"),
])
def test_target_names_are_validated_before_anything_happens(world, name, fragment):
    backend = FakeBackend()
    with pytest.raises(ReembedError, match=fragment):
        _run(world, target=name, embedder=_embedder(world.cfg, backend))
    assert backend.calls == []


def test_an_existing_collection_without_a_fingerprint_is_not_a_target(world):
    world.client.get_or_create_collection("someone_elses")
    with pytest.raises(ReembedError, match="no embedding fingerprint"):
        _run(world, target="someone_elses")


def test_a_target_started_by_another_embedder_or_cap_is_refused(world):
    _run(world)
    other_cap = _embedder(world.cfg, entry={**ENTRY, "max_seq_length": 512})
    with pytest.raises(ReembedError, match="max_seq_length"):
        _run(world, embedder=other_cap)
    other_model = _embedder(world.cfg, entry={**ENTRY, "model": "fake/other"})
    with pytest.raises(ReembedError, match="model"):
        _run(world, embedder=other_model)


def test_a_missing_active_collection_is_an_error(world):
    world.client.delete_collection("active")
    with pytest.raises(ReembedError, match="does not exist"):
        _run(world)


# ------------------------------------------------------------------ the CLI ----

def _main(monkeypatch, cfg, *argv):
    import main
    monkeypatch.setattr(main, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(main, "_bootstrap_logging", lambda c: None)
    args = main.build_parser().parse_args(["index", *argv])
    args.func(args)


@pytest.mark.parametrize("argv, fragment", [
    (["--rebuild-into", "n1", "--append", "f.jsonl"], "cannot be combined"),
    (["--limit", "5"], "only apply with --rebuild-into"),
    (["--embedding", "eg2"], "only apply with --rebuild-into"),
    (["--rebuild-into", "n1", "--embedding", "nope"], "not in embedding.providers"),
])
def test_cli_misuse_is_an_error_line_and_exit_2(world, monkeypatch, capsys, argv, fragment):
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, world.cfg, *argv)
    assert e.value.code == 2 and fragment in capsys.readouterr().err


def test_cli_flags_reach_the_engine(world, monkeypatch, capsys):
    seen = {}

    def fake_rebuild(cfg, target, emb, limit=None):
        seen.update(target=target, label=emb.spec.label(), limit=limit)
        return {"status": "building", "count": 5, "expected": N, "dimension": DIM,
                "build": {"embedded": 5, "chunks_per_s": 1.0}, "verified": {"min_cosine": 1.0}}

    monkeypatch.setattr("src.embeddings.reembed.rebuild_into", fake_rebuild)
    monkeypatch.setattr(Embedder, "from_config", classmethod(lambda cls, cfg, spec=None: _embedder(cfg)))
    _main(monkeypatch, world.cfg, "--rebuild-into", "tryout", "--embedding", "eg2", "--limit", "5")
    assert seen == {"target": "tryout", "label": "eg2 · fake/eg2", "limit": 5}
    assert "left 'building' on purpose" in capsys.readouterr().out


# -------------------------------------------------- bench --collection ----

def test_config_for_collection_points_at_the_collection_and_its_embedder(world):
    _run(world)
    derived = config_for_collection(world.cfg, "eg2_new")
    assert derived.get("paths.collection_name") == "eg2_new" and derived.get("embedding.provider") == "eg2"
    assert world.cfg.get("paths.collection_name") == "active"                     # the original is untouched


def test_config_for_collection_refuses_what_cannot_be_scored_fairly(world):
    with pytest.raises(ValueError, match="no embedding fingerprint"):
        config_for_collection(world.cfg, "nope")

    _run(world, target="smoke", limit=3)
    with pytest.raises(ValueError, match="only a complete chunk collection"):
        config_for_collection(world.cfg, "smoke")

    _run(world)
    changed = Config({**world.cfg.as_dict(), "embedding": {
        "provider": "eg2", "providers": {"eg2": {**ENTRY, "max_seq_length": 512}}}}, world.cfg.project_root)
    with pytest.raises(ValueError, match="max_seq_length"):
        config_for_collection(changed, "eg2_new")

    world.active.delete(ids=["d00"])
    with pytest.raises(ValueError, match="same ids as the active collection"):
        config_for_collection(world.cfg, "eg2_new")


def test_bench_run_scores_the_derived_config_and_a_refusal_is_a_clean_error(tmp_path, monkeypatch, capsys):
    from eval.bench import cli
    from eval.bench.questions import save_suite
    sets = tmp_path / "sets"
    sets.mkdir()
    save_suite(sets / "lexical.yaml", [])
    monkeypatch.setattr(cli, "_load", lambda *a, **k: [SimpleNamespace(id="q1")])
    monkeypatch.setattr(cli, "resolve", lambda spec, path: [("R0-bm25", {})])
    monkeypatch.setattr(cli, "load_config", lambda path=None: "ACTIVE-CFG")
    got = {}

    def stop_at_run_bench(cfg, *a, **k):
        got["cfg"] = cfg
        raise SystemExit("reached run_bench")

    monkeypatch.setattr("eval.bench.runner.run_bench", stop_at_run_bench)
    monkeypatch.setattr("src.embeddings.reembed.config_for_collection",
                        lambda cfg, name: f"{cfg}+{name}")
    parsed = _bench_args("--sets", str(sets), "--collection", "eg2_new")
    with pytest.raises(SystemExit):
        cli._cmd_run(parsed)
    assert got["cfg"] == "ACTIVE-CFG+eg2_new"

    def refuse(cfg, name):
        raise ValueError("collection 'x' has no embedding fingerprint")

    monkeypatch.setattr("src.embeddings.reembed.config_for_collection", refuse)
    with pytest.raises(SystemExit) as e:
        cli._cmd_run(parsed)
    assert e.value.code == 2 and "no embedding fingerprint" in capsys.readouterr().err


def _bench_args(*argv):
    import main
    return main.build_parser().parse_args(["bench", "run", *argv])


def test_the_run_record_names_the_collection_and_the_registry_model_and_cap(world, tmp_path):
    from eval.bench.fingerprint import index_fingerprint
    from eval.bench.report import _header
    rag = SimpleNamespace(retriever=SimpleNamespace(_get_collection=lambda: SimpleNamespace(count=lambda: 3)))
    cfg = Config({**world.cfg.as_dict(), "paths": {**world.cfg.as_dict()["paths"], "collection_name": "eg2_new"}},
                 world.cfg.project_root)
    index = index_fingerprint(cfg, rag)
    assert index["collection"] == "eg2_new"
    assert index["embedding"] == {"provider": "eg2", "model": "fake/eg2", "max_seq_length": 1024}

    meta = {"git": {"sha": "a" * 40, "dirty": False}, "config_digest": "d", "index": index, "split": "dev",
            "n_questions": 1, "suites": ["lexical"], "test_ledger_count": 0}
    assert "collection eg2_new (eg2 · fake/eg2, cap 1024)" in "\n".join(_header({}, meta))
    meta["index"] = {"dense_count": 3}                                            # an older run record
    assert "collection" not in "\n".join(_header({}, meta))
