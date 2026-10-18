"""The console's graph-embedder surface (manage_api.py): the graph_embed job, the
Graph-tab re-index chain, GET /api/canvas/embedder, the Settings key and the
schema.

Run:  python -m pytest tests/ -q

Offline: Chroma stores under tmp_path, a fake embedding backend, no model, no
network. "On disk" means the management_module fixture's own config.yaml (a
copy of config.example.yaml in tmp_path, the console's cwd); the tracked
config.yaml is checked unchanged after every test. The only subprocess is the
real worker running `python -c pass`. The FakeBackend/_vec pattern is copied
from test_reembed.py (tests/ is not a package).
"""
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils.config_loader import Config

pytest.importorskip("chromadb")

ROOT = Path(__file__).resolve().parents[1]
DIM = 8
G2 = {"kind": "openai", "base_url": "http://127.0.0.1:8104/v1", "model": "fake/g2", "dimensions": DIM}
CANVAS = {"nodes": [{"id": "a1", "type": "text", "x": 0, "y": 0, "width": 250, "height": 60,
                     "text": "Alpha is the first node of this small concept map, and it stands alone."}],
          "edges": []}


@pytest.fixture(autouse=True)
def _tracked_config_untouched():
    path = ROOT / "config.yaml"
    before = path.read_bytes() if path.exists() else None
    yield
    assert (path.read_bytes() if path.exists() else None) == before, "a test wrote the tracked config.yaml"


@pytest.fixture(autouse=True)
def _release_chroma_stores():
    """A store's loaded collections hold C-runtime file streams (4 per HNSW
    segment, of 512 per process) until its System stops, and nothing here closes
    them. Stop the stores this test opened, so later fopen users in the same run
    (PyMuPDF in test_web_import_ocr.py) are not starved."""
    from chromadb.api.shared_system_client import SharedSystemClient
    before = set(SharedSystemClient._identifier_to_system)
    yield
    for key in set(SharedSystemClient._identifier_to_system) - before:
        SharedSystemClient._identifier_to_refcount.pop(key, None)
        SharedSystemClient._identifier_to_system.pop(key).stop()


def _vec(text):
    raw = np.frombuffer(hashlib.sha256(text.encode()).digest()[:DIM], dtype=np.uint8).astype(float) - 127.5
    return raw / np.linalg.norm(raw)


class FakeBackend:
    dimension = DIM

    def embed(self, texts):
        return [_vec("g2" + t).tolist() for t in texts]


def _on_disk(m, tmp_path, changes):
    """Write keys into the console's config.yaml: the fixture's tmp copy, its cwd."""
    assert Path.cwd().samefile(tmp_path)
    m._persist_section_keys(tmp_path / "config.yaml", changes)


@pytest.fixture
def vault(management_module, monkeypatch, tmp_path):
    """One indexed canvas; removal and queueing are recorded, never done. The
    recording enqueue builds each job's argv the way the real one does first."""
    m = management_module
    v = tmp_path / "vault"
    (v / "Maps").mkdir(parents=True)
    (v / "Maps" / "m.canvas").write_text(json.dumps(CANVAS), encoding="utf-8")
    monkeypatch.setattr(m, "_vault_root", lambda: v)
    monkeypatch.setattr(m, "CFG", Config({"canvas": {"backup_dir": str(tmp_path / "bk")},
                                          "parser": {"vault_path": str(v)}}, tmp_path))
    monkeypatch.setattr(m, "_indexed_as", lambda rel: {
        "Maps/m.canvas": {"chunks": 1, "jsonls": ["canvas_chunks.jsonl"]}})
    calls = []
    monkeypatch.setattr(m, "_chroma_delete_by_source", lambda s: calls.append(("chroma",)) or 1)
    monkeypatch.setattr(m, "_jsonl_remove_sources", lambda s, n: calls.append(("jsonl",)) or {})
    monkeypatch.setattr(m, "build_manifest", lambda force=False: {})

    class Recorded:
        def __init__(self, kind, prm):
            calls.append(("job", kind, m._build_argv(kind, prm)))
            self.kind = kind

        def public(self):
            return {"kind": self.kind}

    monkeypatch.setattr(m, "enqueue", Recorded)
    return m, calls


# ---------------------------------------------------------------- the job ----

def test_graph_embed_job_argv_limit_and_refusals(management_module, monkeypatch, tmp_path):
    m = management_module
    monkeypatch.setattr(m, "_JOBS", {})
    monkeypatch.setattr(m, "_ORDER", [])
    with pytest.raises(ValueError, match="graph.embedding_provider is null"):
        m._build_argv("graph_embed", {})                                # the shipped default

    _on_disk(m, tmp_path, {"graph.embedding_provider": "eg2"})
    assert m._build_argv("graph_embed", {}) == [sys.executable, "main.py", "graph-embed"]
    assert m._build_argv("graph_embed", {"limit": 20}) == \
        [sys.executable, "main.py", "graph-embed", "--limit", "20"]
    for bad in (0, -3, True, "20", 2.5):
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            m._build_argv("graph_embed", {"limit": bad})

    for value, fragment in (("local", "is the main collection's embedder"),   # shipped main: local
                            ("nope", "is not in embedding.providers")):
        _on_disk(m, tmp_path, {"graph.embedding_provider": value})
        r = m.jobs_create(m.JobIn(kind="graph_embed", params={}))
        assert r.status_code == 400 and fragment in json.loads(r.body)["error"]
    assert m._JOBS == {}                                                # nothing was queued


def test_reindex_queues_graph_embed_third_only_when_configured(vault, tmp_path):
    m, calls = vault
    r = m.canvas_reindex(m.CanvasPathIn(path="Maps/m.canvas"))
    assert [j["kind"] for j in r["jobs"]] == ["ingest_canvas", "index_append"]
    assert r["note"] == "restart serve_api (:8051) once the append job finishes."   # unchanged when null

    calls.clear()
    _on_disk(m, tmp_path, {"graph.embedding_provider": "eg2"})
    r = m.canvas_reindex(m.CanvasPathIn(path="Maps/m.canvas"))
    assert [j["kind"] for j in r["jobs"]] == ["ingest_canvas", "index_append", "graph_embed"]
    assert [c[0] for c in calls] == ["chroma", "jsonl", "job", "job", "job"]
    assert calls[-1][2][-1] == "graph-embed"
    assert r["note"] == "restart serve_api (:8051) once the graph_embed job finishes."


def test_an_empty_embedder_name_reads_as_null_in_the_console(vault, tmp_path):
    m, calls = vault
    _on_disk(m, tmp_path, {"graph.embedding_provider": ""})
    assert '  embedding_provider: ""' in (tmp_path / "config.yaml").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="graph.embedding_provider is null"):
        m._build_argv("graph_embed", {})                                # the job
    r = m.canvas_reindex(m.CanvasPathIn(path="Maps/m.canvas"))         # the re-index chain
    assert [j["kind"] for j in r["jobs"]] == ["ingest_canvas", "index_append"]
    assert r["note"] == "restart serve_api (:8051) once the append job finishes."
    s = m.canvas_embedder()                                             # the status
    assert (s["ok"], s["configured"], s["verdict"]) == (True, None, "off")


@pytest.mark.parametrize("value, fragment", [("local", "is the main collection's embedder"),
                                             ("nope", "is not in embedding.providers")])
def test_reindex_refuses_before_removing_anything_when_the_embedder_is_broken(
        vault, tmp_path, value, fragment):
    m, calls = vault
    _on_disk(m, tmp_path, {"graph.embedding_provider": value})
    r = m.canvas_reindex(m.CanvasPathIn(path="Maps/m.canvas"))
    assert r.status_code == 400 and fragment in json.loads(r.body)["error"]
    assert calls == []                                                  # nothing removed or queued


def test_graph_embed_triggers_the_auto_restart(management_module, monkeypatch, tmp_path):
    m = management_module
    (tmp_path / "jobs").mkdir()
    monkeypatch.setattr(m, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(m, "_JOBS", {})
    monkeypatch.setattr(m, "_ORDER", [])
    monkeypatch.setattr(m, "load_config",
                        lambda *a, **k: Config({"webui": {"auto_restart_rag": True}}, tmp_path))
    restarts = []
    monkeypatch.setattr(m, "_restart_rag_api",
                        lambda: restarts.append(1) or {"port": 8051, "killed_pid": 1, "new_pid": 2})
    jobs = []
    for kind in ("recalibrate", "graph_embed"):          # a kind that never restarts, then this one
        job = m.Job(id=uuid.uuid4().hex[:10], kind=kind, argv=[sys.executable, "-c", "pass"], params={})
        job.log_file = str(tmp_path / "jobs" / f"{job.id}.log")
        m._JOBS[job.id] = job
        m._ORDER.append(job.id)
        jobs.append(job)
    for job in jobs:
        m._QUEUE.put(job.id)                             # the real worker, one job at a time
    log = Path(jobs[1].log_file)
    deadline = time.time() + 60
    while time.time() < deadline and "auto-restarted" not in (
            log.read_text(encoding="utf-8") if log.exists() else ""):
        time.sleep(0.05)
    assert [j.status for j in jobs] == ["done", "done"]
    assert restarts == [1]                               # graph_embed restarted :8051, recalibrate not
    assert "auto-restarted :8051" in log.read_text(encoding="utf-8")
    assert "auto-restart" not in Path(jobs[0].log_file).read_text(encoding="utf-8")


# ------------------------------------------------------------- the status ----

def _store(tmp_path):
    """Main collection 'vault' (4 canvas + 2 markdown chunks), its chunk files and
    a config naming it, with graph.embedding_provider: g2 (a fake)."""
    from src.utils.chroma_client import persistent_client
    root = tmp_path / "store"
    data = root / "data"
    data.mkdir(parents=True)
    canvas = [(f"c{i}", f"canvas node {i}") for i in range(4)]
    md = [(f"m{i}", f"md text {i}") for i in range(2)]
    for name, rows, ft in (("chunks.jsonl", md, "md"), ("canvas_chunks.jsonl", canvas, "canvas")):
        (data / name).write_text("".join(json.dumps({"doc_id": i, "text": t, "metadata": {
            "file_type": ft}}) + "\n" for i, t in rows), encoding="utf-8")
    cfg = Config({"paths": {"chunks_file": str(data / "chunks.jsonl"), "chroma_dir": str(root / "chroma"),
                            "bm25_index": str(root / "bm25.pkl"), "collection_name": "vault"},
                  "embedding": {"provider": "local", "local_model": "fake/main", "batch_size": 4,
                                "providers": {"g2": G2}},
                  "graph": {"enabled": False, "file_type": "canvas", "embedding_provider": "g2"}}, root)
    client = persistent_client(cfg.path("paths.chroma_dir"))
    main = client.get_or_create_collection("vault", metadata={"hnsw:space": "cosine"})
    main.add(ids=[i for i, _ in canvas + md], embeddings=[_vec("main" + t).tolist() for _, t in canvas + md],
             documents=[t for _, t in canvas + md],
             metadatas=[{"file_type": "canvas"} for _ in canvas] + [{"file_type": "md"} for _ in md])
    return SimpleNamespace(cfg=cfg, client=client, data=data)


def test_embedder_status_reports_verdict_choices_and_files_newer(management_module, monkeypatch,
                                                                  tmp_path):
    from src.embeddings import graph_twin
    from src.embeddings.embedder import Embedder
    m = management_module
    w = _store(tmp_path)
    monkeypatch.setattr(m, "load_config", lambda *a, **k: w.cfg)
    monkeypatch.setattr(m, "DATA_DIR", w.data)
    monkeypatch.setattr(m, "build_manifest", lambda force=False: {
        "map.canvas": {"file_types": ["canvas"], "jsonls": ["canvas_chunks.jsonl"]},
        "a.md": {"file_types": ["md"], "jsonls": ["chunks.jsonl"]}})

    s = m.canvas_embedder()
    assert (s["ok"], s["verdict"], s["twin"], s["configured"]) == (True, "missing", None, "g2")
    assert (s["collection"], s["file_type"], s["graph_enabled"]) == ("vault__graph", "canvas", False)
    assert s["main"] == {"collection": "vault", "embedder": "local · fake/main", "chunks": 4}
    assert [(c["value"], c["available"]) for c in s["choices"]] == \
        [("null", True), ("g2", True), ("local", False)]
    assert s["choices"][1]["label"] == "g2 · fake/g2, 8-d"
    assert "main collection's embedder" in s["choices"][2]["reason"]

    emb = Embedder(backend=FakeBackend(), chunks_file=w.cfg.path("paths.chunks_file"),
                   chroma_dir=w.cfg.path("paths.chroma_dir"), bm25_index=w.cfg.path("paths.bm25_index"),
                   collection_name="vault", batch_size=4, spec=graph_twin.graph_spec(w.cfg))
    sidecar = graph_twin.build_twin(w.cfg, emb, client=w.client, progress=lambda _: None)
    built = time.mktime(time.strptime(sidecar["completed_at"], "%Y-%m-%dT%H:%M:%S"))
    for name in ("chunks.jsonl", "canvas_chunks.jsonl"):
        os.utime(w.data / name, (built - 60, built - 60))
    s = m.canvas_embedder()
    assert s["verdict"] == "ready" and s["twin"]["files_newer"] == []
    assert (s["twin"]["count"], s["twin"]["expected"]) == (4, 4) and "written since" not in s["reason"]

    for name in ("chunks.jsonl", "canvas_chunks.jsonl"):    # a whole-lane re-ingest since the build
        os.utime(w.data / name, (built + 60, built + 60))
    s = m.canvas_embedder()
    assert s["verdict"] == "ready" and s["twin"]["files_newer"] == ["canvas_chunks.jsonl"]   # its lane only
    assert "canvas_chunks.jsonl" in s["reason"] and "rebuild it (graph_embed)" in s["reason"]

    bad = Config({**w.cfg.as_dict(), "embedding": {**w.cfg.as_dict()["embedding"], "provider": "nope"}},
                 w.cfg.project_root)
    monkeypatch.setattr(m, "load_config", lambda *a, **k: bad)
    r = m.canvas_embedder()                                  # the main embedder itself is broken
    assert r.status_code == 400 and "embedding.provider = 'nope'" in json.loads(r.body)["error"]

    def broken(cfg, client=None):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(m, "load_config", lambda *a, **k: w.cfg)
    monkeypatch.setattr(graph_twin, "twin_status", broken)
    r = m.canvas_embedder()                                  # only a Chroma failure is a 500
    assert r.status_code == 500
    assert json.loads(r.body) == {"ok": False, "error": "ChromaDB read failed: RuntimeError: database is locked"}


# -------------------------------------------------- settings and the schema ----

def test_graph_embedder_setting_values_are_filled_at_request_time(management_module, monkeypatch):
    m = management_module
    monkeypatch.setattr(m, "_torch_devices", lambda: ({}, ["auto", "cpu"]))
    row = m.settings()["editable"]["graph.embedding_provider"]
    assert row["values"] == ["null", "local", "eg2", "eg2_l512"]        # embedding.providers on disk
    assert (row["value"], row["kind"], row["restart"]) == ("null", "enum", ":8051")   # null reads "null"
    assert m.EDITABLE_SETTINGS["graph.embedding_provider"]["values"] == []         # the spec stays empty

    written = []
    monkeypatch.setattr(m, "_persist_section_keys",
                        lambda path, changes: written.append(dict(changes)) or list(changes))
    for value in ("eg2", "null", "local"):
        assert m.settings_update(m.SettingsIn(changes={"graph.embedding_provider": value}))["ok"] is True
    assert written == [{"graph.embedding_provider": v} for v in ("eg2", "null", "local")]
    r = m.settings_update(m.SettingsIn(changes={"graph.embedding_provider": "openai"}))
    assert r.status_code == 400 and "must be one of" in json.loads(r.body)["error"]
    assert len(written) == 3


def test_schema_tiers_the_route_and_job_kind(management_module):
    from fastapi.routing import APIRoute
    m = management_module
    schema = m.api_schema()
    assert schema["endpoints"]["GET /api/canvas/embedder"]["permission"] == "read"
    kind = schema["job_kinds"]["graph_embed"]
    assert kind["permission"] == "mutating" and set(kind["params"]) == {"limit"}
    assert "from scratch" in kind["note"] and "Never writes the main collection" in kind["note"]
    routes = {(method, r.path) for r in m.app.routes if isinstance(r, APIRoute) for method in r.methods}
    assert ("GET", "/api/canvas/embedder") in routes
    assert "graph.embedding_provider" in m._FINGERPRINT_KEYS
