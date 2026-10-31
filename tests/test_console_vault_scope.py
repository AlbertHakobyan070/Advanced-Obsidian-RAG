"""What the console needs to sort chunk files by age and to scope jobs per vault.

  1. /api/overview reports each chunk file's mtime (`jsonl_mtime`), so the
     Ledger can list the newest lanes first. `by_jsonl` is unchanged.
  2. Every job says which vault it ran against (`vault` in /api/jobs): the one
     active when it was queued, or the one its on-disk record names once it is
     restored. A record from before the field existed has no vault ("") and
     the console counts it as the active vault's.
"""
import json
import os
import time

import pytest


@pytest.fixture
def m(tmp_path, monkeypatch, management_module):
    d = tmp_path / "jobs"
    d.mkdir()
    monkeypatch.setattr(management_module, "JOBS_DIR", d)
    monkeypatch.setattr(management_module, "_JOBS", {})
    monkeypatch.setattr(management_module, "_ORDER", [])
    return management_module


def test_overview_reports_each_chunk_files_mtime(m, tmp_path, monkeypatch):
    old, new = tmp_path / "old_chunks.jsonl", tmp_path / "new_chunks.jsonl"
    for p, t in ((old, 1_000_000), (new, 2_000_000)):
        p.write_text("", encoding="utf-8")
        os.utime(p, (t, t))
    monkeypatch.setattr(m, "chunk_files", lambda: [old, new])
    monkeypatch.setattr(m, "build_manifest", lambda: {})

    def no_store():
        raise RuntimeError("no Chroma in this test")
    monkeypatch.setattr(m, "_collection", no_store)

    def no_rag(*a, **kw):
        raise ConnectionError(":8051 is not part of this test")
    monkeypatch.setattr("requests.get", no_rag)

    o = m.overview()
    assert o["jsonl_mtime"] == {"old_chunks.jsonl": 1_000_000.0,
                                "new_chunks.jsonl": 2_000_000.0}
    assert o["by_jsonl"] == {}, "by_jsonl is left as it was"


def test_a_queued_job_carries_the_active_vault(m, monkeypatch):
    monkeypatch.setattr(m._QUEUE, "put", lambda jid: None)   # nothing runs
    job = m.enqueue("recalibrate", {"dry_run": True})
    assert job.public()["vault"] == str(m.CFG.get("parser.vault_path") or "")


def _record(jobs_dir, jid, **extra):
    rec = {"record_version": 1, "id": jid, "kind": "ingest_md", "status": "done",
           "argv": ["py", "main.py", "ingest-md"], "params": {}, "log_file": f"{jid}.log",
           "returncode": 0, "created": time.time(), "started": None, "ended": None}
    rec.update(extra)
    (jobs_dir / f"{jid}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_a_restored_job_keeps_the_vault_its_record_names(m):
    _record(m.JOBS_DIR, "withv", vault_path="D:/Vaults/Animus")
    _record(m.JOBS_DIR, "nov")                       # a record from before the field
    assert m._load_job_records() == 2
    assert m._JOBS["withv"].public()["vault"] == "D:/Vaults/Animus"
    assert m._JOBS["nov"].public()["vault"] == ""
