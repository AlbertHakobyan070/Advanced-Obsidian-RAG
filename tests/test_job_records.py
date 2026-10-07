"""Job records: the durable half of the job system.

WHAT THIS PROTECTS. `_JOBS` is in-memory and dies with the process, so before
records existed a finished job left only its .log — enough to see WHAT
happened, not HOW. A 2026-09-20 audit found ~97% of this corpus had no recorded
ingest command anywhere, which is what makes those chunk files primary data
rather than regenerable artefacts.

The contracts worth pinning:

  1. A record exists from the moment a job is QUEUED, not when it finishes —
     otherwise a crash mid-run leaves nothing, which is exactly the case the
     feature is for.
  2. It carries enough to replay: argv, cwd, the active vault, and the config
     state, because identical argv under a different config produces different
     chunks and therefore different doc_ids.
  3. Restored jobs come back as history, never as work. A job left `running` by
     a killed process must not look runnable.
  4. A chunk file can be traced back to the command that wrote it. That lookup
     is the whole point.
"""
import json
import time

import pytest


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch, management_module):
    """Point the module's JOBS_DIR at a scratch directory.

    Set explicitly rather than relying on the chdir in `management_module`:
    manage_api resolves JOBS_DIR at import, and the module is cached across the
    whole session, so whichever test imported it first would otherwise own the
    directory for all the others.
    """
    d = tmp_path / "jobs"
    d.mkdir()
    monkeypatch.setattr(management_module, "JOBS_DIR", d)
    return d


def _job(management_module, **kw):
    defaults = dict(id="abc123", kind="ingest_md",
                    argv=["py", "main.py", "ingest-md",
                          "--output", "data/probe_chunks.jsonl"],
                    params={"chunking": "document"})
    defaults.update(kw)
    job = management_module.Job(**defaults)
    job.log_file = str(management_module.JOBS_DIR / f"{job.id}.log")
    return job


# ------------------------------------------------------------- what it holds

def test_a_record_carries_what_a_replay_needs(management_module, jobs_dir):
    m = management_module
    job = _job(m)
    job.status, job.returncode = "done", 0
    m._write_job_record(job)

    rec = json.loads((jobs_dir / "abc123.json").read_text(encoding="utf-8"))

    assert rec["argv"] == job.argv
    assert rec["cwd"], "argv is relative to a working directory; record it"
    assert rec["params"] == {"chunking": "document"}
    # The vault switcher moves DATA_DIR. Replaying against the wrong vault
    # writes one corpus into another, silently.
    assert "vault_path" in rec and "data_dir" in rec
    # Identical argv under a different config yields different doc_ids. The digest
    # is of the console's own config.yaml, which a fresh clone lacks (see conftest).
    if (m.ROOT / "config.yaml").exists():
        assert rec["ingestion"]["config_sha256_16"]
    assert rec["ingestion"]["values"], "the chunk-shaping settings are the point"


def test_the_output_file_is_extracted_so_a_chunk_file_can_be_traced_back(
        management_module):
    out_of = management_module._output_of
    assert out_of(["py", "main.py", "ingest-md", "--output",
                   "data/x_chunks.jsonl"]) == "x_chunks.jsonl"
    assert out_of(["py", "main.py", "index", "--append",
                   "data/y_chunks.jsonl"]) == "y_chunks.jsonl"
    assert out_of(["py", "rebuild_bm25.py"]) is None
    # A flag with no value must not crash the record write.
    assert out_of(["py", "main.py", "ingest-md", "--output"]) is None


def test_bookkeeping_never_takes_a_job_down(management_module, jobs_dir,
                                            monkeypatch):
    """A record that cannot be written is a warning, not a failed ingest."""
    m = management_module
    monkeypatch.setattr(m, "JOBS_DIR", jobs_dir / "does" / "not" / "exist")
    m._write_job_record(_job(m))          # must not raise


# --------------------------------------------------------- when it is written

def test_the_record_exists_before_the_job_runs(management_module, jobs_dir,
                                               monkeypatch):
    """Written at QUEUE time: a process that dies mid-run must still leave
    behind what it was about to do."""
    m = management_module
    queued = []
    monkeypatch.setattr(m._QUEUE, "put", lambda jid: queued.append(jid))

    job = m.enqueue("recalibrate", {"dry_run": True})

    assert queued == [job.id], "the job should have been queued"
    rec = json.loads((jobs_dir / f"{job.id}.json").read_text(encoding="utf-8"))
    assert rec["status"] == "queued"
    assert rec["argv"] == job.argv
    assert "--dry-run" in rec["argv"]


# ------------------------------------------------------------- restoring them

def _write(jobs_dir, jid, status, created, **extra):
    rec = {"record_version": 1, "id": jid, "kind": "ingest_md",
           "status": status, "argv": ["py", "main.py", "ingest-md"],
           "params": {"a": 1}, "cwd": ".", "output": None,
           "log_file": f"{jid}.log", "returncode": 0, "created": created,
           "started": None, "ended": None, "created_iso": "x",
           "vault_path": "", "data_dir": "",
           "ingestion": {"config_sha256_16": "d", "values": {}}}
    rec.update(extra)
    (jobs_dir / f"{jid}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_history_survives_a_restart(management_module, jobs_dir, monkeypatch):
    m = management_module
    monkeypatch.setattr(m, "_JOBS", {})
    monkeypatch.setattr(m, "_ORDER", [])
    _write(jobs_dir, "old00", "done", time.time() - 100)
    _write(jobs_dir, "new00", "failed", time.time())

    assert m._load_job_records() == 2
    assert set(m._JOBS) == {"old00", "new00"}
    # _ORDER is oldest-first; the console reverses it to show newest first.
    assert m._ORDER == ["old00", "new00"]
    assert m._JOBS["old00"].restored is True
    assert m._JOBS["old00"].params == {"a": 1}, "retry rebuilds argv from params"


def test_a_job_killed_mid_run_comes_back_interrupted_not_runnable(
        management_module, jobs_dir, monkeypatch):
    """`running` on disk means the console died, not that the job is alive.
    Leaving that status would make the UI wait forever for output that is never
    coming, and would let it be read as a failure the job never had."""
    m = management_module
    monkeypatch.setattr(m, "_JOBS", {})
    monkeypatch.setattr(m, "_ORDER", [])
    _write(jobs_dir, "mid00", "running", time.time())
    _write(jobs_dir, "que00", "queued", time.time())

    m._load_job_records()

    assert m._JOBS["mid00"].status == "interrupted"
    assert m._JOBS["que00"].status == "interrupted"


def test_an_unreadable_record_is_skipped_not_fatal(management_module, jobs_dir,
                                                   monkeypatch):
    m = management_module
    monkeypatch.setattr(m, "_JOBS", {})
    monkeypatch.setattr(m, "_ORDER", [])
    _write(jobs_dir, "good0", "done", time.time())
    (jobs_dir / "broken.json").write_text("{not json", encoding="utf-8")

    assert m._load_job_records() == 1
    assert set(m._JOBS) == {"good0"}


def test_restoring_twice_does_not_duplicate_history(management_module,
                                                    jobs_dir, monkeypatch):
    m = management_module
    monkeypatch.setattr(m, "_JOBS", {})
    monkeypatch.setattr(m, "_ORDER", [])
    _write(jobs_dir, "once0", "done", time.time())

    m._load_job_records()
    m._load_job_records()

    assert m._ORDER == ["once0"]


# ---------------------------------------------------------------- provenance

def test_provenance_links_a_chunk_file_to_the_command_that_wrote_it(
        management_module, jobs_dir, tmp_path, monkeypatch):
    m = management_module
    chunks = tmp_path / "known_chunks.jsonl"
    chunks.write_text('{"doc_id":"a"}\n{"doc_id":"b"}\n', encoding="utf-8")
    orphan = tmp_path / "orphan_chunks.jsonl"
    orphan.write_text('{"doc_id":"c"}\n', encoding="utf-8")
    monkeypatch.setattr(m, "chunk_files", lambda: [chunks, orphan])

    _write(jobs_dir, "made0", "done", time.time(),
           output="known_chunks.jsonl",
           argv=["py", "main.py", "ingest-md", "--output",
                 "data/known_chunks.jsonl"])

    out = m.jobs_provenance()

    assert out["summary"]["with_a_recorded_command"] == 1
    assert out["summary"]["rows_recorded"] == 2
    assert out["summary"]["rows_unrecorded"] == 1
    assert [e["file"] for e in out["unrecorded"]] == ["orphan_chunks.jsonl"]

    rec = out["recorded"][0]
    assert rec["job_id"] == "made0"
    assert rec["replay"].endswith("--output data/known_chunks.jsonl")


def test_provenance_flags_a_config_that_has_changed_since_the_ingest(
        management_module, jobs_dir, tmp_path, monkeypatch):
    """The command alone is not sufficient to reproduce a chunk file, so an
    entry that implied it was would be worse than no entry."""
    m = management_module
    chunks = tmp_path / "a_chunks.jsonl"
    chunks.write_text('{"doc_id":"a"}\n', encoding="utf-8")
    monkeypatch.setattr(m, "chunk_files", lambda: [chunks])
    monkeypatch.setattr(m, "_ingestion_fingerprint",
                        lambda: {"config_sha256_16": "CURRENT", "values": {}})

    _write(jobs_dir, "stale", "done", time.time(), output="a_chunks.jsonl",
           ingestion={"config_sha256_16": "OLDDIGEST", "values": {}})

    assert m.jobs_provenance()["recorded"][0]["config_changed_since"] is True

    _write(jobs_dir, "stale", "done", time.time(), output="a_chunks.jsonl",
           ingestion={"config_sha256_16": "CURRENT", "values": {}})

    assert m.jobs_provenance()["recorded"][0]["config_changed_since"] is False


def test_provenance_route_is_not_swallowed_by_the_job_id_route(
        management_module):
    """`/api/jobs/provenance` must be declared before `/api/jobs/{jid}`, or
    FastAPI matches it as a job whose id is the literal string 'provenance'."""
    paths = [r.path for r in management_module.app.routes
             if getattr(r, "path", "").startswith("/api/jobs")]
    assert paths.index("/api/jobs/provenance") < paths.index("/api/jobs/{jid}")
