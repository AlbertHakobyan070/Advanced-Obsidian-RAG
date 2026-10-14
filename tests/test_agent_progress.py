"""What an agent waiting on the console and the query API can read instead of
polling blind.

The contracts worth pinning:

  1. A running job's `progress` is the newest counter its log shows, with the
     line it came from, and it is a PHASE (a file, a page, a BM25 bar), never a
     made-up job-wide percentage. tqdm redraws in place with \\r, so only the
     latest redraw counts.
  2. An ETA comes only from how long this kind of job has actually taken
     before. No finished run of the kind, or a job already past its usual
     time, means `eta_s` is null — never a guess.
  3. `/log?wait=` returns as soon as there is something new (output or a status
     change) and does not hang on a job that has already finished.
  4. `/stats` reports per-stage p50/p95 over the calls the API has served, so
     an agent can price a call before it makes it.
"""
import threading
import time
from collections import deque

import pytest


@pytest.fixture
def m(tmp_path, monkeypatch, management_module):
    d = tmp_path / "jobs"
    d.mkdir()
    monkeypatch.setattr(management_module, "JOBS_DIR", d)
    monkeypatch.setattr(management_module, "_JOBS", {})
    monkeypatch.setattr(management_module, "_ORDER", [])
    return management_module


def _add(m, jid, kind="index_append", status="running", log="", **kw):
    job = m.Job(id=jid, kind=kind, argv=["py", "main.py"], params={}, status=status, **kw)
    job.log_file = str(m.JOBS_DIR / f"{jid}.log")
    with open(job.log_file, "w", encoding="utf-8", newline="") as f:
        f.write(log)
    m._JOBS[jid] = job
    m._ORDER.append(jid)
    return job


# ------------------------------------------------------------ progress

def test_progress_is_the_newest_counter_with_its_line(m):
    job = _add(m, "j1", log="12:00 | INFO | pdf_loader | [1/9] a.pdf\n"
                            "12:01 | INFO | pdf_loader | [3/9] c.pdf\n"
                            "OCR on page.number=41/42.\n")
    assert m._progress(job) == {"done": 41, "total": 42,
                                "line": "OCR on page.number=41/42."}


def test_a_tqdm_bar_counts_only_its_latest_redraw(m):
    bar = ("BM25S Compute Scores:  10%|#   | 17000/174939 [00:01<00:09]\r"
           "BM25S Compute Scores:  50%|#####| 87000/174939 [00:05<00:05]")
    job = _add(m, "j1", log="start\n" + bar)
    p = m._progress(job)
    assert (p["done"], p["total"]) == (87000, 174939)
    assert p["line"].startswith("BM25S Compute Scores:  50%")


def test_no_counter_or_an_impossible_one_is_no_progress(m):
    # A path/date-like "2024/10" has done > total; it is not progress.
    job = _add(m, "j1", log="fetched https://x.org/2024/10 ok\nno numbers here\n")
    assert m._progress(job) is None


def test_progress_is_only_reported_while_running(m):
    job = _add(m, "j1", status="done", log="[3/9] c.pdf\n", started=1.0, ended=2.0)
    assert job.public()["progress"] is None


# ------------------------------------------------------------ eta from history

def test_eta_comes_from_the_median_of_past_runs_of_the_same_kind(m):
    for i, dur in enumerate((100.0, 200.0, 300.0)):
        _add(m, f"past{i}", status="done", started=1000.0, ended=1000.0 + dur)
    _add(m, "other", kind="rebuild_bm25", status="done", started=0.0, ended=9999.0)
    job = _add(m, "now", started=time.time() - 50)
    pub = job.public()
    assert pub["typical"] == {"median_s": 200.0, "runs": 3}
    assert 145 <= pub["eta_s"] <= 150
    assert 50 <= pub["elapsed_s"] <= 55


def test_no_history_means_no_eta(m):
    job = _add(m, "now", started=time.time() - 5)
    pub = job.public()
    assert pub["typical"] is None and pub["eta_s"] is None


def test_a_job_past_its_usual_time_has_no_eta(m):
    _add(m, "past", status="done", started=0.0, ended=10.0)
    job = _add(m, "now", started=time.time() - 60)
    pub = job.public()
    assert pub["typical"]["median_s"] == 10.0 and pub["eta_s"] is None


def test_failed_runs_do_not_shape_the_eta(m):
    _add(m, "bad", status="failed", started=0.0, ended=5.0)
    _add(m, "good", status="done", started=0.0, ended=100.0)
    assert m._typical_seconds("index_append") == {"median_s": 100.0, "runs": 1}


# ------------------------------------------------------------ queue position

def test_queue_position_counts_the_queued_jobs_ahead(m):
    _add(m, "run", status="running", started=time.time())
    a = _add(m, "a", status="queued")
    b = _add(m, "b", status="queued")
    assert (a.public()["queue_position"], b.public()["queue_position"]) == (1, 2)
    assert m._JOBS["run"].public()["queue_position"] is None


def test_listing_jobs_with_history_does_not_deadlock(m):
    _add(m, "past", status="done", started=0.0, ended=10.0)
    _add(m, "now", started=time.time())
    assert len(m.jobs_list()["jobs"]) == 2


# ------------------------------------------------------------ log: tail and wait

def test_tail_returns_the_last_lines_as_a_terminal_shows_them(m):
    _add(m, "j1", log="one\ntwo\nbar 1/4 [\rbar 4/4 [\n")
    out = m.jobs_log("j1", tail=2)
    assert out["data"] == "two\nbar 4/4 ["
    assert out["offset"] > 0 and out["job"]["id"] == "j1"


def test_wait_returns_early_when_the_log_grows(m):
    job = _add(m, "j1", log="hello\n")
    size = len("hello\n")

    def later():
        time.sleep(0.6)
        with open(job.log_file, "a", encoding="utf-8", newline="") as f:
            f.write("more\n")
    threading.Thread(target=later).start()
    t0 = time.monotonic()
    out = m.jobs_log("j1", offset=size, wait=10)
    assert time.monotonic() - t0 < 5
    assert out["data"] == "more\n"


def test_wait_on_a_finished_job_returns_at_once(m):
    _add(m, "j1", status="done", log="x\n", started=0.0, ended=1.0)
    t0 = time.monotonic()
    m.jobs_log("j1", offset=2, wait=10)
    assert time.monotonic() - t0 < 1


def test_the_schema_documents_tail_wait_and_the_eta_fields(m):
    eps = m.api_schema()["endpoints"]
    assert {"tail", "wait"} <= set(eps["GET /api/jobs/{id}/log"]["query"])
    assert "eta_s" in eps["GET /api/jobs/{id}"]["purpose"]


# ------------------------------------------------------------ query API latency

def test_stats_reports_latency_per_endpoint_and_stage(monkeypatch):
    import serve_api
    hist = deque(maxlen=50)
    for ms, rr in ((100, 80.0), (200, 150.0), (300, 250.0)):
        hist.append({"endpoint": "/search", "ms": ms, "timings": {"rerank": rr, "fuse": 0.1}})
    hist.append({"endpoint": "/query", "ms": 5000,
                 "timings": {"retrieval": {"rerank": 90.0}, "generation": None}})
    monkeypatch.setattr(serve_api, "_HISTORY", hist)
    monkeypatch.setitem(serve_api._STATE, "stats", {"chunks": 1})

    lat = serve_api.stats()["latency"]["by_endpoint"]
    assert lat["/search"]["calls"] == 3
    assert lat["/search"]["total_ms"] == {"p50": 200, "p95": 300}
    assert lat["/search"]["stages_ms"]["rerank"] == {"p50": 150.0, "p95": 250.0}
    assert lat["/query"]["stages_ms"] == {"retrieval.rerank": {"p50": 90.0, "p95": 90.0}}
    # The cached corpus overview is not polluted by the per-call block.
    assert "latency" not in serve_api._STATE["stats"]


def test_stats_latency_is_empty_before_any_call(monkeypatch):
    import serve_api
    monkeypatch.setattr(serve_api, "_HISTORY", deque(maxlen=50))
    monkeypatch.setitem(serve_api._STATE, "stats", {"chunks": 1})
    assert serve_api.stats()["latency"]["by_endpoint"] == {}
