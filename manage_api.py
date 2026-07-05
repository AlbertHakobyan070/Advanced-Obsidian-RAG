"""
manage_api.py — Corpus management console (backend) for Personal RAG.

Everything serve_api.py deliberately is NOT: ingest, index, OCR passes,
document search/inspection, deletion, uploads — driven from a browser at
http://127.0.0.1:8052 (webui/index.html), with live job logs.

Design rules (they encode this project's hard-won gotchas — don't undo them):

  * NO pipeline in-process. Heavy work runs as SUBPROCESSES of the existing
    entry points (main.py / rebuild_bm25.py / recalibrate_courses.py), one at
    a time, from a queue. ChromaDB is effectively single-writer and two
    concurrent ingests would fight over the JSONLs, so the worker is serial
    by construction. The query endpoint (:8051) stays untouched and warm.
  * Every ChromaDB scan/delete is PAGED in 5000-row batches ("too many SQL
    variables" at ~168K chunks otherwise).
  * JSONL is the source of truth. Deleting from Chroma alone resurrects
    chunks at the next BM25 rebuild — so deletion here removes the rows from
    the JSONL files too, then queues a rebuild.
  * JSONL lines split on "\\n" ONLY (never .splitlines(): some Other/ chunk
    text contains U+2028/U+2029/\\x85 which would shred records).

Run (inside the venv, from project root):
    python -m uvicorn manage_api:app --host 127.0.0.1 --port 8052

Then open http://127.0.0.1:8052 — the console UI is served from webui/.
Restart the QUERY endpoint (:8051) after index-changing jobs finish; it loads
the indexes at startup and stays warm.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Iterator, Optional

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.utils.config_loader import Config, load_config
from src.utils.logger import configure_logging, get_logger

log = get_logger("manage_api")

ROOT = Path(__file__).resolve().parent
CFG: Config = load_config()
configure_logging(level=CFG.get("logging.level", "INFO"), console=True)

JOBS_DIR = CFG.path("webui.jobs_dir") if CFG.get("webui.jobs_dir") else ROOT / "logs" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = CFG.path("paths.chunks_file").parent
MANIFEST_CACHE = (CFG.path("webui.manifest_cache")
                  if CFG.get("webui.manifest_cache")
                  else DATA_DIR / ".manifest_cache.json")
RAG_API = CFG.get("webui.rag_api", "http://127.0.0.1:8051")
COLLECTION = CFG.get("paths.collection_name", "obsidian_vault")
PAGE = 5000                      # ChromaDB paging batch (see module docstring)

app = FastAPI(title="Personal RAG — Management Console", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# JSONL streaming (the "\n"-only rule, without loading 200MB files into RAM)
# ============================================================================

def iter_jsonl_lines(path: Path) -> Iterator[bytes]:
    """Yield raw lines split strictly on b'\\n' (U+2028 etc. stay inside)."""
    buf = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            buf += block
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                yield buf[:nl]
                buf = buf[nl + 1:]
    if buf.strip():
        yield buf


def chunk_files() -> list[Path]:
    """chunks.jsonl + every data/*_chunks.jsonl — same union rebuild_bm25 uses."""
    base = CFG.path("paths.chunks_file")
    files = [base] if base.exists() else []
    files += sorted(p for p in DATA_DIR.glob("*_chunks.jsonl") if p != base)
    return files

# ============================================================================
# Document manifest — per-source_file aggregates, cached per JSONL mtime/size
# ============================================================================

_manifest_lock = threading.Lock()


def _scan_one_jsonl(path: Path) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    for raw in iter_jsonl_lines(path):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        m = rec.get("metadata") or {}
        sf = str(m.get("source_file") or m.get("filename") or "?")
        d = docs.setdefault(sf, {
            "source_file": sf,
            "filename": str(m.get("filename") or Path(sf).stem),
            "chunks": 0,
            "course": "",
            "domain": "",
            "file_types": [],
            "tags": [],
            "jsonls": [path.name],
        })
        d["chunks"] += 1
        c = str(m.get("course_name") or m.get("course") or "")
        if c and c.lower() != "unknown":
            d["course"] = c
        dom = str(m.get("domain") or "")
        if dom and dom.lower() != "general" or not d["domain"]:
            d["domain"] = dom or d["domain"]
        ft = str(m.get("file_type") or "")
        if ft and ft not in d["file_types"]:
            d["file_types"].append(ft)
        tg = m.get("tags") or []
        if isinstance(tg, str):
            tg = [t.strip() for t in tg.split(",") if t.strip()]
        for t in tg:
            if t not in d["tags"] and len(d["tags"]) < 15:
                d["tags"].append(t)
    return docs


MANIFEST_CACHE_VERSION = 2      # bump when _scan_one_jsonl's shape changes


def _load_cache() -> dict:
    try:
        cache = json.loads(MANIFEST_CACHE.read_text(encoding="utf-8"))
        if cache.get("version") != MANIFEST_CACHE_VERSION:
            return {"version": MANIFEST_CACHE_VERSION, "files": {}}
        return cache
    except Exception:
        return {"version": MANIFEST_CACHE_VERSION, "files": {}}


def _save_cache(cache: dict) -> None:
    try:
        MANIFEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    except Exception as e:                                  # non-fatal
        log.warning("manifest cache not saved: %s", e)


def build_manifest(force: bool = False) -> dict:
    """
    {source_file -> {chunks, course, domain, file_types, jsonls}} across all
    chunk files, rebuilt per-file only when a JSONL's (size, mtime) changed.
    First cold build walks every line once (~169K lines, a few seconds off
    the HDD); after that it's a cache read.
    """
    with _manifest_lock:
        cache = {"files": {}} if force else _load_cache()
        changed = False
        seen_names = set()
        for path in chunk_files():
            st = path.stat()
            key = path.name
            seen_names.add(key)
            entry = cache["files"].get(key)
            if entry and entry.get("size") == st.st_size and entry.get("mtime") == st.st_mtime:
                continue
            log.info("manifest: scanning %s ...", key)
            cache["files"][key] = {
                "size": st.st_size, "mtime": st.st_mtime,
                "docs": _scan_one_jsonl(path),
            }
            changed = True
        for gone in set(cache["files"]) - seen_names:
            del cache["files"][gone]
            changed = True
        if changed:
            _save_cache(cache)

        merged: dict[str, dict] = {}
        for key, entry in cache["files"].items():
            for sf, d in entry["docs"].items():
                if sf in merged:
                    t = merged[sf]
                    t["chunks"] += d["chunks"]
                    t["course"] = t["course"] or d["course"]
                    t["domain"] = t["domain"] or d["domain"]
                    for ft in d["file_types"]:
                        if ft not in t["file_types"]:
                            t["file_types"].append(ft)
                    for tg in d.get("tags", []):
                        if tg not in t.setdefault("tags", []):
                            t["tags"].append(tg)
                    if key not in t["jsonls"]:
                        t["jsonls"].append(key)
                else:
                    merged[sf] = {**d, "jsonls": [key]}
        return merged

# ============================================================================
# Job queue — one worker, subprocesses of the existing entry points
# ============================================================================

@dataclass
class Job:
    id: str
    kind: str
    argv: list[str]
    params: dict
    status: str = "queued"          # queued | running | done | failed | cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    returncode: int | None = None
    log_file: str = ""

    def public(self) -> dict:
        d = asdict(self)
        d["argv"] = " ".join(self.argv)
        return d


_JOBS: dict[str, Job] = {}
_ORDER: list[str] = []
_QUEUE: "Queue[str]" = Queue()
_PROCS: dict[str, subprocess.Popen] = {}
_jobs_lock = threading.Lock()


def _safe_rel(p: str, *, default_dir: str = "data") -> str:
    """Constrain user-supplied paths to inside the project (no drive/.. escapes)."""
    p = (p or "").strip().replace("\\", "/")
    if not p:
        raise ValueError("empty path")
    if ".." in p.split("/") or re.match(r"^([A-Za-z]:|/)", p):
        raise ValueError(f"path must be project-relative: {p!r}")
    if "/" not in p:
        p = f"{default_dir}/{p}"
    return p


def _build_argv(kind: str, prm: dict) -> list[str]:
    py = sys.executable
    if kind == "ingest_pdfs":
        argv = [py, "main.py", "ingest-pdfs"]
        if prm.get("only_books"):
            argv.append("--only-books")
        if prm.get("skip_books"):
            argv.append("--skip-books")
        if prm.get("include_path"):
            argv += ["--include-path", str(prm["include_path"])]
        if prm.get("exclude_path"):
            argv += ["--exclude-path", str(prm["exclude_path"])]
        if prm.get("output"):
            argv += ["--output", _safe_rel(str(prm["output"]))]
        if prm.get("max_pages"):
            argv += ["--max-pages", str(int(prm["max_pages"]))]
        if prm.get("pages"):
            from src.ingestion.pdf_loader import parse_page_spec
            parse_page_spec(str(prm["pages"]))       # ValueError -> 400 (caught in jobs_create)
            argv += ["--pages", str(prm["pages"])]
        if prm.get("no_ocr"):
            argv.append("--no-ocr")
        if prm.get("ocr_engine"):
            if prm["ocr_engine"] not in ("auto", "tesseract", "vlm", "none"):
                raise ValueError("ocr_engine must be auto|tesseract|vlm|none")
            argv += ["--ocr-engine", prm["ocr_engine"]]
        if prm.get("chunking"):
            if prm["chunking"] not in ("heading", "fixed"):
                raise ValueError("chunking must be heading|fixed")
            argv += ["--chunking", prm["chunking"]]
        if prm.get("no_images"):
            argv.append("--no-images")
        if prm.get("archive_processed"):
            argv.append("--archive-processed")
        if prm.get("force_domain"):
            argv += ["--force-domain", str(prm["force_domain"])]
        if prm.get("force_tags"):
            tags = prm["force_tags"]
            if isinstance(tags, (list, tuple)):
                tags = ",".join(str(t) for t in tags)
            argv += ["--force-tags", str(tags)]
        return argv
    if kind == "ingest_notebooks":
        argv = [py, "main.py", "ingest-notebooks"]
        if prm.get("output"):
            argv += ["--output", _safe_rel(str(prm["output"]))]
        if prm.get("no_outputs"):
            argv.append("--no-outputs")
        if prm.get("save_figures"):
            argv.append("--save-figures")
        if prm.get("exts"):
            argv += ["--exts", str(prm["exts"])]
        return argv
    if kind == "ingest_code":
        argv = [py, "main.py", "ingest-code"]
        if prm.get("output"):
            argv += ["--output", _safe_rel(str(prm["output"]))]
        if prm.get("include_path"):
            argv += ["--include-path", str(prm["include_path"])]
        if prm.get("exclude_path"):
            argv += ["--exclude-path", str(prm["exclude_path"])]
        if prm.get("exts"):
            argv += ["--exts", str(prm["exts"])]
        return argv
    if kind == "index_append":
        return [py, "main.py", "index", "--append", _safe_rel(str(prm.get("file", "")))]
    if kind == "index_rebuild":
        return [py, "main.py", "index"]
    if kind == "rebuild_bm25":
        return [py, "rebuild_bm25.py"]
    if kind == "build_hype":
        argv = [py, "build_hype.py"]
        if prm.get("include_path"):
            argv += ["--include-path", str(prm["include_path"])]
        if prm.get("file_types"):
            argv += ["--file-types", str(prm["file_types"])]
        if prm.get("questions"):
            argv += ["--questions", str(int(prm["questions"]))]
        if prm.get("max_chunks"):
            argv += ["--max-chunks", str(int(prm["max_chunks"]))]
        if prm.get("dry_run"):
            argv.append("--dry-run")
        return argv
    if kind == "recalibrate":
        argv = [py, "recalibrate_courses.py"]
        if prm.get("dry_run", True):
            argv.append("--dry-run")
        return argv
    if kind == "eval":
        argv = [py, "main.py", "eval"]
        if prm.get("retrieval_only"):
            argv.append("--retrieval-only")
        return argv
    raise ValueError(f"unknown job kind {kind!r}")


def enqueue(kind: str, params: dict) -> Job:
    argv = _build_argv(kind, params or {})
    job = Job(id=uuid.uuid4().hex[:10], kind=kind, argv=argv, params=params or {})
    job.log_file = str(JOBS_DIR / f"{job.id}.log")
    with _jobs_lock:
        _JOBS[job.id] = job
        _ORDER.append(job.id)
    _QUEUE.put(job.id)
    log.info("job %s queued: %s", job.id, " ".join(argv))
    return job


def _worker() -> None:
    while True:
        jid = _QUEUE.get()
        job = _JOBS.get(jid)
        if job is None or job.status == "cancelled":
            continue
        job.status, job.started = "running", time.time()
        try:
            with open(job.log_file, "ab") as lf:
                lf.write((" ".join(job.argv) + "\n\n").encode())
                lf.flush()
                # Child stdout is this log FILE, so Python would pick the
                # locale codepage (cp1251 here) and crash on emoji/box-drawing
                # output from main.py. Force UTF-8 for every job subprocess.
                env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
                proc = subprocess.Popen(
                    job.argv, cwd=str(ROOT), stdout=lf,
                    stderr=subprocess.STDOUT, env=env,
                )
                _PROCS[jid] = proc
                rc = proc.wait()
            job.returncode = rc
            if job.status != "cancelled":
                job.status = "done" if rc == 0 else "failed"
        except Exception as e:
            job.returncode = -1
            job.status = "failed"
            try:
                with open(job.log_file, "ab") as lf:
                    lf.write(f"\n[manage_api] launch failed: {e}\n".encode())
            except OSError:
                pass
        finally:
            job.ended = time.time()
            _PROCS.pop(jid, None)
            log.info("job %s %s (rc=%s)", jid, job.status, job.returncode)


threading.Thread(target=_worker, daemon=True, name="job-worker").start()

# ============================================================================
# ChromaDB helpers (always paged)
# ============================================================================

def _collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CFG.path("paths.chroma_dir")))
    return client.get_collection(COLLECTION)


def _chroma_delete_by_source(source_files: set[str]) -> int:
    """Exact-match delete on metadata.source_file, paged both ways."""
    c = _collection()
    total = c.count()
    ids: list[str] = []
    offset = 0
    while offset < total:
        got = c.get(limit=PAGE, offset=offset, include=["metadatas"])
        for i, m in zip(got["ids"], got["metadatas"]):
            if str((m or {}).get("source_file", "")) in source_files:
                ids.append(i)
        offset += PAGE
    for k in range(0, len(ids), PAGE):
        c.delete(ids=ids[k:k + PAGE])
    return len(ids)


def _jsonl_remove_sources(source_files: set[str], jsonl_names: set[str]) -> dict[str, int]:
    """Stream-rewrite each affected JSONL, dropping matching rows atomically."""
    removed: dict[str, int] = {}
    for path in chunk_files():
        if path.name not in jsonl_names:
            continue
        tmp = path.with_suffix(path.suffix + ".tmp")
        n = 0
        with open(tmp, "wb") as out:
            for raw in iter_jsonl_lines(path):
                if not raw.strip():
                    continue
                keep = True
                try:
                    rec = json.loads(raw.decode("utf-8", errors="replace"))
                    sf = str((rec.get("metadata") or {}).get("source_file", ""))
                    if sf in source_files:
                        keep = False
                except json.JSONDecodeError:
                    pass                      # unparseable line: keep, don't destroy
                if keep:
                    out.write(raw + b"\n")
                else:
                    n += 1
        if n:
            shutil.move(str(tmp), str(path))
            removed[path.name] = n
        else:
            tmp.unlink(missing_ok=True)
    return removed

# ============================================================================
# API models
# ============================================================================

class JobIn(BaseModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class DeleteIn(BaseModel):
    source_files: list[str] = Field(min_length=1)
    rebuild: bool = True            # queue rebuild_bm25 after (keep indexes in sync)


class InboxIngestIn(BaseModel):
    force: bool = False             # ingest even if a file looks already-indexed
    ocr_engine: Optional[str] = None  # optional override (auto|tesseract|vlm|none)
    chunking: Optional[str] = None  # heading|fixed (how oversized sections split)
    # Batch-level metadata (inbox files carry no course path): stamped on every
    # chunk of this batch. domain feeds scope routing; tags feed tag search +
    # the retrieval tag boost.
    domain: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class RetagIn(BaseModel):
    source_files: list[str] = Field(min_length=1)
    domain: Optional[str] = None            # set/replace domain (None = keep)
    course: Optional[str] = None            # set/replace course_name+course_code (None = keep)
    add_tags: list[str] = Field(default_factory=list)
    remove_tags: list[str] = Field(default_factory=list)
    rebuild: bool = True                    # queue rebuild_bm25 after

# ============================================================================
# Routes — console
# ============================================================================

@app.get("/")
def index_page():
    ui = ROOT / "webui" / "index.html"
    if ui.exists():
        return FileResponse(ui)
    return JSONResponse({"error": "webui/index.html not found next to manage_api.py"},
                        status_code=404)


@app.get("/api/overview")
def overview() -> dict:
    manifest = build_manifest()
    by_domain: dict[str, int] = {}
    by_jsonl: dict[str, int] = {}
    total_chunks = 0
    for d in manifest.values():
        total_chunks += d["chunks"]
        by_domain[d["domain"] or "unknown"] = by_domain.get(d["domain"] or "unknown", 0) + d["chunks"]
        for j in d["jsonls"]:
            by_jsonl[j] = by_jsonl.get(j, 0) + d["chunks"]

    chroma_count: Any
    try:
        chroma_count = _collection().count()
    except Exception as e:
        chroma_count = f"unavailable ({type(e).__name__})"

    def _size(p: Path) -> int:
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0

    rag_ready = None
    try:
        import requests
        rag_ready = requests.get(f"{RAG_API}/health", timeout=1.5).json().get("ready")
    except Exception:
        rag_ready = False

    with _jobs_lock:
        recent = [_JOBS[j].public() for j in _ORDER[-6:]][::-1]

    return {
        "files": len(manifest),
        "chunks": total_chunks,
        "by_domain": dict(sorted(by_domain.items(), key=lambda x: -x[1])),
        "by_jsonl": dict(sorted(by_jsonl.items(), key=lambda x: -x[1])),
        "chroma_count": chroma_count,
        "disk": {
            "chroma_db": _size(CFG.path("paths.chroma_dir")),
            "bm25_index": _size(CFG.path("paths.bm25_index")),
            "jsonl_total": sum(_size(p) for p in chunk_files()),
        },
        "rag_api": {"url": RAG_API, "ready": rag_ready},
        "recent_jobs": recent,
        "jsonl_files": [p.name for p in chunk_files()],
    }


@app.get("/api/documents")
def documents(q: str = "", domain: str = "", course: str = "",
              jsonl: str = "", tag: str = "", limit: int = 50, offset: int = 0) -> dict:
    manifest = build_manifest()
    ql = q.strip().lower()
    tagl = tag.strip().lstrip("#").lower()
    rows = []
    for d in manifest.values():
        if ql and ql not in d["source_file"].lower() and ql not in d["filename"].lower():
            continue
        if domain and (d["domain"] or "unknown") != domain:
            continue
        if course and course.lower() not in (d["course"] or "").lower():
            continue
        if jsonl and jsonl not in d["jsonls"]:
            continue
        if tagl and tagl not in [str(t).lower() for t in (d.get("tags") or [])]:
            continue
        rows.append(d)
    rows.sort(key=lambda r: (-r["chunks"], r["source_file"]))
    limit = max(1, min(limit, 500))
    return {"total": len(rows), "rows": rows[offset:offset + limit]}


@app.get("/api/facets")
def facets() -> dict:
    manifest = build_manifest()
    domains: dict[str, int] = {}
    courses: dict[str, int] = {}
    tags: dict[str, int] = {}
    for d in manifest.values():
        domains[d["domain"] or "unknown"] = domains.get(d["domain"] or "unknown", 0) + 1
        if d["course"]:
            courses[d["course"]] = courses.get(d["course"], 0) + 1
        for t in (d.get("tags") or []):
            tags[str(t)] = tags.get(str(t), 0) + 1
    return {
        "domains": dict(sorted(domains.items(), key=lambda x: -x[1])),
        "courses": dict(sorted(courses.items(), key=lambda x: -x[1])[:40]),
        "tags": dict(sorted(tags.items(), key=lambda x: -x[1])[:60]),
        "jsonls": [p.name for p in chunk_files()],
    }


@app.get("/api/documents/preview")
def doc_preview(source_file: str, n: int = 3) -> dict:
    """First n chunk texts for one document (read from the JSONLs)."""
    manifest = build_manifest()
    d = manifest.get(source_file)
    if not d:
        return {"error": "unknown source_file", "chunks": []}
    wanted = min(max(n, 1), 10)
    out = []
    for path in chunk_files():
        if path.name not in d["jsonls"]:
            continue
        for raw in iter_jsonl_lines(path):
            if len(out) >= wanted:
                break
            try:
                rec = json.loads(raw.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            if str((rec.get("metadata") or {}).get("source_file", "")) == source_file:
                t = str(rec.get("text") or "")
                out.append(t[:1500] + ("…" if len(t) > 1500 else ""))
        if len(out) >= wanted:
            break
    return {"source_file": source_file, "chunks": out, "meta": d}


@app.post("/api/documents/delete")
def documents_delete(body: DeleteIn) -> dict:
    """
    Remove documents from the INDEX (never from the vault): paged ChromaDB
    delete + JSONL row removal (so a rebuild can't resurrect them) + optional
    queued BM25 rebuild. Restart the query endpoint afterwards.
    """
    targets = {s for s in body.source_files if s}
    manifest = build_manifest()
    jsonl_names: set[str] = set()
    for sf in targets:
        d = manifest.get(sf)
        if d:
            jsonl_names.update(d["jsonls"])

    try:
        chroma_deleted = _chroma_delete_by_source(targets)
    except Exception as e:
        return {"ok": False, "error": f"ChromaDB delete failed: {type(e).__name__}: {e}"}

    removed = _jsonl_remove_sources(targets, jsonl_names)
    build_manifest(force=False)            # touched files re-scan on next read

    rebuild_job = enqueue("rebuild_bm25", {}).id if body.rebuild else None
    return {
        "ok": True,
        "chroma_deleted": chroma_deleted,
        "jsonl_removed": removed,
        "rebuild_job": rebuild_job,
        "note": "Vault files were NOT touched. Restart serve_api (:8051) once "
                "the rebuild job finishes so the warm pipeline reloads.",
    }

def _retag_meta(m: dict, set_domain: Optional[str], set_course: Optional[str],
                add: list[str], rem: set[str]) -> dict:
    """Apply one retag to a chunk metadata dict, in place. `course` sets BOTH
    course_name and course_code — the loaders keep them equal for folder-map
    matches, and the manifest/eval read course_name first."""
    if set_domain:
        m["domain"] = set_domain
    if set_course:
        m["course_name"] = set_course
        m["course_code"] = set_course
    tags = m.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    tags = [t for t in tags if t.lower() not in rem]
    tags += [t for t in add if t not in tags]
    if tags or "tags" in m:
        m["tags"] = tags
    return m


@app.post("/api/documents/retag")
def documents_retag(body: RetagIn) -> dict:
    """
    Set domain and/or course and/or add/remove tags on whole documents.
    METADATA ONLY — chunk text never changes, so doc_ids (and embeddings)
    are untouched:
      1. stream-rewrite the affected JSONLs (source of truth),
      2. paged ChromaDB metadata update for the same doc_ids,
      3. optional queued rebuild_bm25 (the sparse payload carries its own
         metadata copy — without it, tag boosts won't see sparse-lane hits).
    """
    targets = {s for s in body.source_files if s}
    set_domain = body.domain.strip().lower() if body.domain else None
    set_course = body.course.strip() if body.course and body.course.strip() else None
    add = [t.strip().lstrip("#").lower() for t in body.add_tags if t.strip()]
    rem = {t.strip().lstrip("#").lower() for t in body.remove_tags if t.strip()}
    if not (set_domain or set_course or add or rem):
        return JSONResponse({"ok": False, "error": "nothing to change"},
                            status_code=400)

    manifest = build_manifest()
    jsonl_names: set[str] = set()
    missing = []
    for sf in targets:
        d = manifest.get(sf)
        if d:
            jsonl_names.update(d["jsonls"])
        else:
            missing.append(sf)
    if not jsonl_names:
        return JSONResponse({"ok": False, "error": "no matching documents",
                             "missing": missing}, status_code=404)

    changed_meta: dict[str, dict] = {}       # doc_id -> cleaned new metadata
    rows_changed = 0
    for path in chunk_files():
        if path.name not in jsonl_names:
            continue
        tmp = path.with_suffix(path.suffix + ".tmp")
        n = 0
        with open(tmp, "wb") as out:
            for raw in iter_jsonl_lines(path):
                if not raw.strip():
                    continue
                line = raw
                try:
                    rec = json.loads(raw.decode("utf-8", errors="replace"))
                    m = rec.get("metadata") or {}
                    if str(m.get("source_file", "")) in targets:
                        rec["metadata"] = _retag_meta(m, set_domain, set_course, add, rem)
                        changed_meta[str(rec.get("doc_id", ""))] = rec["metadata"]
                        line = json.dumps(rec, ensure_ascii=False).encode("utf-8")
                        n += 1
                except json.JSONDecodeError:
                    pass                      # unparseable: pass through untouched
                out.write(line + b"\n")
        if n:
            shutil.move(str(tmp), str(path))
            rows_changed += n
        else:
            tmp.unlink(missing_ok=True)

    # Chroma metadata update (paged; only ids that actually exist there —
    # known dedup means a few JSONL rows never got vectors).
    def clean(meta: dict) -> dict:
        out: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = ", ".join(str(x) for x in v)
            else:
                out[k] = str(v)
        return out

    chroma_updated = 0
    try:
        col = _collection()
        ids = [i for i in changed_meta if i]
        for k in range(0, len(ids), PAGE):
            batch = ids[k:k + PAGE]
            found = col.get(ids=batch, include=[])["ids"]
            if found:
                col.update(ids=found,
                           metadatas=[clean(changed_meta[i]) for i in found])
                chroma_updated += len(found)
    except Exception as e:
        return {"ok": False, "rows_changed": rows_changed,
                "error": f"JSONLs updated but ChromaDB update failed "
                         f"({type(e).__name__}: {e}) — retag again to heal.",
                "missing": missing}

    rebuild_job = enqueue("rebuild_bm25", {}).id if body.rebuild else None
    return {"ok": True, "rows_changed": rows_changed,
            "chroma_updated": chroma_updated,
            "missing": missing, "rebuild_job": rebuild_job,
            "note": "Restart serve_api (:8051) after the rebuild finishes so "
                    "warm retrieval sees the new metadata."}


# ---- jobs ----

@app.post("/api/jobs")
def jobs_create(body: JobIn) -> dict:
    try:
        job = enqueue(body.kind, body.params)
    except (ValueError, KeyError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, "job": job.public()}


@app.get("/api/jobs")
def jobs_list() -> dict:
    with _jobs_lock:
        return {"jobs": [_JOBS[j].public() for j in _ORDER[::-1]]}


@app.get("/api/jobs/{jid}")
def jobs_get(jid: str) -> dict:
    job = _JOBS.get(jid)
    if not job:
        return JSONResponse({"error": "no such job"}, status_code=404)
    return job.public()


@app.get("/api/jobs/{jid}/log")
def jobs_log(jid: str, offset: int = 0) -> dict:
    job = _JOBS.get(jid)
    if not job:
        return JSONResponse({"error": "no such job"}, status_code=404)
    p = Path(job.log_file)
    if not p.exists():
        return {"offset": 0, "data": "", "status": job.status}
    size = p.stat().st_size
    offset = max(0, min(offset, size))
    with open(p, "rb") as f:
        f.seek(offset)
        blob = f.read(1 << 16)
    return {"offset": offset + len(blob),
            "data": blob.decode("utf-8", errors="replace"),
            "status": job.status}


@app.post("/api/jobs/{jid}/retry")
def jobs_retry(jid: str) -> dict:
    """
    Re-enqueue a failed/cancelled job with the SAME kind+params. This is the
    sanctioned checkpoint-recovery path: every job here is idempotent by
    design (ingest archives processed PDFs so a retry only touches leftovers;
    index_append upserts deterministic ids, so its committed dense half is
    never duplicated; rebuild_bm25 is derived from the JSONLs). A new job id
    and log file are created; the failed job's log stays for the post-mortem.
    """
    job = _JOBS.get(jid)
    if not job:
        return JSONResponse({"error": "no such job"}, status_code=404)
    if job.status not in ("failed", "cancelled"):
        return JSONResponse({"ok": False,
                             "error": f"job is {job.status} — only failed/"
                                      f"cancelled jobs can be retried"},
                            status_code=409)
    new = enqueue(job.kind, job.params)
    return {"ok": True, "job": new.public(), "retried_from": jid}


@app.post("/api/jobs/{jid}/cancel")
def jobs_cancel(jid: str) -> dict:
    job = _JOBS.get(jid)
    if not job:
        return JSONResponse({"error": "no such job"}, status_code=404)
    proc = _PROCS.get(jid)
    job.status = "cancelled"
    if proc and proc.poll() is None:
        proc.terminate()
    return {"ok": True, "job": job.public()}

# ---- uploads into the vault inbox ----

def _inbox() -> Path:
    vault = Path(CFG.get("pdf.vault_path") or CFG.get("parser.vault_path"))
    inbox = vault / CFG.get("webui.inbox_dir", "00 – AUA_DS/Other/Inbox")
    inbox.mkdir(parents=True, exist_ok=True)
    return inbox


@app.post("/api/upload")
async def upload(files: list[UploadFile]) -> dict:
    """Save files INTO the vault inbox; ingest them with an Inbox-scoped job."""
    inbox = _inbox()
    saved = []
    for uf in files:
        name = re.sub(r"[^\w.\- ()\[\]]", "_", Path(uf.filename or "upload").name)
        dest = inbox / name
        for i in itertools.count(2):        # never overwrite
            if not dest.exists():
                break
            dest = inbox / f"{Path(name).stem} ({i}){Path(name).suffix}"
        with open(dest, "wb") as out:
            while True:
                blob = await uf.read(1 << 20)
                if not blob:
                    break
                out.write(blob)
        saved.append(dest.name)
    return {"ok": True, "saved": saved, "inbox": str(inbox),
            "hint": "Click 'Ingest inbox now' (POST /api/ingest_inbox) to "
                    "index these."}


@app.get("/api/inbox")
def inbox_list() -> dict:
    inbox = _inbox()
    rows = [{"name": f.name, "bytes": f.stat().st_size}
            for f in sorted(inbox.iterdir()) if f.is_file()]
    return {"inbox": str(inbox), "files": rows}


@app.post("/api/ingest_inbox")
def ingest_inbox(body: InboxIngestIn) -> dict:
    """
    The ONE sanctioned way to index inbox uploads. Owns the job parameters
    server-side because the old UI-hardcoded ones (`only_books:true`) silently
    matched 0 files — Inbox is not a book folder, so three jobs ran "done"
    while ingesting nothing.

    Guards, in order:
      * empty inbox            -> 400, nothing queued
      * filename already known -> 409 with the matches (force:true overrides);
                                  doc_ids are path-dependent, so re-ingesting a
                                  file that lives elsewhere in the corpus WOULD
                                  create real duplicates
    Then queues ingest -> index_append (serial worker keeps the order). Each
    batch gets its own timestamped JSONL so a later batch can never clobber an
    earlier one, and processed PDFs are archived to Inbox/_ingested/.
    """
    inbox = _inbox()
    pdfs = [f for f in sorted(inbox.iterdir())
            if f.is_file() and f.suffix.lower() == ".pdf"]
    non_pdfs = [f.name for f in sorted(inbox.iterdir())
                if f.is_file() and f.suffix.lower() != ".pdf"]
    if not pdfs:
        return JSONResponse(
            {"ok": False, "error": "No PDFs in the inbox — drop files in first.",
             "non_pdfs_ignored": non_pdfs}, status_code=400)

    # Duplicate guard: match inbox filenames against everything already indexed.
    manifest = build_manifest()
    known: dict[str, str] = {}
    for sf, d in manifest.items():
        known.setdefault(Path(sf).stem.lower(), sf)
        fn = str(d.get("filename") or "")
        if fn:
            known.setdefault(Path(fn).stem.lower(), sf)
    conflicts = [{"file": f.name, "existing_source": known[f.stem.lower()]}
                 for f in pdfs if f.stem.lower() in known]
    if conflicts and not body.force:
        return JSONResponse(
            {"ok": False,
             "error": f"{len(conflicts)} inbox file(s) look already indexed.",
             "conflicts": conflicts,
             "hint": "Remove them from the inbox, or repeat with force:true "
                     "to ingest anyway (this WILL duplicate their chunks if "
                     "they are the same files)."}, status_code=409)

    vault = Path(CFG.get("pdf.vault_path") or CFG.get("parser.vault_path"))
    include_rel = inbox.relative_to(vault).as_posix()   # exact folder, not "Inbox"
    out = f"data/inbox_{time.strftime('%Y%m%d_%H%M%S')}_chunks.jsonl"
    params: dict[str, Any] = {"include_path": include_rel, "output": out,
                              "no_images": True, "archive_processed": True}
    if body.ocr_engine:
        params["ocr_engine"] = body.ocr_engine
    if body.chunking:
        params["chunking"] = body.chunking   # validated in _build_argv
    if body.domain:
        params["force_domain"] = body.domain.strip().lower()
    if body.tags:
        params["force_tags"] = [t.strip().lstrip("#").lower()
                                for t in body.tags if t.strip()]
    try:
        j1 = enqueue("ingest_pdfs", params)   # ValueError (bad ocr_engine/chunking) -> 400
    except (ValueError, KeyError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    j2 = enqueue("index_append", {"file": out})
    return {"ok": True,
            "files": [f.name for f in pdfs],
            "non_pdfs_ignored": non_pdfs,
            "forced_past_conflicts": conflicts if body.force else [],
            "output": out,
            "jobs": [j1.public(), j2.public()],
            "note": "index --append rebuilds the sparse index itself; restart "
                    "serve_api (:8051) once the append job finishes."}


# ---- vault tree (browse / in-RAG check / retag staging UI) ----

def _vault_root() -> Path:
    return Path(CFG.get("pdf.vault_path") or CFG.get("parser.vault_path"))


def _rag_lookup() -> dict[str, tuple[str, dict]]:
    """manifest keyed by lowercase-posix source_file -> (original_key, doc).
    The original key is what /api/documents/retag and /delete expect."""
    return {sf.replace("\\", "/").lower(): (sf, d)
            for sf, d in build_manifest().items()}

_TREE_EXTS = {".pdf", ".md", ".ipynb", ".py", ".r", ".rmd"}
_TREE_SKIP = {".obsidian", ".trash", ".git", "node_modules",
              ".smart-connections", ".obsidian-git", "_ingested", "_Backups"}


def _file_row(f: Path, rel: str, rag: dict[str, tuple[str, dict]]) -> dict:
    # Inbox PDFs are archived to _ingested/ AFTER indexing, so their disk path
    # gained a segment their indexed source_file doesn't have — strip it.
    hit = (rag.get(rel.lower())
           or rag.get(rel.lower().replace("/_ingested/", "/")))
    key, d = hit if hit else (None, None)
    return {
        "name": f.name, "path": rel, "bytes": f.stat().st_size,
        "ext": f.suffix.lower(),
        "indexable": f.suffix.lower() in _TREE_EXTS,
        "in_rag": bool(d),
        "chunks": d["chunks"] if d else 0,
        "domain": (d.get("domain") or "") if d else "",
        "tags": (d.get("tags") or []) if d else [],
        "source_file": key,                       # exact retag/delete key
    }


@app.get("/api/vault/tree")
def vault_tree(path: str = "") -> dict:
    """
    One folder level of the vault, with per-file in-RAG status. `path` is
    vault-relative posix; '' = the configured browse root (webui.vault_tree_root,
    default '00 – AUA_DS' per the user's spec — the rest of the vault is reachable
    via /api/vault/search below). Read-only: never writes to the vault.
    """
    vault = _vault_root()
    root_rel = str(CFG.get("webui.vault_tree_root", "00 – AUA_DS"))
    base = (vault / (path or root_rel)).resolve()
    if not str(base).lower().startswith(str(vault.resolve()).lower()):
        return JSONResponse({"error": "path escapes the vault"}, status_code=400)
    if not base.is_dir():
        return JSONResponse({"error": f"not a folder: {path}"}, status_code=404)

    rag = _rag_lookup()
    dirs, files = [], []
    for entry in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if entry.name in _TREE_SKIP or entry.name.startswith("."):
            continue
        rel = entry.relative_to(vault).as_posix()
        if entry.is_dir():
            dirs.append({"name": entry.name, "path": rel})
        elif entry.suffix.lower() in _TREE_EXTS:
            files.append(_file_row(entry, rel, rag))
    return {"root": root_rel, "path": path or root_rel,
            "dirs": dirs, "files": files}


@app.get("/api/vault/search")
def vault_search(q: str, limit: int = 60) -> dict:
    """Whole-vault filename search (books outside the tree root get their
    in-RAG check + retag here). Case-insensitive substring; capped."""
    ql = q.strip().lower()
    if len(ql) < 2:
        return JSONResponse({"error": "query too short"}, status_code=400)
    vault = _vault_root()
    rag = _rag_lookup()
    rows = []
    limit = max(1, min(limit, 200))
    # unlike the tree, search DOES look inside _ingested (archived inbox
    # PDFs stay findable + show their in-RAG status via the path-strip above)
    search_skip = _TREE_SKIP - {"_ingested"}
    for ext in _TREE_EXTS:
        for f in vault.rglob(f"*{ext}"):
            if any(part in search_skip or part.startswith(".") for part in f.parts):
                continue
            if ql not in f.name.lower():
                continue
            rows.append(_file_row(f, f.relative_to(vault).as_posix(), rag))
            if len(rows) >= limit:
                return {"rows": rows, "truncated": True}
    return {"rows": rows, "truncated": False}


@app.get("/api/settings")
def settings() -> dict:
    return {
        "rag_api": RAG_API,
        "vault_path": str(CFG.get("pdf.vault_path") or CFG.get("parser.vault_path")),
        "inbox_dir": CFG.get("webui.inbox_dir", "00 – AUA_DS/Other/Inbox"),
        "jsonl_files": [p.name for p in chunk_files()],
        "ocr_engines": ["auto", "tesseract", "vlm", "none"],
    }


@app.get("/api/schema")
def api_schema() -> dict:
    """
    Machine-readable capability map of the MANAGEMENT console, so an agent
    (an agent / Claude Code) can drive corpus operations over JSON the same way
    it drives the query API (:8051) — no browser, no page snapshots.

    Every operation carries a `permission` tier the calling agent MUST honor:
      * read       — safe, no confirmation needed (stats, search, status, logs)
      * mutating   — changes the index; ask the user first (ingest/append/retag/OCR)
      * destructive— removes content; ALWAYS confirm with the user, echo exactly
                     what will be deleted, and never run unprompted.
    This is a POLICY the agent enforces (the local API has no auth) — the tiers
    exist so a toolkit/skill can gate calls. See the rag-ops skill.
    """
    return {
        "service": "personal-rag-management-console",
        "version": app.version,
        "base_url": f"http://127.0.0.1:{CFG.get('webui.port', 8052)}",
        "query_api": RAG_API,
        "permission_tiers": {
            "read": "safe; no confirmation",
            "mutating": "changes the index; ask the user before running",
            "destructive": "removes content; ALWAYS confirm, echo the exact "
                           "targets, never run unprompted",
        },
        "worker": "single serial queue; index-changing jobs run one at a time. "
                  "Restart serve_api (:8051) after any index change so the warm "
                  "pipeline reloads.",
        "endpoints": {
            "GET /api/overview": {
                "permission": "read",
                "purpose": "corpus summary: chunk/doc counts, per-domain + "
                           "per-jsonl breakdown, chroma vector count, disk use, "
                           "rag_api health, last 6 jobs"},
            "GET /api/facets": {
                "permission": "read",
                "purpose": "domain + course + jsonl facet lists for filtering"},
            "GET /api/documents": {
                "permission": "read",
                "purpose": "per-source-file rows (filename, course, domain, tags, "
                           "chunk count, which JSONLs). This is the 'is X indexed "
                           "/ what's its metadata' lookup.",
                "query": {"q": "filename/path substring", "domain": "str?",
                          "course": "str?", "jsonl": "str?", "tag": "str?",
                          "limit": "<=500", "offset": "int"}},
            "GET /api/documents/preview": {
                "permission": "read",
                "purpose": "first n chunk texts of one document",
                "query": {"source_file": "str (exact key from /api/documents)",
                          "n": "1-10"}},
            "GET /api/vault/tree": {
                "permission": "read",
                "purpose": "one folder level of the vault with per-file in-RAG "
                           "status (browse from webui.vault_tree_root)",
                "query": {"path": "vault-relative posix ('' = root)"}},
            "GET /api/vault/search": {
                "permission": "read",
                "purpose": "whole-vault filename search + in-RAG membership check",
                "query": {"q": "filename substring (>=2 chars)", "limit": "<=200"}},
            "GET /api/inbox": {
                "permission": "read",
                "purpose": "files currently staged in the upload inbox"},
            "GET /api/jobs": {"permission": "read",
                              "purpose": "all jobs newest-first with status"},
            "GET /api/jobs/{id}": {"permission": "read",
                                   "purpose": "one job's full record"},
            "GET /api/jobs/{id}/log": {
                "permission": "read",
                "purpose": "tail a job's log from byte offset",
                "query": {"offset": "int (resume point from the last poll)"}},
            "POST /api/upload": {
                "permission": "mutating",
                "purpose": "save PDF(s) INTO the vault inbox (multipart 'files'). "
                           "Does not index — follow with /api/ingest_inbox."},
            "POST /api/ingest_inbox": {
                "permission": "mutating",
                "purpose": "the sanctioned inbox lane: dup-check -> ingest -> "
                           "append (archives processed PDFs). Chains two jobs.",
                "body": {"force": "bool (past the 409 dup guard)",
                         "ocr_engine": "auto|tesseract|vlm|none?",
                         "domain": "str? (stamped on the batch)",
                         "tags": "list[str]?"}},
            "POST /api/jobs": {
                "permission": "mutating",
                "purpose": "queue a job. Kinds + params under `job_kinds` below.",
                "body": {"kind": "str", "params": "dict"}},
            "POST /api/jobs/{id}/retry": {
                "permission": "mutating",
                "purpose": "re-queue a failed/cancelled job (idempotent)"},
            "POST /api/jobs/{id}/cancel": {
                "permission": "mutating",
                "purpose": "terminate a running/queued job"},
            "POST /api/documents/retag": {
                "permission": "mutating",
                "purpose": "metadata-only: set domain and/or course and/or "
                           "add/remove tags on whole documents. doc_ids + "
                           "embeddings UNCHANGED; queues one BM25 rebuild.",
                "body": {"source_files": "list[str] (from /api/documents)",
                         "domain": "str?",
                         "course": "str? (sets course_name+course_code)",
                         "add_tags": "list[str]?",
                         "remove_tags": "list[str]?", "rebuild": "bool"}},
            "POST /api/documents/delete": {
                "permission": "destructive",
                "purpose": "remove documents from the INDEX (paged Chroma delete + "
                           "JSONL row removal + queued rebuild). Vault files are "
                           "NEVER touched. Confirm the exact source_files first.",
                "body": {"source_files": "list[str]", "rebuild": "bool"}},
        },
        "job_kinds": {
            "ingest_pdfs": {
                "permission": "mutating",
                "params": {"include_path": "substr", "exclude_path": "substr",
                           "output": "data/*.jsonl", "max_pages": "int",
                           "pages": '"1-50,60,70-80" (1-based subset)',
                           "ocr_engine": "auto|tesseract|vlm|none",
                           "chunking": "heading|fixed (how oversized sections "
                                       "split; fixed = sliding window for "
                                       "OCR/wall-of-text)",
                           "only_books": "bool", "skip_books": "bool",
                           "no_images": "bool", "force_domain": "str",
                           "force_tags": "csv or list"},
                "note": "ocr_engine=vlm needs the DeepSeek-OCR server on :8100 up "
                        "(that's 'rag ocr'). VLM is ~12s/page on one GPU."},
            "ingest_notebooks": {
                "permission": "mutating",
                "params": {"output": "data/*.jsonl", "no_outputs": "bool",
                           "save_figures": "bool", "exts": ".ipynb,.py,..."},
                "note": "owns .ipynb/.py/.R/.Rmd"},
            "ingest_code": {
                "permission": "mutating",
                "params": {"output": "data/*.jsonl", "include_path": "substr",
                           "exclude_path": "substr", "exts": ".js,.ts,.sql,..."},
                "note": "every language ingest_notebooks doesn't cover; agent-"
                        "project roots need an include_path to be scoped in"},
            "index_append": {"permission": "mutating",
                             "params": {"file": "data/*.jsonl"},
                             "note": "idempotent upsert; also rebuilds sparse"},
            "index_rebuild": {"permission": "mutating", "params": {},
                              "note": "full rebuild from chunks.jsonl — heavy"},
            "rebuild_bm25": {"permission": "mutating", "params": {},
                             "note": "sync sparse after ingest/delete/retag"},
            "build_hype": {"permission": "mutating",
                           "params": {"include_path": "substr",
                                      "file_types": "csv", "questions": "int",
                                      "max_chunks": "int", "dry_run": "bool"},
                           "note": "ONE LLM call per chunk — always scope + "
                                   "--dry-run first"},
            "recalibrate": {"permission": "mutating",
                            "params": {"dry_run": "bool"},
                            "note": "metadata-only course recalibration"},
            "eval": {"permission": "read",
                     "params": {"retrieval_only": "bool (skip generation — "
                                                  "offline, minutes)"},
                     "note": "golden-query suite; the full (generation) mode "
                             "needs FreeLLMAPI up"},
        },
        "restart_after_index_change": "python -m uvicorn serve_api:app --host "
                                      "127.0.0.1 --port 8051",
    }
