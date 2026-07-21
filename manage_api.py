"""
manage_api.py — Corpus management console (backend) for the personal RAG.

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

app = FastAPI(title="the personal RAG — Management Console", version="0.1.0")
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


def _files_csv(files) -> str:
    """Validate + join a filename list for --include-files/--files flags.
    Plain filenames only (the upload sanitizer never produces commas or path
    separators, so anything else here is a caller bug or an escape attempt)."""
    if isinstance(files, str):
        files = [f for f in files.split(",")]
    names = [str(f).strip() for f in (files or []) if str(f).strip()]
    if not names:
        raise ValueError("file list is empty")
    for n in names:
        if n != Path(n).name or "," in n:
            raise ValueError(f"plain filenames only: {n!r}")
    return ",".join(names)


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
        if prm.get("include_files"):
            argv += ["--include-files", _files_csv(prm["include_files"])]
        if prm.get("chunking"):
            if prm["chunking"] not in ("heading", "fixed", "document", "none"):
                raise ValueError("chunking must be heading|fixed|document|none")
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
        if prm.get("include_path"):
            argv += ["--include-path", str(prm["include_path"])]
        if prm.get("include_files"):
            argv += ["--include-files", _files_csv(prm["include_files"])]
        if prm.get("force_domain"):
            argv += ["--force-domain", str(prm["force_domain"])]
        if prm.get("force_tags"):
            tags = prm["force_tags"]
            if isinstance(tags, (list, tuple)):
                tags = ",".join(str(t) for t in tags)
            argv += ["--force-tags", str(tags)]
        return argv
    if kind == "ingest_code":
        argv = [py, "main.py", "ingest-code"]
        if prm.get("output"):
            argv += ["--output", _safe_rel(str(prm["output"]))]
        if prm.get("include_path"):
            argv += ["--include-path", str(prm["include_path"])]
        if prm.get("exclude_path"):
            argv += ["--exclude-path", str(prm["exclude_path"])]
        if prm.get("include_files"):
            argv += ["--include-files", _files_csv(prm["include_files"])]
        if prm.get("exts"):
            argv += ["--exts", str(prm["exts"])]
        if prm.get("force_domain"):
            argv += ["--force-domain", str(prm["force_domain"])]
        if prm.get("force_tags"):
            tags = prm["force_tags"]
            if isinstance(tags, (list, tuple)):
                tags = ",".join(str(t) for t in tags)
            argv += ["--force-tags", str(tags)]
        return argv
    if kind == "ingest_md":
        # Scoped md parse (inbox md lane): include filter + own output are
        # REQUIRED so the canonical chunks.jsonl can never be clobbered.
        if not prm.get("include_path"):
            raise ValueError("ingest_md requires include_path")
        out = _safe_rel(str(prm.get("output") or ""))
        if Path(out).name == "chunks.jsonl":
            raise ValueError("ingest_md must not write chunks.jsonl")
        argv = [py, "main.py", "ingest-md",
                "--include-path", str(prm["include_path"]),
                "--output", out]
        if prm.get("chunking"):
            if prm["chunking"] not in ("heading", "fixed", "document", "none"):
                raise ValueError("chunking must be heading|fixed|document|none")
            argv += ["--chunking", prm["chunking"]]
        if prm.get("force_domain"):
            argv += ["--force-domain", str(prm["force_domain"])]
        if prm.get("force_tags"):
            tags = prm["force_tags"]
            if isinstance(tags, (list, tuple)):
                tags = ",".join(str(t) for t in tags)
            argv += ["--force-tags", str(tags)]
        return argv
    if kind == "fetch_web":
        urls = prm.get("urls") or []
        if isinstance(urls, str):
            urls = [u for u in urls.replace("\n", ",").split(",") if u.strip()]
        urls = [str(u).strip() for u in urls if str(u).strip()]
        if not urls:
            raise ValueError("fetch_web requires urls")
        for u in urls:
            if not re.match(r"^https?://", u):
                raise ValueError(f"only http(s) URLs are fetched: {u!r}")
        backend = prm.get("backend") or "auto"
        if backend not in ("auto", "requests", "crawl4ai", "scrapling"):
            raise ValueError("backend must be auto|requests|crawl4ai|scrapling")
        fmt = prm.get("format") or "md"
        if fmt not in ("md", "pdf"):
            raise ValueError("format must be md|pdf")
        return [py, "main.py", "fetch-web", "--urls", ",".join(urls),
                "--backend", backend, "--format", fmt]
    if kind == "convert_files":
        argv = [py, "main.py", "convert-files",
                "--files", _files_csv(prm.get("files"))]
        if prm.get("ocr_pages"):
            from src.ingestion.pdf_loader import parse_page_spec
            parse_page_spec(str(prm["ocr_pages"]))   # ValueError -> 400
            argv += ["--ocr-pages", str(prm["ocr_pages"])]
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
            # Opt-in autopilot: after a SUCCESSFUL index-changing job, restart
            # the warm query API so it serves the new state. The flag is read
            # fresh from disk so the Settings toggle applies immediately.
            if (job.status == "done"
                    and job.kind in ("index_append", "index_rebuild",
                                     "rebuild_bm25")):
                try:
                    auto = load_config().get("webui.auto_restart_rag", False)
                except Exception:
                    auto = False
                if str(auto).lower() == "true":
                    try:
                        info = _restart_rag_api()
                        msg = (f"\n[manage_api] auto-restarted :{info['port']} "
                               f"(pid {info['killed_pid']} -> {info['new_pid']}) "
                               f"— webui.auto_restart_rag is on\n")
                    except Exception as e:
                        msg = f"\n[manage_api] auto-restart failed: {e}\n"
                    try:
                        with open(job.log_file, "ab") as lf:
                            lf.write(msg.encode())
                    except OSError:
                        pass


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
    chunking: Optional[str] = None  # heading|fixed|document|none (oversized sections)
    # Batch-level metadata (inbox files carry no course path): stamped on every
    # chunk of this batch. domain feeds scope routing; tags feed tag search +
    # the retrieval tag boost.
    domain: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    # Optional subset: restrict the lane to these inbox filenames (the custom-
    # jobs designer routes its custom files elsewhere and sends the rest here).
    # None/empty = the whole inbox, the classic behavior.
    files: Optional[list[str]] = None
    # Optional vault-relative destination folder: files are MOVED there BEFORE
    # the ingest job runs, so source_file (and therefore doc_ids) match the
    # file's final home — moving after ingest would orphan the indexed path.
    # Unset = classic behavior (stay in the inbox, archive to _ingested).
    dest_dir: Optional[str] = None


class InboxDeleteIn(BaseModel):
    names: list[str] = Field(min_length=1)   # plain filenames inside the inbox


class ImportFetchIn(BaseModel):
    urls: list[str] = Field(min_length=1)
    backend: str = "auto"           # auto | requests | crawl4ai | scrapling
    format: str = "md"              # md (markitdown) | pdf (Chromium print)


class ImportConvertIn(BaseModel):
    files: list[str] = Field(min_length=1)   # inbox filenames to convert to .md
    ocr_pages: Optional[str] = None          # e.g. "1-4,9" — OCR these PDF pages too


class ImportPromoteIn(BaseModel):
    names: list[str] = Field(min_length=1)   # _converted .md files -> inbox root


class CustomGroupIn(BaseModel):
    kind: str                                # pdf | code | md | nb
    files: list[str] = Field(min_length=1)   # inbox filenames in this group
    chunking: Optional[str] = None           # heading|fixed|document|none
    ocr_engine: Optional[str] = None         # pdf groups only
    pages: Optional[str] = None              # pdf groups only ("1-50,60")
    domain: Optional[str] = None             # pdf groups only (force_domain)
    tags: list[str] = Field(default_factory=list)  # pdf groups only
    exts: Optional[str] = None               # code groups only (".sql,.js")
    output: Optional[str] = None             # override the timestamped JSONL
    # Vault-relative destination: the group's files MOVE there before the
    # ingest job runs (source_file/doc_ids match the final home). Unset =
    # files stay in the inbox.
    dest_dir: Optional[str] = None


class CustomIngestIn(BaseModel):
    groups: list[CustomGroupIn] = Field(min_length=1)
    force: bool = False             # override the already-indexed dup guard


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

    # Sparse count from the build-time sidecar (never unpickle the payload
    # here — that's a multi-GB RAM spike on the 16 GB box).
    sparse_count = sparse_built = None
    try:
        meta_p = Path(str(CFG.path("paths.bm25_index")) + ".meta.json")
        if meta_p.exists():
            sm = json.loads(meta_p.read_text(encoding="utf-8"))
            sparse_count = sm.get("count")
            sparse_built = sm.get("built_at")
    except Exception:
        pass

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
        "sparse_count": sparse_count,
        "sparse_built": sparse_built,
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
    subset = {n.strip() for n in (body.files or []) if n.strip()} or None
    pdfs = [f for f in sorted(inbox.iterdir())
            if f.is_file() and f.suffix.lower() == ".pdf"
            and (subset is None or f.name in subset)]
    non_pdfs = [f.name for f in sorted(inbox.iterdir())
                if f.is_file() and f.suffix.lower() != ".pdf"
                and (subset is None or f.name in subset)]
    if not pdfs:
        return JSONResponse(
            {"ok": False, "error": "No PDFs in the inbox — drop files in first."
             if subset is None else "None of the requested files are inbox PDFs.",
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
    # uuid tail: per-file metadata sends several of these calls in the same
    # second, and same-name outputs would make the batches clobber each other
    out = (f"data/inbox_{time.strftime('%Y%m%d_%H%M%S')}"
           f"_{uuid.uuid4().hex[:4]}_chunks.jsonl")
    # Validate the enums BEFORE any file moves — a 400 after moving would
    # strand files in the destination with nothing queued for them.
    if body.ocr_engine and body.ocr_engine not in ("auto", "tesseract", "vlm", "none"):
        return JSONResponse({"ok": False,
                             "error": "ocr_engine must be auto|tesseract|vlm|none"},
                            status_code=400)
    if body.chunking and body.chunking not in ("heading", "fixed", "document", "none"):
        return JSONResponse({"ok": False,
                             "error": "chunking must be heading|fixed|document|none"},
                            status_code=400)
    names = [f.name for f in pdfs]
    moved_to = None
    if body.dest_dir:
        # Move FIRST so source_file/doc_ids carry the final path; no archive
        # step afterwards — the destination IS the file's home.
        try:
            dest_abs, dest_rel = _resolve_vault_dest(body.dest_dir)
        except ValueError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        names = _move_to_dest(names, dest_abs)
        include_rel, moved_to = dest_rel, dest_rel
    params: dict[str, Any] = {"include_path": include_rel, "output": out,
                              "no_images": True,
                              "archive_processed": body.dest_dir is None,
                              "include_files": names}
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
            "files": names,
            "moved_to": moved_to,
            "non_pdfs_ignored": non_pdfs,
            "forced_past_conflicts": conflicts if body.force else [],
            "output": out,
            "jobs": [j1.public(), j2.public()],
            "note": "index --append rebuilds the sparse index itself; restart "
                    "serve_api (:8051) once the append job finishes."}


# ---- destination-folder support (move BEFORE ingest, doc_id-stable) ----

def _resolve_vault_dest(dest_dir: str) -> tuple[Path, str]:
    """Validate a vault-relative destination folder; create it if missing.
    Returns (absolute_path, vault_relative_posix). Rejects escapes."""
    vault = _vault_root().resolve()
    rel = (dest_dir or "").strip().replace("\\", "/").strip("/")
    if not rel:
        raise ValueError("empty destination")
    if ".." in rel.split("/") or re.match(r"^([A-Za-z]:|/)", rel):
        raise ValueError(f"destination must be vault-relative: {dest_dir!r}")
    dest = (vault / rel).resolve()
    if not str(dest).lower().startswith(str(vault).lower()):
        raise ValueError("destination escapes the vault")
    dest.mkdir(parents=True, exist_ok=True)
    return dest, dest.relative_to(vault).as_posix()


def _move_to_dest(names: list[str], dest: Path) -> list[str]:
    """Move inbox files into their destination folder (collision-safe rename).
    Returns the FINAL filenames — ingest jobs must scope on these."""
    inbox = _inbox()
    final: list[str] = []
    for name in names:
        src = inbox / name
        tgt = dest / name
        for i in itertools.count(2):
            if not tgt.exists():
                break
            tgt = dest / f"{src.stem} ({i}){src.suffix}"
        shutil.move(str(src), str(tgt))
        final.append(tgt.name)
    return final


# ---- inbox housekeeping + import lane (fetch / convert / promote) ----

def _converted_dir() -> Path:
    d = _inbox() / "_converted"
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.post("/api/inbox/delete")
def inbox_delete(body: InboxDeleteIn) -> dict:
    """Remove added-by-accident files from the inbox (and/or its _converted
    staging). Deletes the FILES ON DISK inside the inbox only — nothing that
    was already indexed is touched (that's /api/documents/delete)."""
    inbox = _inbox()
    conv = _converted_dir()
    removed, missing = [], []
    for name in body.names:
        name = (name or "").strip()
        if not name or Path(name).name != name:
            missing.append(name)
            continue
        p = inbox / name
        if not p.is_file():
            p = conv / name
        if p.is_file():
            p.unlink()
            removed.append(name)
        else:
            missing.append(name)
    return {"ok": True, "removed": removed, "missing": missing}


@app.get("/api/import/converted")
def import_converted() -> dict:
    """List staged conversions (.md and printed .pdf) awaiting promotion."""
    conv = _converted_dir()
    rows = [{"name": f.name, "bytes": f.stat().st_size,
             "ext": f.suffix.lower()}
            for f in sorted(conv.iterdir())
            if f.is_file() and f.suffix.lower() in (".md", ".pdf")]
    return {"dir": str(conv), "files": rows}


@app.get("/api/import/file")
def import_file(name: str, where: str = "converted", download: int = 0):
    """Serve one staged/inbox file for in-console preview (md rendered
    client-side; pdf shown in the browser's viewer with page numbers — that's
    how you pick OCR page ranges). Plain filenames only; read-only.

    download=1 sends it as an attachment instead: a fetched page you want to
    KEEP but not index (save the .md/.pdf and move on) shouldn't have to go
    through the ingest flow to get out of the staging pool.
    """
    if Path(name).name != name or not name:
        return JSONResponse({"error": "plain filenames only"}, status_code=400)
    base = {"converted": _converted_dir(), "inbox": _inbox()}.get(where)
    if base is None:
        return JSONResponse({"error": "where must be converted|inbox"},
                            status_code=400)
    p = base / name
    if not p.is_file() or p.suffix.lower() not in (".md", ".pdf"):
        return JSONResponse({"error": f"no such previewable file: {name}"},
                            status_code=404)
    media = "application/pdf" if p.suffix.lower() == ".pdf" else \
            "text/markdown; charset=utf-8"
    if download:
        # octet-stream so the browser saves rather than renders the .md
        return FileResponse(p, media_type="application/octet-stream",
                            filename=p.name,
                            content_disposition_type="attachment")
    return FileResponse(p, media_type=media,
                        content_disposition_type="inline")


@app.get("/api/import/ocr_scan")
def import_ocr_scan(name: str, where: str = "converted", limit: int = 400) -> dict:
    """Which pages of a staged PDF need OCR? Read-only report, no OCR run.

    Saves scrolling a 700-page book by hand: returns the page ranges with no
    extractable text (paste straight into --pages / the ⚙ OCR range box), the
    'sparse' middle ground worth eyeballing, and a per-page sample so you can
    confirm a flagged page really is a scan before spending an OCR pass on it.

    Uses the SAME threshold the ingest path uses, so its verdict matches what
    ingestion would actually do.
    """
    if Path(name).name != name or not name:
        return {"ok": False, "error": "plain filenames only"}
    base = {"converted": _converted_dir(), "inbox": _inbox()}.get(where)
    if base is None:
        return {"ok": False, "error": "where must be converted|inbox"}
    p = base / name
    if not p.is_file() or p.suffix.lower() != ".pdf":
        return {"ok": False, "error": f"not a staged PDF: {name}"}
    from src.ingestion.ocr_scan import scan_pdf
    try:
        rep = scan_pdf(p, threshold=int(CFG.get("pdf.skip_scanned_threshold", 50)))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    # Per-page rows are only for eyeballing; cap them so a 900-page book does
    # not ship a megabyte of JSON into the browser. The ranges are complete.
    rep["pages_truncated"] = len(rep["pages"]) > limit
    rep["pages"] = rep["pages"][:limit]
    return {"ok": True, **rep}


@app.post("/api/import/fetch")
def import_fetch(body: ImportFetchIn) -> dict:
    """Queue a fetch_web job: pull the URLs into <inbox>/_converted, either as
    markdown (markitdown) or as a printed PDF of the rendered page (headless
    Chromium — LaTeX/tables/code exactly as the site shows them). Nothing is
    indexed."""
    try:
        job = enqueue("fetch_web", {"urls": body.urls, "backend": body.backend,
                                    "format": body.format})
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, "job": job.public(),
            "note": "Outputs land in _converted — preview, then promote to the "
                    "inbox and ingest."}


@app.post("/api/import/convert")
def import_convert(body: ImportConvertIn) -> dict:
    """Queue a convert_files job: markitdown the named inbox files to .md in
    _converted; optional Tesseract OCR for selected PDF pages."""
    inbox = _inbox()
    missing = [n for n in body.files if not (inbox / n).is_file()
               or Path(n).name != n]
    if missing:
        return JSONResponse({"ok": False, "error": "not in inbox",
                             "missing": missing}, status_code=400)
    try:
        job = enqueue("convert_files",
                      {"files": body.files, "ocr_pages": body.ocr_pages})
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, "job": job.public()}


@app.post("/api/import/promote")
def import_promote(body: ImportPromoteIn) -> dict:
    """Move staged _converted .md files into the inbox root so they appear in
    the uploads list and can be routed by the ingest lanes."""
    inbox = _inbox()
    conv = _converted_dir()
    moved, missing = [], []
    for name in body.names:
        name = (name or "").strip()
        src = conv / name
        if not name or Path(name).name != name or not src.is_file():
            missing.append(name)
            continue
        dest = inbox / name
        i = 1
        while dest.exists():
            dest = inbox / f"{Path(name).stem} ({i}){Path(name).suffix}"
            i += 1
        shutil.move(str(src), str(dest))
        moved.append(dest.name)
    return {"ok": True, "moved": moved, "missing": missing}


# ---- custom-jobs designer: per-group file-scoped ingest ----

@app.post("/api/ingest_custom")
def ingest_custom(body: CustomIngestIn) -> dict:
    """
    Compile the custom-jobs plan into the serial job queue. Each group is a
    set of inbox files of one kind with its own parameters:
      pdf  -> ingest_pdfs  (chunking / ocr_engine / pages / domain / tags)
      code -> ingest_code  (chunking n/a; exts subset)
      md   -> ingest_md    (chunking; one job per file — the parser scope is
                            a path substring, so each md file gets its own)
    Every group's output JSONL gets its own index_append job right after it,
    so a failed group never blocks the others' indexing.
    Guards: unknown kinds/files -> 400; already-indexed-looking files -> 409
    unless force (same stem check as the inbox lane).
    """
    inbox = _inbox()
    vault = Path(CFG.get("pdf.vault_path") or CFG.get("parser.vault_path"))
    include_rel = inbox.relative_to(vault).as_posix()
    have = {f.name for f in inbox.iterdir() if f.is_file()}

    problems = []
    dests: dict[int, tuple[Path, str]] = {}
    for gi, g in enumerate(body.groups):
        if g.kind not in ("pdf", "code", "md", "nb"):
            problems.append(f"group {gi}: kind must be pdf|code|md|nb")
        if g.chunking and g.chunking not in ("heading", "fixed", "document", "none"):
            problems.append(f"group {gi}: bad chunking {g.chunking!r}")
        if g.ocr_engine and g.ocr_engine not in ("auto", "tesseract", "vlm", "none"):
            problems.append(f"group {gi}: bad ocr_engine {g.ocr_engine!r}")
        for n in g.files:
            if n not in have:
                problems.append(f"group {gi}: {n!r} is not in the inbox")
        if g.dest_dir:
            try:
                dests[gi] = _resolve_vault_dest(g.dest_dir)
            except ValueError as e:
                problems.append(f"group {gi}: {e}")
    if problems:
        return JSONResponse({"ok": False, "error": "bad plan",
                             "problems": problems}, status_code=400)

    # Dup guard across ALL custom files (stem match against the manifest).
    manifest = build_manifest()
    known: dict[str, str] = {}
    for sf, d in manifest.items():
        known.setdefault(Path(sf).stem.lower(), sf)
        fn = str(d.get("filename") or "")
        if fn:
            known.setdefault(Path(fn).stem.lower(), sf)
    all_files = [n for g in body.groups for n in g.files]
    conflicts = [{"file": n, "existing_source": known[Path(n).stem.lower()]}
                 for n in all_files if Path(n).stem.lower() in known]
    if conflicts and not body.force:
        return JSONResponse(
            {"ok": False,
             "error": f"{len(conflicts)} file(s) look already indexed.",
             "conflicts": conflicts,
             "hint": "force:true ingests anyway (duplicates if same files)."},
            status_code=409)

    ts = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    jobs = []
    try:
        for gi, g in enumerate(body.groups):
            # Destination groups move their files FIRST (doc_ids carry the
            # final path); scope the jobs on the destination + final names.
            scope_rel, names = include_rel, list(g.files)
            if gi in dests:
                dest_abs, dest_rel = dests[gi]
                names = _move_to_dest(names, dest_abs)
                scope_rel = dest_rel
            gdomain = g.domain.strip().lower() if g.domain else None
            gtags = [t.strip().lstrip("#").lower() for t in g.tags if t.strip()] or None
            if g.kind == "md":
                # one scoped parse per md file (path-substring scope)
                for fi, name in enumerate(names):
                    out = f"data/inbox_{ts}_g{gi}f{fi}_md_chunks.jsonl"
                    jobs.append(enqueue("ingest_md", {
                        "include_path": f"{scope_rel}/{name}",
                        "output": out, "chunking": g.chunking,
                        "force_domain": gdomain, "force_tags": gtags}))
                    jobs.append(enqueue("index_append", {"file": out}))
                continue
            out = g.output or f"data/inbox_{ts}_g{gi}_{g.kind}_chunks.jsonl"
            if g.kind == "pdf":
                params: dict[str, Any] = {
                    "include_path": scope_rel, "include_files": names,
                    "output": out, "no_images": True,
                    "archive_processed": gi not in dests}
                if g.chunking:
                    params["chunking"] = g.chunking
                if g.ocr_engine:
                    params["ocr_engine"] = g.ocr_engine
                if g.pages:
                    params["pages"] = g.pages
                if g.domain:
                    params["force_domain"] = g.domain.strip().lower()
                if g.tags:
                    params["force_tags"] = [t.strip().lstrip("#").lower()
                                            for t in g.tags if t.strip()]
                jobs.append(enqueue("ingest_pdfs", params))
            elif g.kind == "nb":                         # .ipynb/.py/.R/.Rmd
                params = {"include_path": scope_rel, "include_files": names,
                          "output": out, "force_domain": gdomain,
                          "force_tags": gtags}
                if g.exts:
                    params["exts"] = g.exts
                jobs.append(enqueue("ingest_notebooks", params))
            else:                                        # code
                params = {"include_path": scope_rel, "include_files": names,
                          "output": out, "force_domain": gdomain,
                          "force_tags": gtags}
                if g.exts:
                    params["exts"] = g.exts
                jobs.append(enqueue("ingest_code", params))
            jobs.append(enqueue("index_append", {"file": out}))
    except (ValueError, KeyError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    return {"ok": True,
            "jobs": [j.public() for j in jobs],
            "forced_past_conflicts": conflicts if body.force else [],
            "note": "Groups run serially; each group's append follows it. "
                    "Restart serve_api (:8051) after the last append. "
                    "Code/md chunks keep path-derived metadata — use retag "
                    "for domain/course/tags afterwards if needed."}


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
    default '00 – AUA_DS' per the author's spec — the rest of the vault is reachable
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


# Editable config surface for the Settings tab. Maps dotted key -> spec.
# Everything here is persisted into config.yaml IN PLACE (comments preserved);
# none of it hot-applies — the response says which services need a restart.
EDITABLE_SETTINGS: dict[str, dict] = {
    "parser.vault_path":            {"kind": "dir",  "restart": ":8051 + :8052",
                                     "label": "Obsidian vault root"},
    "paths.chunks_file":            {"kind": "str",  "restart": ":8052",
                                     "label": "Markdown chunks JSONL (data dir anchor)"},
    "paths.chroma_dir":             {"kind": "str",  "restart": ":8051 + :8052",
                                     "label": "ChromaDB directory (dense vectors)"},
    "paths.bm25_index":             {"kind": "str",  "restart": ":8051 + :8052",
                                     "label": "BM25 index pickle (sparse)"},
    "paths.collection_name":        {"kind": "str",  "restart": ":8051 + :8052",
                                     "label": "Chroma collection name"},
    "embedding.local_model":        {"kind": "str",  "restart": ":8051",
                                     "label": "Embedding model (HF id or local path)"},
    # Free text on purpose (any HF cross-encoder id works); the console offers
    # the known-good ones as suggestions with their measured cost.
    "retrieval.cross_encoder_model": {"kind": "str", "restart": ":8051",
                                     "label": "Cross-encoder rerank model"},
    "retrieval.cross_encoder_max_length": {"kind": "str", "restart": ":8051",
                                     "label": "Cross-encoder max tokens per pair"},
    "retrieval.cross_encoder_device": {"kind": "enum", "restart": ":8051",
                                     "values": ["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
                                     "label": "Cross-encoder device"},
    "retrieval.rerank_mode":        {"kind": "enum", "restart": ":8051",
                                     "values": ["cross_encoder", "lexical", "none"],
                                     "label": "Default rerank method"},
    "parser.chunking":              {"kind": "enum", "restart": ":8052",
                                     "values": ["heading", "fixed", "document", "none"],
                                     "label": "Default chunking strategy"},
    # Values are filled in at request time from the `providers:` registry —
    # see _provider_choices(). Switching this rewrites generation.model too
    # (see settings_update), because a provider and a stale model id from the
    # previous provider is the one combination that fails at call time.
    "generation.provider":          {"kind": "enum", "restart": ":8051",
                                     "values": [],
                                     "label": "Generation backend (providers: registry)"},
    "generation.base_url":          {"kind": "str",  "restart": ":8051",
                                     "label": "Generation endpoint (legacy providers only)"},
    "generation.model":             {"kind": "str",  "restart": ":8051",
                                     "label": "Generation model id"},
    "webui.inbox_dir":              {"kind": "str",  "restart": ":8052",
                                     "label": "Inbox folder (vault-relative)"},
    "webui.auto_restart_rag":       {"kind": "enum", "restart": "none (read per job)",
                                     "values": ["true", "false"],
                                     "label": "Auto-restart :8051 after index-changing jobs"},
}


def _provider_choices(disk_cfg) -> tuple[list[str], list[dict]]:
    """Selectable generation backends + their status, for the Settings tab.

    Returns (names, details). Names = the `providers:` registry plus the
    reserved legacy names, so the dropdown can never strand an existing config
    on a value it cannot re-select.

    details carries `key_present` — whether the env var each provider names is
    actually set — because "I switched to MiniMax and it 500s" is almost always
    an unset key. The key VALUE is never read or returned, only its presence.
    """
    from src.llm.llm_client import LLMClient

    registry = disk_cfg.get("providers", {}) or {}
    names = [n for n in registry if n not in LLMClient.RESERVED_PROVIDERS]
    details = []
    for name in names:
        spec = registry[name] or {}
        env = spec.get("api_key_env")
        details.append({
            "name": name,
            "kind": spec.get("kind", "openai"),
            "base_url": spec.get("base_url"),
            "model": spec.get("model"),
            "api_key_env": env,
            "key_optional": bool(spec.get("api_key_optional")),
            "key_present": bool(os.environ.get(env)) if env else True,
        })
    return names + list(LLMClient.RESERVED_PROVIDERS), details


def _persist_section_keys(cfg_path: Path, changes: dict[str, Any]) -> list[str]:
    """Rewrite `section.leaf` values in config.yaml IN PLACE, preserving
    comments and layout. Section-aware (unlike persist_config_values) so keys
    that repeat across sections — vault_path, model — stay unambiguous: the
    leaf must appear exactly once WITHIN its top-level section block."""
    text = cfg_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    written: list[str] = []
    for dotted, value in changes.items():
        section, _, leaf = dotted.partition(".")
        sval = ("true" if value else "false") if isinstance(value, bool) else str(value)
        if any(c in sval for c in "\r\n"):
            raise ValueError(f"{dotted}: newlines not allowed")
        # quote strings that YAML would mangle (paths with ':' etc.)
        if not re.fullmatch(r"[\w.\-/]+", sval):
            sval = '"' + sval.replace('"', '\\"') + '"'
        sec_re = re.compile(rf"^{re.escape(section)}\s*:\s*(#.*)?$")
        key_re = re.compile(
            rf"^(?P<pre>[ \t]+{re.escape(leaf)}[ \t]*:[ \t]*)(?P<val>[^#\r\n]*?)(?P<post>[ \t]*(?:#[^\r\n]*)?)$")
        hits = []
        in_sec = False
        for i, ln in enumerate(lines):
            if sec_re.match(ln):
                in_sec = True
                continue
            if in_sec and ln and not ln[0] in " \t#":
                in_sec = False               # next top-level key ends the block
            if in_sec:
                m = key_re.match(ln)
                if m:
                    hits.append((i, m))
        if len(hits) != 1:
            raise ValueError(f"{dotted}: found {len(hits)} matches in "
                             f"{cfg_path.name}; refusing to rewrite ambiguously")
        i, m = hits[0]
        lines[i] = m.group("pre") + sval + m.group("post")
        written.append(dotted)
    if written:
        cfg_path.write_text("\n".join(lines), encoding="utf-8")
    return written


class SettingsIn(BaseModel):
    changes: dict[str, Any] = Field(min_length=1)


@app.get("/api/settings")
def settings() -> dict:
    # Re-read config.yaml fresh: after a save (no restart yet) the boot-time
    # CFG is stale, and the Settings tab must show what's ON DISK.
    from src.utils.config_loader import load_config as _load
    disk_cfg = _load()
    provider_names, provider_details = _provider_choices(disk_cfg)
    # Reranker suggestions + whether torch can actually reach a GPU. The device
    # dropdown offering "cuda" on a CPU-only torch build would be a trap, so the
    # console reports what torch really sees rather than what the box has.
    from src.retrieval.reranker import KNOWN_RERANKERS
    try:
        import torch
        gpu = {"torch": torch.__version__,
               "cuda_build": torch.version.cuda,
               "available": bool(torch.cuda.is_available()),
               "devices": [torch.cuda.get_device_name(i)
                           for i in range(torch.cuda.device_count())]
               if torch.cuda.is_available() else []}
    except Exception as e:                       # torch missing/broken
        gpu = {"torch": None, "cuda_build": None, "available": False,
               "devices": [], "error": str(e)}
    editable = {}
    for key, spec in EDITABLE_SETTINGS.items():
        editable[key] = {**spec, "value": disk_cfg.get(key)}
    editable["generation.provider"]["values"] = provider_names
    return {
        "rag_api": RAG_API,
        "providers": provider_details,
        "judge_provider": disk_cfg.get("eval.judge.provider"),
        "rerankers": [{"id": k, **v} for k, v in KNOWN_RERANKERS.items()],
        "gpu": gpu,
        "vault_path": str(CFG.get("pdf.vault_path") or CFG.get("parser.vault_path")),
        "inbox_dir": CFG.get("webui.inbox_dir", "00 – AUA_DS/Other/Inbox"),
        "jsonl_files": [p.name for p in chunk_files()],
        "ocr_engines": ["auto", "tesseract", "vlm", "none"],
        "config_path": str(ROOT / "config.yaml"),
        "editable": editable,
    }


@app.post("/api/settings")
def settings_update(body: SettingsIn) -> dict:
    """Persist whitelisted config values into config.yaml (comment-preserving,
    section-aware). Nothing hot-applies: the response lists which services to
    restart. Vault path must exist; enums are validated; unknown keys 400."""
    from src.utils.config_loader import load_config as _load
    disk_cfg = _load()
    provider_names, _ = _provider_choices(disk_cfg)

    changes: dict[str, Any] = {}
    restarts: set[str] = set()
    notes: list[str] = []
    for key, value in body.changes.items():
        spec = EDITABLE_SETTINGS.get(key)
        if not spec:
            return JSONResponse({"ok": False, "error": f"unknown setting {key!r}"},
                                status_code=400)
        if key == "generation.provider":          # enum filled at request time
            spec = {**spec, "values": provider_names}
        sval = str(value).strip()
        if not sval:
            return JSONResponse({"ok": False, "error": f"{key}: empty value"},
                                status_code=400)
        if spec["kind"] == "enum" and sval not in spec["values"]:
            return JSONResponse(
                {"ok": False,
                 "error": f"{key} must be one of {spec['values']}"}, status_code=400)
        if spec["kind"] == "dir" and not Path(sval).is_dir():
            return JSONResponse(
                {"ok": False,
                 "error": f"{key}: directory does not exist: {sval}"}, status_code=400)
        # No boot-time-CFG "unchanged" skip here: CFG goes stale after a save
        # without a restart, and rewriting an identical value is harmless.
        changes[key] = sval
        restarts.add(spec["restart"])
    # Switching backend without switching model sends the OLD provider's model
    # id to the NEW endpoint, which fails at call time with a confusing 4xx.
    # Carry the new provider's default model along unless the caller set one.
    new_provider = changes.get("generation.provider")
    if new_provider and "generation.model" not in changes:
        spec = (disk_cfg.get("providers", {}) or {}).get(new_provider) or {}
        default_model = spec.get("model")
        if default_model and default_model != disk_cfg.get("generation.model"):
            changes["generation.model"] = str(default_model)
            restarts.add(EDITABLE_SETTINGS["generation.model"]["restart"])
            notes.append(f"generation.model set to {default_model!r} to match "
                         f"provider {new_provider!r}")
        env = spec.get("api_key_env")
        if env and not spec.get("api_key_optional") and not os.environ.get(env):
            notes.append(f"WARNING: {env} is not set in this environment — "
                         f"{new_provider} will fail until you export it "
                         f"(or add it to .env)")

    if not changes:
        return {"ok": True, "written": [], "note": "nothing changed"}
    try:
        written = _persist_section_keys(ROOT / "config.yaml", changes)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    note = ("Saved to config.yaml. Restart to apply: "
            + "; ".join(sorted(restarts))
            + ". Changing the embedding model REQUIRES a full re-embed "
              "of the corpus (main.py index) before search works again.")
    if notes:
        note += "  |  " + "  |  ".join(notes)
    return {"ok": True, "written": written, "note": note}


# ---- folder browser (Settings path pickers) ----

@app.get("/api/browse")
def browse(path: str = "") -> dict:
    """
    One level of the LOCAL filesystem, folders only — powers the Settings
    tab's path pickers and the vault switcher (a browser page can't open a
    native folder dialog for server-side paths). '' lists the drives.
    Read-only; never creates or touches anything.
    """
    if not path:
        import string
        drives = [f"{d}:/" for d in string.ascii_uppercase
                  if Path(f"{d}:/").exists()]
        return {"path": "", "parent": None, "dirs": drives}
    p = Path(path)
    if not p.is_dir():
        return JSONResponse({"error": f"not a folder: {path}"}, status_code=404)
    dirs = []
    try:
        for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if entry.is_dir() and not entry.name.startswith((".", "$")):
                dirs.append(entry.name)
    except PermissionError:
        return JSONResponse({"error": f"no permission: {path}"}, status_code=403)
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "dirs": dirs}


# ---- vault switcher (Obsidian-style: every vault ever opened stays listed,
#      and each vault remembers its own index/path settings) ----

# The per-vault settings snapshot: everything that must travel WITH a vault.
# Each corpus needs its own index trio — reusing another vault's indexes
# retrieves nonsense (same warning the Settings tab shows).
VAULT_KEYS = ["parser.vault_path", "paths.chunks_file", "paths.chroma_dir",
              "paths.bm25_index", "paths.collection_name", "webui.inbox_dir"]

# Registry lives at a vault-INDEPENDENT path (DATA_DIR follows the per-vault
# chunks_file, so it moves on switch — the registry must not move with it).
_VAULT_REGISTRY = ROOT / "data" / ".vault_registry.json"
_vault_lock = threading.Lock()


def _load_vaults() -> list[dict]:
    try:
        return json.loads(_VAULT_REGISTRY.read_text(encoding="utf-8"))["vaults"]
    except Exception:
        return []


def _save_vaults(vaults: list[dict]) -> None:
    _VAULT_REGISTRY.write_text(
        json.dumps({"vaults": vaults}, indent=2, ensure_ascii=False),
        encoding="utf-8")


def _fresh_settings_snapshot() -> dict[str, str]:
    from src.utils.config_loader import load_config as _load
    disk = _load()
    return {k: str(disk.get(k) or "") for k in VAULT_KEYS}


def _norm_vault(p: str) -> str:
    return str(p).replace("\\", "/").rstrip("/").lower()


class VaultSwitchIn(BaseModel):
    path: str                       # vault root folder (absolute)
    label: Optional[str] = None     # display name; defaults to the folder name


class VaultForgetIn(BaseModel):
    path: str                       # registry entry to drop (files untouched)


@app.get("/api/vaults")
def vaults_list() -> dict:
    snap = _fresh_settings_snapshot()
    cur = _norm_vault(snap["parser.vault_path"])
    with _vault_lock:
        vaults = _load_vaults()
        known = {_norm_vault(v["path"]) for v in vaults}
        if cur and cur not in known:      # current vault self-registers
            vaults.append({"path": snap["parser.vault_path"],
                           "label": Path(snap["parser.vault_path"]).name,
                           "last_used": time.strftime("%Y-%m-%d %H:%M"),
                           "settings": snap})
            _save_vaults(vaults)
    rows = [{**{k: v for k, v in v.items() if k != "settings"},
             "current": _norm_vault(v["path"]) == cur} for v in vaults]
    return {"vaults": rows, "current": snap["parser.vault_path"]}


@app.post("/api/vaults/switch")
def vaults_switch(body: VaultSwitchIn) -> dict:
    """
    Switch the console (and, after restarts, the whole RAG) to another vault.
    Obsidian-style: the CURRENT vault's settings are snapshotted into the
    registry first, then the target's last-known settings are restored — or,
    for a never-seen vault, a fresh per-vault index trio is scaffolded next to
    the current one (empty corpus is a valid state; ingest fills it).
    Nothing hot-applies: restart :8051 + :8052 after switching.
    """
    target = Path(body.path)
    if not target.is_dir():
        return JSONResponse({"ok": False,
                             "error": f"vault folder does not exist: {body.path}"},
                            status_code=400)
    snap = _fresh_settings_snapshot()
    cur_key = _norm_vault(snap["parser.vault_path"])
    tgt_key = _norm_vault(str(target))

    with _vault_lock:
        vaults = _load_vaults()
        by_key = {_norm_vault(v["path"]): v for v in vaults}
        # 1. snapshot the current vault's state (its "last session")
        if cur_key:
            cur = by_key.get(cur_key)
            if cur is None:
                cur = {"path": snap["parser.vault_path"],
                       "label": Path(snap["parser.vault_path"]).name}
                vaults.append(cur)
                by_key[cur_key] = cur
            cur["settings"] = snap
            cur["last_used"] = time.strftime("%Y-%m-%d %H:%M")
        # 2. restore (or scaffold) the target's state
        tgt = by_key.get(tgt_key)
        scaffolded = False
        if tgt is None or not tgt.get("settings"):
            slug = re.sub(r"[^\w\-]+", "_", target.name).strip("_").lower() or "vault"
            chroma_root = Path(snap["paths.chroma_dir"]).parent
            settings = {
                "parser.vault_path": str(target).replace("\\", "/"),
                "paths.chunks_file": f"data/vaults/{slug}/chunks.jsonl",
                "paths.chroma_dir": (chroma_root / f"vault_{slug}" / "chroma_db"
                                     ).as_posix(),
                "paths.bm25_index": (chroma_root / f"vault_{slug}" / "bm25_index.pkl"
                                     ).as_posix(),
                "paths.collection_name": snap["paths.collection_name"] or "obsidian_vault",
                "webui.inbox_dir": snap["webui.inbox_dir"] or "Inbox",
            }
            scaffolded = True
            if tgt is None:
                tgt = {"path": str(target), "label": body.label or target.name}
                vaults.append(tgt)
            tgt["settings"] = settings
        if body.label:
            tgt["label"] = body.label
        tgt["last_used"] = time.strftime("%Y-%m-%d %H:%M")
        try:
            written = _persist_section_keys(ROOT / "config.yaml", tgt["settings"])
        except ValueError as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
        _save_vaults(vaults)

    if scaffolded:
        Path(tgt["settings"]["paths.chunks_file"]).parent.mkdir(
            parents=True, exist_ok=True)
    return {"ok": True, "written": written, "scaffolded": scaffolded,
            "note": ("New vault: empty per-vault indexes were scaffolded — "
                     "ingest to fill them. " if scaffolded else
                     "Restored this vault's last-known settings. ")
                    + "Restart :8051 + :8052 to apply."}


@app.post("/api/vaults/forget")
def vaults_forget(body: VaultForgetIn) -> dict:
    """Drop a vault from the registry (files/indexes on disk untouched).
    The currently active vault cannot be forgotten."""
    snap = _fresh_settings_snapshot()
    if _norm_vault(body.path) == _norm_vault(snap["parser.vault_path"]):
        return JSONResponse({"ok": False,
                             "error": "that vault is currently active"},
                            status_code=400)
    with _vault_lock:
        vaults = _load_vaults()
        kept = [v for v in vaults if _norm_vault(v["path"]) != _norm_vault(body.path)]
        if len(kept) == len(vaults):
            return JSONResponse({"ok": False, "error": "not in the registry"},
                                status_code=404)
        _save_vaults(kept)
    return {"ok": True, "forgotten": body.path}


# ---- service restart lane (:8051) ----

def _rag_port() -> int:
    from urllib.parse import urlparse
    return urlparse(RAG_API).port or 8051


def _pid_on_port(port: int) -> int | None:
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[3] == "LISTENING" \
                and parts[1].endswith(f":{port}"):
            return int(parts[4])
    return None


_RESTART_LOCK = threading.Lock()


def _restart_rag_api() -> dict:
    """Kill whatever listens on the query-API port and relaunch serve_api
    detached, inheriting THIS console's environment (rag.bat sets the HF
    cache vars — a console started bare would hand :8051 a broken env, which
    is exactly the failure the launcher comments warn about). Windows-only:
    the Docker deployment restarts via the container, not this lane."""
    if os.name != "nt":
        raise RuntimeError("service restart is the Windows lane; in Docker "
                           "restart the container instead")
    port = _rag_port()
    with _RESTART_LOCK:
        old_pid = _pid_on_port(port)
        if old_pid:
            subprocess.run(["taskkill", "/PID", str(old_pid), "/F"],
                           capture_output=True)
            for _ in range(20):                      # wait for the port to free
                if _pid_on_port(port) is None:
                    break
                time.sleep(0.5)
        log_path = ROOT / "logs" / "serve_api_8051.out.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        lf = open(log_path, "ab")
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "serve_api:app",
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=flags)
    log.info("restarted :%d (old pid %s -> new pid %d)", port, old_pid, proc.pid)
    return {"port": port, "killed_pid": old_pid, "new_pid": proc.pid}


@app.post("/api/service/restart")
def service_restart() -> dict:
    """Restart the warm query API. The pipeline reloads indexes + models from
    scratch, so /health flips ready after ~1–3 minutes."""
    try:
        info = _restart_rag_api()
    except (RuntimeError, OSError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, **info,
            "note": "Warm pipeline reloading — poll /health until ready "
                    "(~1–3 min; models + both indexes load from scratch)."}


@app.get("/api/schema")
def api_schema() -> dict:
    """
    Machine-readable capability map of the MANAGEMENT console, so an agent
    (the local agent / Claude Code) can drive corpus operations over JSON the same way
    it drives the query API (:8051) — no browser, no page snapshots.

    Every operation carries a `permission` tier the calling agent MUST honor:
      * read       — safe, no confirmation needed (stats, search, status, logs)
      * mutating   — changes the index; ask the author first (ingest/append/retag/OCR)
      * destructive— removes content; ALWAYS confirm with the author, echo exactly
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
            "mutating": "changes the index; ask the author before running",
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
                         "chunking": "heading|fixed|document|none?",
                         "domain": "str? (stamped on the batch)",
                         "tags": "list[str]?",
                         "files": "list[str]? (restrict to these inbox PDFs; "
                                  "unset = whole inbox)",
                         "dest_dir": "str? vault-relative folder — files MOVE "
                                     "there BEFORE ingest (doc_ids carry the "
                                     "final path); unset = stay in inbox, "
                                     "archive to _ingested"}},
            "POST /api/ingest_custom": {
                "permission": "mutating",
                "purpose": "custom-jobs designer lane: per-group file-scoped "
                           "ingest (pdf/code/md kinds, each with its own "
                           "params) + an append per group. Same 409 dup guard "
                           "as the inbox lane.",
                "body": {"groups": "[{kind: pdf|code|md|nb, files: [names], "
                                   "chunking?, ocr_engine?, pages?, domain?, "
                                   "tags?, exts?, output?, dest_dir? "
                                   "(vault-relative; move-then-ingest)}]. "
                                   "nb = .ipynb/.py/.R/.Rmd via ingest_notebooks; "
                                   "domain/tags now stamp every kind, not just "
                                   "pdf.",
                         "force": "bool"}},
            "POST /api/inbox/delete": {
                "permission": "mutating",
                "purpose": "remove staged files from the inbox / _converted "
                           "folder ON DISK (index untouched — that's "
                           "/api/documents/delete)",
                "body": {"names": "list[str] (plain filenames)"}},
            "GET /api/import/converted": {
                "permission": "read",
                "purpose": "list staged .md conversions in <inbox>/_converted"},
            "POST /api/import/fetch": {
                "permission": "mutating",
                "purpose": "queue fetch_web: pull http(s) URLs into _converted "
                           "as markdown (markitdown) or as a printed PDF of "
                           "the rendered page (headless Chromium — keeps "
                           "LaTeX/tables/code as the site shows them). "
                           "Nothing indexed.",
                "body": {"urls": "list[str]",
                         "backend": "auto|requests|crawl4ai|scrapling",
                         "format": "md|pdf"}},
            "GET /api/import/file": {
                "permission": "read",
                "purpose": "serve one staged/inbox .md or .pdf for preview "
                           "(pdf shows page numbers -> pick OCR page ranges)",
                "query": {"name": "plain filename", "where": "converted|inbox"}},
            "GET /api/browse": {
                "permission": "read",
                "purpose": "one level of the local filesystem, folders only "
                           "('' = drives) — powers the Settings path pickers"},
            "GET /api/vaults": {
                "permission": "read",
                "purpose": "vault registry (Obsidian-style): every vault ever "
                           "opened, with labels + which one is active"},
            "POST /api/vaults/switch": {
                "permission": "mutating",
                "purpose": "snapshot the current vault's settings, restore the "
                           "target's last-known ones (or scaffold a fresh "
                           "per-vault index trio for a new vault), persist to "
                           "config.yaml. Restart :8051 + :8052 after.",
                "body": {"path": "vault root folder", "label": "str?"}},
            "POST /api/vaults/forget": {
                "permission": "mutating",
                "purpose": "drop a non-active vault from the registry "
                           "(nothing on disk is touched)",
                "body": {"path": "registry entry"}},
            "POST /api/import/convert": {
                "permission": "mutating",
                "purpose": "queue convert_files: markitdown inbox files to .md "
                           "in _converted; optional PDF-page OCR",
                "body": {"files": "list[str]", "ocr_pages": '"1-4,9"?'}},
            "POST /api/import/promote": {
                "permission": "mutating",
                "purpose": "move _converted .md files into the inbox root so "
                           "the ingest lanes can pick them up",
                "body": {"names": "list[str]"}},
            "GET /api/settings": {
                "permission": "read",
                "purpose": "runtime info + the editable config surface (paths, "
                           "models, defaults) with current values"},
            "POST /api/settings": {
                "permission": "mutating",
                "purpose": "persist whitelisted config values into config.yaml "
                           "(comment-preserving). NOTHING hot-applies — the "
                           "response says which services to restart. Changing "
                           "the embedding model needs a FULL RE-EMBED (ask "
                           "the operator).",
                "body": {"changes": "{dotted.key: value} from GET editable"}},
            "POST /api/service/restart": {
                "permission": "mutating",
                "purpose": "kill + relaunch the warm query API (:8051) so it "
                           "serves the current indexes. /health flips ready "
                           "after ~1-3 min. Windows lane only (Docker: restart "
                           "the container). webui.auto_restart_rag=true does "
                           "this automatically after index-changing jobs."},
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
                           "include_files": "list[str] (exact filenames)",
                           "output": "data/*.jsonl", "max_pages": "int",
                           "pages": '"1-50,60,70-80" (1-based subset)',
                           "ocr_engine": "auto|tesseract|vlm|none",
                           "chunking": "heading|fixed|document|none (how "
                                       "oversized sections split; fixed = "
                                       "sliding window for OCR walls, document "
                                       "= element-aware [code/tables/lists "
                                       "never cut], none = no splitting)",
                           "only_books": "bool", "skip_books": "bool",
                           "no_images": "bool", "force_domain": "str",
                           "force_tags": "csv or list"},
                "note": "ocr_engine=vlm needs the DeepSeek-OCR server on :8100 up "
                        "(that's 'rag ocr'). VLM is ~12s/page on one GPU."},
            "ingest_notebooks": {
                "permission": "mutating",
                "params": {"output": "data/*.jsonl", "no_outputs": "bool",
                           "save_figures": "bool", "exts": ".ipynb,.py,...",
                           "include_path": "substr (file-scoped custom jobs)",
                           "include_files": "list[str] (exact filenames)",
                           "force_domain": "str", "force_tags": "csv or list"},
                "note": "owns .ipynb + .py + .R + .Rmd — it has the Python "
                        "ast/`# %%` cell splitter, which raw code ingestion "
                        "lacks (that is WHY .py/.R live here, not in "
                        "ingest_code). File-scopable since session 15."},
            "ingest_code": {
                "permission": "mutating",
                "params": {"output": "data/*.jsonl", "include_path": "substr",
                           "exclude_path": "substr",
                           "include_files": "list[str] (exact filenames)",
                           "exts": ".js,.ts,.sql,...",
                           "force_domain": "str", "force_tags": "csv or list"},
                "note": "every language ingest_notebooks doesn't cover "
                        "(.js/.ts/.sql/.go/.java/.c/.cpp/.rs/… — NOT "
                        ".py/.R/.ipynb/.Rmd); agent-project roots need an "
                        "include_path to be scoped in"},
            "ingest_md": {
                "permission": "mutating",
                "params": {"include_path": "substr (REQUIRED)",
                           "output": "data/*.jsonl (REQUIRED, never chunks.jsonl)",
                           "chunking": "heading|fixed|document|none",
                           "force_domain": "str", "force_tags": "csv or list"},
                "note": "SCOPED markdown parse (inbox md lane); guarded so the "
                        "vault-wide chunks.jsonl can never be clobbered"},
            "fetch_web": {
                "permission": "mutating",
                "params": {"urls": "list[str] (http/https only)",
                           "backend": "auto|requests|crawl4ai|scrapling",
                           "format": "md|pdf (pdf = Chromium page print)"},
                "note": "writes .md/.pdf to <inbox>/_converted; indexes nothing"},
            "convert_files": {
                "permission": "mutating",
                "params": {"files": "list[str] (inbox filenames)",
                           "ocr_pages": '"1-4,9"? (Tesseract, PDFs only)'},
                "note": "markitdown any-file -> .md into <inbox>/_converted"},
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
