"""
manage_api.py — Corpus management console (backend) for Noetrix.

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

import copy
import itertools
import hashlib
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

import yaml
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from eval.bench.questions import (
    STATUSES, SUITES, SchemaError, load_sets, question_from_dict, question_to_dict,
    save_suite, sets_lock)
from src.utils.branding import CONSOLE_API_TITLE, CONSOLE_SERVICE
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import Config, load_config
from src.utils.logger import configure_logging, get_logger

log = get_logger("manage_api")

ROOT = Path(__file__).resolve().parent
CFG: Config = load_config()
# A no-op in practice: get_logger() above already configured logging with the
# defaults, and the first configuration wins (see src/utils/logger.py).
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

app = FastAPI(title=CONSOLE_API_TITLE, version="0.2.0")
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
    # True for a job rebuilt from its on-disk record rather than run by this
    # process. The console shows it so nobody waits for output that is never
    # coming, and so "interrupted" is readable as "the console died", not "the
    # job failed".
    restored: bool = False

    def public(self) -> dict:
        d = asdict(self)
        d["argv"] = " ".join(self.argv)
        d["record"] = _job_record_path(self.id).name
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


def _vault_data_path(p: str, *, default_dir: str = "data") -> str:
    """_safe_rel, then ANCHORED TO THE ACTIVE VAULT's data directory.

    `_safe_rel` returns a project-relative "data/x.jsonl", and main.py resolves
    that against its CWD — the project root. That is only correct while the
    active vault's chunks_file also lives under the project. After a vault
    switch it does not: chunks_file moves (e.g. to G:/ANIMUS/Animus Data), so
    DATA_DIR, chunk_files() and the BM25 union all follow it while ingest jobs
    kept writing into the PREVIOUS vault's data folder. Two consequences, both
    silent: the new vault's Ledger showed nothing, and the stray *_chunks.jsonl
    got swept into the OLD vault's sparse union on its next rebuild — one
    vault's material indexed into another's corpus.

    Validation still happens on the RELATIVE form first, so drive letters and
    `..` are rejected before anything is joined.
    """
    rel = _safe_rel(p, default_dir=default_dir)
    head, _, tail = rel.partition("/")
    if head == default_dir and tail:
        return str((DATA_DIR / tail).resolve())
    return rel


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


# Every chunk-writing lane's canonical chunk file: (the config key that names it,
# the loader's built-in default). One table, so every output guard agrees on which
# file names are protected. "markdown" is the whole-vault parse, chunks.jsonl.
_LANE_CHUNK_FILES = {
    "pdf": ("pdf.output_file", "pdf_chunks.jsonl"),
    "notebooks": ("notebooks.output_file", "ipynb_chunks.jsonl"),
    "code": ("code.output_file", "code_chunks.jsonl"),
    "canvas": ("canvas.output_file", "canvas_chunks.jsonl"),
    "markdown": ("paths.chunks_file", "chunks.jsonl"),
}


def _refuse_other_lanes_file(out: str | None, lane: str) -> None:
    """Refuse an --output that is ANOTHER lane's canonical chunk file.

    Every loader opens its output with "w", so a run pointed at a different
    lane's file (or at chunks.jsonl) replaces that lane's rows with its own: they
    drop out of the sparse index, which is re-derived from the JSONLs, while they
    stay in the dense one. This holds for a whole-lane run as much as a scoped
    one. `lane`'s OWN file is the caller's business (a whole-lane run may write
    it; a scoped run is refused by the caller's own guard). Names are compared
    case-insensitively because NTFS is, and against the file each lane would
    write itself (config's output file, else the built-in default).
    """
    if out is None:
        return
    name = Path(out).name.lower()
    owners = sorted(other for other, (key, default) in _LANE_CHUNK_FILES.items()
                    if other != lane
                    and Path(CFG.get(key) or default).name.lower() == name)
    if owners:
        raise ValueError(
            f"{Path(out).name} is the chunk file of the {' and '.join(owners)} lane, "
            f"and the loader truncates its output: this {lane} run would wipe that "
            f"lane's rows out of the sparse index while they stay in the dense one. "
            f"Give this run its own output file (name it *_chunks.jsonl so the "
            f"sparse rebuild picks it up).")


def _lane_output(cfg_key: str, default_name: str, prm: dict,
                 scope: tuple[str, ...]) -> list[str]:
    """The ["--output", path] pair for a lane whose loader opens its output
    with "w", or [] to leave the loader's own default in force.

    A run narrowed by any `scope` param must not write the lane's canonical
    file. The loader replaces what it writes, so a run scoped to one folder,
    file or page range truncates the whole-lane file down to that scope. The
    dense index keeps every chunk (append upserts, it never deletes) while
    build_sparse_union re-derives the sparse half from the JSONLs and loses the
    rest — a silent dense/sparse drift, from a job that reported success. The
    same guard ingest_canvas and ingest_md carry. The canonical name is the one
    the loader would pick itself (config's `output_file`, else its built-in
    default), compared case-insensitively because NTFS is. Whatever the scope, a
    run may not take ANOTHER lane's canonical file either
    (_refuse_other_lanes_file).
    """
    canonical = Path(CFG.get(cfg_key) or default_name).name
    out = _vault_data_path(str(prm["output"])) if prm.get("output") else None
    if (any(prm.get(k) for k in scope)
            and (out is None or Path(out).name.lower() == canonical.lower())):
        raise ValueError(
            f"a scoped {cfg_key.partition('.')[0]} run must not write "
            f"{canonical} — the loader truncates its output, so everything "
            f"outside this scope would drop out of the sparse index while "
            f"staying in the dense one, and out of this lane's only chunk "
            f"file. Give this run its own output file (name it *_chunks.jsonl "
            f"so the sparse rebuild picks it up).")
    _refuse_other_lanes_file(out, cfg_key.partition(".")[0])
    return ["--output", out] if out else []


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
        argv += _lane_output("pdf.output_file", "pdf_chunks.jsonl", prm,
                             ("include_path", "exclude_path", "include_files",
                              "only_books", "skip_books", "max_pages", "pages"))
        if prm.get("max_pages"):
            argv += ["--max-pages", str(int(prm["max_pages"]))]
        if prm.get("pages"):
            from src.ingestion.pdf_loader import parse_page_spec
            parse_page_spec(str(prm["pages"]))       # ValueError -> 400 (caught in jobs_create)
            argv += ["--pages", str(prm["pages"])]
        if prm.get("no_ocr"):
            argv.append("--no-ocr")
        if prm.get("ocr_engine"):
            if prm["ocr_engine"] not in ("auto", "tesseract", "paddle", "vlm", "none"):
                raise ValueError("ocr_engine must be auto|tesseract|paddle|vlm|none")
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
        argv += _lane_output("notebooks.output_file", "ipynb_chunks.jsonl", prm,
                             ("include_path", "include_files", "exts"))
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
        argv += _lane_output("code.output_file", "code_chunks.jsonl", prm,
                             ("include_path", "exclude_path", "include_files", "exts"))
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
    if kind == "ingest_canvas":
        argv = [py, "main.py", "ingest-canvas"]
        out = _vault_data_path(str(prm.get("output") or "data/canvas_chunks.jsonl"))
        # A SCOPED run must not write the canonical file. The loader opens its
        # output with "w", so a run scoped to one folder truncates the
        # whole-vault file down to that folder. The dense index keeps every
        # chunk (append upserts, it never deletes) while build_sparse_union
        # re-derives the sparse half from the JSONLs and loses the rest — a
        # silent dense/sparse drift, from a job that reported success. Same
        # guard ingest_md carries for chunks.jsonl.
        if prm.get("include_path") and Path(out).name == "canvas_chunks.jsonl":
            raise ValueError(
                "a scoped canvas run must not write canvas_chunks.jsonl — the "
                "loader truncates its output, so every canvas outside this "
                "scope would drop out of the sparse index while staying in "
                "the dense one. Give this run its own output file.")
        _refuse_other_lanes_file(out, "canvas")
        if prm.get("output") or prm.get("include_path"):
            argv += ["--output", out]
        if prm.get("include_path"):
            argv += ["--include-path", str(prm["include_path"])]
        if prm.get("max_chunk_size"):
            argv += ["--max-chunk-size", str(int(prm["max_chunk_size"]))]
        if prm.get("chunking"):
            if prm["chunking"] not in ("heading", "fixed", "document", "none"):
                raise ValueError("chunking must be heading|fixed|document|none")
            argv += ["--chunking", str(prm["chunking"])]
        if prm.get("context_depth") is not None:
            depth = int(prm["context_depth"])
            if depth not in (0, 1, 2):
                raise ValueError("context_depth must be 0, 1 or 2")
            argv += ["--context-depth", str(depth)]
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
        out = _vault_data_path(str(prm.get("output") or ""))
        if Path(out).name == "chunks.jsonl":
            raise ValueError("ingest_md must not write chunks.jsonl")
        _refuse_other_lanes_file(out, "markdown")
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
        if backend not in ("auto", "requests", "crawl4ai", "scrapling", "crawlee"):
            raise ValueError("backend must be auto|requests|crawl4ai|scrapling|crawlee")
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
        return [py, "main.py", "index", "--append",
                _vault_data_path(str(prm.get("file", "")))]
    if kind == "index_rebuild":
        # DESTRUCTIVE in effect: `main.py index` deletes the dense collection
        # and rebuilds both indexes from chunks.jsonl ALONE, so every appended
        # lane (pdf / notebook / code / canvas / inbox files) drops out — see
        # Embedder.build_indexes. Hence the "destructive" tier in api_schema
        # and the warning in the Ingest tab's hint.
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


# ---------------------------------------------------------------------------
#  Job records — the durable half of the job system
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. `_JOBS` is in-memory and dies with the process, so until now
# the only trace a finished job left was its .log file. That was enough to see
# WHAT happened and not enough to see HOW: a 2026-09-20 audit found that ~97%
# of this corpus — every large lane — had no recorded ingest command anywhere,
# which makes those chunk files primary data rather than regenerable artefacts.
#
# A record is written beside the log at QUEUE time (so a crash still leaves
# one) and rewritten when the job ends. It carries the argv, so the run can be
# repeated by hand, plus the two things argv alone does not capture:
#
#   * WHICH VAULT was active. The vault switcher moves DATA_DIR, and replaying
#     an ingest against the wrong one writes a corpus into its neighbour —
#     silently, which is the trap `_vault_data_path` exists to prevent.
#   * WHAT THE CONFIG SAID. The loaders read chunk sizes, splitter choice and
#     the taxonomy from config.yaml, so identical argv under a different config
#     produces different chunks and different doc_ids. The digest answers "has
#     anything changed since?" and the subset answers "changed from what?"
#     without needing the old file.
#
# It is a record, not a guarantee: it cannot reproduce a vault whose FILES have
# changed. What it does is make a replay's divergence visible instead of
# silent.

_JOB_RECORD_VERSION = 1

# Keys that actually shape chunks, and therefore doc_ids. Kept explicit rather
# than dumping the whole config: this file is written on every job, config.yaml
# carries machine paths, and an enumerated list says which settings the author
# believed were load-bearing at the time.
_FINGERPRINT_KEYS = (
    "parser.chunking", "parser.max_chunk_size", "parser.min_chunk_size",
    "parser.overlap_size", "parser.skip_dirs",
    "pdf.chunking", "pdf.max_chunk_size", "pdf.min_chunk_size",
    "pdf.overlap_size", "pdf.ocr_engine",
    "code.max_chunk_size", "code.min_chunk_size", "code.overlap_size",
    "notebooks.max_chunk_size", "notebooks.min_chunk_size",
    "notebooks.overlap_size",
    "canvas.chunking", "canvas.max_chunk_size", "canvas.min_chunk_size",
    "canvas.chunk_overlap", "canvas.context_depth",
    "embedding.local_model", "embedding.provider",
    "paths.collection_name",
)


def _ingestion_fingerprint() -> dict:
    """The config state a replay would need to match, read fresh from disk."""
    cfg_path = ROOT / "config.yaml"
    try:
        raw = cfg_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()[:16]
    except OSError:
        digest = None
    try:
        disk = load_config()
        values = {k: disk.get(k) for k in _FINGERPRINT_KEYS}
    except Exception:                       # never fail a job over bookkeeping
        values = {}
    return {"config_sha256_16": digest, "values": values}


def _output_of(argv: list[str]) -> str | None:
    """The --output this job writes, so a chunk file can be traced back to the
    command that produced it. That lookup is the whole point of the record."""
    for flag in ("--output", "--append"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return Path(argv[i + 1]).name
    return None


def _job_record_path(jid: str) -> Path:
    return JOBS_DIR / f"{jid}.json"


def _write_job_record(job: "Job") -> None:
    """Persist (or refresh) a job's record. Bookkeeping must never take a job
    down with it, so every failure here is logged and swallowed."""
    try:
        rec = {
            "record_version": _JOB_RECORD_VERSION,
            "id": job.id,
            "kind": job.kind,
            "status": job.status,
            "argv": list(job.argv),
            "params": job.params,
            "cwd": str(ROOT),
            "output": _output_of(job.argv),
            "log_file": Path(job.log_file).name,
            "returncode": job.returncode,
            "created": job.created,
            "started": job.started,
            "ended": job.ended,
            "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                         time.localtime(job.created)),
            "vault_path": str(CFG.get("parser.vault_path") or ""),
            "data_dir": str(DATA_DIR),
            "ingestion": _ingestion_fingerprint(),
        }
        p = _job_record_path(job.id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str),
                       encoding="utf-8")
        tmp.replace(p)                      # atomic: never a half-written record
    except Exception as e:
        log.warning("could not write job record for %s: %s", job.id, e)


def _load_job_records(limit: int = 200) -> int:
    """Rehydrate finished jobs from disk at import, newest first.

    Restored jobs exist so the console's history survives a restart and so a
    past run can be inspected and re-queued (`/api/jobs/{id}/retry` rebuilds
    the argv from `params`). They are never re-run on their own: they come back
    with their recorded terminal status, and a job left `running` by a killed
    process is marked `interrupted` rather than resurrected — nothing is
    waiting on it and the worker queue is empty at this point.
    """
    try:
        records = sorted(JOBS_DIR.glob("*.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    except OSError:
        return 0
    restored: list[Job] = []
    for p in records:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            job = Job(id=str(rec["id"]), kind=str(rec["kind"]),
                      argv=list(rec.get("argv") or []),
                      params=dict(rec.get("params") or {}))
            status = str(rec.get("status") or "done")
            job.status = "interrupted" if status in ("queued", "running") else status
            job.created = float(rec.get("created") or p.stat().st_mtime)
            job.started = rec.get("started")
            job.ended = rec.get("ended")
            job.returncode = rec.get("returncode")
            job.log_file = str(JOBS_DIR / (rec.get("log_file") or f"{job.id}.log"))
            job.restored = True
            restored.append(job)
        except Exception as e:
            log.warning("skipping unreadable job record %s: %s", p.name, e)
    restored.sort(key=lambda j: j.created)          # _ORDER is oldest-first
    with _jobs_lock:
        for job in restored:
            if job.id not in _JOBS:
                _JOBS[job.id] = job
                _ORDER.append(job.id)
    if restored:
        log.info("restored %d job record(s) from %s", len(restored), JOBS_DIR)
    return len(restored)


def enqueue(kind: str, params: dict) -> Job:
    argv = _build_argv(kind, params or {})
    job = Job(id=uuid.uuid4().hex[:10], kind=kind, argv=argv, params=params or {})
    job.log_file = str(JOBS_DIR / f"{job.id}.log")
    with _jobs_lock:
        _JOBS[job.id] = job
        _ORDER.append(job.id)
    # Written BEFORE the job is queued: a process that dies mid-run must still
    # leave behind what it was about to do.
    _write_job_record(job)
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
            # Rewrite the record with the terminal status, so a later session
            # can tell a completed ingest from one that never finished.
            _write_job_record(job)
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


# Restore history BEFORE the worker starts: the queue is empty at this point,
# so there is no chance of a restored entry being picked up and re-run.
_load_job_records()

threading.Thread(target=_worker, daemon=True, name="job-worker").start()

# ============================================================================
# ChromaDB helpers (always paged)
# ============================================================================

def _collection():
    client = persistent_client(CFG.path("paths.chroma_dir"))
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
    ocr_engine: Optional[str] = None  # optional override (auto|tesseract|paddle|vlm|none)
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
    backend: str = "auto"           # auto | requests | crawl4ai | scrapling | crawlee
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
        # Without a Cache-Control header the browser falls back to HEURISTIC
        # freshness: with only Last-Modified to go on it treats a file edited
        # weeks ago as fresh for days and serves the stale copy WITHOUT ASKING.
        # That is the "my handler isn't firing" trap that has cost this project
        # time in two separate sessions.
        #
        # `no-cache` means revalidate before reuse. FileResponse sends an ETag
        # but does NOT implement conditional responses (that lives in
        # StaticFiles), so in practice every load re-sends the file — measured
        # 200, not 304. Over loopback, for a single-user admin console, that is
        # a few hundred KB nobody notices, and it is the right trade against a
        # console that silently runs last week's JavaScript.
        return FileResponse(ui, headers={"Cache-Control": "no-cache"})
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


@app.get("/api/jobs/provenance")
def jobs_provenance() -> dict:
    """Which command produced each chunk file — and which ones nothing records.

    This answers the question that motivated job records at all. A chunk file
    with no record is not regenerable: the flags it was ingested with (scope,
    chunking strategy, forced domain/tags) are gone, so the JSONL itself is the
    only surviving copy of that decision and must be treated as primary data,
    not as a build artefact.

    Records only exist for jobs run AFTER this was added, and only for jobs run
    through the console — a `main.py` invocation from a terminal leaves none.
    `unrecorded` is therefore expected to be large at first and to shrink; it
    is a backlog, not a fault.
    """
    by_output: dict[str, list[dict]] = {}
    for p in sorted(JOBS_DIR.glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out = rec.get("output")
        if out:
            by_output.setdefault(out, []).append(rec)

    recorded, unrecorded = [], []
    for f in chunk_files():
        runs = sorted(by_output.get(f.name, []),
                      key=lambda r: r.get("created") or 0, reverse=True)
        rows = None
        try:
            with open(f, "rb") as fh:
                rows = sum(1 for line in fh if line.strip())
        except OSError:
            pass
        entry = {"file": f.name, "rows": rows,
                 "bytes": f.stat().st_size if f.exists() else None}
        if runs:
            latest = runs[0]
            entry.update({
                "job_id": latest.get("id"),
                "kind": latest.get("kind"),
                "status": latest.get("status"),
                "when": latest.get("created_iso"),
                "replay": " ".join(latest.get("argv") or []),
                "cwd": latest.get("cwd"),
                "vault_path": latest.get("vault_path"),
                "config_sha256_16": (latest.get("ingestion") or {}).get("config_sha256_16"),
                "runs": len(runs),
            })
            recorded.append(entry)
        else:
            unrecorded.append(entry)

    current = _ingestion_fingerprint().get("config_sha256_16")
    for e in recorded:
        # A replay under a different config can produce different chunks and
        # therefore different doc_ids, so say so rather than implying the
        # command alone is sufficient.
        e["config_changed_since"] = bool(
            e.get("config_sha256_16") and e["config_sha256_16"] != current)

    return {
        "config_sha256_16": current,
        "recorded": recorded,
        "unrecorded": unrecorded,
        "summary": {
            "files": len(recorded) + len(unrecorded),
            "with_a_recorded_command": len(recorded),
            "rows_recorded": sum(e["rows"] or 0 for e in recorded),
            "rows_unrecorded": sum(e["rows"] or 0 for e in unrecorded),
        },
        "note": "A file under `unrecorded` cannot be regenerated from records. "
                "Treat it as primary data: never truncate or rebuild it "
                "casually, and prove any change that would alter doc_ids "
                "set-identical instead of planning to re-ingest.",
    }


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
    inbox = vault / (_vault_rel(CFG.get("webui.inbox_dir")) or "Inbox")
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
    if body.ocr_engine and body.ocr_engine not in ("auto", "tesseract", "paddle", "vlm", "none"):
        return JSONResponse({"ok": False,
                             "error": "ocr_engine must be auto|tesseract|paddle|vlm|none"},
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
        if g.ocr_engine and g.ocr_engine not in ("auto", "tesseract", "paddle", "vlm", "none"):
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


def _vault_rel(p: str) -> str:
    """Normalise a vault-RELATIVE path so it can be safely joined to the root.

    `Path(vault) / "/"` does not mean "the vault root" — a leading separator
    makes the right-hand side ABSOLUTE and pathlib throws the base away
    entirely, yielding `G:\\`. The containment check then rejects it and the
    whole Vault tab renders one red line, which is what "the tree shows
    nothing" turned out to be. Backslashes are folded too: these values are
    written by hand and by Windows users.
    """
    s = str(p or "").replace("\\", "/").strip()
    while s.startswith("/"):
        s = s[1:]
    return s.rstrip("/")


def _vault_contains(vault: Path, candidate: Path) -> bool:
    """True only when candidate is the vault itself or a real descendant.

    String-prefix checks are not path containment: ``C:\\Vault2`` starts with
    ``C:\\Vault`` even though it is a sibling.  ``relative_to`` compares path
    components and therefore closes that escape without platform-specific
    separator or case logic here.
    """
    try:
        candidate.relative_to(vault)
        return True
    except ValueError:
        return False


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
    vault-relative posix; '' = the configured browse root (webui.vault_tree_root),
    which defaults to the VAULT ROOT — any other default names a folder that
    exists in one particular vault and nowhere else. The rest of the vault is
    reachable via /api/vault/search below. Read-only: never writes to the vault.
    """
    vault = _vault_root()
    if not vault.is_dir():
        return JSONResponse(
            {"error": f"the configured vault folder does not exist: {vault}. "
                      f"Pick one in Settings -> Vaults."}, status_code=404)
    vault_resolved = vault.resolve()
    root_rel = _vault_rel(CFG.get("webui.vault_tree_root"))
    here = _vault_rel(path) or root_rel
    base = (vault / here).resolve() if here else vault_resolved
    if not _vault_contains(vault_resolved, base):
        return JSONResponse({"error": "path escapes the vault"}, status_code=400)
    if not base.is_dir():
        return JSONResponse(
            {"error": f"not a folder in this vault: "
                      f"{here or '(vault root)'}"}, status_code=404)

    rag = _rag_lookup()
    dirs, files = [], []
    for entry in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if entry.name in _TREE_SKIP or entry.name.startswith("."):
            continue
        entry_rel = entry.relative_to(vault).as_posix()
        if entry.is_dir():
            dirs.append({"name": entry.name, "path": entry_rel})
        elif entry.suffix.lower() in _TREE_EXTS:
            files.append(_file_row(entry, entry_rel, rag))
    return {"root": root_rel, "path": here, "vault": str(vault),
            "dirs": dirs, "files": files}


# ---- graph / canvas lane: scope discovery + dry-run preview ----
#
# Canvas ingestion is NOT "one more file type", and the console shouldn't file
# it as one. Canvases live in several unrelated trees, their filenames almost
# never carry a course keyword, and the thing worth checking before committing
# is the GRAPH (how many chunks actually carry edges), which no other lane has.
# These two endpoints serve the Ingest tab's Graph section.

# Vault-wide canvas scan, cached. Walking the vault is the slow part and the
# Ingest tab asks for it on every visit; the console's rescan button sends
# refresh=1 when the operator knows files changed.
_CANVAS_SCAN: dict[str, dict] = {}
_CANVAS_SCAN_TTL = 300.0


class CanvasPreviewIn(BaseModel):
    include_path: str | None = None
    max_chunk_size: int | None = Field(default=None, ge=200, le=20000)
    chunking: str | None = None
    context_depth: int | None = Field(default=None, ge=0, le=2)
    min_chunk_size: int | None = Field(default=None, ge=1, le=5000)


@app.get("/api/canvas/folders")
def canvas_folders(refresh: int = 0) -> dict:
    """Where the canvases actually are, with counts.

    A free-text include_path box asks the operator to guess a substring that
    matches trees they cannot see. This lists the real ones instead.
    """
    vault = _vault_root()
    if not vault.is_dir():
        return JSONResponse(
            {"error": f"the configured vault folder does not exist: {vault}"},
            status_code=404)
    # Cached: this walks the whole vault, and the Ingest tab asks on every
    # visit. `refresh=1` (the console's ⟳ rescan) bypasses it.
    cached = _CANVAS_SCAN.get(str(vault))
    if cached and not refresh and (time.time() - cached["at"]) < _CANVAS_SCAN_TTL:
        return {**cached["result"], "cached": True,
                "age_s": int(time.time() - cached["at"])}

    from src.ingestion.canvas_loader import iter_canvas_files
    t0 = time.time()
    roots: dict[str, int] = {}
    total = 0
    for f in iter_canvas_files(vault):
        rel = f.relative_to(vault).parts
        key = rel[0] if len(rel) > 1 else "(vault root)"
        roots[key] = roots.get(key, 0) + 1
        total += 1
    result = {
        "vault": str(vault),
        "total": total,
        "ms": int((time.time() - t0) * 1000),
        "folders": [{"path": k, "canvases": v}
                    for k, v in sorted(roots.items(), key=lambda kv: -kv[1])],
    }
    _CANVAS_SCAN[str(vault)] = {"at": time.time(), "result": result}
    return {**result, "cached": False}


@app.post("/api/canvas/preview")
def canvas_preview(body: CanvasPreviewIn) -> dict:
    """Run the canvas loader for real, WITHOUT touching the index.

    Writes to a scratch directory and deletes it. It must never write into
    `data/`: build_sparse_union globs `data/*_chunks.jsonl`, so a preview file
    parked there would silently join the sparse index while the dense half
    knows nothing about it.
    """
    import tempfile
    from src.ingestion.canvas_loader import CanvasLoader, decode_canvas_edges

    tmp = Path(tempfile.mkdtemp(prefix="canvas_preview_"))
    try:
        loader = CanvasLoader.from_config(CFG)
        loader.output_file = tmp / "preview.jsonl"
        if body.include_path:
            loader.include_path = body.include_path.strip().lower()
        if body.min_chunk_size is not None:
            loader.min_chunk = int(body.min_chunk_size)
        if body.max_chunk_size is not None:
            loader.max_chunk = int(body.max_chunk_size)
        if body.chunking:
            from src.ingestion.obsidian_parser import CHUNKING_STRATEGIES
            if body.chunking not in CHUNKING_STRATEGIES:
                return JSONResponse(
                    {"error": f"chunking must be one of {CHUNKING_STRATEGIES}"},
                    status_code=400)
            loader.chunking = body.chunking
        if body.context_depth is not None:
            loader.context_depth = int(body.context_depth)
        try:
            loader.ingest_vault(verbose=False)
        except ValueError as e:
            # A malformed .canvas names itself; that is worth seeing BEFORE a
            # real run rather than as a failed job.
            return JSONResponse({"error": str(e)}, status_code=400)

        with_edges = 0
        richest = None
        rows = 0
        with open(loader.output_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                rec = json.loads(line)
                edges = decode_canvas_edges(rec["metadata"].get("canvas_edges"))
                if edges:
                    with_edges += 1
                if richest is None or len(edges) > richest[0]:
                    richest = (len(edges), rec)
        st = loader.stats
        chars = st["chars_total"]
        sample = richest[1] if richest else None
        return {
            "chunks": rows,
            "files_found": st["files_found"],
            "files_skipped": st["files_skipped"],
            "nodes_skipped": st["nodes_skipped"],
            "nodes_split": st["nodes_split"],
            "edges_total": st["edges_total"],
            "with_edges": with_edges,
            "with_edges_pct": round(100.0 * with_edges / rows, 1) if rows else 0.0,
            "chars_total": chars,
            "chars_context": st["chars_context"],
            "context_pct": round(100.0 * st["chars_context"] / chars, 1) if chars else 0.0,
            "settings": {"include_path": loader.include_path,
                         "min_chunk_size": loader.min_chunk,
                         "max_chunk_size": loader.max_chunk,
                         "chunking": loader.chunking,
                         "context_depth": loader.context_depth},
            "sample": {
                "doc_id": sample["doc_id"],
                "source_file": sample["metadata"].get("source_file"),
                "edges": decode_canvas_edges(sample["metadata"].get("canvas_edges")),
                "domain": sample["metadata"].get("domain"),
                "course_name": sample["metadata"].get("course_name"),
                "text": sample["text"][:2500],
            } if sample else None,
            "note": "nothing was indexed; this ran the real loader into a "
                    "scratch file and deleted it",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
            # Components BELOW the vault root only. The absolute path also
            # carries the directories above it, and this check additionally
            # rejects any component starting with "." — so a vault stored
            # under a dotted folder would make search return nothing.
            rel = f.relative_to(vault)
            if any(part in search_skip or part.startswith(".")
                   for part in rel.parts):
                continue
            if ql not in f.name.lower():
                continue
            rows.append(_file_row(f, rel.as_posix(), rag))
            if len(rows) >= limit:
                return {"rows": rows, "truncated": True}
    return {"rows": rows, "truncated": False}


# Editable config surface for the Settings tab. Maps dotted key -> spec.
# Everything here is persisted into config.yaml IN PLACE (comments preserved);
# none of it hot-applies — the response says which services need a restart.
EDITABLE_SETTINGS: dict[str, dict] = {
    "parser.vault_path":            {"kind": "dir",  "restart": ":8051 + :8052",
                                     "label": "Vault / documents root"},
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
    # Values are filled in at request time from what torch ACTUALLY reports —
    # see _device_choices(). Offering "cuda" on a CPU-only build is a trap: the
    # only symptom is a query that silently runs many times slower.
    "retrieval.cross_encoder_device": {"kind": "enum", "restart": ":8051",
                                     "values": [],
                                     "label": "Cross-encoder device"},
    "retrieval.rerank_mode":        {"kind": "enum", "restart": ":8051",
                                     "values": ["cross_encoder", "lexical",
                                                "http", "none", "laya"],
                                     "label": "Default rerank method"},
    # Free text: it IS the feature. Blank turns it off.
    "retrieval.rerank_instruction": {"kind": "str",  "restart": ":8051",
                                     "allow_empty": True,
                                     "label": "Rerank instruction (ranking criterion)"},
    "retrieval.rerank_instruction_format": {"kind": "enum", "restart": ":8051",
                                     "values": ["prefix", "instruct"],
                                     "label": "Rerank instruction format"},
    "pdf.ocr_engine":               {"kind": "enum", "restart": "none (read per job)",
                                     "values": ["auto", "tesseract", "paddle", "vlm", "none"],
                                     "label": "OCR engine for scanned pages"},
    # Values are filled in at request time from pdf.vlm_ocr_presets.
    "pdf.vlm_ocr.preset":           {"kind": "enum", "restart": "none (read per job)",
                                     "values": [],
                                     "label": "VLM-OCR preset (vision model)"},
    "pdf.vlm_ocr.base_url":         {"kind": "str",  "restart": "none (read per job)",
                                     "label": "VLM-OCR endpoint (OpenAI-compatible)"},
    "pdf.paddle_ocr.base_url":      {"kind": "str",  "restart": "none (read per job)",
                                     "label": "PaddleOCR sidecar endpoint"},
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
    # kind "relpath": must stay INSIDE the vault. A leading separator makes the
    # value absolute, and `Path(vault) / "/x"` discards the vault entirely —
    # which silently emptied the whole Vault tab.
    "webui.inbox_dir":              {"kind": "relpath", "restart": ":8052",
                                     "label": "Inbox folder (vault-relative)"},
    "webui.vault_tree_root":        {"kind": "relpath", "restart": ":8052",
                                     "allow_empty": True,
                                     "label": "Vault-tab browse root (blank = vault root)"},
    "webui.auto_restart_rag":       {"kind": "enum", "restart": "none (read per job)",
                                     "values": ["true", "false"],
                                     "label": "Auto-restart :8051 after index-changing jobs"},
    # The feature flag behind the Experimental features panel. It is an
    # ordinary editable setting so it uses the SAME writer, validator and
    # restart reporting as everything else — an experimental feature with its
    # own bespoke save path would be the one setting nobody had tested.
    "graph.enabled":                {"kind": "enum", "restart": ":8051 + reload this page",
                                     "values": ["true", "false"],
                                     "label": "Graph RAG mode (canvas traversal)"},
    # Likewise the flag behind the Laya reranker's row below. It has no console
    # controls to hide, so no page reload is needed — only the :8051 restart.
    "retrieval.laya.enabled":       {"kind": "enum", "restart": ":8051",
                                     "values": ["true", "false"],
                                     "label": "Laya reranker (experimental)"},
}


# Features that are built and tested but NOT part of the default workflow, and
# the config flag that turns each one on. The console renders this list; it does
# not hardcode any feature, so the next experimental feature is a row here plus
# its own flag in EDITABLE_SETTINGS.
#
# `key` must name a real EDITABLE_SETTINGS entry, so the flag is written by the
# ordinary settings path (see the note on graph.enabled above) — asserted by
# tests/test_experimental_features.py rather than left to a reviewer to notice.
EXPERIMENTAL_FEATURES: list[dict] = [
    {
        "id": "graph_rag",
        "key": "graph.enabled",
        "label": "Graph RAG · canvas traversal",
        "what": "A second query mode that walks the edges you drew between "
                "Obsidian canvas nodes, then stops and asks a human what the "
                "retrieved nodes are for: merge them into the previous "
                "result, answer from the graph alone, or take the raw nodes "
                "with no LLM at all.",
        "why_off": "It works and is covered by tests, but it has never been "
                   "scored against the golden set — so its retrieval "
                   "benefit is unmeasured. Until it is, it stays out of the "
                   "default Query tab instead of implying an answer quality "
                   "nobody has checked.",
        "unaffected": "Ordinary Ask / Search never touch it, and the canvas "
                      "lane keeps competing in normal retrieval either way. "
                      "Canvas INGESTION is not gated: that is how the lane "
                      "gets indexed in the first place.",
        "surfaces": ["Query tab: the Graph RAG panel",
                     "API: POST /graph/expand"],
    },
    {
        "id": "laya_rerank",
        "key": "retrieval.laya.enabled",
        "label": "Laya reranker · fine-tuned relevance scorer",
        "what": "A fifth rerank method, laya: a ~420M-parameter Laya model "
                "fine-tuned on your own notes (trained off-machine, see "
                "docs/laya-finetune.md) gives every candidate passage a "
                "probability that it answers the question, and that "
                "probability orders the pool. Pick it per call (rerank: laya) "
                "or as the default rerank method. It needs the laya package "
                "and a checkpoint in models/laya-noetrix.",
        "why_off": "It has never been scored against the eval sets, so whether "
                   "it ranks better is unmeasured, and it must pass the spec "
                   "§11 graduation rule first: beat the current default on "
                   "dev with a paired CI that excludes zero, then hold on "
                   "test. Upstream presents the base model as something to "
                   "specialise, not a zero-shot ranker, so any benefit rests "
                   "on the fine-tune alone.",
        "unaffected": "Ordinary Ask / Search keep the configured reranker "
                      "either way. While this is off, rerank: laya is "
                      "refused with a readable error, never answered by "
                      "another reranker, and nothing is imported or loaded "
                      "at startup.",
        "surfaces": ["API: rerank=laya on /search, /query, /compare",
                     "Config: retrieval.rerank_mode: laya",
                     "Bench: the laya-rerank config"],
    },
]


def _provider_choices(disk_cfg) -> tuple[list[str], list[dict]]:
    """Selectable generation backends + their status, for the Settings tab.

    Returns (names, details). Names = the `providers:` registry plus the
    reserved legacy names, so the dropdown can never strand an existing config
    on a value it cannot re-select.

    details carries `key_present` and `key_compatible` — whether the env var
    each provider names is set and, when the provider declares a required
    prefix, whether it is the right credential type.  The key VALUE is never
    returned.
    """
    from src.llm.llm_client import LLMClient

    registry = disk_cfg.get("providers", {}) or {}
    names = [n for n in registry if n not in LLMClient.RESERVED_PROVIDERS]
    details = []
    for name in names:
        spec = registry[name] or {}
        env = spec.get("api_key_env")
        value = os.environ.get(env, "") if env else ""
        prefix = str(spec.get("api_key_prefix") or "")
        present = bool(value) if env else True
        compatible = (not prefix or value.startswith(prefix)) if present else None
        optional = bool(spec.get("api_key_optional"))
        available = bool(
            (not env)
            or (not present and optional)
            or (present and compatible is not False)
        )
        details.append({
            "name": name,
            "label": spec.get("label") or name,
            "description": spec.get("description"),
            "kind": spec.get("kind", "openai"),
            "base_url": spec.get("base_url"),
            "model": spec.get("model"),
            "api_key_env": env,
            "key_optional": optional,
            "key_present": present,
            "key_compatible": compatible,
            "available": available,
        })
    return names + list(LLMClient.RESERVED_PROVIDERS), details


def _hub_credentials() -> list[dict]:
    """Non-provider credentials the console can manage, and their real state.

    `source` matters: huggingface_hub reads a token from an environment
    variable OR from a stored login file, and a stale one in EITHER place
    breaks every model download with a misleading "not found". The console has
    to be able to show which one is in play, not just whether an env var is set.
    """
    rows = []
    for env, why in _EXTRA_KEY_ENVS.items():
        value = os.environ.get(env, "")
        rows.append({"env": env, "why": why, "present": bool(value)})
    stored = None
    try:
        from huggingface_hub import constants, get_token
        if get_token():
            stored = str(constants.HF_TOKEN_PATH)
    except Exception:                            # hub not installed / moved
        pass
    return [{"credentials": rows, "stored_token_path": stored}][0]


def _with_current(choices: list[str], current: Any) -> list[str]:
    """Enum members + whatever the config already holds.

    A dropdown built purely from what this machine can do would STRAND a config
    written on a different machine: the value shows as selected, is not in the
    list, and the first save silently rewrites it to whatever was first. The
    provider dropdown solves this by always appending the reserved names; this
    is the same guarantee for device-shaped enums."""
    cur = str(current or "").strip()
    return choices if not cur or cur in choices else choices + [cur]


def _taxonomy(cfg) -> dict:
    """What this vault calls its folder-derived label, for the console's
    wording. The METADATA keys stay course_code / course_name whatever this
    says — renaming them would invalidate every stored chunk and force a full
    re-embed, so only the human-facing label is configurable."""
    label = str(cfg.get("taxonomy.label") or "course")
    return {
        "label": label,
        "label_plural": str(cfg.get("taxonomy.label_plural") or label + "s"),
        "detect_from_path": bool(cfg.get("taxonomy.detect_from_path", True)),
    }


def _torch_devices() -> tuple[dict, list[str]]:
    """What torch can ACTUALLY reach here, and the device values worth offering.

    Returns (info, choices). The choices list is the whole point: the dropdown
    used to offer cuda/cuda:0/cuda:1 unconditionally, which on a CPU-only build
    (every Docker image here, and every Mac) silently falls back to CPU. Apple
    Silicon reports `mps` when torch runs natively on the host — never inside a
    Linux container, which cannot reach Metal.
    """
    choices = ["auto", "cpu"]
    try:
        import torch
        info = {"torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "available": bool(torch.cuda.is_available()),
                "devices": [torch.cuda.get_device_name(i)
                            for i in range(torch.cuda.device_count())]
                if torch.cuda.is_available() else [],
                "mps": False}
        try:
            info["mps"] = bool(torch.backends.mps.is_available())
        except Exception:                        # older torch, no mps backend
            pass
        if info["available"]:
            choices.append("cuda")
            choices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        if info["mps"]:
            choices.append("mps")
    except Exception as e:                       # torch missing/broken
        info = {"torch": None, "cuda_build": None, "available": False,
                "devices": [], "mps": False, "error": str(e)}
    return info, choices


_YAML_KEY_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z0-9_][A-Za-z0-9_.\-]*)"
    r"(?P<pre_tail>[ \t]*:[ \t]*)(?P<val>[^#\r\n]*?)"
    r"(?P<post>[ \t]*(?:#[^\r\n]*)?)$")


def _yaml_key_index(lines: list[str]) -> list[tuple[int, str, "re.Match[str]"]]:
    """Index every `key: value` line as (line_no, FULL dotted path, match).

    Nesting comes from indentation, so `generation.model` and
    `generation.local.model` are two different paths instead of two hits on the
    same leaf name. That distinction is the whole point: matching on the last
    segment made every provider switch fail with "found 2 matches", because
    `generation:` carries both its own `model` and the legacy `local:` block's.

    Blank lines, comments and sequence items are skipped. A sequence item never
    holds a scalar this writer is allowed to touch, and letting `- name: x`
    push the indent stack would mis-parent every key after it.
    """
    index: list[tuple[int, str, re.Match[str]]] = []
    stack: list[tuple[int, str]] = []                 # (indent width, key)
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        m = _YAML_KEY_RE.match(ln)
        if not m:
            continue
        indent = len(m.group("indent").expandtabs(8))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = ".".join([k for _, k in stack] + [m.group("key")])
        index.append((i, path, m))
        stack.append((indent, m.group("key")))
    return index


def _persist_section_keys(cfg_path: Path, changes: dict[str, Any]) -> list[str]:
    """Rewrite dotted keys in config.yaml IN PLACE, preserving comments and
    layout.

    Keys are matched on their FULL path (see _yaml_key_index), so a leaf name
    that repeats at different depths or in different sections — `model`,
    `base_url`, `vault_path` — is never ambiguous. The safety guarantee is
    unchanged: a path that does not resolve to exactly one line raises instead
    of guessing which one was meant.
    """
    text = cfg_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    index = _yaml_key_index(lines)
    written: list[str] = []
    for dotted, value in changes.items():
        sval = ("true" if value else "false") if isinstance(value, bool) else str(value)
        if any(c in sval for c in "\r\n"):
            raise ValueError(f"{dotted}: newlines not allowed")
        # quote strings that YAML would mangle (paths with ':' etc.)
        if not re.fullmatch(r"[\w.\-/]+", sval):
            sval = '"' + sval.replace('"', '\\"') + '"'
        hits = [(i, m) for i, path, m in index if path == dotted]
        if not hits:
            raise ValueError(f"{dotted}: no such key in {cfg_path.name} — add it "
                             f"to the file before setting it from the console")
        if len(hits) > 1:
            where = ", ".join(f"line {i + 1}" for i, _ in hits)
            raise ValueError(f"{dotted}: found {len(hits)} matches in "
                             f"{cfg_path.name} ({where}); refusing to rewrite "
                             f"ambiguously")
        i, m = hits[0]
        lines[i] = m.group("indent") + m.group("key") + m.group("pre_tail") \
            + sval + m.group("post")
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
    # Reranker suggestions + whether torch can actually reach an accelerator.
    from src.retrieval.reranker import KNOWN_RERANKERS, RERANK_PROFILES
    gpu, device_choices = _torch_devices()
    vlm_presets = sorted((disk_cfg.get("pdf.vlm_ocr_presets") or {}).keys())
    editable = {}
    for key, spec in EDITABLE_SETTINGS.items():
        editable[key] = {**spec, "value": disk_cfg.get(key)}
    editable["generation.provider"]["values"] = provider_names
    editable["retrieval.cross_encoder_device"]["values"] = _with_current(
        device_choices, disk_cfg.get("retrieval.cross_encoder_device"))
    # "null" is a real, selectable value: it turns the preset OFF and falls back
    # to the explicit pdf.vlm_ocr.* keys, which is how this config behaved
    # before presets existed.
    editable["pdf.vlm_ocr.preset"]["values"] = ["null"] + vlm_presets
    # YAML null round-trips as the literal string "null" through the dropdown,
    # so an untouched "no preset" config doesn't read back as a pending change.
    editable["pdf.vlm_ocr.preset"]["value"] = (
        disk_cfg.get("pdf.vlm_ocr.preset") or "null")
    return {
        "rag_api": RAG_API,
        "providers": provider_details,
        "hub": _hub_credentials(),
        "judge_provider": disk_cfg.get("eval.judge.provider"),
        "rerankers": [{"id": k, **v} for k, v in KNOWN_RERANKERS.items()],
        "rerank_profiles": [{"id": k, **v} for k, v in RERANK_PROFILES.items()],
        "gpu": gpu,
        "vault_path": str(CFG.get("pdf.vault_path") or CFG.get("parser.vault_path")),
        "inbox_dir": CFG.get("webui.inbox_dir") or "Inbox",
        "jsonl_files": [p.name for p in chunk_files()],
        "ocr_engines": ["auto", "tesseract", "paddle", "vlm", "none"],
        "taxonomy": _taxonomy(disk_cfg),
        "config_path": str(ROOT / "config.yaml"),
        "editable": editable,
        # Read off DISK (not boot-time CFG) for the same reason `editable` is:
        # after a save with no restart yet, the panel must show what was saved.
        "experimental": [
            {**feat, "on": bool(disk_cfg.get(feat["key"]))}
            for feat in EXPERIMENTAL_FEATURES
        ],
    }


@app.post("/api/settings")
def settings_update(body: SettingsIn) -> dict:
    """Persist whitelisted config values into config.yaml (comment-preserving,
    section-aware). Nothing hot-applies: the response lists which services to
    restart. Vault path must exist; enums are validated; unknown keys 400."""
    from src.utils.config_loader import load_config as _load
    disk_cfg = _load()
    provider_names, _ = _provider_choices(disk_cfg)
    _, device_choices = _torch_devices()

    changes: dict[str, Any] = {}
    restarts: set[str] = set()
    notes: list[str] = []
    for key, value in body.changes.items():
        spec = EDITABLE_SETTINGS.get(key)
        if not spec:
            return JSONResponse({"ok": False, "error": f"unknown setting {key!r}"},
                                status_code=400)
        # Enums whose members depend on this machine / this config are filled in
        # at request time, exactly as they are in GET /api/settings.
        if key == "generation.provider":
            spec = {**spec, "values": provider_names}
        elif key == "retrieval.cross_encoder_device":
            spec = {**spec, "values": _with_current(
                device_choices, disk_cfg.get("retrieval.cross_encoder_device"))}
        elif key == "pdf.vlm_ocr.preset":
            spec = {**spec, "values": ["null"] + sorted(
                (disk_cfg.get("pdf.vlm_ocr_presets") or {}).keys())}
        sval = str(value).strip()
        if not sval and not spec.get("allow_empty"):
            return JSONResponse({"ok": False, "error": f"{key}: empty value"},
                                status_code=400)
        if spec["kind"] == "enum" and sval not in spec["values"]:
            return JSONResponse(
                {"ok": False,
                 "error": f"{key} must be one of {spec['values']}"}, status_code=400)
        if spec["kind"] == "relpath":
            norm = _vault_rel(sval)
            if norm != sval.replace("\\", "/").strip():
                return JSONResponse(
                    {"ok": False,
                     "error": f"{key} is relative to the vault root — drop the "
                              f"leading separator (use {norm!r}, or blank for "
                              f"the vault root)"}, status_code=400)
            if ".." in norm.split("/"):
                return JSONResponse(
                    {"ok": False, "error": f"{key}: '..' is not allowed"},
                    status_code=400)
            sval = norm
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
    if new_provider:
        spec = (disk_cfg.get("providers", {}) or {}).get(new_provider) or {}
        if "generation.model" not in changes:
            default_model = spec.get("model")
            if default_model and default_model != disk_cfg.get("generation.model"):
                changes["generation.model"] = str(default_model)
                restarts.add(EDITABLE_SETTINGS["generation.model"]["restart"])
                notes.append(
                    f"generation.model set to {default_model!r} to match "
                    f"provider {new_provider!r}")
        env = spec.get("api_key_env")
        if env:
            key_value = os.environ.get(env, "")
            expected_prefix = str(spec.get("api_key_prefix") or "")
            if not key_value and not spec.get("api_key_optional"):
                notes.append(f"WARNING: {env} is not set in this environment — "
                             f"{new_provider} will fail until you export it "
                             f"(or add it to .env)")
            elif expected_prefix and not key_value.startswith(expected_prefix):
                notes.append(
                    f"WARNING: {env} has the wrong credential type for "
                    f"{new_provider}; it must begin with {expected_prefix!r}")

    # A known cross-encoder's hard context window is not a performance hint:
    # sentence-transformers can accept a larger configured max_length and then
    # fail inside the model with an opaque position-embedding IndexError. Catch
    # that invalid pair before it is persisted and before :8051 is restarted.
    reranker_keys = {
        "retrieval.cross_encoder_model",
        "retrieval.cross_encoder_max_length",
    }
    if reranker_keys.intersection(changes):
        from src.retrieval.reranker import KNOWN_RERANKERS

        model = str(changes.get(
            "retrieval.cross_encoder_model",
            disk_cfg.get("retrieval.cross_encoder_model"),
        ))
        raw_length = changes.get(
            "retrieval.cross_encoder_max_length",
            disk_cfg.get("retrieval.cross_encoder_max_length"),
        )
        try:
            max_length = int(raw_length)
        except (TypeError, ValueError):
            return JSONResponse(
                {"ok": False,
                 "error": "retrieval.cross_encoder_max_length must be an integer"},
                status_code=400,
            )
        if max_length < 1:
            return JSONResponse(
                {"ok": False,
                 "error": "retrieval.cross_encoder_max_length must be positive"},
                status_code=400,
            )
        known = KNOWN_RERANKERS.get(model)
        if known and max_length > int(known["context_length"]):
            return JSONResponse(
                {"ok": False,
                 "error": "retrieval.cross_encoder_max_length="
                          f"{max_length} exceeds {model!r}'s "
                          f"{known['context_length']}-token context limit"},
                status_code=400,
            )

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


# ---- provider API keys (.env writer) ----

class ProviderKeyIn(BaseModel):
    env: str                       # the env-var NAME, e.g. MINIMAX_API_KEY
    value: str = ""                # "" clears it


def _env_file() -> Path:
    return ROOT / ".env"


# Credentials that are not generation providers but that this system genuinely
# needs to be able to set. Kept as an explicit allowlist for the same reason
# the provider names are: the .env writer must never become a way to set
# arbitrary environment variables for every job the worker spawns.
#
# HF_TOKEN earns its place because a WRONG one is actively harmful and almost
# undiagnosable from outside: the hub answers 401 for an unauthorized request,
# huggingface_hub reports that as "repository not found", and every public
# embedding or reranker model then looks like it does not exist. Being able to
# clear it from the console is as important as being able to set it.
_EXTRA_KEY_ENVS = {
    "HF_TOKEN": "Hugging Face token — only needed for GATED or PRIVATE models. "
                "Public embedding and reranker models need none, and an "
                "invalid one makes them all fail as 'not found'.",
    "HUGGING_FACE_HUB_TOKEN": "Legacy name for HF_TOKEN; set only if a tool "
                              "you use still reads this one.",
}


def _known_key_envs(disk_cfg) -> set[str]:
    """Env-var names this config actually reads a key from.

    The writer accepts ONLY these. A console endpoint that writes arbitrary
    NAME=VALUE pairs into .env is a way to set PATH or PYTHONPATH for every
    job the worker spawns; restricting it to names the provider registry
    already declares keeps it to what it is for.
    """
    names = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}   # the reserved lanes
    names |= set(_EXTRA_KEY_ENVS)
    for spec in (disk_cfg.get("providers", {}) or {}).values():
        env = (spec or {}).get("api_key_env")
        if env:
            names.add(str(env))
    for dotted in ("pdf.vlm_ocr.api_key_env",):
        env = disk_cfg.get(dotted)
        if env:
            names.add(str(env))
    return names


def _write_env_var(path: Path, name: str, value: str) -> str:
    """Set or remove NAME=value in .env, preserving every other line.

    Returns "set" | "cleared". The file is rewritten whole (it is a handful of
    lines), and an existing assignment is replaced IN PLACE rather than
    appended, so the file cannot accumulate duplicate names where the last one
    silently wins.
    """
    lines = (path.read_text(encoding="utf-8").split("\n")
             if path.exists() else [])
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=")
    kept, replaced = [], False
    for line in lines:
        if pattern.match(line):
            if value and not replaced:
                kept.append(f"{name}={value}")
                replaced = True
            continue                      # drop old/duplicate assignments
        kept.append(line)
    if value and not replaced:
        if kept and kept[-1].strip():
            kept.append("")
        kept.append(f"{name}={value}")
        kept.append("")
    path.write_text("\n".join(kept), encoding="utf-8")
    return "set" if value else "cleared"


@app.post("/api/providers/key")
def provider_key(body: ProviderKeyIn) -> dict:
    """Store a provider API key in .env (gitignored) and apply it in-process.

    The value is never read back by any endpoint — /api/settings reports only
    whether the variable is SET. Jobs inherit os.environ, so setting it here
    also fixes the currently-running console without a restart; :8051 is a
    separate process and still needs one.
    """
    from src.utils.config_loader import load_config as _load
    disk_cfg = _load()
    name = body.env.strip()
    if name not in _known_key_envs(disk_cfg):
        return JSONResponse(
            {"ok": False, "error": f"{name!r} is not an api_key_env named by "
                                   f"this config's providers"}, status_code=400)
    value = body.value.strip()
    if any(c in value for c in "\r\n"):
        return JSONResponse({"ok": False, "error": "key contains a newline"},
                            status_code=400)
    expected_prefix = None
    for spec in (disk_cfg.get("providers", {}) or {}).values():
        spec = spec or {}
        if spec.get("api_key_env") == name and spec.get("api_key_prefix"):
            expected_prefix = str(spec["api_key_prefix"])
            break
    if value and expected_prefix and not value.startswith(expected_prefix):
        return JSONResponse(
            {"ok": False,
             "error": f"{name} has the wrong key type; this provider expects "
                      f"a key beginning with {expected_prefix!r}"},
            status_code=400)
    try:
        action = _write_env_var(_env_file(), name, value)
    except OSError as e:
        return JSONResponse({"ok": False, "error": f"cannot write .env: {e}"},
                            status_code=500)
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)
    log.info("provider key %s %s (value not logged)", name, action)
    return {"ok": True, "env": name, "action": action,
            "note": f"{name} {action} in .env. Restart :8051 for the query API "
                    f"to pick it up (this console already has it)."}


# ---- reranker preflight ----

def _model_cache_roots() -> list[Path]:
    """Every directory a cached model could actually be sitting in.

    There is no single answer, which is the whole reason this exists. The
    reranker is loaded by sentence-transformers, which honours
    SENTENCE_TRANSFORMERS_HOME **over** HF_HOME; meanwhile HF_HUB_CACHE can be
    set machine-wide to a third location that overrides HF_HOME for
    huggingface_hub itself. On this project's own machine those really are three
    different drives, so probing only the hub's idea of the cache reports a
    model as absent while the pipeline loads it fine from elsewhere.
    """
    roots: list[Path] = []

    def add(p: str | None, *suffix: str) -> None:
        if not p:
            return
        cand = Path(p).joinpath(*suffix)
        if cand not in roots:
            roots.append(cand)

    add(os.environ.get("SENTENCE_TRANSFORMERS_HOME"))
    add(os.environ.get("HF_HUB_CACHE"))
    add(os.environ.get("HUGGINGFACE_HUB_CACHE"))
    add(os.environ.get("HF_HOME"), "hub")
    add(os.environ.get("HF_HOME"), "st")
    try:
        from huggingface_hub import constants
        add(constants.HF_HUB_CACHE)
    except Exception:                            # hub layout changed
        pass
    add(str(Path.home() / ".cache" / "huggingface"), "hub")
    return roots


def _model_is_cached(model: str) -> tuple[bool, str | None]:
    """(is_cached, where). `where` is None when nothing could be inspected.

    Matches the `models--org--name` directory layout both caches share, and
    requires an actual weights file — a metadata-only directory (a previous
    failed download, or hub's `.no_exist` bookkeeping) is not a cached model.
    """
    slug = "models--" + model.replace("/", "--")
    looked = False
    for root in _model_cache_roots():
        try:
            if not root.is_dir():
                continue
            looked = True
            snaps = root / slug / "snapshots"
            if not snaps.is_dir():
                continue
            for snap in snaps.iterdir():
                for name in ("model.safetensors", "pytorch_model.bin",
                             "model.onnx"):
                    if (snap / name).exists():
                        return True, str(snap)
        except OSError:                          # unreadable drive, permissions
            continue
    return False, (str(len(_model_cache_roots())) + " roots" if looked else None)


class RerankCheckIn(BaseModel):
    model: Optional[str] = None          # default: whatever config.yaml holds
    max_length: Optional[int] = None


@app.post("/api/rerank/check")
def rerank_check(body: RerankCheckIn) -> dict:
    """Can this cross-encoder actually be used here, and what will it cost?

    Switching reranker is a config write plus a :8051 restart, and until now
    every way it could go wrong looked identical from the console: the restart
    just never reported ready. The two real causes are boring and both
    detectable BEFORE the restart —

      * the id does not exist / the host cannot reach the hub (typo, offline,
        a container with no egress);
      * the weights are not in the local cache yet, so the next :8051 boot
        spends several minutes downloading before it can answer anything.

    No weights are downloaded here: this asks the hub for metadata and looks in
    the cache. A model that is unreachable AND uncached is reported as not
    usable, because that restart WILL fail.
    """
    from src.utils.config_loader import load_config as _load
    from src.retrieval.reranker import KNOWN_RERANKERS

    disk_cfg = _load()
    model = (body.model or disk_cfg.get("retrieval.cross_encoder_model") or "").strip()
    if not model:
        return JSONResponse({"ok": False, "error": "no cross-encoder configured"},
                            status_code=400)
    raw_len = body.max_length if body.max_length is not None \
        else disk_cfg.get("retrieval.cross_encoder_max_length")
    warnings: list[str] = []

    known = KNOWN_RERANKERS.get(model)
    try:
        max_length = int(raw_len)
    except (TypeError, ValueError):
        max_length = None
        warnings.append("retrieval.cross_encoder_max_length is not an integer")
    if known and max_length and max_length > int(known["context_length"]):
        warnings.append(
            f"max_length={max_length} exceeds {model}'s "
            f"{known['context_length']}-token limit — the model raises an "
            f"opaque position-embedding error at query time")

    cached, cache_hit = _model_is_cached(model)
    if cache_hit is None:
        warnings.append("could not inspect the local model cache")

    reachable, detail = None, None
    try:
        from huggingface_hub import model_info
        info = model_info(model)
        reachable = True
        detail = f"{info.id} ({info.pipeline_tag or 'no pipeline tag'})"
    except Exception as e:                       # offline, 404, auth, timeout
        reachable = False
        detail = f"{type(e).__name__}: {e}"
        # A STALE CREDENTIAL LOOKS EXACTLY LIKE A MISSING MODEL. The hub answers
        # 401 for an unauthorized request and huggingface_hub surfaces that as
        # RepositoryNotFoundError, so an expired token makes every public
        # cross-encoder report "does not exist" — and the download :8051
        # attempts at boot fails the same way, with the same misleading message.
        #
        # Retry anonymously to find out which it actually is. The credential can
        # come from an environment variable OR from a stored login file, so this
        # does not condition on the env vars alone; it asks where the token
        # actually came from only to name it in the warning.
        try:
            info = model_info(model, token=False)
        except Exception as e2:
            detail = (f"{type(e).__name__}: {e} "
                      f"(anonymous retry also failed: {type(e2).__name__})")
        else:
            reachable = True
            detail = f"{info.id} (reachable anonymously)"
            where = ", ".join(
                v for v in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
                            "HUGGINGFACEHUB_API_TOKEN")
                if os.environ.get(v))
            if not where:
                try:
                    from huggingface_hub import constants
                    where = f"the stored login at {constants.HF_TOKEN_PATH}"
                except Exception:
                    where = "a stored Hugging Face login"
            warnings.append(
                f"the Hugging Face hub REJECTED the credential from {where}, "
                f"but this model is public and reachable without one. Fix or "
                f"remove that credential — otherwise :8051 hits the same 401 "
                f"downloading the weights, and reports it as 'model not found'")

    usable = bool(cached or reachable)
    if not usable:
        warnings.append(
            "not in the local cache and the hub is not reachable — restarting "
            ":8051 with this model would fail to build the pipeline")
    elif not cached:
        warnings.append(
            "not cached yet: the next :8051 start downloads the weights first, "
            "so 'ready' can take several minutes on a slow link")

    return {"ok": True, "model": model, "usable": usable, "cached": cached,
            "cache_path": cache_hit if cached else None,
            "reachable": reachable, "hub": detail,
            "max_length": max_length,
            "context_length": known["context_length"] if known else None,
            "known": bool(known), "warnings": warnings}


# ---- OCR engines (status + warm-up) ----

@app.get("/api/ocr/status")
def ocr_status() -> dict:
    """What OCR this install can actually do, right now.

    Two engines with completely different shapes, and the console kept
    conflating them:
      * tesseract — a BINARY in this image/machine. Either present or not.
      * vlm       — a vision model served OVER HTTP somewhere else. Nothing to
                    install; it is up or it is down, and it needs warming.

    Read-only and never raises: every probe is wrapped, because "the OCR panel
    500s" is a worse failure than "the endpoint is down".
    """
    from src.utils.config_loader import load_config as _load
    disk_cfg = _load()

    # --- tesseract ---
    # Ask the ingest path's OWN resolver rather than reimplementing it: it also
    # locates and sets TESSDATA_PREFIX, which is the difference between "the
    # binary is on PATH" and "OCR will actually run". A panel that disagreed
    # with what ingestion does would be worse than no panel.
    from src.ingestion.pdf_loader import detect_ocr_engine
    detected = detect_ocr_engine()
    exe = shutil.which("tesseract")
    tess: dict = {"available": detected == "tesseract", "path": exe,
                  "version": None, "languages": [],
                  "tessdata": os.environ.get("TESSDATA_PREFIX"),
                  "detected": detected}
    if exe:
        try:
            out = subprocess.run([exe, "--version"], capture_output=True,
                                 text=True, timeout=10).stdout
            tess["version"] = (out.splitlines() or [""])[0].strip()
        except Exception as e:
            tess["version"] = f"(version probe failed: {e})"
        try:
            out = subprocess.run([exe, "--list-langs"], capture_output=True,
                                 text=True, timeout=10).stdout
            tess["languages"] = [ln.strip() for ln in out.splitlines()[1:]
                                 if ln.strip()]
        except Exception:
            pass

    # --- vlm endpoint ---
    presets = disk_cfg.get("pdf.vlm_ocr_presets") or {}
    vlm: dict = {"configured": False, "reachable": False, "base_url": None,
                 "model": None, "preset": disk_cfg.get("pdf.vlm_ocr.preset"),
                 "presets": sorted(presets), "models_served": [], "error": None}
    try:
        from src.ingestion.ocr_vlm import VLMOCR
        client = VLMOCR.from_config(disk_cfg)
        vlm.update({"configured": True, "base_url": client.base_url,
                    "model": client.model})
        import requests
        r = requests.get(f"{client.base_url}/models",
                         headers=client._headers(), timeout=5)
        vlm["reachable"] = r.status_code < 500
        if r.ok:
            data = r.json()
            vlm["models_served"] = [m.get("id") for m in (data.get("data") or [])
                                    if isinstance(m, dict)][:20]
    except Exception as e:
        vlm["error"] = f"{type(e).__name__}: {e}"

    # --- paddle sidecar ---
    # A separate container by design, so "configured" and "reachable" are two
    # genuinely different states here: the config can name a sidecar that is
    # simply stopped, which is the normal way to give its RAM back between
    # ingests. The panel says which one it is instead of "OCR is broken".
    #
    # There is a THIRD state, and it is the one that used to be invisible: a
    # container that is up and answering while its engine cannot import at all
    # (a missing system library, a base image that moved under the wheels). The
    # sidecar now answers 503 with `engine_importable: false` for that, so this
    # reports readiness separately rather than folding it into "reachable".
    paddle: dict = {"configured": False, "reachable": False, "base_url": None,
                    "lang": None, "langs": [], "engine_importable": None,
                    "engine_error": None, "error": None,
                    # CPU sidecar or GPU one, which engine it defaults to, and
                    # the models behind it. All None on a sidecar predating
                    # those fields — unknown, not broken.
                    "device": None, "paddleocr_version": None,
                    "pipeline": None, "pipelines": {}}
    if disk_cfg.get("pdf.paddle_ocr") or {}:
        try:
            from src.ingestion.ocr_paddle import PaddleOCRClient
            client = PaddleOCRClient.from_config(disk_cfg)
            paddle.update({"configured": True, "base_url": client.base_url,
                           "lang": client.lang})
            import requests
            r = requests.get(f"{client.base_url}/health", timeout=5)
            # An answer of ANY status means the container is up. Deriving this
            # from the status code would file the 503 above under "container
            # stopped" — the exact conflation this panel exists to avoid.
            paddle["reachable"] = True
            try:
                body = r.json()
            except Exception:
                body = None
            if isinstance(body, dict):
                paddle["langs"] = body.get("langs") or []
                # Absent on a sidecar older than this field: unknown, not
                # broken. Only a literal false means "up but cannot OCR".
                imp = body.get("engine_importable")
                paddle["engine_importable"] = (None if imp is None else bool(imp))
                paddle["engine_error"] = body.get("engine_import_error")
                # The sidecar has reported these since the GPU lane landed, but
                # nothing read them, so "is the fast sidecar actually running?"
                # still meant reading `docker ps`. They are the difference
                # between a 3s page and a 5s one (device), between PP-OCRv4 and
                # PP-OCRv6 (version), and between plain text and markdown with
                # LaTeX (pipeline) — none of which is visible in a transcript
                # until you already suspect it.
                paddle["device"] = body.get("device")
                paddle["paddleocr_version"] = body.get("paddleocr_version")
                paddle["pipeline"] = body.get("pipeline")
                paddle["pipelines"] = body.get("pipelines") or {}
            elif not r.ok:
                paddle["engine_error"] = f"HTTP {r.status_code}"
        except Exception as e:
            paddle["error"] = f"{type(e).__name__}: {e}"

    engine = str(disk_cfg.get("pdf.ocr_engine", "auto"))
    # What the ingest path would ACTUALLY resolve to, which is not always what
    # ocr_engine says: an off-process engine with no config block falls back to
    # auto-detection, and "auto"/"tesseract" without tessdata resolves to no
    # OCR at all.
    if engine == "none" or not disk_cfg.get("pdf.ocr_enabled", True):
        effective = "none"
    elif engine == "vlm":
        effective = "vlm" if vlm["configured"] else (detected or "none")
    elif engine == "paddle":
        effective = "paddle" if paddle["configured"] else (detected or "none")
    else:
        effective = detected or "none"

    return {"engine": engine, "effective": effective,
            "ocr_enabled": bool(disk_cfg.get("pdf.ocr_enabled", True)),
            "language": disk_cfg.get("pdf.ocr_language", "eng"),
            "tesseract": tess, "vlm": vlm, "paddle": paddle,
            "launch_hint": disk_cfg.get("pdf.vlm_ocr.launch_hint")}


@app.post("/api/ocr/warm")
def ocr_warm() -> dict:
    """Load the vision model by sending it one tiny real page.

    A GET /models answers "is the server up", not "is the model loaded" — a
    llama.cpp server answers /models instantly while the first real image
    request still pays the full weight-load. This sends a 64x64 white PNG
    through the exact ocr_image() path, so a success here means the next
    ingest page will be fast rather than a cold-start timeout.
    """
    from src.utils.config_loader import load_config as _load
    try:
        from src.ingestion.ocr_vlm import VLMOCR
        client = VLMOCR.from_config(_load())
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"},
                            status_code=400)
    import base64
    # 64x64 white PNG, inline so warming needs no scratch file and no Pillow.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAAPElEQVR4nO3BMQEAAADC"
        "oPVPbQwfoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAOA3AAABAAAB"
        "5Vd8AAAAAElFTkSuQmCC")
    t0 = time.time()
    try:
        text = client.ocr_image(png)
    except Exception as e:
        return JSONResponse(
            {"ok": False, "base_url": client.base_url, "model": client.model,
             "error": f"{type(e).__name__}: {e}"}, status_code=502)
    dt = round(time.time() - t0, 1)
    return {"ok": True, "base_url": client.base_url, "model": client.model,
            "seconds": dt, "returned_chars": len(text),
            "note": f"Model answered in {dt}s and is now warm. A blank page "
                    f"returning no text is expected and fine."}


# ---- folder browser (Settings path pickers) ----

def _browse_roots(windows: bool | None = None) -> list[dict]:
    """The '' listing: where a path picker starts.

    `windows` is injectable so the POSIX branch is testable from Windows.
    Patching os.name globally is not an option — pathlib reads it, and a
    Windows process then builds PosixPath objects and dies.

    Windows gets drive letters. POSIX (the Docker/Mac deployment) has none, so
    it gets the places a vault can actually live — '/' plus the home dir, the
    current vault, and /Volumes for external disks. Returning drive letters on
    Linux is what made the vault switcher a dead end in the container: every
    'A:/'..'Z:/' probe failed, the list came back empty, and "Use this folder"
    stayed disabled with nothing to click.
    """
    if windows is None:
        windows = os.name == "nt"
    if windows:
        import string
        return [{"name": f"{d}:", "path": f"{d}:/"}
                for d in string.ascii_uppercase if Path(f"{d}:/").exists()]
    roots: list[dict] = []
    seen: set[str] = set()

    def add(label: str, p: str | Path | None) -> None:
        if not p:
            return
        q = Path(p)
        if not q.is_dir():
            return
        key = str(q)
        if key in seen:
            return
        seen.add(key)
        roots.append({"name": f"{label} — {key}" if label else key, "path": key})

    add("home", os.environ.get("HOME"))
    add("vault", CFG.get("parser.vault_path"))
    add("external drives", "/Volumes")          # macOS mounts live here
    add("", "/")
    return roots


@app.get("/api/browse")
def browse(path: str = "") -> dict:
    """
    One level of the LOCAL filesystem, folders only — powers the Settings
    tab's path pickers and the vault switcher (a browser page can't open a
    native folder dialog for a SERVER-side path, and in Docker the server only
    sees what is bind-mounted). '' lists the roots. Read-only.

    Every entry carries its FULL path, joined server-side. The client used to
    join with a literal '\\', which produced '/vault\\Foo' on Linux — a path
    that is never a directory, so the first click 404'd.
    """
    if not path:
        return {"path": "", "parent": None, "sep": os.sep,
                "dirs": _browse_roots()}
    p = Path(path)
    if not p.is_dir():
        return JSONResponse({"error": f"not a folder: {path}"}, status_code=404)
    dirs = []
    try:
        for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if entry.is_dir() and not entry.name.startswith((".", "$")):
                dirs.append({"name": entry.name, "path": str(entry)})
    except PermissionError:
        return JSONResponse({"error": f"no permission: {path}"}, status_code=403)
    parent = str(p.parent) if p.parent != p else ""
    return {"path": str(p), "parent": parent, "sep": os.sep, "dirs": dirs}


# ---- vault switcher (every vault ever opened stays listed,
#      and each vault remembers its own index/path settings) ----

# The per-vault settings snapshot: everything that must travel WITH a vault.
# Each corpus needs its own index trio — reusing another vault's indexes
# retrieves nonsense (same warning the Settings tab shows).
#
# vault_tree_root and manifest_cache are here because they are per-CORPUS, not
# per-install: the tree root is a folder inside one particular vault, and the
# manifest cache is keyed by that vault's source_files. Leaving them global
# meant vault B was browsed at vault A's root and served vault A's cached
# in-RAG status.
VAULT_KEYS = ["parser.vault_path", "paths.chunks_file", "paths.chroma_dir",
              "paths.bm25_index", "paths.collection_name", "webui.inbox_dir",
              "webui.vault_tree_root", "webui.manifest_cache",
              # The small-to-big parent sidecar is built FROM one corpus's
              # markdown; pointing vault B at vault A's parents swaps in text
              # from the wrong vault whenever parent_context is on.
              "retrieval.parents_file"]

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


def _shared_chroma_root(chroma_dir: str) -> Path:
    """Return the install-wide parent used for per-vault index directories.

    When the current vault already uses ``.../vault_animus/chroma_db``, taking
    one parent would put the next vault at
    ``.../vault_animus/vault_next/chroma_db``.  Peel an existing ``vault_*``
    layer so every vault remains a sibling under the shared index root.
    """
    root = Path(chroma_dir).parent
    if root.name.lower().startswith("vault_"):
        root = root.parent
    return root


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
    The CURRENT vault's settings are snapshotted into the
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
        slug = re.sub(r"[^\w\-]+", "_", target.name).strip("_").lower() or "vault"
        chroma_root = _shared_chroma_root(snap["paths.chroma_dir"])
        if tgt is None or not tgt.get("settings"):
            settings = {
                "parser.vault_path": str(target).replace("\\", "/"),
                "paths.chunks_file": f"data/vaults/{slug}/chunks.jsonl",
                "paths.chroma_dir": (chroma_root / f"vault_{slug}" / "chroma_db"
                                     ).as_posix(),
                "paths.bm25_index": (chroma_root / f"vault_{slug}" / "bm25_index.pkl"
                                     ).as_posix(),
                "paths.collection_name": snap["paths.collection_name"] or "obsidian_vault",
                "webui.inbox_dir": snap["webui.inbox_dir"] or "Inbox",
                # A fresh vault gets its OWN root and cache. The tree root is
                # empty = browse from the vault root: any other default names a
                # folder that exists in one specific vault and nowhere else.
                "webui.vault_tree_root": "",
                "webui.manifest_cache": f"data/vaults/{slug}/.manifest_cache.json",
                "retrieval.parents_file": f"data/vaults/{slug}/parents_md.jsonl",
            }
            scaffolded = True
            if tgt is None:
                tgt = {"path": str(target), "label": body.label or target.name}
                vaults.append(tgt)
            tgt["settings"] = settings
        else:
            # BACKFILL. A snapshot taken before a key joined VAULT_KEYS does not
            # contain it, and _persist_section_keys only writes what it is
            # given — so restoring that vault would silently LEAVE the previous
            # vault's value in config.yaml. That is how a returning vault ends
            # up reading another vault's manifest cache or parents sidecar.
            # Derive from THIS vault's own chunks_file rather than the slug:
            # both files have always lived beside chunks.jsonl, so a vault
            # registered before these keys existed gets its real paths back
            # ("data/chunks.jsonl" -> "data/parents_md.jsonl"), not an empty
            # new sidecar that would make parent_context silently find nothing.
            own = Path(tgt["settings"].get("paths.chunks_file")
                       or f"data/vaults/{slug}/chunks.jsonl").parent.as_posix()
            defaults = {
                "webui.vault_tree_root": "",
                "webui.manifest_cache": f"{own}/.manifest_cache.json",
                "retrieval.parents_file": f"{own}/parents_md.jsonl",
            }
            missing = [k for k in VAULT_KEYS if k not in tgt["settings"]]
            for key in missing:
                tgt["settings"][key] = defaults.get(key, snap.get(key, ""))
            if missing:
                log.info("vault %s: backfilled %s", tgt.get("label"), missing)
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

# Where the POSIX lane looks for serve_api's pid. docker-entrypoint.sh runs
# serve_api under a supervisor loop and writes its pid here, so signalling that
# pid restarts the query API without restarting the whole container.
_SERVE_PID_FILE = Path(os.environ.get("RAG_SERVE_PID_FILE")
                       or (ROOT / "logs" / "serve_api.pid"))
# The supervisor bumps this counter on every launch. It is the only reliable
# proof that a relaunch HAPPENED: pids can repeat, and "the pid file changed"
# is not the same statement as "a new process started".
_SERVE_GEN_FILE = Path(str(_SERVE_PID_FILE) + ".generation")


def _serve_api_log_path() -> Path:
    """Where the query API's startup output lands, on either platform.

    The console needs this because a failed :8051 boot is invisible otherwise:
    in the container the traceback goes to the supervisor's stdout, which the
    browser cannot see.
    """
    log_dir = Path(os.environ.get("RAG_LOG_DIR") or (ROOT / "logs"))
    return log_dir / f"serve_api_{_rag_port()}.out.log"


def _read_generation() -> int | None:
    try:
        return int(_SERVE_GEN_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _restart_rag_api_posix() -> dict:
    """Signal the supervised serve_api process; its supervisor relaunches it.

    Deliberately NOT a "find the process and hope" heuristic: without the
    pid-file contract this raises and says what to do instead. A silent no-op
    here would be worse than the old hard error — the console would report a
    restart that never happened, and every setting saved afterwards would look
    applied while :8051 kept serving the old config.
    """
    import signal

    port = _rag_port()
    if not _SERVE_PID_FILE.exists():
        raise RuntimeError(
            f"no serve_api pid file at {_SERVE_PID_FILE} — this lane needs the "
            f"supervised entrypoint (docker-entrypoint.sh). Restart the "
            f"container instead: docker compose restart rag")
    try:
        pid = int(_SERVE_PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as e:
        raise RuntimeError(f"unreadable pid file {_SERVE_PID_FILE}: {e}")
    old_gen = _read_generation()
    with _RESTART_LOCK:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            raise RuntimeError(
                f"serve_api pid {pid} is not running — the supervisor should "
                f"have restarted it already; check the container logs")
        except PermissionError as e:
            raise RuntimeError(f"cannot signal pid {pid}: {e}")
        # Wait for the supervisor to record a NEW launch. The old pid dying is
        # not enough — the relaunch is what makes :8051 come back. 30s, not the
        # old 10s: a crashed serve_api takes the supervisor's 5s crash penalty
        # first, and giving up before that reported a failure that had not
        # happened yet.
        new_pid, new_gen, relaunched = pid, old_gen, False
        for _ in range(60):
            time.sleep(0.5)
            gen = _read_generation()
            try:
                new_pid = int(_SERVE_PID_FILE.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                continue
            # Either signal is enough; a supervisor without the counter file
            # (an older bundle) still works off the pid change alone.
            if (gen is not None and old_gen is not None and gen != old_gen) \
                    or new_pid != pid:
                new_gen, relaunched = gen, True
                break
    if not relaunched:
        raise RuntimeError(
            f"signalled serve_api (pid {pid}) but the supervisor did not "
            f"relaunch it within 30s. The container's entrypoint is what "
            f"relaunches it — if this bundle predates the supervised "
            f"entrypoint, restart the service instead: "
            f"docker compose restart rag")
    log.info("restarted :%d via SIGTERM (old pid %s -> new pid %s, gen %s)",
             port, pid, new_pid, new_gen)
    return {"port": port, "killed_pid": pid, "new_pid": new_pid,
            "generation": new_gen}


def _restart_rag_api() -> dict:
    """Kill whatever listens on the query-API port and relaunch serve_api
    detached, inheriting THIS console's environment (rag.bat sets the HF
    cache vars — a console started bare would hand :8051 a broken env, which
    is exactly the failure the launcher comments warn about).

    Two lanes: Windows kills by port and relaunches itself; POSIX (the
    Docker/Mac deployment) signals the supervised process — see above. Without
    the POSIX lane nothing saved in Settings or the vault switcher could ever
    be applied from the console in the container."""
    if os.name != "nt":
        return _restart_rag_api_posix()
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
        log_path = _serve_api_log_path()
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
        return JSONResponse({"ok": False, "error": str(e),
                             "log": str(_serve_api_log_path())},
                            status_code=400)
    return {"ok": True, **info,
            "log": str(_serve_api_log_path()),
            "note": "Warm pipeline reloading — poll /health until ready "
                    "(~1–3 min; models + both indexes load from scratch). "
                    "If it does not come back, GET /api/service/log says why."}


@app.get("/api/service/log")
def service_log(lines: int = 80) -> dict:
    """Tail the query API's startup log.

    This is the missing half of the restart lane. A :8051 that never comes back
    is almost always a pipeline that cannot build — a cross-encoder that could
    not be downloaded, a model id with a typo, an index path that moved — and
    that traceback lives in a file (Windows) or the supervisor's stdout
    (container) which the browser cannot reach. Without it the console could
    only report "still not ready", which is a symptom, not a cause.
    """
    path = _serve_api_log_path()
    lines = max(1, min(int(lines), 500))
    if not path.exists():
        return {"ok": True, "path": str(path), "exists": False, "lines": [],
                "note": "no log yet — :8051 has not been started by this "
                        "console (or by the container entrypoint) since boot"}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return JSONResponse({"ok": False, "error": f"cannot read {path}: {e}"},
                            status_code=500)
    tail = text.splitlines()[-lines:]
    return {"ok": True, "path": str(path), "exists": True, "lines": tail}


# ---- eval review queue (Eval tab: verify / reject / edit drafted questions) ----
#
# eval/sets/*.yaml holds the labelled questions (local data about the vault). The
# schema, the loader and the writer live in eval.bench.questions; this section only
# decides when a write may happen and keeps two console tabs from tearing a file.

EVAL_SETS_DIR = CFG.path("eval.sets_dir", "eval/sets")
_eval_lock = threading.Lock()    # a review write is load -> change -> replace: one at a time
_EVAL_ACTIONS = {"verify": "verified", "reject": "rejected", "edit": "edited"}


class EvalReviewIn(BaseModel):
    action: str                           # verify | reject | edit
    record: Optional[dict] = None         # edit only: the whole replacement record


@app.exception_handler(SchemaError)
def _eval_schema_error(_request, exc: SchemaError):
    # A sets file the loader rejects (a hand edit with a typo): its SchemaError names
    # the file, the question and the problem, so say that instead of a bare 500.
    return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


# Parsed sets, keyed on the directory and every file's name, mtime and size. Every
# review request reads them, and re-parsing ~400 questions cost ~2 s per call.
_eval_memo: tuple | None = None


def _eval_questions() -> list:
    # include_rejected: a rejected question still lives in its file, and a rewrite
    # that left it out would delete it for good.
    global _eval_memo
    files = sorted(EVAL_SETS_DIR.glob("*.yaml")) if EVAL_SETS_DIR.exists() else []
    sig = (str(EVAL_SETS_DIR),
           tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files))
    if _eval_memo is None or _eval_memo[0] != sig:
        _eval_memo = (sig, load_sets(EVAL_SETS_DIR, include_rejected=True))
    # Copies: the review path mutates provenance before it writes, and a write that
    # fails must not leave the memo holding a status that never reached disk.
    return copy.deepcopy(_eval_memo[1])


def _eval_cache() -> dict:
    """`bench validate --write-cache` output, {qid: {gold_texts, findings}}: read
    only here, and as old as the last validate run. {} if it was never written; a
    corrupt file raises instead of passing for "no findings"."""
    path = EVAL_SETS_DIR / ".review_cache.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _eval_save(new, questions: list) -> None:
    """Write `new` over its namesake and rewrite that WHOLE sets file, atomically.

    The file is found by the id inside it, not assumed to be <suite>.yaml: a suite
    may span several files, and rewriting the wrong one would duplicate ids.
    (load_sets has already validated every file, so each is a list of records.)
    """
    for path in sorted(EVAL_SETS_DIR.glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        ids = {d["id"] for d in yaml.safe_load(text)} if new.id in text else set()
        if new.id in ids:
            break
    else:
        raise LookupError(f"{new.id} is in no *.yaml under {EVAL_SETS_DIR}")
    # save_suite is atomic itself (temp file, then an os.replace retried while a
    # Windows scanner holds the file), so a reader sees the old file or the new one.
    save_suite(path, [new if q.id == new.id else q for q in questions if q.id in ids])


@app.get("/api/eval/progress")
def eval_progress() -> dict:
    """Per suite: its quota and how many questions sit in each review status
    (`total` counts all four), plus a `totals` row summing the suites."""
    rows = {s: {"quota": n, "total": 0, **dict.fromkeys(STATUSES, 0)}
            for s, n in SUITES.items()}
    for q in _eval_questions():
        rows[q.suite]["total"] += 1
        rows[q.suite][q.provenance["status"]] += 1
    return {**rows, "totals": {k: sum(r[k] for r in rows.values())
                               for k in ("quota", "total", *STATUSES)}}


@app.get("/api/eval/questions")
def eval_questions(suite: str = "", status: str = "", split: str = "") -> list[dict]:
    """The review queue, one row per question; an empty filter matches everything.
    n_errors / n_warnings count the question's cached validator findings."""
    cache = _eval_cache()
    rows = []
    for q in _eval_questions():
        st = q.provenance["status"]
        if (suite and q.suite != suite) or (status and st != status) \
                or (split and q.split != split):
            continue
        levels = [f["level"] for f in cache.get(q.id, {}).get("findings", [])]
        rows.append({"id": q.id, "suite": q.suite, "tier": q.tier, "split": q.split,
                     "status": st, "author": q.provenance["author"],
                     "question": q.question, "n_errors": levels.count("error"),
                     "n_warnings": levels.count("warn")})
    return rows


@app.get("/api/eval/questions/{qid}")
def eval_question(qid: str) -> dict:
    """One question as stored, with its gold chunk texts and validator findings
    from the cache (empty lists when the cache does not cover it)."""
    q = next((q for q in _eval_questions() if q.id == qid), None)
    if q is None:
        return JSONResponse({"error": f"no such question: {qid}"}, status_code=404)
    cached = _eval_cache().get(qid, {})
    return {"record": question_to_dict(q), "gold_texts": cached.get("gold_texts", []),
            "findings": cached.get("findings", [])}


@app.post("/api/eval/questions/{qid}")
def eval_review(qid: str, body: EvalReviewIn) -> dict:
    """Verify, reject or edit one question and write its sets file back.

    An edit goes through the same validator the loader uses: a record that breaks
    the schema is a 400 carrying the SchemaError text, and nothing is written. The
    server sets only provenance.status and reviewed_at; the record keeps its author,
    so an edited draft is still a draft (`owner` marks the operator's own).
    """
    status = _EVAL_ACTIONS.get(body.action)
    if status is None:
        return JSONResponse(
            {"ok": False, "error": f"action must be one of {', '.join(_EVAL_ACTIONS)}, "
                                   f"got {body.action!r}"}, status_code=400)
    if (body.action == "edit") != (body.record is not None):
        return JSONResponse(
            {"ok": False, "error": "send `record` with edit, and only with edit"},
            status_code=400)
    # The load is inside both locks: two tabs (the thread lock) or a `bench draft`
    # appending meanwhile (the cross-process sets lock) must never be overwritten
    # by a stale copy.
    with _eval_lock, sets_lock(EVAL_SETS_DIR):
        questions = _eval_questions()
        old = next((q for q in questions if q.id == qid), None)
        if old is None:
            return JSONResponse({"ok": False, "error": f"no such question: {qid}"},
                                status_code=404)
        new = old
        if body.action == "edit":
            try:
                new = question_from_dict(body.record, "edit")
            except SchemaError as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
            if new.split is None:
                new.split = old.split    # a form that omits the split keeps the stored one
            # The split is locked like the id: it is assigned once by `bench split`,
            # and moving a question between dev and test would leak the sealed set.
            if (new.id, new.suite, new.split) != (old.id, old.suite, old.split):
                return JSONResponse(
                    {"ok": False, "error": f"an edit cannot change id, suite or split "
                                           f"({old.id}, {old.suite}, {old.split})"},
                    status_code=400)
        new.provenance["status"] = status
        new.provenance["reviewed_at"] = time.strftime("%Y-%m-%d")
        _eval_save(new, questions)
    return {"ok": True, "record": question_to_dict(new)}


@app.get("/api/schema")
def api_schema() -> dict:
    """
    Machine-readable capability map of the MANAGEMENT console, so an agent
    (the local agent / Claude Code) can drive corpus operations over JSON the same way
    it drives the query API (:8051) — no browser, no page snapshots.

    Every operation carries a `permission` tier the calling agent MUST honor:
      * read       — safe, no confirmation needed (stats, search, status, logs)
      * mutating   — changes local state or invokes a potentially billed
                     external action; ask the operator first
                     (config/secrets/ingest/append/retag/OCR warm)
      * destructive— removes content; ALWAYS confirm with the operator, echo
                     what will be deleted, and never run unprompted.
    This is a POLICY the agent enforces (the local API has no auth) — the tiers
    exist so a toolkit/skill can gate calls. See the rag-ops skill.
    """
    return {
        "service": CONSOLE_SERVICE,
        "version": app.version,
        "base_url": f"http://127.0.0.1:{CFG.get('webui.port', 8052)}",
        "query_api": RAG_API,
        "permission_tiers": {
            "read": "safe; no confirmation",
            "mutating": "changes local state or invokes an external action; "
                        "ask the operator before running",
            "destructive": "removes content; ALWAYS confirm, echo the exact "
                           "targets, never run unprompted",
        },
        "worker": "single serial queue; index-changing jobs run one at a time. "
                  "Restart serve_api (:8051) after any index change so the warm "
                  "pipeline reloads.",
        "endpoints": {
            "GET /api/schema": {
                "permission": "read",
                "purpose": "this machine-readable management capability map"},
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
            "GET /api/canvas/folders": {
                "permission": "read",
                "purpose": "which vault trees hold .canvas files, with counts "
                           "— the scope picker for the Graph ingest lane"},
            "POST /api/canvas/preview": {
                "permission": "read",
                "purpose": "dry-run the canvas loader into a scratch file and "
                           "report node/edge counts, the share of chunks that "
                           "carry edges, the context-inflation cost and one "
                           "composed sample. Indexes nothing",
                "body": {"include_path": "substr?",
                         "min_chunk_size": "int?", "max_chunk_size": "int?",
                         "chunking": "heading|fixed|document|none",
                         "context_depth": "0|1|2"}},
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
                         "ocr_engine": "auto|tesseract|paddle|vlm|none?",
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
                         "backend": "auto|requests|crawl4ai|scrapling|crawlee",
                         "format": "md|pdf"}},
            "GET /api/import/file": {
                "permission": "read",
                "purpose": "serve one staged/inbox .md or .pdf for preview "
                           "(pdf shows page numbers -> pick OCR page ranges)",
                "query": {"name": "plain filename", "where": "converted|inbox"}},
            "GET /api/import/ocr_scan": {
                "permission": "read",
                "purpose": "inspect a staged PDF and report which pages appear "
                           "to need OCR before conversion or ingest",
                "query": {"name": "plain PDF filename",
                          "where": "converted|inbox",
                          "limit": "<=400 pages"}},
            "GET /api/browse": {
                "permission": "read",
                "purpose": "one level of the local filesystem, folders only "
                           "('' = drives) — powers the Settings path pickers"},
            "GET /api/vaults": {
                "permission": "read",
                "purpose": "vault registry: every vault ever "
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
            "GET /api/jobs/provenance": {
                "permission": "read",
                "purpose": "which command produced each chunk file, and which "
                           "files nothing records. A file under `unrecorded` "
                           "is NOT regenerable — the flags it was ingested "
                           "with are gone, so the JSONL is the only surviving "
                           "copy of that decision and must be treated as "
                           "primary data. `config_changed_since` warns that a "
                           "recorded command would no longer reproduce the "
                           "same chunks, because chunk sizes and the splitter "
                           "come from config.yaml rather than from argv."},
            "GET /api/settings": {
                "permission": "read",
                "purpose": "runtime info + the editable config surface (paths, "
                           "models, defaults) with current values, plus "
                           "`experimental`: the features that are built but "
                           "not part of the default workflow, each with its "
                           "config flag and whether it is on"},
            "POST /api/settings": {
                "permission": "mutating",
                "purpose": "persist whitelisted config values into config.yaml "
                           "(comment-preserving). NOTHING hot-applies — the "
                           "response says which services to restart. Changing "
                           "the embedding model needs a FULL RE-EMBED (ask "
                           "the operator).",
                "body": {"changes": "{dotted.key: value} from GET editable"}},
            "POST /api/providers/key": {
                "permission": "mutating",
                "purpose": "set or clear one api_key_env already named by the "
                           "provider registry. Writes .env, never returns the "
                           "secret, and rejects declared credential-prefix "
                           "mismatches.",
                "body": {"env": "configured environment-variable name",
                         "value": "secret value, or blank to clear"}},
            "POST /api/rerank/check": {
                "permission": "read",
                "purpose": "preflight a cross-encoder id before switching to "
                           "it: is it reachable, is it already in the local "
                           "model cache, and is max_length within its context "
                           "limit. Downloads nothing. Call this before writing "
                           "retrieval.cross_encoder_model — an unreachable, "
                           "uncached model makes the next :8051 start fail.",
                "body": {"model": "cross-encoder id (default: the configured one)",
                         "max_length": "int? (default: the configured one)"}},
            "GET /api/ocr/status": {
                "permission": "read",
                "purpose": "resolved Tesseract/VLM OCR capability, model "
                           "reachability, and active OCR configuration"},
            "POST /api/ocr/warm": {
                "permission": "mutating",
                "purpose": "send one tiny image request to the configured VLM "
                           "OCR endpoint so readiness can be verified before a "
                           "large ingest job; may allocate or bill the model"},
            "POST /api/service/restart": {
                "permission": "mutating",
                "purpose": "kill + relaunch the warm query API (:8051) so it "
                           "serves the current indexes. /health flips ready "
                           "after model/index loading. Native Windows and POSIX "
                           "hosts are supported; the container's supervised "
                           "entrypoint makes this work without restarting the "
                           "container. Returns the relaunch generation counter; "
                           "if the supervisor does not relaunch within 30s this "
                           "fails loudly instead of reporting a restart that "
                           "did not happen. "
                           "webui.auto_restart_rag=true does this automatically "
                           "after index-changing jobs."},
            "GET /api/service/log?lines=": {
                "permission": "read",
                "purpose": "tail the query API's startup log. This is how you "
                           "diagnose a :8051 that restarts but never reports "
                           "ready — the pipeline build error (bad model id, "
                           "undownloadable reranker, moved index path) is "
                           "here, and GET /health on :8051 carries the same "
                           "reason as {state: failed, error}."},
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
            "GET /api/eval/progress": {
                "permission": "read",
                "purpose": "eval review-queue progress: per suite {quota, total, "
                           "draft, verified, edited, rejected}, plus a `totals` "
                           "row summing the suites"},
            "GET /api/eval/questions": {
                "permission": "read",
                "purpose": "the eval question review queue, one row per question "
                           "with its status and the validator's error/warning "
                           "counts (empty filter = no filter)",
                "query": {"suite": "str?",
                          "status": "draft|verified|edited|rejected?",
                          "split": "dev|test?"}},
            "GET /api/eval/questions/{qid}": {
                "permission": "read",
                "purpose": "one eval question as stored, plus its gold chunk "
                           "texts and validator findings from the review cache "
                           "(empty lists when the cache does not cover it)"},
            "POST /api/eval/questions/{qid}": {
                "permission": "mutating",
                "purpose": "verify, reject or edit one eval question: sets "
                           "provenance.status + reviewed_at and rewrites its "
                           "sets file atomically. An invalid edit is a 400 "
                           "carrying the schema error and writes nothing; an "
                           "edit may not change id, suite or split.",
                "body": {"action": "verify|reject|edit",
                         "record": "dict (edit only: the whole replacement "
                                   "record)"}},
        },
        "job_kinds": {
            "ingest_pdfs": {
                "permission": "mutating",
                "params": {"include_path": "substr", "exclude_path": "substr",
                           "include_files": "list[str] (exact filenames)",
                           "output": "data/*.jsonl", "max_pages": "int",
                           "pages": '"1-50,60,70-80" (1-based subset)',
                           "ocr_engine": "auto|tesseract|paddle|vlm|none",
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
            "ingest_canvas": {
                "permission": "mutating",
                "params": {"output": "data/*.jsonl", "include_path": "substr",
                           "max_chunk_size": "int? (split oversized node "
                                             "bodies; omit = no splitting)",
                           "chunking": "heading|fixed|document|none",
                           "context_depth": "0|1|2 (how much of a NEIGHBOUR is "
                                            "inlined; >0 duplicates text and "
                                            "inflates the index)",
                           "force_domain": "str", "force_tags": "csv or list"},
                "note": ".canvas files only — one chunk per text node, with "
                        "that node's edges flattened into the chunk as a "
                        "Connections footer plus aligned edge metadata. "
                        "GET /api/canvas/folders lists where canvases live; "
                        "POST /api/canvas/preview dry-runs the loader without "
                        "indexing anything"},
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
                           "backend": "auto|requests|crawl4ai|scrapling|crawlee",
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
            "index_rebuild": {"permission": "destructive", "params": {},
                              "note": "DELETES the dense collection and rebuilds "
                                      "both indexes from chunks.jsonl ALONE — "
                                      "every appended lane (pdf / notebook / "
                                      "code / canvas / inbox files) drops out "
                                      "until its JSONL is re-appended, a fresh "
                                      "re-embed each. Heavy; disaster recovery "
                                      "or an embedding-model change only"},
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
