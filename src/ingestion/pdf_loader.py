"""
pdf_loader.py — PDF ingestion for Personal RAG, matched to the vault parser.

Produces the SAME JSONL shape as obsidian_parser.py so the embedder/indexer
need zero changes:
    {"doc_id": "<16hex>", "text": "<context-header + content>", "metadata": {...}}

Design priorities (in order of how much they were emphasized):
  1. LaTeX / math       — formula boxes are detected and preserved verbatim.
  2. Code w/ indentation — code blocks kept as fenced markdown; indentation intact.
  3. OCR                — scanned/image pages auto-detected; OCR used IF an engine
                          is available, otherwise the page is flagged & skipped
                          (never crashes for lack of Tesseract).
  4. Disk awareness      — C: has ~15GB free, A: has ~680GB. ALL heavy I/O
                          (HF cache, temp images, intermediate files) routed to A:.
                          Images are NOT written to disk by default (we don't embed
                          images, so writing them just burns space) — only their
                          presence is noted in metadata.

Chapter-aware chunking:
  - Uses the PDF's table of contents (toc_items) when present to tag each chunk
    with its chapter/section — far better than page numbers for retrieval.
  - Falls back to markdown headers (## detected via font size by pymupdf4llm)
    when there's no embedded TOC.
  - Splits to the same size constraints as the vault parser (200/3000/150).

Course detection:
  - Reuses FOLDER_COURSE_MAP + DOMAIN_MAP from obsidian_parser (single source of
    truth). Books live in <Course>/Books/<file>.pdf, so the parent-of-parent
    folder usually names the course.

Usage:
    from src.ingestion.pdf_loader import PDFLoader
    loader = PDFLoader.from_config(cfg)
    loader.ingest_vault()                     # walk vault, write data/pdf_chunks.jsonl
    # or one file:
    docs = loader.load_pdf(Path("path/to/book.pdf"))
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Reuse the vault parser's course knowledge — single source of truth.
from src.ingestion.obsidian_parser import (
    FOLDER_COURSE_MAP,
    DOMAIN_MAP,
    COURSE_MAP,
    clean_text,
    split_section,
    build_context_header,
)
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Disk routing — keep everything off the tiny C: drive
# ─────────────────────────────────────────────────────────────────────

def route_caches_to_disk(scratch_dir: Path) -> None:
    """
    Point HuggingFace, temp, and PyMuPDF scratch at a roomy disk BEFORE any
    heavy library import does its own caching. Idempotent.

    On Windows, C: is often the small drive. Anything that would default to
    %LOCALAPPDATA% or %TEMP% on C: gets redirected here.
    """
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    hf_home = scratch_dir / "hf_cache"
    tmp_dir = scratch_dir / "tmp"
    for d in (hf_home, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Only set if not already set, so the user can override.
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(hf_home / "st"))
    os.environ.setdefault("TMPDIR", str(tmp_dir))    # unix
    os.environ.setdefault("TEMP", str(tmp_dir))      # windows
    os.environ.setdefault("TMP", str(tmp_dir))       # windows
    log.info("Caches routed to %s (HF_HOME=%s, TMP=%s)", scratch_dir, hf_home, tmp_dir)


# ─────────────────────────────────────────────────────────────────────
# Data structure (mirrors obsidian_parser.Document)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PDFChunk:
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            sig = f"{self.metadata.get('source_file', '')}::{self.text[:500]}"
            self.doc_id = hashlib.sha256(sig.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "text": self.text, "metadata": self.metadata}


# ─────────────────────────────────────────────────────────────────────
# OCR engine discovery — graceful, never fatal
# ─────────────────────────────────────────────────────────────────────

def _ensure_tessdata() -> bool:
    """True if Tesseract's tessdata is locatable. If TESSDATA_PREFIX isn't
    already set, try the standard install locations and set it in-process so the
    user doesn't have to configure an env var by hand."""
    cur = os.environ.get("TESSDATA_PREFIX")
    if cur and Path(cur).is_dir():
        return True
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tessdata",
        "/opt/homebrew/share/tessdata",
        "/usr/local/share/tessdata",
    ]
    import shutil
    exe = shutil.which("tesseract")
    if exe:
        candidates.insert(0, str(Path(exe).resolve().parent / "tessdata"))
    for c in candidates:
        if c and Path(c).is_dir():
            os.environ["TESSDATA_PREFIX"] = c
            return True
    return False


def detect_ocr_engine() -> Optional[str]:
    """
    Return the name of an available OCR engine, or None. Probes cheaply, never raises.

    Tesseract is preferred over RapidOCR: pymupdf4llm's Tesseract adaptor is the
    stable, highest-recognition-quality path, whereas its bundled RapidOCR
    adaptor is version-fragile and currently breaks against shipping
    rapidocr-onnxruntime builds. NOTE: pymupdf4llm auto-selects its adaptor from
    what's INSTALLED — so to actually use Tesseract you must uninstall
    rapidocr-onnxruntime; otherwise pymupdf4llm will still reach for the broken
    RapidOCR/rapidtess adaptor regardless of what this function returns.
    """
    # Tesseract first (locates + sets TESSDATA_PREFIX if a standard install exists).
    if _ensure_tessdata():
        return "tesseract"
    import shutil
    if shutil.which("tesseract"):
        return "tesseract"
    # Fallback only: RapidOCR (pure-pip, but adaptor is fragile — see docstring).
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return "rapidocr"
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Script detection — the author skips all Armenian-script PDFs.
# Ported from scout_pdfs.py so the loader is an authoritative, always-on gate
# (independent of whether the dedup skip-list was regenerated).
# ─────────────────────────────────────────────────────────────────────

def detect_script(text: str) -> str:
    """Dominant script by letter counts: latin / cyrillic / armenian / mixed / empty."""
    if not text.strip():
        return "empty"
    counts = {"latin": 0, "cyrillic": 0, "armenian": 0}
    for ch in text:
        o = ord(ch)
        if 0x0041 <= o <= 0x024F:
            counts["latin"] += 1
        elif 0x0400 <= o <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x0530 <= o <= 0x058F:
            counts["armenian"] += 1
    total = sum(counts.values())
    if total == 0:
        return "empty"
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    top, top_n = ranked[0]
    if top_n / total < 0.60:
        return f"mixed ({top}+{ranked[1][0]})"
    return top


def should_skip_script(script: str, skip_scripts: set[str]) -> bool:
    """True if the dominant script is skip-listed, including mixed docs that
    contain a skip-listed script (e.g. 'mixed (armenian+latin)')."""
    if not skip_scripts:
        return False
    if script in skip_scripts:
        return True
    if script.startswith("mixed"):
        return any(s in script for s in skip_scripts)
    return False


def parse_page_spec(spec: Optional[str], page_count: Optional[int] = None) -> list[int]:
    """
    Parse a 1-based page selector like "1-50,60,70-80" into a sorted list of
    0-based page indices (what pymupdf4llm's pages= wants). Inclusive ranges +
    singletons; whitespace-tolerant; duplicates collapse.

    When page_count is given, out-of-range pages are dropped (so "1-9999" on a
    300-page book reads pages 1-300, not an error). Raises ValueError on a
    malformed token so the CLI/console can reject the spec up front (validate
    format with page_count=None; the per-PDF clamp happens later in load_pdf).
    """
    if not spec or not spec.strip():
        return []
    pages: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, _, b = tok.partition("-")
            a, b = a.strip(), b.strip()
            if not (a.isdigit() and b.isdigit()):
                raise ValueError(f"bad page range {tok!r} - use e.g. 70-80")
            lo, hi = int(a), int(b)
            if lo < 1 or hi < lo:
                raise ValueError(f"bad page range {tok!r} - 1-based, low<=high")
            pages.update(range(lo, hi + 1))
        else:
            if not tok.isdigit():
                raise ValueError(f"bad page number {tok!r}")
            n = int(tok)
            if n < 1:
                raise ValueError(f"page numbers are 1-based (got {tok!r})")
            pages.add(n)
    if page_count is not None:
        pages = {p for p in pages if p <= page_count}
    return sorted(p - 1 for p in pages)          # 1-based spec -> 0-based indices


def load_skip_set(skip_list_file: Optional[Path]) -> set[str]:
    """Load a dedup skip-list (JSON array of vault-relative paths) as a posix set."""
    if not skip_list_file:
        return set()
    p = Path(skip_list_file)
    if not p.exists():
        log.warning("skip-list not found: %s (nothing will be skipped by list)", p)
        return set()
    try:
        items = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("could not read skip-list %s: %s", p, e)
        return set()
    return {str(x).replace("\\", "/").lower() for x in items}


# ─────────────────────────────────────────────────────────────────────
# The loader
# ─────────────────────────────────────────────────────────────────────

class PDFLoader:
    # Folder names (case-insensitive) that contain books/readings
    BOOK_FOLDER_NAMES = {"books", "book", "readings", "reading", "textbooks", "textbook"}

    def __init__(
        self,
        vault_path: Path,
        output_file: Path,
        scratch_dir: Path,
        min_chunk: int = 200,
        max_chunk: int = 3000,
        overlap: int = 150,
        scanned_char_threshold: int = 50,
        ocr_enabled: bool = True,
        ocr_language: str = "eng",
        ocr_engine_pref: str = "auto",
        vlm_ocr=None,
        only_book_folders: bool = False,
        skip_books: bool = False,
        include_path: str | None = None,
        exclude_path: str | None = None,
        max_pages_per_pdf: int | None = None,
        skip_list_file: Path | None = None,
        skip_folders: list[str] | None = None,
        skip_scripts: list[str] | None = None,
        script_sample_pages: int = 3,
        extract_images: bool = False,
        figures_dir: Path | None = None,
        image_min_frac: float = 0.08,
    ):
        self.vault_path = Path(vault_path)
        self.output_file = Path(output_file)
        self.scratch_dir = Path(scratch_dir)
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.overlap = overlap
        # Chunking strategy for oversized page-sections ('heading' | 'fixed');
        # validated at use in split_section, overridable via --chunking.
        self.chunking = "heading"
        self.scanned_char_threshold = scanned_char_threshold
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.only_book_folders = only_book_folders
        # skip_books is the inverse of only_book_folders: process everything
        # EXCEPT book folders. Since the books pass found books via book-token
        # folders, this guarantees already-ingested books are not re-processed.
        self.skip_books = skip_books
        # include_path: if set, only process PDFs whose relative path contains
        # this substring (case-insensitive), e.g. "Current Courses".
        self.include_path = include_path.lower() if include_path else None
        # exclude_path: skip PDFs whose relative path contains this substring,
        # e.g. "Current Courses" to avoid re-processing the lecture pass.
        self.exclude_path = exclude_path.lower() if exclude_path else None
        # include_files: exact lowercase FILENAMES to process (file-scoped
        # custom jobs from the console). None = no filename filter.
        self.include_files: set[str] | None = None
        self.max_pages_per_pdf = max_pages_per_pdf
        # page_spec: an arbitrary 1-based page selector ("1-50,60,70-80") that
        # trims a PDF to a substantive subset (drop front-matter/index/back-
        # matter). None = read from page 1 (subject to max_pages_per_pdf). When
        # set, it takes precedence over max_pages_per_pdf. Kept pages are chunked
        # exactly as before, so their doc_ids are stable — a re-ingest with a
        # WIDER range only ADDS pages (upsert); it never removes earlier ones.
        self.page_spec: str | None = None
        # archive_processed: after a PDF yields >=1 chunk, move it into an
        # _ingested/ folder next to it (used by the console's inbox lane so a
        # later batch never re-processes — or worse, silently clobbers — the
        # previous batch's files). _ingested is excluded from discovery.
        self.archive_processed = False
        # Batch-level metadata overrides (inbox lane): files dropped in the
        # inbox carry no course-folder path, so path-derived domain is always
        # "general". force_domain/force_tags stamp every chunk written this
        # run — metadata only, so doc_ids are unaffected.
        self.force_domain: str | None = None
        self.force_tags: list[str] = []

        # Dedup skip-list (from scout_dedup.py) + folder/script gates.
        self.skip_set = load_skip_set(skip_list_file)
        self.skip_folders = [s.replace("\\", "/").lower() for s in (skip_folders or [])]
        self.skip_scripts = set(skip_scripts if skip_scripts is not None else ["armenian"])
        self.script_sample_pages = script_sample_pages

        # Figure extraction: save diagrams/plots to disk and link them to the
        # chunk on their page. Routed to a roomy disk (figures persist; they are
        # NOT cache, so they go under data/, not scratch).
        self.extract_images = extract_images
        self.figures_dir = Path(figures_dir) if figures_dir else Path("data/figures")
        self.image_min_frac = image_min_frac

        # OCR engine resolution. "auto" keeps the historical behavior
        # (Tesseract-first probe); "vlm" routes scanned pages through the
        # configured vision endpoint (pdf.vlm_ocr — Unlimited-OCR reference);
        # "tesseract" pins the classic path; "none" disables OCR outright.
        self.vlm_ocr = vlm_ocr
        self.ocr_engine_pref = (ocr_engine_pref or "auto").lower()
        self.ocr_engine = self._resolve_ocr_engine(self.ocr_engine_pref)

        self.stats = {
            "pdfs_found": 0,
            "pdfs_processed": 0,
            "pdfs_failed": 0,
            "pages_total": 0,
            "pages_ocr": 0,
            "pages_scanned_skipped": 0,
            "pdfs_skipped_list": 0,
            "pdfs_skipped_book": 0,
            "pdfs_skipped_nonbook": 0,
            "pdfs_skipped_include": 0,
            "pdfs_skipped_excluded": 0,
            "pdfs_skipped_folder": 0,
            "pdfs_skipped_script": 0,
            "chunks_total": 0,
            "chunks_by_domain": {},
            "chunks_by_course": {},
            "formula_chunks": 0,
            "code_chunks": 0,
            "figures_extracted": 0,
            "chunks_with_figures": 0,
        }

    def _resolve_ocr_engine(self, pref: str) -> Optional[str]:
        """Map an engine preference to what is actually usable, never raising.

        Fallback ladder keeps the existing contract (text pages always survive):
          vlm w/o a configured client  -> auto-detect (tesseract) with a warning
          tesseract w/o tessdata       -> None (pages stay sparse) with a warning
        """
        if not self.ocr_enabled or pref == "none":
            return None
        if pref == "vlm":
            if self.vlm_ocr is not None:
                return "vlm"
            log.warning("pdf.ocr_engine=vlm but no pdf.vlm_ocr block is "
                        "configured — falling back to auto-detection.")
            return detect_ocr_engine()
        if pref == "tesseract":
            if _ensure_tessdata():
                return "tesseract"
            log.warning("pdf.ocr_engine=tesseract but tessdata was not found — "
                        "OCR disabled (scanned pages will be sparse).")
            return None
        return detect_ocr_engine()

    def set_ocr_engine(self, pref: str) -> None:
        """Re-resolve the engine (used by the --ocr-engine CLI flag)."""
        self.ocr_engine_pref = (pref or "auto").lower()
        self.ocr_enabled = self.ocr_engine_pref != "none"
        self.ocr_engine = self._resolve_ocr_engine(self.ocr_engine_pref)

    @classmethod
    def from_config(cls, cfg: Config) -> "PDFLoader":
        scratch = cfg.get("pdf.scratch_dir") or str(cfg.project_root / "data" / "scratch")
        route_caches_to_disk(Path(scratch))
        loader = cls(
            vault_path=Path(cfg.get("pdf.vault_path") or cfg.get("parser.vault_path")),
            output_file=cfg.path("pdf.output_file") if cfg.get("pdf.output_file")
                        else cfg.project_root / "data" / "pdf_chunks.jsonl",
            scratch_dir=Path(scratch),
            min_chunk=cfg.get("pdf.min_chunk_size", cfg.get("parser.min_chunk_size", 200)),
            max_chunk=cfg.get("pdf.max_chunk_size", cfg.get("parser.max_chunk_size", 3000)),
            overlap=cfg.get("pdf.overlap_size", cfg.get("parser.overlap_size", 150)),
            scanned_char_threshold=cfg.get("pdf.skip_scanned_threshold", 50),
            ocr_enabled=cfg.get("pdf.ocr_enabled", True),
            ocr_language=cfg.get("pdf.ocr_language", "eng"),
            ocr_engine_pref=cfg.get("pdf.ocr_engine", "auto"),
            vlm_ocr=cls._build_vlm(cfg),
            only_book_folders=cfg.get("pdf.only_book_folders", False),
            skip_books=cfg.get("pdf.skip_books", False),
            include_path=cfg.get("pdf.include_path"),
            exclude_path=cfg.get("pdf.exclude_path"),
            max_pages_per_pdf=cfg.get("pdf.max_pages_per_pdf"),
            skip_list_file=cfg.path("pdf.skip_list_file") if cfg.get("pdf.skip_list_file") else None,
            skip_folders=cfg.get("pdf.skip_folders", []),
            skip_scripts=cfg.get("pdf.skip_scripts", ["armenian"]),
            script_sample_pages=cfg.get("pdf.script_sample_pages", 3),
            extract_images=cfg.get("pdf.extract_images", False),
            figures_dir=cfg.path("pdf.figures_dir") if cfg.get("pdf.figures_dir")
                        else cfg.project_root / "data" / "figures",
            image_min_frac=cfg.get("pdf.image_min_frac", 0.08),
        )
        loader.chunking = str(cfg.get("pdf.chunking",
                                      cfg.get("parser.chunking", "heading"))).lower()
        return loader

    @staticmethod
    def _build_vlm(cfg: Config):
        """VLMOCR client iff pdf.vlm_ocr is configured (no network at build)."""
        if not (cfg.get("pdf.vlm_ocr") or {}):
            return None
        from src.ingestion.ocr_vlm import VLMOCR
        return VLMOCR.from_config(cfg)

    # ---- discovery ----

    def _is_book_path(self, f: Path) -> bool:
        """True if any path part contains a book/reading/textbook token.
        Catches 'Books' as well as 'NA Books', 'ML Books', 'DS Books 2025 Spring'."""
        for part in f.parts:
            toks = re.split(r"[\s_\-]+", part.lower())
            if any(t in self.BOOK_FOLDER_NAMES for t in toks):
                return True
        return False

    def discover_pdfs(self) -> list[Path]:
        """Find all PDFs in the vault. If only_book_folders, restrict to Books/ dirs.
        Applies the dedup skip-list and any folder-level skip rules.

        Filter order matters for DIAGNOSTICS only (selection is a conjunction):
        scope filters (include/exclude/skip-list/folders) run first, the book
        filters last — so pdfs_skipped_nonbook/_book count drops INSIDE your
        scope, and a 0-found summary points at the filter that actually bit.
        """
        pdfs = []
        total_seen = 0
        for f in self.vault_path.rglob("*.pdf"):
            # _ingested = archive folders for already-processed inbox files
            # (see archive_processed) — never rediscover those.
            if any(part in {".obsidian", ".trash", ".git", "_ingested"} for part in f.parts):
                continue
            total_seen += 1
            rel_posix = f.relative_to(self.vault_path).as_posix().lower()
            if self.include_path and self.include_path not in rel_posix:
                self.stats["pdfs_skipped_include"] += 1
                continue
            if self.include_files and f.name.lower() not in self.include_files:
                self.stats["pdfs_skipped_include"] += 1
                continue
            if self.exclude_path and self.exclude_path in rel_posix:
                self.stats["pdfs_skipped_excluded"] += 1
                continue
            if rel_posix in self.skip_set:
                self.stats["pdfs_skipped_list"] += 1
                continue
            if any(folder in rel_posix for folder in self.skip_folders):
                self.stats["pdfs_skipped_folder"] += 1
                continue
            if self.only_book_folders and not self._is_book_path(f):
                self.stats["pdfs_skipped_nonbook"] += 1
                continue
            if self.skip_books and self._is_book_path(f):
                self.stats["pdfs_skipped_book"] += 1
                continue
            pdfs.append(f)
        self.stats["pdfs_found"] = len(pdfs)
        if not pdfs:
            s = self.stats
            log.warning(
                "discover_pdfs found 0 PDFs under %s (saw %d .pdf total before "
                "filters). Where they went: %d outside --include-path %r, "
                "%d hit --exclude-path, %d on the dedup skip-list, %d in skipped "
                "folders, %d dropped by --only-books (non-book path), %d dropped "
                "by --skip-books. If total is 0, pdf.vault_path is wrong. If "
                "'dropped by --only-books' ate everything, your target folder is "
                "not a book folder — drop --only-books (inbox ingests must NOT "
                "set it).",
                self.vault_path, total_seen,
                s["pdfs_skipped_include"], self.include_path,
                s["pdfs_skipped_excluded"], s["pdfs_skipped_list"],
                s["pdfs_skipped_folder"], s["pdfs_skipped_nonbook"],
                s["pdfs_skipped_book"],
            )
        return sorted(pdfs)

    # ---- course detection from path ----

    def _detect_course(self, filepath: Path) -> dict:
        """
        Resolve course/domain from the file path, reusing the vault parser's maps.
        Strategy: walk path parts; the part that isn't a book-folder name and
        matches FOLDER_COURSE_MAP wins. Books live in <Course>/Books/<file>.
        """
        parts = list(filepath.relative_to(self.vault_path).parts)
        # Try each path part against the folder map (case-insensitive)
        for part in parts:
            name = FOLDER_COURSE_MAP.get(part.lower().strip())
            if name:
                domain = DOMAIN_MAP.get(name, "general")
                return {"course_code": name, "course_name": name, "domain": domain}
        # Try AUA course codes embedded in path (e.g. "CS 362")
        for part in parts:
            m = re.search(r"(CS|DS|ENGS|BSDS|ECON)\s*\d{2,3}", part, re.IGNORECASE)
            if m:
                code = re.sub(r"\s+", " ", re.sub(r"(\D)(\d)", r"\1 \2", m.group(0).upper())).strip()
                name = COURSE_MAP.get(code, code)
                domain = DOMAIN_MAP.get(name, "general")
                return {"course_code": code, "course_name": name, "domain": domain}
        return {"course_code": "unknown", "course_name": "unknown", "domain": "general"}

    # ---- chapter mapping from TOC ----

    @staticmethod
    def _build_page_to_chapter(toc_items: list, page_count: int) -> dict[int, str]:
        """
        Build a {page_number: chapter_title} map from a PDF's table of contents.
        toc_items entries look like [level, title, page] (1-based pages).
        We forward-fill: every page inherits the most recent chapter heading.
        """
        if not toc_items:
            return {}
        # Sort by page, keep top 2 levels (chapter + section)
        entries = []
        for item in toc_items:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            level, title, page = item[0], item[1], item[2]
            if level <= 2 and isinstance(page, int) and page > 0:
                entries.append((page, str(title).strip()))
        entries.sort()
        page_map: dict[int, str] = {}
        cur = ""
        ei = 0
        for pno in range(1, page_count + 1):
            while ei < len(entries) and entries[ei][0] <= pno:
                cur = entries[ei][1]
                ei += 1
            if cur:
                page_map[pno] = cur
        return page_map

    # ---- per-PDF extraction ----

    def load_pdf(self, filepath: Path) -> list[PDFChunk]:
        """Extract one PDF into chunks. Returns [] on unreadable/empty files."""
        import pymupdf
        import pymupdf4llm

        rel_path = str(filepath.relative_to(self.vault_path))
        course_meta = self._detect_course(filepath)

        # First pass: open with pymupdf to read TOC + decide per-page OCR need
        try:
            doc = pymupdf.open(filepath)
        except Exception as e:
            log.warning("Cannot open %s: %s", rel_path, e)
            self.stats["pdfs_failed"] += 1
            return []

        page_count = doc.page_count
        toc = doc.get_toc() if hasattr(doc, "get_toc") else []
        page_to_chapter = self._build_page_to_chapter(toc, page_count)

        # Script gate (defense-in-depth, independent of the dedup skip-list):
        # sample the first few pages and skip the whole PDF if its dominant
        # script is skip-listed (the author: drop all Armenian-script content).
        if self.skip_scripts:
            sample = []
            for pno in range(min(self.script_sample_pages, page_count)):
                try:
                    sample.append(doc[pno].get_text("text")[:2000])
                except Exception:
                    pass
            script = detect_script("\n".join(sample))
            if should_skip_script(script, self.skip_scripts):
                log.info("%s: skipped (%s script)", rel_path, script)
                self.stats["pdfs_skipped_script"] += 1
                doc.close()
                return []

        # Decide which pages are "scanned" (little/no extractable text)
        scanned_pages = self._find_scanned_pages(doc)
        # Extract figures while the doc is open (diagrams/plots saved to disk,
        # linked to their page so the chunk on that page can surface them).
        page_to_figures = self._extract_page_figures(doc, filepath)
        doc.close()

        # Which pages to read. A page_spec (arbitrary 1-based subset) wins over
        # max_pages (a from-page-1 cap). Both produce a 0-based `page_list`.
        if self.page_spec:
            page_list = parse_page_spec(self.page_spec, page_count)
            if not page_list:
                log.warning("%s: --pages %r selected no pages in a %d-page PDF "
                            "— skipped.", rel_path, self.page_spec, page_count)
                return []
        else:
            n_pages_to_read = page_count
            if self.max_pages_per_pdf:
                n_pages_to_read = min(page_count, self.max_pages_per_pdf)
            page_list = list(range(n_pages_to_read))
        # Only scanned pages we actually read matter for OCR + accounting.
        page_set = set(page_list)
        scanned_pages = [p for p in scanned_pages if p in page_set]

        # Decide OCR strategy for this document
        use_ocr = bool(scanned_pages) and self.ocr_enabled and self.ocr_engine is not None
        # VLM engine: pymupdf4llm extracts with OCR OFF (its adaptor is never
        # touched), then scanned pages are re-parsed through the vision
        # endpoint and injected back — see _apply_vlm_ocr below.
        vlm_mode = use_ocr and self.ocr_engine == "vlm"
        pymupdf_ocr = use_ocr and not vlm_mode
        if scanned_pages and not use_ocr:
            self.stats["pages_scanned_skipped"] += len(scanned_pages)
            log.warning(
                "%s: %d scanned page(s) and no OCR engine — those pages will be sparse",
                rel_path, len(scanned_pages),
            )

        # Second pass: pymupdf4llm extraction to markdown with page chunks
        try:
            md_chunks = pymupdf4llm.to_markdown(
                str(filepath),
                pages=page_list,
                page_chunks=True,
                show_progress=False,
                # OCR: only turns on for pages MuPDF judges to need it; we gate
                # the whole feature on engine availability above. (VLM mode
                # keeps this OFF — the vision endpoint handles scanned pages.)
                use_ocr=pymupdf_ocr,
                ocr_language=self.ocr_language,
                # Keep code blocks (don't strip mono-spaced text)
                ignore_code=False,
                # We do NOT write images to disk — we don't embed them, and C: is small.
                write_images=False,
                embed_images=False,
                table_strategy="lines_strict",
            )
        except TypeError:
            # Older/newer signature mismatch — retry with a minimal safe call
            md_chunks = pymupdf4llm.to_markdown(
                str(filepath), pages=page_list, page_chunks=True, show_progress=False,
            )
        except Exception as e:
            # An OCR/adaptor failure must NOT cost us the readable text pages.
            # Retry once with OCR off so text-based pages still get extracted;
            # only the scanned pages go sparse.
            if pymupdf_ocr:
                log.warning("%s: OCR extraction failed (%s); retrying without OCR "
                            "(text pages kept, scanned pages sparse).", rel_path, e)
                try:
                    md_chunks = pymupdf4llm.to_markdown(
                        str(filepath), pages=page_list, page_chunks=True,
                        show_progress=False, use_ocr=False,
                        ignore_code=False, write_images=False,
                        embed_images=False, table_strategy="lines_strict",
                    )
                    use_ocr = False
                    pymupdf_ocr = False
                    self.stats["pages_scanned_skipped"] += len(scanned_pages)
                except Exception as e2:
                    log.warning("Extraction failed for %s: %s", rel_path, e2)
                    self.stats["pdfs_failed"] += 1
                    return []
            else:
                log.warning("Extraction failed for %s: %s", rel_path, e)
                self.stats["pdfs_failed"] += 1
                return []

        self.stats["pages_total"] += page_count
        if vlm_mode:
            # scanned_pages is already restricted to the pages we actually read.
            done = self._apply_vlm_ocr(filepath, md_chunks, scanned_pages)
            self.stats["pages_ocr"] += done
            self.stats["pages_scanned_skipped"] += len(scanned_pages) - done
        else:
            self.stats["pages_ocr"] += len(scanned_pages) if use_ocr else 0

        chunks = self._chunks_from_md(
            md_chunks, rel_path, filepath, course_meta, page_to_chapter, page_to_figures
        )
        if chunks:
            self.stats["pdfs_processed"] += 1
        return chunks

    def _apply_vlm_ocr(self, filepath: Path, md_chunks, scanned_pages: list[int]) -> int:
        """
        VLM-mode step 2: parse the scanned pages through the vision endpoint
        (self.vlm_ocr) and inject the returned markdown into the matching
        page_chunks entries, so the normal chunker downstream never knows the
        difference. Returns the number of pages successfully OCR'd; pages that
        fail keep their (near-empty) extracted text and stay sparse — exactly
        the graceful-degradation contract the classic path has.
        """
        if not scanned_pages or self.vlm_ocr is None:
            return 0
        texts = self.vlm_ocr.ocr_pages(filepath, scanned_pages)
        if not texts:
            return 0
        for md in md_chunks:
            meta_in = md.get("metadata", {}) or {}
            pno = self._safe_int(meta_in.get("page_number"))
            if pno is None:
                continue
            parsed = texts.get(pno - 1)          # page_number is 1-based
            if parsed:
                md["text"] = parsed
                # Scanned pages have no layout boxes, so the box-class
                # has_formula detection in _chunks_from_md can't fire on
                # them — mark the entry so a text-level check runs instead.
                md["vlm_injected"] = True
        return len(texts)

    def _find_scanned_pages(self, doc) -> list[int]:
        """Pages whose extractable text is below threshold = likely scanned images."""
        scanned = []
        for pno in range(doc.page_count):
            try:
                txt = doc[pno].get_text("text")
            except Exception:
                txt = ""
            if len(txt.strip()) < self.scanned_char_threshold:
                scanned.append(pno)
        return scanned

    # ---- figure extraction ----

    @staticmethod
    def _safe_dirname(filepath: Path) -> str:
        """Stable, collision-free subfolder name for a book's figures."""
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", filepath.stem)[:60]
        h = hashlib.sha256(str(filepath).encode()).hexdigest()[:8]
        return f"{stem}_{h}"

    def _extract_page_figures(self, doc, filepath: Path) -> dict[int, list[str]]:
        """
        Save figures (diagrams, plots) above the size threshold to disk and return
        {page_number(1-based): [image_path, ...]}. Small images (logos, icons,
        bullets) are filtered out by area fraction. Repeated images (same xref,
        e.g. a running header logo) are saved once. Never raises on a bad image.
        """
        if not self.extract_images:
            return {}
        try:
            figdir = self.figures_dir / self._safe_dirname(filepath)
            figdir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.warning("Cannot create figures dir for %s: %s", filepath.name, e)
            return {}

        page_to_figs: dict[int, list[str]] = {}
        seen_xrefs: set[int] = set()
        for pno in range(doc.page_count):
            page = doc[pno]
            page_area = abs(page.rect.width * page.rect.height) or 1.0
            figs: list[str] = []
            try:
                images = page.get_images(full=True)
            except Exception:
                images = []
            for img in images:
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                # Size filter: how much of the page does this image occupy?
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                if not rects:
                    continue
                frac = max(abs(r.width * r.height) for r in rects) / page_area
                if frac < self.image_min_frac:
                    continue
                seen_xrefs.add(xref)
                try:
                    ext = doc.extract_image(xref)
                    data = ext.get("image")
                    if not data:
                        continue
                    fpath = figdir / f"p{pno + 1:04d}_x{xref}.{ext.get('ext', 'png')}"
                    fpath.write_bytes(data)
                except Exception:
                    continue
                figs.append(str(fpath))
            if figs:
                page_to_figs[pno + 1] = figs
                self.stats["figures_extracted"] += len(figs)
        return page_to_figs

    # ---- markdown chunks -> PDFChunk with context headers ----

    def _chunks_from_md(
        self, md_chunks, rel_path, filepath, course_meta, page_to_chapter,
        page_to_figures=None,
    ) -> list[PDFChunk]:
        out: list[PDFChunk] = []
        book_title = filepath.stem

        for md in md_chunks:
            text = (md.get("text") or "").strip()
            if not text:
                continue
            meta_in = md.get("metadata", {}) or {}
            page_no = self._safe_int(meta_in.get("page_number"))

            # Detect special content via page_boxes class tags. VLM-injected
            # pages have no boxes (they were scanned), so detect LaTeX
            # markers in the parsed markdown instead.
            box_classes = {b.get("class") for b in md.get("page_boxes", []) if isinstance(b, dict)}
            has_formula = "formula" in box_classes or bool(
                md.get("vlm_injected")
                # $...$ must contain a LaTeX-ish char so currency spans
                # ("5$ ... 10$") don't count as math.
                and re.search(
                    r"\$[^$\n]*[\\^_={}][^$\n]*\$|\\\[|\\begin\{|\\frac|\\sum|\\int",
                    text,
                )
            )
            has_code = bool(re.search(r"```|\n {4,}\S|\bdef \w+\(|\bclass \w+", text))

            chapter = page_to_chapter.get(page_no, "") if page_no else ""

            # Clean while PRESERVING code/latex: we deliberately do NOT run the
            # vault's clean_text on fenced code; we only normalize blank lines.
            cleaned = self._clean_preserving_code(text)

            # Build heading stack: [book title, chapter]
            heading_stack = [book_title]
            if chapter and chapter != book_title:
                heading_stack.append(chapter)

            base_meta = {
                "source_file": rel_path,
                "filename": book_title,
                "file_type": "pdf",
                "vault_path": str(self.vault_path),
                "tags": [],
                "wikilinks": [],
                **course_meta,
                "heading": chapter or book_title,
                "heading_level": 1,
                "heading_path": " > ".join(heading_stack),
            }
            if page_no:
                base_meta["page_start"] = page_no
                base_meta["page_end"] = page_no
            if chapter:
                base_meta["chapter"] = chapter
            if has_formula:
                base_meta["has_formula"] = True
            if has_code:
                base_meta["has_code"] = True
            # Figures on this page: store as a delimited STRING (ChromaDB metadata
            # values must be scalars, not lists), so the path survives retrieval
            # and the UI can render the diagram next to the chunk.
            figs = (page_to_figures or {}).get(page_no, []) if page_no else []
            if figs:
                base_meta["has_figure"] = True
                base_meta["figure_count"] = len(figs)
                base_meta["figure_images"] = ";".join(figs)

            context = build_context_header(heading_stack, base_meta)
            sub_chunks = split_section(cleaned, self.max_chunk, self.overlap,
                                       self.chunking)

            for i, piece in enumerate(sub_chunks):
                if len(piece.strip()) < self.min_chunk:
                    continue
                meta = {**base_meta}
                if len(sub_chunks) > 1:
                    meta["chunk_part"] = f"{i + 1}/{len(sub_chunks)}"
                chunk = PDFChunk(text=context + piece, metadata=meta)
                out.append(chunk)
                self._update_stats(meta)

        return out

    @staticmethod
    def _clean_preserving_code(text: str) -> str:
        """
        Light cleanup that NEVER touches code/LaTeX content.
        - Collapse 3+ blank lines to 2 (outside code fences).
        - Strip image embeds (we don't keep images) but keep alt text.
        Indentation inside fenced blocks is preserved byte-for-byte.
        """
        # Remove markdown image embeds -> keep alt text only
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"[image: \1]", text)
        # Remove pymupdf4llm's "picture intentionally omitted" placeholders — figures
        # are extracted and linked via metadata separately, so these are just noise.
        text = re.sub(r"\*\*==>\s*picture.*?omitted\s*<==\*\*", "", text, flags=re.IGNORECASE)
        # Collapse excess blank lines but only when not inside a code fence
        out_lines = []
        in_fence = False
        blank_run = 0
        for line in text.split("\n"):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                out_lines.append(line)
                blank_run = 0
                continue
            if not in_fence and not line.strip():
                blank_run += 1
                if blank_run > 2:
                    continue
            else:
                blank_run = 0
            out_lines.append(line)
        return "\n".join(out_lines).strip()

    @staticmethod
    def _safe_int(v) -> int | None:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _update_stats(self, meta: dict):
        self.stats["chunks_total"] += 1
        d = meta.get("domain", "general")
        self.stats["chunks_by_domain"][d] = self.stats["chunks_by_domain"].get(d, 0) + 1
        c = meta.get("course_code", "unknown")
        self.stats["chunks_by_course"][c] = self.stats["chunks_by_course"].get(c, 0) + 1
        if meta.get("has_formula"):
            self.stats["formula_chunks"] += 1
        if meta.get("has_code"):
            self.stats["code_chunks"] += 1
        if meta.get("has_figure"):
            self.stats["chunks_with_figures"] += 1

    # ---- vault-wide ingest ----

    def ingest_vault(self, verbose: bool = True) -> Path:
        """Walk the vault, extract every PDF, stream results to the output JSONL."""
        pdfs = self.discover_pdfs()
        if verbose:
            log.info("Found %d PDF(s). OCR engine: %s", len(pdfs), self.ocr_engine or "NONE")

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        seen_ids: set[str] = set()
        dupes = 0
        with open(self.output_file, "w", encoding="utf-8") as out_f:
            for idx, pdf in enumerate(pdfs, 1):
                rel = pdf.relative_to(self.vault_path)
                if verbose:
                    log.info("[%d/%d] %s", idx, len(pdfs), rel)
                try:
                    chunks = self.load_pdf(pdf)
                except Exception as e:
                    log.warning("  failed: %s", e)
                    self.stats["pdfs_failed"] += 1
                    continue
                pdf_written = 0
                if self.force_domain or self.force_tags:
                    for ch in chunks:
                        m = ch.metadata if hasattr(ch, "metadata") else {}
                        if self.force_domain:
                            m["domain"] = self.force_domain
                        if self.force_tags:
                            have = m.get("tags") or []
                            if isinstance(have, str):
                                have = [t.strip() for t in have.split(",") if t.strip()]
                            m["tags"] = have + [t for t in self.force_tags
                                                if t not in have]
                for ch in chunks:
                    # Skip chunks whose doc_id already appeared this run. Two chunks
                    # can collide when their (source_file + first 500 chars) match
                    # (repeated headers, overlap windows). ChromaDB upsert rejects an
                    # intra-batch duplicate id, so we filter here at the source.
                    if ch.doc_id in seen_ids:
                        dupes += 1
                        continue
                    seen_ids.add(ch.doc_id)
                    out_f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
                    written += 1
                    pdf_written += 1
                if self.archive_processed and pdf_written:
                    out_f.flush()          # chunks are on disk before the PDF moves
                    self._archive_pdf(pdf)
        if dupes:
            log.info("Skipped %d duplicate-id chunk(s) at write time.", dupes)

        if verbose:
            self._print_stats(written)
        return self.output_file

    def _archive_pdf(self, pdf: Path) -> None:
        """Move a successfully-ingested PDF into <its folder>/_ingested/.
        Never fatal: a locked/undeletable file just stays put with a warning."""
        try:
            dest_dir = pdf.parent / "_ingested"
            dest_dir.mkdir(exist_ok=True)
            dest = dest_dir / pdf.name
            for i in itertools.count(2):
                if not dest.exists():
                    break
                dest = dest_dir / f"{pdf.stem} ({i}){pdf.suffix}"
            shutil.move(str(pdf), str(dest))
            log.info("  archived -> %s", dest.relative_to(self.vault_path))
        except OSError as e:
            log.warning("  could not archive %s (left in place): %s", pdf.name, e)

    def _print_stats(self, written: int):
        s = self.stats
        print(f"\n{'=' * 60}")
        print("  PDF INGESTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  PDFs found:        {s['pdfs_found']}")
        print(f"  PDFs processed:    {s['pdfs_processed']}")
        print(f"  PDFs failed:       {s['pdfs_failed']}")
        print(f"  Skipped (list):    {s['pdfs_skipped_list']}")
        print(f"  Skipped (book):    {s['pdfs_skipped_book']}")
        print(f"  Skipped (non-book):{s['pdfs_skipped_nonbook']}")
        print(f"  Skipped (include): {s['pdfs_skipped_include']}")
        print(f"  Skipped (excl):    {s['pdfs_skipped_excluded']}")
        print(f"  Skipped (folder):  {s['pdfs_skipped_folder']}")
        print(f"  Skipped (script):  {s['pdfs_skipped_script']}")
        print(f"  Pages total:       {s['pages_total']}")
        print(f"  Pages OCR'd:       {s['pages_ocr']}")
        print(f"  Scanned skipped:   {s['pages_scanned_skipped']}")
        print(f"  Chunks written:    {written}")
        print(f"  Formula chunks:    {s['formula_chunks']}")
        print(f"  Code chunks:       {s['code_chunks']}")
        print(f"  Figures extracted: {s['figures_extracted']}  "
              f"(in {s['chunks_with_figures']} chunks)")
        print(f"\n  Chunks by domain:")
        for d, n in sorted(s["chunks_by_domain"].items(), key=lambda x: -x[1]):
            print(f"    {d:.<28} {n}")
        print(f"\n  Chunks by course:")
        for c, n in sorted(s["chunks_by_course"].items(), key=lambda x: -x[1]):
            print(f"    {c:.<28} {n}")
        print(f"{'=' * 60}\n")
        print(f"  Output: {self.output_file}")
        print(f"  Next:   python main.py index --append\n")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config_loader import load_config

    ap = argparse.ArgumentParser(description="Ingest vault PDFs into RAG chunks.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--vault", default=None, help="Override vault path")
    ap.add_argument("--only-books", action="store_true",
                    help="Only process PDFs inside Books/ folders")
    ap.add_argument("--max-pages", type=int, default=None,
                    help="Cap pages per PDF (useful for huge books / testing)")
    ap.add_argument("--no-ocr", action="store_true", help="Disable OCR entirely")
    ap.add_argument("--skip-list", default=None,
                    help="Path to dedup_skiplist.json (overrides config pdf.skip_list_file)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    loader = PDFLoader.from_config(cfg)
    if args.vault:
        loader.vault_path = Path(args.vault)
    if args.only_books:
        loader.only_book_folders = True
    if args.max_pages:
        loader.max_pages_per_pdf = args.max_pages
    if args.no_ocr:
        loader.ocr_enabled = False
        loader.ocr_engine = None
    if args.skip_list:
        loader.skip_set = load_skip_set(Path(args.skip_list))

    loader.ingest_vault()


if __name__ == "__main__":
    main()
