"""
code_loader.py — Raw source-code ingestion for every language the notebook
loader does NOT cover.

WHY THIS EXISTS (P3, session 9)
  .py / .R / .Rmd / .ipynb already go through ipynb_loader (that's where the
  ast-based Python splitter lives). The gap was every OTHER language —
  JavaScript/TypeScript, SQL, Go, Java, C/C++, Rust, shell, Ruby, PHP, C#,
  Kotlin, Swift, Scala, Lua, … — which had no loader at all. This adds a
  dedicated lane so the author's own scripts in those languages become retrievable.

  Separate command (`ingest-code`) and JSONL (data/code_chunks.jsonl) keep the
  notebook and script corpora legible, but the internals reuse the same shared
  helpers pdf_loader/ipynb_loader use (course detection, size splitting, the
  context header, the Armenian script gate) so behavior is identical:
      {"doc_id": "<16hex>", "text": "<context-header + fenced code>", "metadata": {...}}
  doc_id uses the same deterministic scheme, so `index --append` upserts are
  idempotent.

SPLITTING (no Tree-sitter — E1's verdict was stdlib-only)
  A file that fits max_chunk becomes one language-fenced chunk. A larger file is
  split at TOP-LEVEL boundaries with a light, language-agnostic heuristic:
  unindented declaration lines (function/class/struct/func/fn/CREATE TABLE/…)
  and blank-line paragraph boundaries, greedily merged up to a char budget so
  small helpers share a chunk and no line is ever cut mid-way. A body that alone
  exceeds the budget falls through to the shared size-splitter (today's .R
  behavior) — never a crash, never a lost function.

DISCOVERY GUARDS (session-8 decision)
  Build/dependency dirs (node_modules, venv, dist, target, vendor, _Backups …)
  are always skipped so third-party code never enters the corpus. The vault's
  agent-project roots (Workspace1, Workspace2, …) are skipped BY DEFAULT and only
  ingested when an explicit --include-path scopes them in.

USAGE
    python main.py ingest-code                         # walk vault -> data/code_chunks.jsonl
    python main.py ingest-code --include-path "Capstone"
    python main.py index --append data/code_chunks.jsonl
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the vault parser's course knowledge + chunking — single source of truth,
# exactly like pdf_loader / ipynb_loader do.
from src.ingestion.obsidian_parser import (
    FOLDER_COURSE_MAP,
    DOMAIN_MAP,
    COURSE_MAP,
    split_large_chunk,
    build_context_header,
)
# Reuse the script gate so "skip Armenian" behaves identically everywhere.
from src.ingestion.pdf_loader import detect_script, should_skip_script
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

# extension -> markdown fence label (also stored as metadata `language`).
# .py/.r/.rmd/.ipynb are DELIBERATELY absent — the notebook loader owns them.
DEFAULT_LANGUAGE_MAP: dict[str, str] = {
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".sql": "sql",
    ".go": "go",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".rs": "rust",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".kt": "kotlin", ".kts": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".lua": "lua",
    ".pl": "perl", ".pm": "perl",
    ".jl": "julia",
    ".ps1": "powershell",
}

# Directories that must never be walked into (third-party / generated / backups).
# node_modules/vendor/site-packages = dependencies; dist/build/target/.next =
# build output; _Backups = timestamped skill-script copies (session-8).
_SKIP_DIRS = {
    ".git", ".hg", ".svn", ".obsidian", ".trash", ".ipynb_checkpoints",
    "node_modules", "bower_components", "vendor", "site-packages",
    "venv", ".venv", "env", ".env", "__pycache__",
    "dist", "build", "out", "target", ".next", ".nuxt", ".output",
    "_Backups", ".gradle", ".idea", ".vscode",
}

# Vault roots that host agent-project code (mostly duplicates / tooling) — skipped
# by default so they don't flood the coursework corpus, ingested only when an
# explicit --include-path scopes them in (session-8 --skip-roots parity).
_DEFAULT_SKIP_ROOTS = {
    "workspace1", "workspace2", "workspace3", "documents", "rag", "logs",
    "handoffs", "assets",
}

# Minified / generated files: single enormous lines or *.min.* — no retrieval
# value, huge chunks. Skipped by name pattern.
_GENERATED_RE = re.compile(r"\.(min|bundle|generated|lock)\.", re.IGNORECASE)

# Top-level declaration starts, across the common languages. Matched only on
# UNINDENTED lines, so it can't fire on nested code. Deliberately broad — a
# false positive just adds one more (safe) split boundary.
_DECL_RE = re.compile(
    r"^(?:export\s+|default\s+|public\s+|private\s+|protected\s+|internal\s+|"
    r"static\s+|final\s+|abstract\s+|open\s+|sealed\s+|override\s+|pub\s+|"
    r"async\s+|unsafe\s+|extern\s+|inline\s+|func\s+)*"
    r"(?:function|class|interface|enum|struct|trait|impl|record|fn|func|type|"
    r"module|namespace|package|def|sub|proc|procedure)\b"
    r"|^(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|\([^)]*\)\s*\{)",
    re.IGNORECASE,
)
_SQL_DECL_RE = re.compile(
    r"^\s*(?:CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SELECT|WITH|MERGE|GRANT|"
    r"TRUNCATE|REPLACE)\b",
    re.IGNORECASE,
)

_MIN_SCRIPT_SAMPLE = 12
_PIECE_SEP = "\n\n\x00piece\x00\n\n"


def _split_code_source(source: str, budget: int, min_size: int = 0,
                       sql: bool = False) -> list[str]:
    """
    Split source at top-level boundaries into <=budget-char segments.

    A boundary is an UNINDENTED line that either matches a declaration keyword
    (_DECL_RE, or _SQL_DECL_RE for SQL) or starts a new paragraph (follows a
    blank line). Segments between boundaries are greedily merged up to `budget`
    so small helpers share a chunk; a single segment larger than budget is left
    intact for the caller's size-splitter. Never cuts inside a line.
    """
    lines = source.split("\n")
    boundaries: set[int] = {0}
    prev_blank = True
    for i, ln in enumerate(lines):
        if not ln.strip():
            prev_blank = True
            continue
        unindented = ln[:1] not in (" ", "\t")
        if unindented and (_DECL_RE.match(ln)
                           or (sql and _SQL_DECL_RE.match(ln))
                           or prev_blank):
            boundaries.add(i)
        prev_blank = False

    cuts = sorted(boundaries)
    segments = ["\n".join(lines[a:b])
                for a, b in zip(cuts, cuts[1:] + [len(lines)])]

    merged: list[str] = []
    cur = ""
    for seg in segments:
        if cur and len(cur) + len(seg) + 1 > budget:
            merged.append(cur)
            cur = seg
        else:
            cur = f"{cur}\n{seg}" if cur else seg
    if cur.strip():
        if merged and len(cur.strip()) < min_size:
            merged[-1] = f"{merged[-1]}\n{cur}"
        else:
            merged.append(cur)
    return [m for m in merged if m.strip()]


@dataclass
class CodeChunk:
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            sig = f"{self.metadata.get('source_file', '')}::{self.text[:500]}"
            self.doc_id = hashlib.sha256(sig.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "text": self.text, "metadata": self.metadata}


class CodeLoader:
    def __init__(
        self,
        vault_path: Path,
        output_file: Path,
        min_chunk: int = 200,
        max_chunk: int = 3000,
        overlap: int = 150,
        skip_scripts: list[str] | None = None,
        language_map: dict[str, str] | None = None,
        exts: set[str] | None = None,
        include_path: str | None = None,
        exclude_path: str | None = None,
        skip_roots: set[str] | None = None,
        max_file_bytes: int = 1_500_000,
    ):
        self.vault_path = Path(vault_path)
        self.output_file = Path(output_file)
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.overlap = overlap
        self.skip_scripts = set(skip_scripts if skip_scripts is not None else ["armenian"])
        self.language_map = dict(language_map or DEFAULT_LANGUAGE_MAP)
        # exts = the subset to actually ingest (defaults to every mapped ext).
        self.exts = {e.lower() for e in exts} if exts else set(self.language_map)
        self.include_path = include_path.lower() if include_path else None
        self.exclude_path = exclude_path.lower() if exclude_path else None
        # Agent-project roots skipped by default; an explicit --include-path
        # scopes them back in (the user asked for that tree specifically).
        self.skip_roots = (set(skip_roots) if skip_roots is not None
                           else set(_DEFAULT_SKIP_ROOTS))
        self.max_file_bytes = max_file_bytes

        self.stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_failed": 0,
            "files_skipped_script": 0,
            "files_skipped_generated": 0,
            "files_skipped_large": 0,
            "by_ext": {},
            "by_language": {},
            "chunks_total": 0,
            "chunks_by_domain": {},
            "chunks_by_course": {},
        }

    @classmethod
    def from_config(cls, cfg: Config) -> "CodeLoader":
        vault = (cfg.get("code.vault_path")
                 or cfg.get("notebooks.vault_path")
                 or cfg.get("pdf.vault_path")
                 or cfg.get("parser.vault_path"))
        out = (cfg.path("code.output_file") if cfg.get("code.output_file")
               else cfg.project_root / "data" / "code_chunks.jsonl")
        lang_override = cfg.get("code.language_map", None) or {}
        language_map = {**DEFAULT_LANGUAGE_MAP,
                        **{str(k).lower(): str(v) for k, v in lang_override.items()}}
        exts = cfg.get("code.extensions", None)
        skip_roots = cfg.get("code.skip_roots", None)
        return cls(
            vault_path=Path(vault),
            output_file=out,
            min_chunk=cfg.get("code.min_chunk_size", cfg.get("parser.min_chunk_size", 200)),
            max_chunk=cfg.get("code.max_chunk_size", cfg.get("parser.max_chunk_size", 3000)),
            overlap=cfg.get("code.overlap_size", cfg.get("parser.overlap_size", 150)),
            skip_scripts=cfg.get("code.skip_scripts", cfg.get("pdf.skip_scripts", ["armenian"])),
            language_map=language_map,
            exts={str(e).lower() for e in exts} if exts else None,
            include_path=cfg.get("code.include_path"),
            exclude_path=cfg.get("code.exclude_path"),
            skip_roots={str(r).lower() for r in skip_roots} if skip_roots is not None else None,
            max_file_bytes=cfg.get("code.max_file_bytes", 1_500_000),
        )

    # ---- discovery ----

    def discover_files(self) -> list[Path]:
        found = []
        # skip_roots only apply when NOT explicitly scoping a path in
        apply_root_skip = self.include_path is None
        for f in self.vault_path.rglob("*"):
            if f.suffix.lower() not in self.exts:
                continue
            # Directories named like files exist in the vault (a real
            # "PSS2_Solutions.sql/" folder) — rglob returns them and open()
            # then dies with EACCES on Windows. Files only.
            if not f.is_file():
                continue
            if any(part in _SKIP_DIRS for part in f.parts):
                continue
            rel_posix = f.relative_to(self.vault_path).as_posix().lower()
            # include_files: exact-filename scope for file-scoped custom jobs
            # (set post-construction by main.py; None = no filter).
            if getattr(self, "include_files", None) \
                    and f.name.lower() not in self.include_files:
                continue
            if self.include_path and self.include_path not in rel_posix:
                continue
            if self.exclude_path and self.exclude_path in rel_posix:
                continue
            if _GENERATED_RE.search(f.name):
                self.stats["files_skipped_generated"] += 1
                continue
            if apply_root_skip:
                first = f.relative_to(self.vault_path).parts[0].lower()
                if first in self.skip_roots:
                    continue
            found.append(f)
        self.stats["files_found"] = len(found)
        return sorted(found)

    # ---- course detection (mirrors pdf_loader / ipynb_loader) ----

    def _detect_course(self, filepath: Path) -> dict:
        parts = list(filepath.relative_to(self.vault_path).parts)
        for part in parts:
            name = FOLDER_COURSE_MAP.get(part.lower().strip())
            if name:
                return {"course_code": name, "course_name": name,
                        "domain": DOMAIN_MAP.get(name, "general")}
        for part in parts:
            m = re.search(r"(CS|DS|ENGS|BSDS|ECON)\s*\d{2,3}", part, re.IGNORECASE)
            if m:
                code = re.sub(r"\s+", " ", re.sub(r"(\D)(\d)", r"\1 \2", m.group(0).upper())).strip()
                name = COURSE_MAP.get(code, code)
                return {"course_code": code, "course_name": name,
                        "domain": DOMAIN_MAP.get(name, "general")}
        return {"course_code": "unknown", "course_name": "unknown", "domain": "general"}

    # ---- rendering ----

    def _fence(self, source: str, lang: str) -> str:
        """Language-fence the source, splitting at top-level boundaries first
        when it can't fit one chunk (piece-separated so no segment is size-split
        through a function body)."""
        source = source.rstrip()
        if len(source) + len(lang) + 8 <= self.max_chunk:
            return f"```{lang}\n{source}\n```"
        budget = max(self.max_chunk - 200, 500)
        segs = _split_code_source(source, budget, self.min_chunk,
                                  sql=(lang == "sql"))
        return _PIECE_SEP.join(f"```{lang}\n{s}\n```" for s in segs)

    # ---- per-file ----

    def load_file(self, filepath: Path) -> list[CodeChunk]:
        rel_path = str(filepath.relative_to(self.vault_path))
        ext = filepath.suffix.lower()
        lang = self.language_map.get(ext, ext.lstrip(".") or "text")
        self.stats["by_ext"][ext] = self.stats["by_ext"].get(ext, 0) + 1

        try:
            if filepath.stat().st_size > self.max_file_bytes:
                self.stats["files_skipped_large"] += 1
                log.info("%s: skipped (%.1f MB > cap)", rel_path,
                         filepath.stat().st_size / 1e6)
                return []
        except OSError:
            pass

        try:
            raw_text = filepath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.warning("Cannot read %s: %s", rel_path, e)
            self.stats["files_failed"] += 1
            return []
        if not raw_text.strip():
            return []

        # Whole-file Armenian gate: a dominantly Armenian source (rare — comments
        # only) is dropped rather than embedding un-vectorizable text.
        if (self.skip_scripts and len(raw_text.strip()) >= _MIN_SCRIPT_SAMPLE
                and should_skip_script(detect_script(raw_text), self.skip_scripts)):
            log.info("%s: skipped (%s script)", rel_path, detect_script(raw_text))
            self.stats["files_skipped_script"] += 1
            return []

        md = self._fence(raw_text, lang)
        course_meta = self._detect_course(filepath)
        title = filepath.name                          # keep extension (foo.js)
        base_meta = {
            "source_file": rel_path,
            "filename": title,
            "file_type": "code",                       # broad type -> code lane
            "language": lang,                          # specific language label
            "vault_path": str(self.vault_path),
            "tags": [],
            "wikilinks": [],
            **course_meta,
            "heading": title,
            "heading_level": 1,
            "heading_path": title,
            "has_code": True,
        }
        context = build_context_header([title], base_meta)

        sub_chunks: list[str] = []
        for piece in md.split(_PIECE_SEP):
            sub_chunks.extend(split_large_chunk(piece, self.max_chunk, self.overlap))

        out: list[CodeChunk] = []
        for i, piece in enumerate(sub_chunks):
            if len(piece.strip()) < self.min_chunk:
                continue
            meta = {**base_meta}
            if len(sub_chunks) > 1:
                meta["chunk_part"] = f"{i + 1}/{len(sub_chunks)}"
            out.append(CodeChunk(text=context + piece, metadata=meta))
            self._update_stats(meta)
        if out:
            self.stats["files_processed"] += 1
            self.stats["by_language"][lang] = self.stats["by_language"].get(lang, 0) + 1
        return out

    def _update_stats(self, meta: dict):
        self.stats["chunks_total"] += 1
        d = meta.get("domain", "general")
        self.stats["chunks_by_domain"][d] = self.stats["chunks_by_domain"].get(d, 0) + 1
        c = meta.get("course_code", "unknown")
        self.stats["chunks_by_course"][c] = self.stats["chunks_by_course"].get(c, 0) + 1

    # ---- vault-wide ----

    def ingest_vault(self, verbose: bool = True) -> Path:
        files = self.discover_files()
        if verbose:
            log.info("Found %d code file(s) across %s.", len(files), sorted(self.exts))
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        seen_ids: set[str] = set()
        dupes = 0
        with open(self.output_file, "w", encoding="utf-8") as out_f:
            for idx, f in enumerate(files, 1):
                if verbose:
                    log.info("[%d/%d] %s", idx, len(files), f.relative_to(self.vault_path))
                for ch in self.load_file(f):
                    if ch.doc_id in seen_ids:
                        dupes += 1
                        continue
                    seen_ids.add(ch.doc_id)
                    out_f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
                    written += 1
        if dupes:
            log.info("Skipped %d duplicate-id chunk(s) at write time.", dupes)
        if verbose:
            self._print_stats(written)
        return self.output_file

    def _print_stats(self, written: int):
        s = self.stats
        print(f"\n{'=' * 60}")
        print("  CODE INGESTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Files found:         {s['files_found']}")
        print(f"  Files processed:     {s['files_processed']}")
        print(f"  Files failed:        {s['files_failed']}")
        print(f"  Skipped (script):    {s['files_skipped_script']}")
        print(f"  Skipped (generated): {s['files_skipped_generated']}")
        print(f"  Skipped (too large): {s['files_skipped_large']}")
        print(f"  Chunks written:      {written}")
        print(f"\n  Chunks by language:")
        for l, n in sorted(s["by_language"].items(), key=lambda x: -x[1]):
            print(f"    {l:.<28} {n}")
        print(f"\n  Chunks by domain:")
        for d, n in sorted(s["chunks_by_domain"].items(), key=lambda x: -x[1]):
            print(f"    {d:.<28} {n}")
        print(f"\n  Chunks by course:")
        for c, n in sorted(s["chunks_by_course"].items(), key=lambda x: -x[1]):
            print(f"    {c:.<28} {n}")
        print(f"{'=' * 60}")
        print(f"  Output: {self.output_file}")
        print(f"  Next:   python main.py index --append {self.output_file}\n")


def main():
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils.config_loader import load_config

    ap = argparse.ArgumentParser(description="Ingest raw source-code files into RAG chunks.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--vault", default=None, help="Override vault root")
    ap.add_argument("--output", default=None, help="Override output JSONL path")
    ap.add_argument("--include-path", default=None,
                    help="Only files whose path contains SUBSTR (also scopes agent "
                         "roots back in)")
    ap.add_argument("--exclude-path", default=None, help="Skip files whose path contains SUBSTR")
    ap.add_argument("--exts", default=None,
                    help="Comma-separated subset, e.g. '.js,.ts,.sql' (default: all mapped)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    loader = CodeLoader.from_config(cfg)
    if args.vault:
        loader.vault_path = Path(args.vault)
    if args.output:
        loader.output_file = Path(args.output)
    if args.include_path:
        loader.include_path = args.include_path.lower()
    if args.exclude_path:
        loader.exclude_path = args.exclude_path.lower()
    if args.exts:
        loader.exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}
    loader.ingest_vault()


if __name__ == "__main__":
    main()
