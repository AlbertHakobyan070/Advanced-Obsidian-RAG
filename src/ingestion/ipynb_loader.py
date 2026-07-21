"""
ipynb_loader.py — Native code & notebook ingestion for Personal RAG.

WHY THIS EXISTS
  Notebooks exported to PDF lose code indentation, cell boundaries, and table
  structure — exactly the things you want preserved for "how did I solve X"
  retrieval. This reads native source directly and renders clean markdown
  (markdown cells verbatim, code fenced with the kernel language, text outputs
  optionally appended), then chunks it the SAME way pdf_loader does.

  Despite the filename, this handles a family of code/notebook formats:
      .ipynb  — Jupyter notebooks (cells → fenced markdown, text outputs kept)
      .py     — Python scripts (whole file fenced; split on `# %%` cell markers)
      .R      — R scripts (whole file fenced as ```r)
      .Rmd    — R Markdown (already markdown; light passthrough)

  Emits the identical JSONL shape as obsidian_parser.py and pdf_loader.py:
      {"doc_id": "<16hex>", "text": "<context-header + content>", "metadata": {...}}
  so the existing embedder/indexer need zero changes. doc_id uses the same
  deterministic scheme, so `index --append` upserts are idempotent.

ARMENIAN HANDLING (per-cell, not per-file)
  Code cells are kept (they're Latin/English — Python/R keywords, identifiers).
  Markdown prose and text outputs are script-detected PER CELL: a dominantly
  Armenian cell has its body replaced with a short placeholder rather than
  embedding garbage vectors (bge-small is English-only) or dropping the whole
  notebook. A file is skipped ENTIRELY only when it has no substantial code AND
  its prose is Armenian (a pure Armenian write-up — low value here).

FIGURES (Tier 1, opt-in via --save-figures, default OFF)
  Notebook image outputs (PNG base64) are decoded to
  data/notebook_figures/<notebook>/ and linked to the chunk via the SAME scalar
  metadata keys pdf_loader uses (has_figure, figure_count, figure_images=";"-joined
  paths), so app.py renders them with no new logic. The plot itself is not
  embedded — the surrounding code/markdown text is what makes the chunk findable.

USAGE
    python main.py ingest-notebooks                       # walk vault → data/ipynb_chunks.jsonl
    python main.py ingest-notebooks --save-figures        # also extract image outputs
    python main.py ingest-notebooks --no-outputs          # skip cell outputs
    # standalone equivalent:
    python -m src.ingestion.ipynb_loader
    # then:
    python main.py index --append data/ipynb_chunks.jsonl
"""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the vault parser's course knowledge + chunking — single source of truth,
# exactly like pdf_loader does.
from src.ingestion.obsidian_parser import (
    FOLDER_COURSE_MAP,
    DOMAIN_MAP,
    COURSE_MAP,
    split_large_chunk,
    build_context_header,
    apply_forced_meta,
)
# Reuse the script gate from pdf_loader so "skip Armenian" behaves identically.
from src.ingestion.pdf_loader import detect_script, should_skip_script
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

SUPPORTED_EXTS = {".ipynb", ".py", ".r", ".rmd"}
ARMENIAN_PLACEHOLDER = "[Armenian text omitted]"
# A cell/output shorter than this isn't worth script-detecting (e.g. a number,
# a one-word print) — too little signal, and never the source of Armenian noise.
_MIN_SCRIPT_SAMPLE = 12

# Hard chunk boundary between ast-split code segments. load_file() splits the
# rendered markdown on this BEFORE size-splitting, so a segment that fits the
# budget becomes exactly one chunk — split_large_chunk would otherwise cut at
# any blank line inside a function body once the file exceeds max_chunk.
_PIECE_SEP = "\n\n\x00piece\x00\n\n"


def _split_python_source(source: str, budget: int, min_size: int = 0) -> list[str]:
    """
    Split Python source at top-level def/class boundaries (stdlib ast — E1).

    Returns contiguous line-range segments covering the whole file: a segment
    starts at a top-level def/class (its first decorator, so decorators and
    docstrings stay attached) and runs until the next one. Adjacent segments
    are greedily merged up to `budget` chars, so small helpers share a chunk.
    A body is never cut — a function appears whole inside exactly one segment
    (unless that single def alone exceeds budget; the caller's size-splitter
    then handles it as before).

    Fallbacks keep old behavior: syntax errors or a file with no top-level
    defs return [source] unchanged. Lines are split on "\\n" only (project
    rule — source strings can contain U+2028/U+2029, which ast does not count
    as line breaks but str.splitlines() does).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [source]

    starts: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = node.lineno
            if node.decorator_list:
                first = min(first, min(d.lineno for d in node.decorator_list))
            starts.add(first - 1)                      # ast lineno is 1-based
    if not starts:
        return [source]

    lines = source.split("\n")
    cuts = sorted(starts)
    if not cuts or cuts[0] != 0:
        cuts = [0] + cuts
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
        # Don't strand a tiny tail (it would fall under min_chunk and be
        # dropped, losing a whole function) — fold it into the last group.
        if merged and len(cur.strip()) < min_size:
            merged[-1] = f"{merged[-1]}\n{cur}"
        else:
            merged.append(cur)
    return [m for m in merged if m.strip()]


@dataclass
class NBChunk:
    text: str
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            sig = f"{self.metadata.get('source_file', '')}::{self.text[:500]}"
            self.doc_id = hashlib.sha256(sig.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "text": self.text, "metadata": self.metadata}


def _placeholder_if_armenian(text: str, skip_scripts: set[str]) -> tuple[str, bool]:
    """Return (text_or_placeholder, was_placeholdered).

    Replaces a dominantly skip-listed-script block with a placeholder so we keep
    structure without embedding un-vectorizable text. Short/empty blocks pass
    through untouched (too little to detect reliably)."""
    if not skip_scripts or len(text.strip()) < _MIN_SCRIPT_SAMPLE:
        return text, False
    if should_skip_script(detect_script(text), skip_scripts):
        return ARMENIAN_PLACEHOLDER, True
    return text, False


class NotebookLoader:
    def __init__(
        self,
        vault_path: Path,
        output_file: Path,
        min_chunk: int = 200,
        max_chunk: int = 3000,
        overlap: int = 150,
        include_outputs: bool = True,
        max_output_chars: int = 2000,
        skip_scripts: list[str] | None = None,
        save_figures: bool = False,
        figures_dir: Path | None = None,
        exts: set[str] | None = None,
        include_path: str | None = None,
        include_files: set[str] | None = None,
    ):
        self.vault_path = Path(vault_path)
        self.output_file = Path(output_file)
        # Scoping (file-routable custom jobs; None = whole vault, the classic
        # behavior). include_path = a path substring; include_files = exact
        # filenames. Both mirror code_loader's scoping so the designer can route
        # individual .py/.R/.ipynb/.Rmd files just like it routes code.
        self.include_path = include_path
        self.include_files = include_files
        # Per-file / batch metadata overrides (inbox lane), stamped at write —
        # metadata only, doc_ids unaffected. Same shape as the other loaders.
        self.force_domain: str | None = None
        self.force_tags: list[str] = []
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.overlap = overlap
        self.include_outputs = include_outputs
        self.max_output_chars = max_output_chars
        self.skip_scripts = set(skip_scripts if skip_scripts is not None else ["armenian"])
        self.save_figures = save_figures
        self.figures_dir = Path(figures_dir) if figures_dir else Path("data/notebook_figures")
        self.exts = exts if exts is not None else set(SUPPORTED_EXTS)

        self.stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_failed": 0,
            "files_skipped_script": 0,
            "by_ext": {},
            "cells_total": 0,
            "cells_placeholdered": 0,
            "figures_saved": 0,
            "chunks_total": 0,
            "chunks_with_figures": 0,
            "chunks_by_domain": {},
            "chunks_by_course": {},
        }

    @classmethod
    def from_config(cls, cfg: Config) -> "NotebookLoader":
        vault = (cfg.get("notebooks.vault_path")
                 or cfg.get("pdf.vault_path")
                 or cfg.get("parser.vault_path"))
        out = (cfg.path("notebooks.output_file") if cfg.get("notebooks.output_file")
               else cfg.project_root / "data" / "ipynb_chunks.jsonl")
        figs = (cfg.path("notebooks.figures_dir") if cfg.get("notebooks.figures_dir")
                else cfg.project_root / "data" / "notebook_figures")
        exts = cfg.get("notebooks.extensions", None)
        return cls(
            vault_path=Path(vault),
            output_file=out,
            min_chunk=cfg.get("notebooks.min_chunk_size", cfg.get("parser.min_chunk_size", 200)),
            max_chunk=cfg.get("notebooks.max_chunk_size", cfg.get("parser.max_chunk_size", 3000)),
            overlap=cfg.get("notebooks.overlap_size", cfg.get("parser.overlap_size", 150)),
            include_outputs=cfg.get("notebooks.include_outputs", True),
            max_output_chars=cfg.get("notebooks.max_output_chars", 2000),
            skip_scripts=cfg.get("notebooks.skip_scripts", cfg.get("pdf.skip_scripts", ["armenian"])),
            save_figures=cfg.get("notebooks.save_figures", False),
            figures_dir=figs,
            exts={e.lower() for e in exts} if exts else None,
        )

    # ---- discovery ----

    def discover_files(self) -> list[Path]:
        # Standard vault metadata dirs + committed virtual environments / package
        # dirs that should never be indexed (library code, not user work).
        # _Backups holds timestamped copies of the Workspace1 skill scripts — same
        # functions under different paths would flood the code lane with
        # near-duplicates the write-time dedup can't catch (different doc_ids).
        skip_parts = {
            ".ipynb_checkpoints", ".obsidian", ".trash", ".git",
            "venv", ".venv", "env", ".env",
            "site-packages", "node_modules", "__pycache__",
            "_Backups",
        }
        inc = (self.include_path or "").replace("\\", "/").lower() or None
        found = []
        for f in self.vault_path.rglob("*"):
            if f.suffix.lower() not in self.exts:
                continue
            if any(part in skip_parts for part in f.parts):
                continue
            if self.include_files is not None and f.name not in self.include_files:
                continue
            if inc and inc not in f.relative_to(self.vault_path).as_posix().lower():
                continue
            found.append(f)
        self.stats["files_found"] = len(found)
        return sorted(found)

    # ---- course detection from path (mirrors pdf_loader._detect_course) ----

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

    # ---- shared helpers ----

    @staticmethod
    def _src(cell: dict) -> str:
        """Cell source can be a list of lines or a single string."""
        s = cell.get("source", "")
        return "".join(s) if isinstance(s, list) else (s or "")

    def _save_figure(self, b64: str, fmt: str, nb_stem: str, idx: int) -> str | None:
        """Decode a base64 image output to data/notebook_figures/<nb>/, return its
        path (posix, relative to project) or None on failure."""
        try:
            raw = base64.b64decode(b64)
        except Exception:
            return None
        ext = "png" if "png" in fmt else ("jpg" if "jpe" in fmt or "jpg" in fmt else "img")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", nb_stem)[:80]
        dest_dir = self.figures_dir / safe
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"fig_{idx:03d}.{ext}"
        try:
            dest.write_bytes(raw)
        except Exception:
            return None
        self.stats["figures_saved"] += 1
        return dest.as_posix()

    def _outputs_to_md(self, cell: dict, nb_stem: str, fig_counter: list[int],
                       fig_paths: list[str]) -> str:
        """Render text outputs; optionally save image outputs to disk. Drops
        inline base64 from the text. Caps length. Placeholders Armenian text."""
        if not self.include_outputs:
            return ""
        pieces = []
        for out in cell.get("outputs", []) or []:
            otype = out.get("output_type")
            if otype == "stream":
                txt = out.get("text", "")
                pieces.append("".join(txt) if isinstance(txt, list) else txt)
            elif otype in ("execute_result", "display_data"):
                data = out.get("data", {}) or {}
                # Image payloads: optionally persist, never embed the base64 text.
                if self.save_figures:
                    for mime, payload in data.items():
                        if mime.startswith("image/"):
                            b64 = "".join(payload) if isinstance(payload, list) else payload
                            fig_counter[0] += 1
                            p = self._save_figure(b64, mime, nb_stem, fig_counter[0])
                            if p:
                                fig_paths.append(p)
                txt = data.get("text/plain", "")
                pieces.append("".join(txt) if isinstance(txt, list) else txt)
            elif otype == "error":
                tb = out.get("traceback", []) or []
                clean = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in tb]
                pieces.append("\n".join(clean))
        text = "\n".join(p for p in pieces if p and p.strip())
        if not text:
            return ""
        text, was_ph = _placeholder_if_armenian(text, self.skip_scripts)
        if was_ph:
            self.stats["cells_placeholdered"] += 1
            return f"\n_Output:_ {ARMENIAN_PLACEHOLDER}\n"
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + "\n... [output truncated]"
        return f"\n_Output:_\n```\n{text}\n```\n"

    # ---- per-format rendering: each returns (markdown, n_cells, has_code) ----

    def _render_ipynb(self, raw_text: str, nb_stem: str,
                      fig_paths: list[str]) -> tuple[str, int, bool]:
        nb = json.loads(raw_text)
        lang = (
            nb.get("metadata", {}).get("kernelspec", {}).get("language")
            or nb.get("metadata", {}).get("language_info", {}).get("name")
            or "python"
        )
        cells = nb.get("cells", []) or []
        out_lines: list[str] = []
        has_code = False
        fig_counter = [0]
        for cell in cells:
            ctype = cell.get("cell_type")
            src = self._src(cell)
            if ctype == "markdown":
                if src.strip():
                    body, was_ph = _placeholder_if_armenian(src.rstrip(), self.skip_scripts)
                    if was_ph:
                        self.stats["cells_placeholdered"] += 1
                    out_lines.append(body)
            elif ctype == "code":
                if src.strip():
                    has_code = True
                    body = src.rstrip()
                    # Cells are natural boundaries already; ast-split only a
                    # cell that alone exceeds max_chunk (E1).
                    if lang.startswith("python"):
                        out_lines.append(self._fence_python(body))
                    else:
                        out_lines.append(f"```{lang}\n{body}\n```")
                    out_lines.append(
                        self._outputs_to_md(cell, nb_stem, fig_counter, fig_paths)
                    )
            elif ctype == "raw" and src.strip():
                out_lines.append(src.rstrip())
        md = "\n\n".join(line for line in out_lines if line and line.strip())
        return md, len(cells), has_code

    def _code_budget(self) -> int:
        """Char budget for one ast segment: max_chunk minus headroom for the
        context header and fence markers load_file adds around/atop it."""
        return max(self.max_chunk - 200, 500)

    def _fence_python(self, block: str) -> str:
        """Fence a python block; ast-split it first when it can't fit one
        chunk (E1). Multi-segment results are joined with the hard piece
        separator so no segment gets size-split through a function body."""
        if len(block) + 20 <= self.max_chunk:
            return f"```python\n{block}\n```"
        segs = _split_python_source(block, self._code_budget(), self.min_chunk)
        return _PIECE_SEP.join(f"```python\n{s}\n```" for s in segs)

    def _render_script(self, raw_text: str, lang: str) -> tuple[str, int, bool]:
        """Plain .py/.R: fence the whole file. Split on `# %%` cell markers (py)
        so a long script becomes several logical blocks the chunker can split on;
        blocks (or the whole file) that exceed max_chunk are further split at
        top-level def/class boundaries via ast (E1). .R is untouched — no
        stdlib parser, marginal corpus. Prose detection happens later on the
        rendered text via the gate."""
        text = raw_text.rstrip()
        if not text.strip():
            return "", 0, False
        # Jupyter/VSCode `# %%` percent-cells (python). Keep markers as separators.
        if lang == "python" and re.search(r"^\s*#\s*%%", text, re.MULTILINE):
            blocks = re.split(r"^\s*#\s*%%.*$", text, flags=re.MULTILINE)
            blocks = [b.strip() for b in blocks if b.strip()]
            md = "\n\n".join(self._fence_python(b) for b in blocks)
            return md, len(blocks), True
        if lang == "python":
            md = self._fence_python(text)
            return md, md.count(_PIECE_SEP) + 1, True
        md = f"```{lang}\n{text}\n```"
        return md, 1, True

    def _render_rmd(self, raw_text: str) -> tuple[str, int, bool]:
        """R Markdown is already markdown with ```{r} fences. Pass through;
        normalize the chunk fence label so retrieval sees ```r not ```{r}."""
        text = raw_text.rstrip()
        if not text.strip():
            return "", 0, False
        normalized = re.sub(r"```\{r[^}]*\}", "```r", text)
        has_code = "```r" in normalized
        return normalized, 1, has_code

    # ---- per-file ----

    def load_file(self, filepath: Path) -> list[NBChunk]:
        rel_path = str(filepath.relative_to(self.vault_path))
        ext = filepath.suffix.lower()
        self.stats["by_ext"][ext] = self.stats["by_ext"].get(ext, 0) + 1
        try:
            raw_text = filepath.read_text(encoding="utf-8")
        except Exception as e:
            log.warning("Cannot read %s: %s", rel_path, e)
            self.stats["files_failed"] += 1
            return []

        fig_paths: list[str] = []
        try:
            if ext == ".ipynb":
                md, n_cells, has_code = self._render_ipynb(raw_text, filepath.stem, fig_paths)
                file_type = "ipynb"
            elif ext == ".py":
                md, n_cells, has_code = self._render_script(raw_text, "python")
                file_type = "py"
            elif ext == ".r":
                md, n_cells, has_code = self._render_script(raw_text, "r")
                file_type = "r"
            elif ext == ".rmd":
                md, n_cells, has_code = self._render_rmd(raw_text)
                file_type = "rmd"
            else:
                return []
        except Exception as e:
            log.warning("Cannot parse %s: %s", rel_path, e)
            self.stats["files_failed"] += 1
            return []

        self.stats["cells_total"] += n_cells
        if not md.strip():
            return []

        # File-level skip: ONLY for no-code files that (a) actually contained
        # skip-listed-script prose we placeholdered, and (b) have essentially no
        # real content left after removing those placeholders. Code-bearing files
        # always survive. A short English-only note is NOT skipped here (it had no
        # placeholdering); it's handled normally and may simply yield 0 chunks if
        # below min_chunk. We can't judge by detect_script(md) directly because
        # the placeholder text itself is Latin.
        if self.skip_scripts and not has_code and ARMENIAN_PLACEHOLDER in md:
            residual = md.replace(ARMENIAN_PLACEHOLDER, "")
            residual = re.sub(r"[#>*`\-_=\s]+", "", residual)
            if len(residual) < 50:
                log.info("%s: skipped (no code + skip-listed prose only)", rel_path)
                self.stats["files_skipped_script"] += 1
                return []

        course_meta = self._detect_course(filepath)
        title = filepath.stem
        base_meta = {
            "source_file": rel_path,
            "filename": title,
            "file_type": file_type,
            "vault_path": str(self.vault_path),
            "tags": [],
            "wikilinks": [],
            **course_meta,
            "heading": title,
            "heading_level": 1,
            "heading_path": title,
            "has_code": has_code,
            "n_cells": n_cells,
        }
        # Figures are file-level here (notebook outputs aren't tied to a chunk
        # offset). Attach to every chunk of this file so the UI can surface them.
        if fig_paths:
            base_meta["has_figure"] = True
            base_meta["figure_count"] = len(fig_paths)
            base_meta["figure_images"] = ";".join(fig_paths)

        context = build_context_header([title], base_meta)
        # Hard piece boundaries first (ast-split code segments — E1), then the
        # usual size-splitting within each piece. No separator = old behavior.
        sub_chunks = []
        for piece in md.split(_PIECE_SEP):
            sub_chunks.extend(split_large_chunk(piece, self.max_chunk, self.overlap))

        out: list[NBChunk] = []
        for i, piece in enumerate(sub_chunks):
            if len(piece.strip()) < self.min_chunk:
                continue
            meta = {**base_meta}
            if len(sub_chunks) > 1:
                meta["chunk_part"] = f"{i + 1}/{len(sub_chunks)}"
            out.append(NBChunk(text=context + piece, metadata=meta))
            self._update_stats(meta)
        if out:
            self.stats["files_processed"] += 1
        return out

    def _update_stats(self, meta: dict):
        self.stats["chunks_total"] += 1
        d = meta.get("domain", "general")
        self.stats["chunks_by_domain"][d] = self.stats["chunks_by_domain"].get(d, 0) + 1
        c = meta.get("course_code", "unknown")
        self.stats["chunks_by_course"][c] = self.stats["chunks_by_course"].get(c, 0) + 1
        if meta.get("has_figure"):
            self.stats["chunks_with_figures"] += 1

    # ---- vault-wide ----

    def ingest_vault(self, verbose: bool = True) -> Path:
        files = self.discover_files()
        if verbose:
            log.info("Found %d file(s) across %s.", len(files), sorted(self.exts))
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
                    if self.force_domain or self.force_tags:
                        apply_forced_meta(ch.metadata, self.force_domain, self.force_tags)
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
        print("  NOTEBOOK / CODE INGESTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Files found:         {s['files_found']}")
        print(f"  Files processed:     {s['files_processed']}")
        print(f"  Files failed:        {s['files_failed']}")
        print(f"  Skipped (script):    {s['files_skipped_script']}")
        print(f"  Cells/blocks seen:   {s['cells_total']}")
        print(f"  Cells placeholdered: {s['cells_placeholdered']}")
        print(f"  Figures saved:       {s['figures_saved']}  (in {s['chunks_with_figures']} chunks)")
        print(f"  Chunks written:      {written}")
        print(f"\n  Files by type:")
        for e, n in sorted(s["by_ext"].items(), key=lambda x: -x[1]):
            print(f"    {e:.<28} {n}")
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

    ap = argparse.ArgumentParser(description="Ingest native notebooks & code into RAG chunks.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--vault", default=None, help="Override vault root")
    ap.add_argument("--output", default=None, help="Override output JSONL path")
    ap.add_argument("--no-outputs", action="store_true", help="Do not include cell outputs")
    ap.add_argument("--save-figures", action="store_true",
                    help="Decode notebook image outputs to data/notebook_figures/ "
                         "and link them via metadata (default OFF)")
    ap.add_argument("--exts", default=None,
                    help="Comma-separated subset, e.g. '.ipynb,.py' (default: all four)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    loader = NotebookLoader.from_config(cfg)
    if args.vault:
        loader.vault_path = Path(args.vault)
    if args.output:
        loader.output_file = Path(args.output)
    if args.no_outputs:
        loader.include_outputs = False
    if args.save_figures:
        loader.save_figures = True
    if args.exts:
        loader.exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}

    loader.ingest_vault()


if __name__ == "__main__":
    main()
