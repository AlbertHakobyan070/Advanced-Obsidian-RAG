"""Assembled studio pages -> PDF-lane chunks -> one NON-canonical chunk file.

The chunks are built by the PDF lane's own `_chunks_from_md`, so they carry the
same context header, page_start/page_end, chapter (from the PDF's TOC),
course/domain detection, has_formula/has_figure and figure_images metadata a
whole-book ingest would give them. Each also gets `ocr_source: ocr_studio`.

The output file is `data/ocrstudio_<slug>_<key>_chunks.jsonl`: its own file per
document, never a lane's canonical one (the loader-truncation trap), and named
*_chunks.jsonl so the sparse union picks it up beside the dense append.

`source_file` is the vault-relative path in the OS form, exactly as the PDF
lane writes it, so "replace" can find and remove that document's old chunks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.ocr_studio.pages import Store, assemble_page, atomic_write, doc_key, page_figures, store_root
from src.ocr_studio.render import IMAGE_EXTS

PAGE_SETS = ("done", "all")


def output_name(rel: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(rel.replace("\\", "/")).stem)[:40].strip("_")
    return f"ocrstudio_{stem or 'doc'}_{doc_key(rel)[:8]}_chunks.jsonl"


def build_chunks(cfg, vault: Path, rel: str, pages: str = "done",
                 store: Store | None = None) -> tuple[list, dict]:
    """(chunks, info) for one document from its saved studio pages. Writes nothing."""
    import pymupdf
    from src.ingestion.pdf_loader import PDFLoader

    if pages not in PAGE_SETS:
        raise ValueError(f"pages must be one of {PAGE_SETS}")
    vault = Path(vault)
    filepath = vault / rel
    store = store or Store(store_root(cfg))
    saved = store.pages(rel)
    working = set(store.page_set(rel) or saved)          # no set = every saved page
    selected = {n: p for n, p in saved.items()
                if n in working and (pages == "all" or p.get("status") == "done")}

    # Built directly, not via PDFLoader.from_config: that also routes the HF and
    # temp caches for a heavy ingest, which must not happen inside the console.
    loader = PDFLoader(
        vault_path=vault, output_file=Path("unused.jsonl"), scratch_dir=Path("."),
        min_chunk=cfg.get("pdf.min_chunk_size", cfg.get("parser.min_chunk_size", 200)),
        max_chunk=cfg.get("pdf.max_chunk_size", cfg.get("parser.max_chunk_size", 3000)),
        overlap=cfg.get("pdf.overlap_size", cfg.get("parser.overlap_size", 150)),
        ocr_enabled=False,
    )
    loader.chunking = str(cfg.get("pdf.chunking", cfg.get("parser.chunking", "heading"))).lower()

    with pymupdf.open(str(filepath)) as doc:
        page_count = doc.page_count
        toc = doc.get_toc() if filepath.suffix.lower() == ".pdf" else []
    md_chunks, figures = [], {}
    for n, page in selected.items():
        text = assemble_page(page)
        if text.strip():
            # remote_ocr_injected: these pages have no layout boxes, so the
            # chunker detects formulas from the LaTeX in the text instead.
            md_chunks.append({"text": text, "metadata": {"page_number": n},
                              "remote_ocr_injected": True})
        figures[n] = page_figures(page)

    rel_os = str(filepath.relative_to(vault))
    chunks = loader._chunks_from_md(md_chunks, rel_os, filepath, loader._detect_course(filepath),
                                    loader._build_page_to_chapter(toc, page_count), figures)
    is_image = filepath.suffix.lower() in IMAGE_EXTS
    for c in chunks:
        c.metadata["ocr_source"] = "ocr_studio"
        if is_image:
            c.metadata["file_type"] = "image"
    return chunks, {"pages_saved": len(saved), "pages_selected": len(selected),
                    "pages_with_text": len(md_chunks), "chunks": len(chunks),
                    "source_file": rel_os}


def write_chunks(chunks: list, out: Path) -> Path:
    data = "".join(json.dumps(c.to_dict(), ensure_ascii=False) + "\n" for c in chunks)
    atomic_write(Path(out), data.encode("utf-8"))
    return Path(out)
