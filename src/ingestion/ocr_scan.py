"""
ocr_scan.py — Which pages of a PDF actually need OCR?

Answers the question you would otherwise answer by scrolling a 700-page
textbook yourself: which pages carry no extractable text (scans, photographed
pages, image-only figures) and would therefore go into the index EMPTY unless
an OCR engine handles them.

This is a REPORT, not an ingest step. It opens the PDF read-only, measures the
extractable text per page, and returns page ranges — you then eyeball the few
suspicious ones in the preview and decide what to OCR. Nothing is written,
nothing is indexed, no OCR engine is contacted.

The threshold is the SAME `pdf.skip_scanned_threshold` the ingest path uses
(PDFLoader._find_scanned_pages), so a page reported here is exactly a page the
real ingest would treat as scanned — a report that disagreed with the ingester
would be worse than no report.

    from src.ingestion.ocr_scan import scan_pdf
    rep = scan_pdf(Path("book.pdf"), threshold=50)
    rep["needs_ocr_spec"]   # "12-18,241,690-712"  -> paste into --pages
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

# A page with some text but very little of it is usually a plate/figure page
# with a caption, or a chapter divider — worth flagging separately from a
# fully empty page, because those are the ones where OCR pays off least.
SPARSE_MULTIPLIER = 4


def _ranges(pages: list[int]) -> str:
    """[1,2,3,7,9,10] -> '1-3,7,9-10' (1-based, ready for --pages)."""
    if not pages:
        return ""
    out, start, prev = [], pages[0], pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
            continue
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = p
    out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(out)


def scan_pdf(path: Path, threshold: int = 50,
             sample_chars: int = 90) -> dict[str, Any]:
    """Report per-page extractable-text volume and which pages need OCR.

    Page numbers in the result are **1-based**, matching what a PDF viewer
    shows and what `--pages` expects — the loader works 0-based internally and
    that off-by-one is exactly the kind of thing that wastes an OCR pass.
    """
    import fitz  # PyMuPDF — already a dependency of pdf_loader

    doc = fitz.open(path)
    try:
        pages: list[dict[str, Any]] = []
        for pno in range(doc.page_count):
            try:
                txt = doc[pno].get_text("text") or ""
            except Exception as e:                    # a corrupt page
                log.warning("%s p%d: %s", path.name, pno + 1, e)
                txt = ""
            n = len(txt.strip())
            has_images = False
            try:
                has_images = bool(doc[pno].get_images(full=False))
            except Exception:
                pass
            if n < threshold:
                verdict = "needs_ocr"
            elif n < threshold * SPARSE_MULTIPLIER:
                verdict = "sparse"
            else:
                verdict = "ok"
            pages.append({
                "page": pno + 1,
                "chars": n,
                "has_images": has_images,
                "verdict": verdict,
                "sample": " ".join(txt.split())[:sample_chars],
            })
    finally:
        doc.close()

    need = [p["page"] for p in pages if p["verdict"] == "needs_ocr"]
    sparse = [p["page"] for p in pages if p["verdict"] == "sparse"]
    total = len(pages)
    return {
        "file": path.name,
        "page_count": total,
        "threshold": threshold,
        "needs_ocr": need,
        "needs_ocr_spec": _ranges(need),
        "sparse": sparse,
        "sparse_spec": _ranges(sparse),
        "n_needs_ocr": len(need),
        "n_sparse": len(sparse),
        "pct_needs_ocr": round(100.0 * len(need) / total, 1) if total else 0.0,
        # A born-digital PDF has ~0 flagged pages; a photographed book has
        # ~100%. The middle is the interesting case: a text PDF with inserted
        # scanned plates, where OCRing only those pages is the cheap win.
        "kind": ("born_digital" if not need else
                 "fully_scanned" if len(need) == total else "mixed"),
        "pages": pages,
    }
