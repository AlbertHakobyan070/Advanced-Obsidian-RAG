"""One call shape over the OCR engines this repo already has.

    run(cfg, engine, png, lang) -> {"text", "format", "confidence", "lines", "ms"}

`lines` are [{"text", "bbox": [x0, y0, x1, y1] in the CROP's pixels, "conf"}].

The engine is named explicitly per region and there is NO fallback here, unlike
the ingest lane's ladder (pdf_loader._resolve_ocr_engine): an engine that is not
configured, not reachable or not able to read the language raises EngineError on
that region, and the region shows the error. Quietly answering with a different
engine would put text under a label that did not produce it.

The clients are the ingest lane's own (VLMOCR, PaddleOCRClient), built from the
same config keys, so the studio and a whole-book ingest cannot drift apart.
Tesseract runs in-process through PyMuPDF, the same route web_import uses.
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

from PIL import Image

ENGINES = {
    "tesseract": "Tesseract (in-process, PyMuPDF)",
    "paddle": "PaddleOCR sidecar, PP-OCR (plain text)",
    "paddle-vl": "PaddleOCR sidecar, VL (markdown, tables, LaTeX)",
    "vlm": "VLM-OCR endpoint (markdown)",
}


class EngineError(RuntimeError):
    """The chosen engine cannot OCR this region; the message says why."""


def tessdata_dir() -> Path | None:
    """The tessdata folder Tesseract would use, or None. Uses the ingest lane's
    own lookup, which honours TESSDATA_PREFIX (rag.bat sets it)."""
    from src.ingestion.pdf_loader import _ensure_tessdata
    if _ensure_tessdata():
        return Path(os.environ["TESSDATA_PREFIX"])
    return None


def tesseract_langs() -> list[str]:
    td = tessdata_dir()
    if td is None:
        return []
    return sorted(p.stem for p in td.glob("*.traineddata") if p.stem != "osd")


def _fit(png: bytes, max_edge) -> tuple[bytes, float]:
    """Downscale so the longest edge is <= max_edge. Returns (png, factor)."""
    if not max_edge:
        return png, 1.0
    img = Image.open(io.BytesIO(png))
    edge = max(img.size)
    if edge <= int(max_edge):
        return png, 1.0
    f = int(max_edge) / edge
    img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), f


def _http_error(name: str, url: str, e: Exception) -> EngineError:
    import requests
    if isinstance(e, requests.exceptions.ConnectionError):
        return EngineError(f"{name} is unreachable at {url} — is it running?")
    if isinstance(e, requests.exceptions.Timeout):
        return EngineError(f"{name} at {url} timed out")
    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
        try:
            detail = e.response.json().get("error") or e.response.text
        except ValueError:
            detail = e.response.text
        return EngineError(f"{name} answered HTTP {e.response.status_code}: {str(detail)[:300]}")
    return EngineError(f"{name} failed: {type(e).__name__}: {e}")


def ocr_tesseract(png: bytes, lang: str) -> dict:
    import pymupdf
    td = tessdata_dir()
    if td is None:
        raise EngineError("Tesseract's tessdata folder was not found — set TESSDATA_PREFIX "
                          "to it (rag.bat does)")
    lang = (lang or "eng").strip()
    missing = [code for code in lang.split("+") if not (td / f"{code}.traineddata").is_file()]
    if missing:
        raise EngineError(f"Tesseract has no language {'+'.join(missing)!r} in {td} "
                          f"(installed: {', '.join(tesseract_langs()) or 'none'})")
    pix = pymupdf.Pixmap(png)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    pix.set_dpi(300, 300)
    try:
        doc = pymupdf.open("pdf", pix.pdfocr_tobytes(language=lang, tessdata=str(td)))
    except Exception as e:
        raise EngineError(f"Tesseract failed: {type(e).__name__}: {e}") from e
    with doc:
        page = doc[0]
        scale = pix.width / (page.rect.width or 1.0)
        lines = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                if text:
                    lines.append({"text": text, "conf": None,
                                  "bbox": [round(v * scale, 1) for v in line["bbox"]]})
        return {"text": page.get_text("text").strip(), "format": "plain",
                "confidence": None, "lines": lines}


def ocr_paddle(cfg, png: bytes, lang: str, pipeline: str) -> dict:
    if not (cfg.get("pdf.paddle_ocr") or {}):
        raise EngineError("the PaddleOCR sidecar is not configured (pdf.paddle_ocr)")
    from src.ingestion.ocr_paddle import PaddleOCRClient
    client = PaddleOCRClient.from_config(cfg)
    if lang:
        client.lang = lang
    client.pipeline = pipeline
    png, f = _fit(png, client.max_edge_px)
    try:
        data = client.ocr_response(png, boxes=True)
    except Exception as e:
        raise _http_error(f"the PaddleOCR sidecar ({pipeline})", client.base_url, e) from e
    lines = []
    for b in data.get("boxes") or []:
        box = b.get("box") or []
        if len(box) == 4:
            lines.append({"text": str(b.get("text") or ""), "conf": b.get("score"),
                          "bbox": [round(v / f, 1) for v in box]})
    scores = [ln["conf"] for ln in lines if isinstance(ln["conf"], (int, float))]
    conf = data.get("confidence")
    if conf is None and scores:
        conf = sum(scores) / len(scores)
    return {"text": str(data.get("text") or "").strip(),
            "format": "markdown" if pipeline == "vl" else "plain",
            "confidence": round(float(conf), 4) if conf is not None else None,
            "lines": lines}


def ocr_vlm(cfg, png: bytes) -> dict:
    if not (cfg.get("pdf.vlm_ocr") or {}):
        raise EngineError("no VLM-OCR endpoint is configured (pdf.vlm_ocr)")
    from src.ingestion.ocr_vlm import VLMOCR
    try:
        client = VLMOCR.from_config(cfg)
    except ValueError as e:                    # a preset name not in the table
        raise EngineError(str(e)) from e
    png, _ = _fit(png, client.max_edge_px)
    try:
        text = client.ocr_image(png)
    except Exception as e:
        raise _http_error(f"the VLM-OCR endpoint ({client.model})", client.base_url, e) from e
    return {"text": text, "format": "markdown", "confidence": None, "lines": []}


def run(cfg, engine: str, png: bytes, lang: str = "") -> dict:
    """OCR one image with the named engine. Raises EngineError, never falls back."""
    t0 = time.time()
    if engine == "tesseract":
        out = ocr_tesseract(png, lang)
    elif engine == "paddle":
        out = ocr_paddle(cfg, png, lang, "ocr")
    elif engine == "paddle-vl":
        out = ocr_paddle(cfg, png, "", "vl")   # one multilingual model: no language
    elif engine == "vlm":
        out = ocr_vlm(cfg, png)
    else:
        raise EngineError(f"unknown engine {engine!r}; expected one of {sorted(ENGINES)}")
    out["ms"] = int((time.time() - t0) * 1000)
    return out


# ---- layout proposals -------------------------------------------------------
#
# Proposals are ordinary editable boxes in page points; nothing is saved until
# the operator saves the page. Two sources:
#
#   layout  pymupdf-layout (already installed, the model pymupdf4llm's
#           page_boxes come from). A page without a text layer is OCR'd by
#           Tesseract first — that is how pymupdf4llm reads a scan — so that
#           source needs Tesseract for scans.
#   paddle  the sidecar's detected text lines (return_boxes), merged into
#           blocks. Text only: the "ocr" pipeline detects lines, not figures.

_LAYOUT_TYPES = {"picture": "figure", "table": "table", "formula": "formula",
                 "page-header": "ignore", "page-footer": "ignore"}
PROPOSAL_SOURCES = ("layout", "paddle")


def _propose_layout(path: Path, page: int, scanned: bool, lang: str) -> list[dict]:
    import pymupdf4llm
    md = pymupdf4llm.to_markdown(str(path), pages=[page - 1], page_chunks=True,
                                 show_progress=False, use_ocr=scanned,
                                 ocr_language=lang or "eng", write_images=False,
                                 embed_images=False)
    boxes = (md[0].get("page_boxes") if md else None) or []
    return [{"type": _LAYOUT_TYPES.get(b.get("class"), "text"),
             "bbox": [float(v) for v in b["bbox"]], "hint": b.get("class")}
            for b in boxes if b.get("bbox")]


def _merge_lines(lines: list[dict]) -> list[list[float]]:
    """Text-line boxes -> paragraph-ish blocks: a line joins the block above it
    when the vertical gap is under ~one line height and the two overlap
    horizontally."""
    blocks: list[list[float]] = []
    for ln in sorted(lines, key=lambda ln: (ln["bbox"][1], ln["bbox"][0])):
        x0, y0, x1, y1 = ln["bbox"]
        h = max(1.0, y1 - y0)
        for b in reversed(blocks):
            if y0 - b[3] < 0.9 * h and min(x1, b[2]) > max(x0, b[0]):
                b[:] = [min(b[0], x0), min(b[1], y0), max(b[2], x1), max(b[3], y1)]
                break
        else:
            blocks.append([x0, y0, x1, y1])
    return blocks


def _propose_paddle(cfg, path: Path, page: int, lang: str) -> list[dict]:
    from src.ocr_studio import render
    dpi = int(cfg.get("pdf.paddle_ocr.dpi") or 200)
    img, scale = render.render(path, page, dpi, cfg.get("pdf.paddle_ocr.max_edge_px"))
    out = ocr_paddle(cfg, render.to_png(img), lang, "ocr")
    if not out["lines"]:
        raise EngineError("the sidecar returned no line boxes — it predates return_boxes "
                          "(rebuild it from ocr-sidecar/) or found no text")
    return [{"type": "text", "bbox": [round(v / scale, 1) for v in b], "hint": "paddle-lines"}
            for b in _merge_lines(out["lines"])]


def propose(cfg, path: Path, page: int, source: str, scanned: bool, lang: str = "") -> list[dict]:
    if source == "layout":
        return _propose_layout(path, page, scanned, lang)
    if source == "paddle":
        return _propose_paddle(cfg, path, page, lang)
    raise ValueError(f"source must be one of {PROPOSAL_SOURCES}")


def status(cfg) -> dict:
    """What each engine can do right now: availability, its languages, why not.

    The language lists are what the region picker offers, so a region can only
    ask an engine for a language it actually has. Never raises."""
    tl = tesseract_langs()
    out = {"tesseract": {"label": ENGINES["tesseract"], "available": bool(tl), "langs": tl,
                         "error": None if tl else "tessdata not found (set TESSDATA_PREFIX)"}}

    # The two remote probes run side by side: on Windows a refused localhost
    # port takes ~2 s to fail, and the tab should not wait for both in turn.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as ex:
        paddle_f, vlm_f = ex.submit(_paddle_status, cfg), ex.submit(_vlm_status, cfg)
        out["paddle"], out["paddle-vl"] = paddle_f.result()
        out["vlm"] = vlm_f.result()
    return out


def _paddle_status(cfg) -> tuple[dict, dict]:
    import requests
    paddle = {"label": ENGINES["paddle"], "available": False, "langs": [], "error": None}
    vl = {"label": ENGINES["paddle-vl"], "available": False, "langs": [], "error": None}
    if not (cfg.get("pdf.paddle_ocr") or {}):
        paddle["error"] = vl["error"] = "not configured (pdf.paddle_ocr)"
        return paddle, vl
    url = str(cfg.get("pdf.paddle_ocr.base_url") or "http://127.0.0.1:8103").rstrip("/")
    try:
        r = requests.get(f"{url}/health", timeout=3)
        body = r.json() if r.content else {}
    except Exception as e:
        paddle["error"] = vl["error"] = f"sidecar unreachable at {url} ({type(e).__name__})"
        return paddle, vl
    if r.status_code >= 400 or not body.get("engine_importable", True):
        paddle["error"] = vl["error"] = (
            f"sidecar at {url} is up but cannot OCR: "
            f"{body.get('engine_import_error') or 'HTTP %d' % r.status_code}")
        return paddle, vl
    paddle.update(available=True, langs=list(body.get("langs") or []))
    v = (body.get("pipelines") or {}).get("vl") or {}
    vl["available"] = bool(v.get("available"))
    vl["error"] = None if vl["available"] else (v.get("reason") or "this sidecar has no VL pipeline")
    return paddle, vl


def _vlm_status(cfg) -> dict:
    vlm = {"label": ENGINES["vlm"], "available": False, "langs": [], "error": None, "model": None}
    if not (cfg.get("pdf.vlm_ocr") or {}):
        vlm["error"] = "not configured (pdf.vlm_ocr)"
        return vlm
    try:
        from src.ingestion.ocr_vlm import VLMOCR
        client = VLMOCR.from_config(cfg)
        vlm["model"] = client.model
        vlm["available"] = client.probe()
        if not vlm["available"]:
            vlm["error"] = f"endpoint unreachable at {client.base_url}"
    except Exception as e:
        vlm["error"] = f"{type(e).__name__}: {e}"
    return vlm
