"""
ocr_vlm.py — VLM-based OCR for scanned pages (Baidu Unlimited-OCR et al).

Instead of classic OCR (Tesseract: character recognition, plain text out),
this renders a scanned page to an image and asks a vision-language DOCUMENT-
PARSING model for markdown. Baidu's Unlimited-OCR (June 2026, 3B, the
DeepSeek-OCR successor) is the reference target: its "document parsing."
prompt returns structured markdown — headings, tables, formulas — which is
exactly the shape pdf_loader's chunker already eats. That's the quality
argument over Tesseract on textbook scans.

The model is NOT loaded in-process. It is reached over any OpenAI-compatible
/v1/chat/completions endpoint that accepts image_url content, which keeps this
box's 16GB-RAM / CPU constraints out of the ingest process and lets the same
code hit any of:

  * SGLang / vLLM serving baidu/Unlimited-OCR on a real GPU (the reference
    stack wants CUDA 12.9 — see the HF card),
  * llama.cpp / LM Studio / Ollama serving one of its GGUF quantizations
    locally (the CPU-viable path on the author's machine; slow but overnight-able),
  * FreeLLMAPI (:3001) fronting a free hosted vision model (Llama 4 Scout,
    Gemini) — zero local setup, works today, rate-limited.

Config (pdf.vlm_ocr.*) picks the endpoint; pdf.ocr_engine: "vlm" turns it on.
Everything is fail-soft: an unreachable endpoint or a bad page falls back to
"page stays sparse" (or Tesseract, if pdf_loader resolved that instead) — an
ingest pass is never lost to the OCR layer, matching the existing contract.

Usage:
    vlm = VLMOCR.from_config(cfg)
    texts = vlm.ocr_pages(Path("book.pdf"), scanned_pages=[3, 4, 9])
    # -> {3: "…markdown…", 9: "…"}  (4 missing = that page failed)
"""
from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Optional

from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

# Model echo / wrapper noise worth stripping from parses.
_FENCE_RE = re.compile(r"^```(?:markdown|md)?\s*|\s*```$", re.IGNORECASE)
_IMG_TOKEN_RE = re.compile(r"<image>\s*", re.IGNORECASE)


class VLMOCR:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8100/v1",
        model: str = "Unlimited-OCR",
        api_key: str | None = None,
        prompt: str = "<image>document parsing.",
        dpi: int = 200,
        timeout: float = 300.0,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        max_pages_per_pdf: int | None = None,
        max_edge_px: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.prompt = prompt
        self.dpi = dpi
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_pages_per_pdf = max_pages_per_pdf
        # Cap on the rendered page's longest edge. DPI is relative to the
        # page's PHYSICAL size, so a large-format page can blow past the
        # vision encoder's memory budget at a DPI that is safe for smaller
        # pages (llama.cpp's SAM encode graph OOM-crashes an 8GB GPU
        # somewhere between ~1200px and ~1650px longest edge).
        self.max_edge_px = max_edge_px
        self._down = False   # endpoint declared dead for this run after a
                             # connection-level failure — skip remaining pages fast

    @classmethod
    def from_config(cls, cfg: Config) -> "VLMOCR":
        key_env = cfg.get("pdf.vlm_ocr.api_key_env", "OPENAI_API_KEY")
        return cls(
            base_url=cfg.get("pdf.vlm_ocr.base_url", "http://127.0.0.1:8100/v1"),
            model=cfg.get("pdf.vlm_ocr.model", "Unlimited-OCR"),
            api_key=cfg.secret(key_env) if key_env else None,
            prompt=cfg.get("pdf.vlm_ocr.prompt", "<image>document parsing."),
            dpi=int(cfg.get("pdf.vlm_ocr.dpi", 200)),
            timeout=float(cfg.get("pdf.vlm_ocr.timeout", 300)),
            max_tokens=int(cfg.get("pdf.vlm_ocr.max_tokens", 8192)),
            temperature=float(cfg.get("pdf.vlm_ocr.temperature", 0.0)),
            max_pages_per_pdf=cfg.get("pdf.vlm_ocr.max_pages_per_pdf", None),
            max_edge_px=cfg.get("pdf.vlm_ocr.max_edge_px", None),
        )

    # ---- endpoint ----

    def probe(self) -> bool:
        """Cheap reachability check (GET /models). Never raises."""
        import requests
        try:
            r = requests.get(f"{self.base_url}/models",
                             headers=self._headers(), timeout=5)
            return r.status_code < 500
        except Exception:
            return False

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def ocr_image(self, png_bytes: bytes) -> str:
        """
        One page image -> parsed markdown. Raises on transport/HTTP errors so
        the caller can count the page as failed; returns "" on an empty parse.
        """
        import requests

        b64 = base64.b64encode(png_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        }
        r = requests.post(f"{self.base_url}/chat/completions",
                          json=payload, headers=self._headers(),
                          timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        text = (((data.get("choices") or [{}])[0].get("message") or {})
                .get("content") or "")
        text = _IMG_TOKEN_RE.sub("", text)
        text = _FENCE_RE.sub("", text.strip()).strip()
        return text

    # ---- per-document driver ----

    def ocr_pages(self, filepath: Path, scanned_pages: list[int]) -> dict[int, str]:
        """
        Render each 0-based page in `scanned_pages` at self.dpi and OCR it.
        Returns {page_index: markdown} for the pages that succeeded; a missing
        key means that page failed (it stays sparse, exactly like the no-OCR
        path). A connection-level failure marks the endpoint down for the rest
        of this run so a 400-page book doesn't wait out 400 timeouts.
        """
        import pymupdf
        import requests

        if self._down or not scanned_pages:
            return {}

        pages = list(scanned_pages)
        if self.max_pages_per_pdf and len(pages) > self.max_pages_per_pdf:
            log.warning(
                "%s: %d scanned pages exceed vlm_ocr.max_pages_per_pdf=%d — "
                "OCRing the first %d, the rest stay sparse.",
                filepath.name, len(pages), self.max_pages_per_pdf,
                self.max_pages_per_pdf,
            )
            pages = pages[: self.max_pages_per_pdf]

        out: dict[int, str] = {}
        try:
            doc = pymupdf.open(filepath)
        except Exception as e:
            log.warning("VLM OCR: cannot reopen %s (%s)", filepath.name, e)
            return {}

        try:
            for pno in pages:
                try:
                    page = doc[pno]
                    zoom = self.dpi / 72
                    if self.max_edge_px:
                        edge = max(page.rect.width, page.rect.height) or 1.0
                        zoom = min(zoom, self.max_edge_px / edge)
                    mat = pymupdf.Matrix(zoom, zoom)
                    png = page.get_pixmap(matrix=mat).tobytes("png")
                except Exception as e:
                    log.warning("VLM OCR: render failed %s p.%d (%s)",
                                filepath.name, pno + 1, e)
                    continue
                try:
                    text = self.ocr_image(png)
                    if text:
                        out[pno] = text
                except requests.exceptions.ConnectionError as e:
                    log.warning(
                        "VLM OCR endpoint unreachable at %s (%s) — skipping the "
                        "remaining scanned pages this run. Is the vision server "
                        "up?", self.base_url, type(e).__name__,
                    )
                    self._down = True
                    break
                except Exception as e:
                    log.warning("VLM OCR failed on %s p.%d (%s: %s)",
                                filepath.name, pno + 1, type(e).__name__, e)
        finally:
            doc.close()
        return out
