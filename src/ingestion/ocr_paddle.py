"""
ocr_paddle.py — PaddleOCR over HTTP, as a third OCR lane for scanned pages.

Why a third lane. The two existing ones sit at opposite extremes:

  * `tesseract` is a BINARY in this process's image. Cheap, always there, and
    weakest on the material that actually needs OCR — multi-column scans,
    tables, and anything with formulas.
  * `vlm` is a vision-language model asked to *parse the document*. Much better
    output, but it needs a served model, and on CPU it is minutes per page.

PaddleOCR sits in between: a real detection+recognition pipeline (far better
than Tesseract on skewed, multi-column, and non-Latin scans), small enough to
run on CPU at seconds per page, and — critically for the container build — able
to live in its OWN image. Its dependency tree (paddlepaddle) is large and
conflicts with nothing in the RAG image if it never enters it.

So this client speaks to a sidecar rather than importing paddle. That is the
same shape as the VLM lane, and it is what makes the OCR engine a container you
can stop when you are not ingesting:

    docker compose --profile ocr up -d paddleocr   # ingest day
    docker compose stop paddleocr                  # give the RAM back

`VLMOCR` already owns the part that is identical between the two — render each
scanned page at a DPI, drive the pages, mark the endpoint down after a
connection failure so a 400-page book does not wait out 400 timeouts. This
subclasses it and replaces only the wire call, so the two lanes cannot drift.

Config (`pdf.paddle_ocr.*`) picks the endpoint; `pdf.ocr_engine: paddle` turns
it on.

Usage:
    ocr = PaddleOCRClient.from_config(cfg)
    texts = ocr.ocr_pages(Path("book.pdf"), scanned_pages=[3, 4, 9])
    # -> {3: "…text…", 9: "…"}   (4 missing = that page failed)
"""
from __future__ import annotations

import base64

from src.ingestion.ocr_vlm import VLMOCR
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


class PaddleOCRClient(VLMOCR):
    """PaddleOCR sidecar client.

    The sidecar contract is deliberately tiny (see `paddleocr/server.py` in the
    Docker bundle):

        GET  /health -> {"ok": true, "engine": "paddleocr", "langs": [...]}
        POST /ocr    {"image_b64": "<png>", "lang": "en"}
                     -> {"text": "...", "lines": 12, "ms": 1830}

    It is NOT an OpenAI-compatible endpoint, which is the whole reason this
    class exists: reusing the VLM client with a different URL would send
    chat-completion payloads to a service that has no idea what they are.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8103",
                 lang: str = "en", dpi: int = 200, timeout: float = 120.0,
                 max_pages_per_pdf: int | None = None,
                 max_edge_px: int | None = 2000):
        super().__init__(
            base_url=base_url,
            model=f"paddleocr:{lang}",
            api_key=None,
            prompt="",                       # unused: this lane sends no prompt
            dpi=dpi,
            timeout=timeout,
            max_pages_per_pdf=max_pages_per_pdf,
            max_edge_px=max_edge_px,
        )
        self.lang = lang

    @classmethod
    def from_config(cls, cfg: Config) -> "PaddleOCRClient":
        def pick(key, default):
            v = cfg.get(f"pdf.paddle_ocr.{key}")
            return default if v is None else v

        return cls(
            base_url=str(pick("base_url", "http://127.0.0.1:8103")),
            lang=str(pick("lang", "en")),
            dpi=int(pick("dpi", 200)),
            timeout=float(pick("timeout", 120)),
            max_pages_per_pdf=pick("max_pages_per_pdf", None),
            max_edge_px=pick("max_edge_px", 2000),
        )

    # ---- endpoint ----

    def probe(self) -> bool:
        """Cheap reachability check. Never raises."""
        import requests
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code < 500
        except Exception:
            return False

    def ocr_image(self, png_bytes: bytes) -> str:
        """One page image -> recognised text.

        Raises on transport/HTTP errors so the inherited driver can count the
        page as failed and, for a connection error, stop early.
        """
        import requests

        payload = {
            "image_b64": base64.b64encode(png_bytes).decode("ascii"),
            "lang": self.lang,
        }
        r = requests.post(f"{self.base_url}/ocr", json=payload,
                          timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError(f"unexpected /ocr response type: {type(data).__name__}")
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return str(data.get("text") or "").strip()
