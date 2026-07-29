"""
PaddleOCR sidecar — one endpoint, one job: image bytes in, text out.

Why this is a separate container and not an OCR engine inside the RAG image:

  * paddlepaddle plus the PaddleOCR model set is bigger than the entire rest of
    the RAG image, and it is needed only while INGESTING scanned PDFs. Keeping
    it out means the query path does not carry it in RAM or on disk.
  * it can therefore be stopped and started independently:
        docker compose --profile ocr up -d paddleocr
        docker compose stop paddleocr
    which is the point — a laptop running the whole stack wants that memory
    back once the scanned books are in the index.

The contract is deliberately tiny and NOT OpenAI-shaped, because this is not a
chat model and pretending otherwise would invite the wrong client:

    GET  /health -> {"ok": true, "engine": "paddleocr", "langs": [...],
                     "engine_importable": true, "engine_import_error": null,
                     "device": "gpu:0", "paddleocr_version": "3.7.0",
                     "loaded": ["en"]}          # langs whose engine is BUILT
                 -> 503 with the same body when the engine cannot import
    POST /ocr    {"image_b64": "<base64 png>", "lang": "en"}
                 -> {"text": "...", "lines": 12, "ms": 1830, "device": "gpu:0"}

Errors are returned as HTTP 4xx/5xx with a JSON `error` field. The client
(src/ingestion/ocr_paddle.py) counts a failed page as "stays sparse", exactly
like every other OCR lane, so a bad page never costs the whole ingest.

ONE FILE, TWO IMAGES. Dockerfile pins PaddleOCR 2.9.1 on CPU (the portable
default, and what the Mac bundle ships); Dockerfile.gpu pins 3.7.0 on CUDA.
Their APIs differ in both directions — the constructor keywords and the result
shape — so this file detects the installed major rather than being forked. A
fork would have been two files that drift, which is the failure this codebase
keeps paying for elsewhere.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import subprocess
import sys
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paddleocr-sidecar")

DEFAULT_LANG = os.environ.get("PADDLE_OCR_LANG", "en")

# "cpu", "gpu", or "gpu:N". Only the GPU image sets it; the CPU image leaves it
# empty and lets PaddleOCR pick, which keeps the portable build free of any
# device wiring it cannot honour.
#
# N is the index INSIDE the container. When compose reserves a specific card
# with device_ids, that card is renumbered to 0 here whatever its host index —
# so "gpu:0" is normally right even when the host card is #1.
DEFAULT_DEVICE = (os.environ.get("PADDLE_OCR_DEVICE") or "").strip()

app = FastAPI(title="PaddleOCR sidecar", version="1.0.0")


def _paddleocr_version() -> str | None:
    """Installed PaddleOCR version string, or None if it cannot be read."""
    try:
        from importlib.metadata import version
        return str(version("paddleocr"))
    except Exception:
        return None


def _paddleocr_major() -> int:
    """Installed PaddleOCR major version, or 0 when it cannot be determined.

    The 2.x and 3.x lines disagree about both the constructor keywords and the
    result shape, so everything below branches on this rather than guessing
    from whichever attribute happens to exist.
    """
    try:
        from importlib.metadata import version
        return int(str(version("paddleocr")).split(".")[0])
    except Exception:
        try:
            import paddleocr
            return int(str(getattr(paddleocr, "__version__", "0")).split(".")[0])
        except Exception:
            return 0


# One engine per language, built on first use. Loading is slow (weights come off
# disk and are JIT-compiled), so it happens once and is then reused; a lock
# keeps two concurrent first-requests from building the same engine twice and
# doubling peak memory on a machine that is already tight.
_ENGINES: dict[str, object] = {}
_LOCK = threading.Lock()


def _engine(lang: str):
    with _LOCK:
        if lang not in _ENGINES:
            from paddleocr import PaddleOCR
            major = _paddleocr_major()
            log.info("loading PaddleOCR %s.x for lang=%s device=%s "
                     "(first call is slow)", major or "?", lang,
                     DEFAULT_DEVICE or "auto")
            t0 = time.time()

            # Correcting pages scanned upside down or 90 degrees off is the
            # single biggest quality difference against a naive OCR call, and
            # it is common in phone-photographed material. Both lines can do
            # it; they spell it differently, and passing the wrong spelling is
            # a TypeError, not a warning.
            if major >= 3:
                kwargs = {"lang": lang, "use_textline_orientation": True}
                # 3.x takes the device as a constructor argument. Passing it
                # explicitly rather than relying on auto-detection means a GPU
                # image that cannot reach its card fails loudly here instead of
                # silently running on CPU at a tenth of the speed.
                if DEFAULT_DEVICE:
                    kwargs["device"] = DEFAULT_DEVICE
            else:
                # 2.x: show_log exists here and is gone in 3.x.
                kwargs = {"lang": lang, "use_angle_cls": True,
                          "show_log": False}
                if DEFAULT_DEVICE:
                    kwargs["use_gpu"] = DEFAULT_DEVICE.startswith("gpu")

            _ENGINES[lang] = PaddleOCR(**kwargs)
            log.info("loaded in %.1fs", time.time() - t0)
        return _ENGINES[lang]


def _run_ocr(engine, image):
    """Recognise one image, returning the raw per-line texts.

    The two majors return genuinely different structures:

      2.x  [[ [box, (text, confidence)], ... ]]   nested tuples, one list per
                                                  image, [[None]] for a blank
                                                  page
      3.x  [ {"rec_texts": [...], "rec_scores": [...], ...} ]   one dict-like
                                                  result per image

    Both are read defensively, so a point release that reshuffles a key returns
    a sparse page rather than a 500.
    """
    if _paddleocr_major() >= 3:
        # predict() is the 3.x entry point; ocr() survives as a deprecated
        # alias that emits a warning on every call.
        result = engine.predict(image)
        lines: list[str] = []
        for page in (result or []):
            texts = None
            if isinstance(page, dict) or hasattr(page, "get"):
                texts = page.get("rec_texts")
            if texts is None:
                texts = getattr(page, "rec_texts", None)
            for text in (texts or []):
                if text and str(text).strip():
                    lines.append(str(text).strip())
        return lines

    result = engine.ocr(image, cls=True)
    lines = []
    for page in (result or []):
        for entry in (page or []):
            try:
                text = entry[1][0]
            except (TypeError, IndexError):
                continue
            if text and str(text).strip():
                lines.append(str(text).strip())
    return lines


# Whether `import paddle` works AT ALL in this image. Checked once, then cached
# for the life of the container: an import that works does not stop working,
# and one that fails needs a rebuild, not a retry.
_IMPORT_OK: bool | None = None
_IMPORT_ERR: str | None = None
_IMPORT_LOCK = threading.Lock()


def _check_importable() -> tuple[bool, str | None]:
    """Can this container's OCR engine be imported? Runs in a SUBPROCESS.

    /health used to answer 200 without ever touching paddle, so the one thing
    that actually breaks this image — an import-time failure of a large C++
    extension — was invisible until the first real page came in as a 500, with
    the compose healthcheck reporting "healthy" the whole time.

    The child process is not paranoia, it is the only shape that works: those
    failures do not all raise. A missing libgomp.so.1 raises ImportError, but a
    paddlepaddle built against a different glibc calls abort() inside
    libpaddle.so, and no `except` can catch SIGABRT. Checking in-process would
    mean a health probe that KILLS the sidecar it is checking — and with
    `restart: unless-stopped`, a container that crash-loops on every probe. A
    child turns both shapes into an exit code and a line of stderr.

    It also keeps paddle's ~250MB out of the server process until something
    actually asks for OCR, which is the whole reason this sidecar is separate.

    Import only: no PaddleOCR(...), so no engine build and no weight download
    (that is ~130s on first run and belongs on the first real page, not here).
    """
    global _IMPORT_OK, _IMPORT_ERR
    with _IMPORT_LOCK:
        if _IMPORT_OK is None:
            t0 = time.time()
            try:
                # Import in the SAME ORDER as the /ocr path: numpy and PIL
                # first, then paddleocr. That order is not cosmetic. A cold
                # `import paddle, paddleocr` raises "zlib.error: Error -2 while
                # decompressing data" out of pyclipper's extension, while the
                # identical import after numpy and PIL are loaded succeeds and
                # goes on to OCR a page correctly. A probe that used the cold
                # order reported engine_importable:false on a container that
                # could transcribe perfectly — a false negative is just the
                # old lie with the sign flipped, and this endpoint exists to
                # answer "can this container OCR", not "does a different
                # import order happen to work".
                proc = subprocess.run(
                    [sys.executable, "-c",
                     "import numpy, PIL.Image; import paddle, paddleocr"],
                    capture_output=True, text=True, timeout=180)
            except Exception as e:
                # Never let the probe itself be the outage: report it as "not
                # importable" with the reason instead of 500ing /health.
                _IMPORT_OK, _IMPORT_ERR = False, f"{type(e).__name__}: {e}"
            else:
                _IMPORT_OK = proc.returncode == 0
                if not _IMPORT_OK:
                    # The last stderr line is the useful one: for an ImportError
                    # it is the exception, for a heap corruption it is glibc's
                    # message. A killed process may print nothing at all, so
                    # name the signal — "exit -11" alone tells an operator
                    # nothing, "SIGSEGV" says the native library is the problem.
                    tail = [ln.strip() for ln in (proc.stderr or proc.stdout or
                                                  "").splitlines() if ln.strip()]
                    if proc.returncode < 0:
                        try:
                            import signal
                            how = signal.Signals(-proc.returncode).name
                        except ValueError:
                            how = f"signal {-proc.returncode}"
                        how = f"the import CRASHED ({how})"
                    else:
                        how = f"exit {proc.returncode}"
                    _IMPORT_ERR = f"{tail[-1] if tail else 'no output'} — {how}"
            log.info("engine importable=%s in %.1fs%s", _IMPORT_OK,
                     time.time() - t0,
                     f" — {_IMPORT_ERR}" if _IMPORT_ERR else "")
        return _IMPORT_OK, _IMPORT_ERR


# Warm the check at startup so the first health probe answers instantly rather
# than waiting out the import (and tripping the healthcheck's 5s timeout).
threading.Thread(target=_check_importable, daemon=True).start()


class OCRIn(BaseModel):
    image_b64: str
    lang: str | None = None


@app.get("/health")
def health():
    importable, err = _check_importable()
    body = {
        "ok": bool(importable),
        "engine": "paddleocr",
        # Readiness, as opposed to "the HTTP server answered". These are two
        # different questions and this endpoint used to answer only the easy
        # one.
        "engine_importable": bool(importable),
        "engine_import_error": err,
        "default_lang": DEFAULT_LANG,
        # Reported so the console can tell a GPU sidecar from a CPU one
        # without the operator inspecting the image, and so a GPU build that
        # fell back to CPU is visible rather than merely slow.
        "device": DEFAULT_DEVICE or "auto",
        "paddleocr_version": _paddleocr_version(),
        # Languages whose engine is BUILT (first /ocr call per language), which
        # is not the same as readiness — an empty list is the normal state of a
        # freshly started, perfectly healthy sidecar.
        "loaded": sorted(_ENGINES),
        # The languages PaddleOCR ships recognition models for. Listed so the
        # console can show them without the operator reading upstream docs.
        "langs": ["en", "ch", "fr", "german", "korean", "japan", "ru", "es",
                  "pt", "it", "ar", "hi", "ta", "te", "ka", "latin", "cyrillic",
                  "devanagari"],
    }
    if not importable:
        # 503 rather than 200-with-a-flag: the compose healthcheck is a bare
        # `curl -fsS`, and every other consumer likewise treats a status code
        # as the answer. A consumer that never learns about the new field must
        # still not be told this container is ready.
        return JSONResponse(body, status_code=503)
    return body


@app.post("/ocr")
def ocr(body: OCRIn):
    lang = (body.lang or DEFAULT_LANG).strip() or DEFAULT_LANG
    try:
        raw = base64.b64decode(body.image_b64, validate=True)
    except Exception as e:
        return JSONResponse({"error": f"image_b64 is not valid base64: {e}"},
                            status_code=400)
    if not raw:
        return JSONResponse({"error": "image_b64 is empty"}, status_code=400)

    t0 = time.time()
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        lines = _run_ocr(_engine(lang), np.array(img))
    except Exception as e:
        log.exception("OCR failed")
        return JSONResponse({"error": f"{type(e).__name__}: {e}"},
                            status_code=500)

    return {"text": "\n".join(lines), "lines": len(lines),
            "lang": lang, "ms": int((time.time() - t0) * 1000),
            # Which device actually did the work. Without this a GPU sidecar
            # quietly running on CPU is indistinguishable from a fast one,
            # and the only symptom is a page rate nobody has a baseline for.
            "device": DEFAULT_DEVICE or "auto"}
