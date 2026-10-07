"""
web_import.py — Online-source fetcher + any-file→markdown converter.

Two lanes, both writing .md files into the inbox's `_converted/` staging
subfolder (they do NOT index anything — promote the outputs to the inbox and
run the normal ingest lane):

  fetch_urls()     pull web pages (requests | crawl4ai | scrapling | crawlee
                   backends, "auto" = best installed of crawl4ai > scrapling >
                   requests; crawlee is opt-in by name) and convert the HTML
                   to markdown via markitdown.
  convert_files()  convert already-uploaded inbox files (pdf/docx/pptx/xlsx/
                   html/…) to markdown via markitdown; optional Tesseract OCR
                   for selected PDF pages when the text layer is missing.

Design rules honored:
  * Optional backends fail READABLE: naming a backend that isn't installed
    raises with the pip command to fix it; "auto" quietly falls back down the
    chain and plain requests is always available.
  * Nothing here touches the indexes — output is staged .md only.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)

FETCH_BACKENDS = ("auto", "requests", "crawl4ai", "scrapling", "crawlee")

_SAFE_NAME = re.compile(r"[^\w.\- ()\[\]]")


def _slug_for(url: str) -> str:
    tail = re.sub(r"[?#].*$", "", url).rstrip("/").rsplit("/", 1)[-1] or "page"
    host = re.sub(r"^https?://(www\.)?", "", url).split("/", 1)[0]
    name = f"{host}__{tail}"[:120]
    return _SAFE_NAME.sub("_", name)


def _unique(dest_dir: Path, stem: str, suffix: str) -> Path:
    p = dest_dir / f"{stem}{suffix}"
    i = 1
    while p.exists():
        p = dest_dir / f"{stem} ({i}){suffix}"
        i += 1
    return p


def _markitdown():
    try:
        from markitdown import MarkItDown
    except ImportError as e:
        raise RuntimeError(
            "markitdown is not installed in this venv — "
            "pip install \"markitdown[pdf,docx,pptx,xlsx]\"") from e
    return MarkItDown(enable_plugins=False)


# ---------------------------------------------------------------- fetching

def _fetch_requests(url: str) -> tuple[bytes, str]:
    import requests
    r = requests.get(url, timeout=45, headers={
        "User-Agent": "Mozilla/5.0 (personal-rag importer)"})
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    return r.content, ctype


def _fetch_crawl4ai(url: str) -> tuple[bytes, str]:
    try:
        import asyncio
        from crawl4ai import AsyncWebCrawler
    except ImportError as e:
        raise RuntimeError(
            "crawl4ai backend requested but not installed — "
            "pip install crawl4ai (heavy: pulls playwright)") from e

    async def _run():
        async with AsyncWebCrawler() as crawler:
            res = await crawler.arun(url=url)
            return res
    res = asyncio.run(_run())
    md = getattr(res, "markdown", None)
    if md:                                    # crawl4ai already made markdown
        return str(md).encode("utf-8"), "text/markdown"
    return (res.html or "").encode("utf-8"), "text/html"


def _fetch_scrapling(url: str) -> tuple[bytes, str]:
    try:
        from scrapling.fetchers import Fetcher
    except ImportError as e:
        raise RuntimeError(
            "scrapling backend requested but not installed — "
            "pip install scrapling") from e
    page = Fetcher.get(url)
    return page.html_content.encode("utf-8"), "text/html"


def _fetch_crawlee(url: str) -> tuple[bytes, str]:
    try:
        import asyncio
        from crawlee import Request
        from crawlee.crawlers import PlaywrightCrawler
        from crawlee.storage_clients import MemoryStorageClient
    except ImportError as e:
        raise RuntimeError(
            "crawlee backend requested but not installed — "
            "pip install \"crawlee[playwright,beautifulsoup]\" "
            "(it drives Chromium: python -m playwright install chromium)") from e

    async def _run():
        got: dict = {}
        # Memory storage: crawlee's default is a ./storage folder in the working
        # directory, which for a CLI run is the repo root.
        crawler = PlaywrightCrawler(max_requests_per_crawl=1,
                                    storage_client=MemoryStorageClient())

        @crawler.router.default_handler
        async def _page(context):
            got["html"] = await context.page.content()

        @crawler.failed_request_handler
        async def _failed(context, error):    # retries spent; crawlee only logs it
            got["error"] = error

        # always_enqueue: crawlee caches its storages process-wide, so a URL an
        # earlier fetch in this run already handled would be skipped silently.
        await crawler.run([Request.from_url(url, always_enqueue=True)])
        return got
    got = asyncio.run(_run())
    if "html" not in got:
        err = got.get("error")
        raise RuntimeError(
            f"crawlee could not fetch {url}: "
            f"{err if err is not None else 'no page was returned'}") from err
    return got["html"].encode("utf-8"), "text/html"


def fetch_one(url: str, backend: str = "auto") -> tuple[bytes, str]:
    """Return (payload_bytes, content_type). Explicit missing backend raises
    readable; 'auto' tries crawl4ai -> scrapling -> requests, using whatever
    exists (requests is a hard dependency of the project). crawlee is never in
    that chain: it runs only when asked for by name."""
    if backend not in FETCH_BACKENDS:
        raise ValueError(f"backend must be one of {FETCH_BACKENDS}")
    if backend == "requests":
        return _fetch_requests(url)
    if backend == "crawl4ai":
        return _fetch_crawl4ai(url)
    if backend == "scrapling":
        return _fetch_scrapling(url)
    if backend == "crawlee":
        return _fetch_crawlee(url)
    for fn in (_fetch_crawl4ai, _fetch_scrapling):
        try:
            return fn(url)
        except RuntimeError:                  # backend not installed
            continue
        except Exception as e:                # backend installed but failed
            log.warning("auto backend %s failed on %s: %s",
                        fn.__name__, url, e)
            continue
    return _fetch_requests(url)


def _print_pdf(url: str, dest: Path) -> None:
    """Render the live page in headless Chromium and print it to PDF.
    Keeps the site's own rendering — LaTeX (KaTeX/MathJax), tables, syntax-
    highlighted code — which markdown conversion necessarily flattens."""
    try:
        from playwright.sync_api import Error as PWError, sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "PDF fetch needs playwright — pip install playwright && "
            "python -m playwright install chromium") from e
    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch()
            except PWError as e:
                # The single most common failure in a fresh env: the browser
                # binary was never downloaded, or it lives at a path the
                # process can't see (PLAYWRIGHT_BROWSERS_PATH). Say exactly that
                # instead of surfacing a raw "chromium error".
                raise RuntimeError(
                    "headless Chromium is not available — run "
                    "`python -m playwright install chromium`. If it is already "
                    "installed elsewhere, point PLAYWRIGHT_BROWSERS_PATH at that "
                    f"cache before launching. (playwright said: {e})") from e
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 1024})
                page.goto(url, wait_until="networkidle", timeout=90_000)
                page.wait_for_timeout(1500)   # late math/highlight rendering
                page.pdf(path=str(dest), format="A4", print_background=True,
                         margin={"top": "14mm", "bottom": "14mm",
                                 "left": "12mm", "right": "12mm"})
            finally:
                browser.close()
    except Exception:
        dest.unlink(missing_ok=True)          # never leave a partial PDF staged
        raise


def fetch_urls(urls: list[str], dest_dir: Path, backend: str = "auto",
               fmt: str = "md") -> list[dict]:
    """Fetch every URL and write one .md (or printed .pdf) per page into
    dest_dir. Returns per-URL result dicts; a failed URL reports its error and
    the run continues (partial success is the useful behavior for link
    batches). fmt="pdf" prints the rendered page via headless Chromium —
    URLs that ARE a PDF are saved as-is either way in that mode."""
    if fmt not in ("md", "pdf"):
        raise ValueError("fmt must be md|pdf")
    dest_dir.mkdir(parents=True, exist_ok=True)
    md = _markitdown() if fmt == "md" else None
    out: list[dict] = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        if not re.match(r"^https?://", url):
            out.append({"url": url, "ok": False,
                        "error": "only http(s) URLs are fetched"})
            continue
        try:
            if fmt == "pdf":
                stem = _slug_for(url)
                dest = _unique(dest_dir, stem, ".pdf")
                if url.lower().split("?")[0].endswith(".pdf"):
                    payload, _ = _fetch_requests(url)   # already a PDF
                    dest.write_bytes(payload)
                else:
                    _print_pdf(url, dest)
                out.append({"url": url, "ok": True, "file": dest.name,
                            "bytes": dest.stat().st_size})
                log.info("printed %s -> %s (%d bytes)", url, dest.name,
                         dest.stat().st_size)
                continue
            payload, ctype = fetch_one(url, backend)
            stem = _slug_for(url)
            if "markdown" in ctype:
                text = payload.decode("utf-8", errors="replace")
            else:
                # hand markitdown the raw payload with a type-appropriate
                # temp suffix (html pages, PDFs linked directly, docx, …)
                suffix = ".html"
                if "pdf" in ctype or url.lower().endswith(".pdf"):
                    suffix = ".pdf"
                tmp = _unique(dest_dir, f"_fetch_{stem}", suffix)
                tmp.write_bytes(payload)
                try:
                    text = md.convert(str(tmp)).text_content
                finally:
                    tmp.unlink(missing_ok=True)
            header = f"# {stem}\n\n> Source: {url}\n> Fetched: " \
                     f"{time.strftime('%Y-%m-%d %H:%M')}\n\n"
            dest = _unique(dest_dir, stem, ".md")
            dest.write_text(header + text, encoding="utf-8")
            out.append({"url": url, "ok": True, "file": dest.name,
                        "chars": len(text)})
            log.info("fetched %s -> %s (%d chars)", url, dest.name, len(text))
        except Exception as e:
            out.append({"url": url, "ok": False,
                        "error": f"{type(e).__name__}: {e}"})
            log.warning("fetch failed for %s: %s", url, e)
    return out


# ---------------------------------------------------------------- converting

def _ocr_pdf_pages(pdf_path: Path, pages_spec: str, lang: str = "eng") -> str:
    """OCR the selected 1-based pages of a PDF with Tesseract (the same engine
    the ingest lane uses) and return them as markdown-ish text. `lang` is the
    Tesseract language code; the caller reads it from pdf.ocr_language, so this
    helper never re-loads config.yaml.

    Runs through PyMuPDF's built-in Tesseract hook, not pytesseract: that
    package is in neither requirements.txt nor the runtime venv, so the old
    implementation raised on every call. PyMuPDF is already a hard dependency."""
    import fitz                                # pymupdf
    from src.ingestion.pdf_loader import parse_page_spec
    doc = fitz.open(pdf_path)
    try:
        idxs = parse_page_spec(pages_spec, page_count=doc.page_count)
        parts = []
        for i in idxs:
            page = doc[i]
            try:
                # full=True OCRs the whole page image, as the old pixmap ->
                # Tesseract path did; the default only fills in image areas and
                # keeps the page's own text layer, which is what the caller is
                # bypassing. 200 dpi because PyMuPDF's default of 72 is
                # unreadable to Tesseract.
                tp = page.get_textpage_ocr(language=lang, dpi=200, full=True)
            except (RuntimeError, fitz.mupdf.FzErrorLibrary) as e:
                # Both are what PyMuPDF raises when Tesseract or its tessdata is
                # unusable: RuntimeError when none can be found, FzErrorLibrary
                # (not a RuntimeError) when the folder lacks the language. Neither
                # message names the env var or the language, so say both here.
                raise RuntimeError(
                    f"Tesseract OCR failed on page {i + 1}: it needs tessdata for "
                    f"language {lang!r} - set TESSDATA_PREFIX to the tessdata "
                    f"folder that holds it. PyMuPDF said: {e}") from e
            # textpage= is load-bearing: without it get_text() quietly returns
            # the page's own text layer, which for a scan is empty.
            txt = page.get_text("text", textpage=tp)
            parts.append(f"## Page {i + 1}\n\n{txt.strip()}")
        return "\n\n".join(parts)
    finally:
        doc.close()


def convert_files(files: list[str], inbox: Path, dest_dir: Path,
                  ocr_pages: str = "", ocr_language: str = "eng") -> list[dict]:
    """Convert named inbox files to markdown into dest_dir via markitdown.
    `ocr_pages` (e.g. \"1-4,9\") additionally OCRs those pages of each PDF and
    appends the OCR text — for scanned pages markitdown's text layer misses —
    in `ocr_language` (a Tesseract code; the CLI passes pdf.ocr_language).
    Names must be plain filenames living in the inbox (no path parts)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    md = _markitdown()
    out: list[dict] = []
    for name in files:
        name = (name or "").strip()
        if not name or Path(name).name != name:
            out.append({"file": name, "ok": False,
                        "error": "plain inbox filenames only"})
            continue
        src = inbox / name
        if not src.is_file():
            out.append({"file": name, "ok": False, "error": "not in inbox"})
            continue
        try:
            text = md.convert(str(src)).text_content or ""
            if ocr_pages and src.suffix.lower() == ".pdf":
                ocr_txt = _ocr_pdf_pages(src, ocr_pages, ocr_language)
                text += f"\n\n---\n\n# OCR (pages {ocr_pages})\n\n{ocr_txt}"
            dest = _unique(dest_dir, src.stem, ".md")
            header = f"# {src.stem}\n\n> Converted from: {name}\n\n"
            dest.write_text(header + text, encoding="utf-8")
            out.append({"file": name, "ok": True, "output": dest.name,
                        "chars": len(text)})
            log.info("converted %s -> %s (%d chars)", name, dest.name, len(text))
        except Exception as e:
            out.append({"file": name, "ok": False,
                        "error": f"{type(e).__name__}: {e}"})
            log.warning("convert failed for %s: %s", name, e)
    return out
