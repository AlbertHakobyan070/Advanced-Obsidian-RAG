"""PDF scout: resolve a chapter / page range, cut it into sections and size-capped units.

No LLM, no network. Every failure raises ScoutError with the fix in the message.
Page numbers are 1-based PDF pages everywhere (`doc[i]` is page `i + 1`).
"""
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from .config import KMConfig

CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s*\d+[-.:]?\d*")
_PREFIX_RE = re.compile(r"^(chapter\s+\d+|\d+)[.:)]?\s+")
_LETTERS_RE = re.compile(r"[^\W\d_]{2}")
_HEADING_MAX = 100     # longer lines are prose, not headings


class ScoutError(ValueError):
    pass


@dataclass
class Section:
    id: str
    title: str
    page_start: int
    page_end: int


@dataclass
class Heading:
    title: str
    page: int


@dataclass
class Unit:
    id: str
    section_id: str
    page_start: int
    page_end: int
    text: str
    chars: int


@dataclass
class Scout:
    pdf_path: Path
    vault_rel: str
    sha256: str
    book_title: str
    chapter_title: str
    page_start: int
    page_end: int
    pages: dict[int, str]
    sections: list[Section]
    units: list[Unit]
    captions: dict[int, list[str]]
    image_counts: dict[int, int]
    scanned_pages: list[int]
    toc_used: bool
    headings: list[Heading] = field(default_factory=list)


def _collapse(s: str) -> str:
    return " ".join(s.split())


def _norm(title: str) -> str:
    """Lowercase, collapse whitespace, drop a leading 'Chapter N.' / 'N ' prefix."""
    return _PREFIX_RE.sub("", _collapse(title).lower())


def _read(doc, nums, kcfg: KMConfig):
    pages, captions, image_counts, scanned = {}, {}, {}, []
    for n in nums:
        page = doc[n - 1]
        text = page.get_text("text")
        pages[n] = text
        if len(text.strip()) < kcfg.scanned_threshold:
            scanned.append(n)
        caps = [ln.strip() for ln in text.splitlines() if CAPTION_RE.match(ln.strip())]
        if caps:
            captions[n] = caps
        image_counts[n] = len(page.get_images())   # embedded image references on the page
    return pages, captions, image_counts, scanned


def _headings(doc, nums, ratio: float) -> list[Heading]:
    """Lines set at least `ratio` times the body font size, which needs no outline. A line's size
    is its smallest span (a drop cap does not make a sentence a heading); consecutive heading lines
    of one size on one page are one wrapped heading; a line on 3+ pages is a running header."""
    lines, sizes = [], Counter()
    for n in nums:
        for block in doc[n - 1].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if spans:
                    for s in spans:
                        sizes[round(s["size"], 1)] += len(s["text"])
                    lines.append((n, round(min(s["size"] for s in spans), 1),
                                  " ".join(s["text"].strip() for s in spans)))
    if not sizes:
        return []
    floor = sizes.most_common(1)[0][0] * ratio             # the body size: most characters are set in it
    pages_of = defaultdict(set)
    for n, _, t in lines:
        pages_of[t].add(n)
    out, prev = [], None
    for n, size, t in lines:
        hit = size >= floor and len(t) <= _HEADING_MAX and _LETTERS_RE.search(t) and len(pages_of[t]) < 3
        if hit and prev == (n, size):
            out[-1].title += " " + t
        elif hit:
            out.append(Heading(t, n))
        prev = (n, size) if hit else None
    return out


def _chapter_range(toc, chapter: str, n_pages: int):
    want = _norm(chapter)
    for i, (level, title, page) in enumerate(toc):
        if _norm(title) == want:
            end = n_pages
            for lvl2, _, p2 in toc[i + 1:]:
                if lvl2 <= level:
                    end = p2 - 1
                    break
            return i, level, _collapse(title), page, max(page, end)
    raise ScoutError(f"chapter {chapter!r} not found in the PDF outline; check the title or pass --pages A-B")


def _sections(toc, ch_i, ch_level, start, end, window_pages) -> list[Section]:
    subs = []
    for lvl, title, page in toc[ch_i + 1:]:
        if lvl <= ch_level or page > end:
            break
        if lvl == ch_level + 1:
            subs.append((_collapse(title), page))
    if subs:
        # first section absorbs any lead-in pages so the sections cover the whole range
        subs[0] = (subs[0][0], start)
        return [Section(f"sec-{i}", t, p, max(p, (subs[i][1] - 1) if i < len(subs) else end))
                for i, (t, p) in enumerate(subs, start=1)]
    wins = [(a, min(a + window_pages - 1, end)) for a in range(start, end + 1, window_pages)]
    return [Section(f"sec-{i}", f"Pages {a}-{b}", a, b) for i, (a, b) in enumerate(wins, start=1)]


def _units(sec: Section, pages: dict[int, str], cap: int) -> list[Unit]:
    out, cur = [], []

    def flush():
        if cur:
            text = "\n".join(pages[p] for p in cur)
            out.append(Unit(f"{sec.id}-u{len(out) + 1}", sec.id, cur[0], cur[-1], text, len(text)))
            cur.clear()

    for p in range(sec.page_start, sec.page_end + 1):
        if cur and len("\n".join(pages[q] for q in cur + [p])) > cap:
            flush()
        cur.append(p)
    flush()
    return out


def scout(pdf: Path, kcfg: KMConfig, *, chapter: str | None = None,
          pages: tuple[int, int] | None = None) -> Scout:
    if (chapter is None) == (pages is None):
        raise ScoutError("pass exactly one of --chapter or --pages")
    pdf = Path(pdf)
    if not pdf.is_absolute():       # the contract (CLI and job kind): absolute, or relative to parser.vault_path
        pdf = kcfg.vault_root / pdf
    if not pdf.is_file():
        raise ScoutError(f"PDF not found: {pdf}")
    if pdf.suffix.casefold() != ".pdf":
        raise ScoutError(f"{pdf} is not a .pdf file; the map harness reads PDFs only")
    try:
        vault_rel = pdf.resolve().relative_to(kcfg.vault_root.resolve()).as_posix()
    except ValueError:
        raise ScoutError(f"{pdf} is outside the vault; set parser.vault_path in config.yaml to a folder that contains it")
    with fitz.open(pdf) as doc:
        n_pages = len(doc)
        toc = doc.get_toc()
        if chapter is not None:
            if not toc:
                raise ScoutError("this PDF has no outline (table of contents); pass --pages A-B instead of --chapter")
            ch_i, ch_level, chapter_title, start, end = _chapter_range(toc, chapter, n_pages)
            sections = _sections(toc, ch_i, ch_level, start, end, kcfg.window_pages)
        else:
            start, end = pages
            if not 1 <= start <= end <= n_pages:
                raise ScoutError(f"--pages {start}-{end} is outside the PDF (1-{n_pages})")
            chapter_title = f"Pages {start}-{end}"
            sections = _sections([], 0, 0, start, end, kcfg.window_pages)
        texts, captions, image_counts, scanned = _read(doc, range(start, end + 1), kcfg)
        headings = _headings(doc, range(start, end + 1), kcfg.heading_min_ratio)
        book_title = _collapse(doc.metadata.get("title") or "")[:60].strip() or pdf.stem
    if len(scanned) / len(texts) > kcfg.max_scanned_ratio:
        raise ScoutError(f"{len(scanned)} of {len(texts)} pages in {start}-{end} have no text layer; "
                         "OCR is needed first (scanned pages exceed knowledge_map.scout.max_scanned_ratio)")
    return Scout(
        pdf_path=pdf, vault_rel=vault_rel, sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        book_title=book_title, chapter_title=chapter_title, page_start=start, page_end=end,
        pages=texts, sections=sections,
        units=[u for s in sections for u in _units(s, texts, kcfg.max_unit_chars)],
        captions=captions, image_counts=image_counts, scanned_pages=scanned,
        toc_used=chapter is not None, headings=headings,
    )


def inventory(folder: Path, kcfg: KMConfig) -> list[dict]:
    """Per-PDF counts for choosing what to map. Counts only; no LLM."""
    folder = Path(folder)
    rows = []
    for pdf in sorted(folder.rglob("*.pdf")):
        with fitz.open(pdf) as doc:
            texts, captions, image_counts, scanned = _read(doc, range(1, len(doc) + 1), kcfg)
            rows.append({
                "file": pdf.relative_to(folder).as_posix(),
                "pages": len(doc),
                "text_pages": len(texts) - len(scanned),
                "scanned_pages": len(scanned),
                "toc_chapters": sum(1 for e in doc.get_toc() if e[0] == 1),
                "captions": sum(len(c) for c in captions.values()),
                "images": sum(image_counts.values()),
            })
    return rows
