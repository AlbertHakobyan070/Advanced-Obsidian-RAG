"""
locate.py — find an extractor's verbatim quote in PDF page text.

WHY THIS EXISTS
  Map pointers are derived, not trusted. The LLM returns a quote; this module
  finds it in the page text and the pages it lands on become the pointer. A
  quote that cannot be found exactly is a failed node, never a guessed page.

THE MATCH
  Exact substring match after `norm` on BOTH sides — NFKC (ligatures), soft
  hyphens dropped, a line-end hyphen joins its word, curly quotes straightened,
  remaining hyphens ignored, dashes count as a space, whitespace collapsed,
  case folded. No fuzzy matching, ever: a paraphrase is `not_found`.

PAGES
  Keys are 1-based PDF page indices (as pdf_loader stores page_start /
  page_end), never printed page labels. Pages are joined with one space, so a
  quote may run across ONE page break; its span is (first_page, last_page).
  A hit spanning three or more pages (e.g. across a short caption page) is
  not an occurrence at all, so a quote with only such hits is not_found.

SEARCH ORDER
  The node's section pages first; the chapter range only when the section has
  zero occurrences (it catches a section whose page offset is wrong). One start
  page → located; occurrences starting on different pages → ambiguous.
"""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable

_QUOTES = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
})


def norm(text: str) -> str:
    """Comparison form of `text`; quotes and page text both go through this."""
    t = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    t = re.sub(r"(\w)[-‐‑][ \t]*\r?\n[ \t]*(\w)", r"\1\2", t)  # line-end hyphen (ASCII or U+2010/11) joins its word
    t = t.translate(_QUOTES)                                         # ‘’‚‛′ -> '   “”„‟″ -> "
    t = re.sub(r"[-\u2010\u2011\u2012]", "", t)                      # hyphens ignored on both sides
    t = re.sub(r"[\u2013\u2014\u2015]", " ", t)                      # dashes count as a space
    return re.sub(r"\s+", " ", t).strip().casefold()


class PageText:
    """Normalised page text, joined once, with each page's start offset."""

    def __init__(self, pages: dict[int, str]):
        self.pages = sorted(pages)
        self.starts: list[int] = []
        parts: list[str] = []
        offset = 0
        for page in self.pages:
            part = norm(pages[page])
            self.starts.append(offset)
            parts.append(part)
            offset += len(part) + 1                                  # the " " joiner
        self.text = " ".join(parts)

    def _page_at(self, offset: int) -> int:
        return self.pages[bisect_right(self.starts, offset) - 1]

    def _spans(self, quote: str, lo: int, hi: int) -> list[tuple[int, int]]:
        """(start, end) offsets of every occurrence starting on a page in [lo, hi]."""
        needle = norm(quote)
        if not needle:                                               # "" would match everywhere
            return []
        spans = []
        i = self.text.find(needle)
        while i != -1:
            if lo <= self._page_at(i) <= hi:
                spans.append((i, i + len(needle)))
            i = self.text.find(needle, i + 1)                        # overlapping hits count too
        return spans

    def find(self, quote: str, lo: int, hi: int) -> list[tuple[int, int]]:
        """Every occurrence starting on a page in [lo, hi], as (first_page, last_page)."""
        return [(self._page_at(s), self._page_at(e - 1)) for s, e in self._spans(quote, lo, hi)]

    def context(self, quote: str, page: int, chars: int = 400) -> str:
        """Normalised text `chars` either side of the quote's first occurrence on `page`."""
        spans = self._spans(quote, page, page)
        if not spans:
            raise ValueError(f"quote not found on page {page}: {quote!r}")
        s, e = spans[0]
        return self.text[max(0, s - chars):e + chars]


@dataclass(frozen=True)
class Located:
    status: str                    # located | not_found | ambiguous | bad_length
    pages: tuple[int, ...]         # empty unless located


def locate(quote: str, text: PageText, section: tuple[int, int],
           chapter: tuple[int, int], words: tuple[int, int] = (5, 30)) -> Located:
    """Find `quote` in the section, else the chapter, and derive its pages."""
    if not words[0] <= len(quote.split()) <= words[1]:
        return Located("bad_length", ())
    def occurrences(lo: int, hi: int) -> list[tuple[int, int]]:
        # At most one page break: a wider hit does not count as an occurrence.
        return [(f, l) for f, l in text.find(quote, lo, hi) if l - f <= 1]

    hits = occurrences(*section) or occurrences(*chapter)
    if not hits:
        return Located("not_found", ())
    if len({first for first, _ in hits}) > 1:
        return Located("ambiguous", ())
    # One start page (possibly several hits on it): the span is every page they touch.
    return Located("located", tuple(sorted({p for f, l in hits for p in range(f, l + 1)})))


def page_ranges(pages: Iterable[int]) -> list[tuple[int, int]]:
    """Merge pages into sorted runs of consecutive numbers."""
    ranges: list[tuple[int, int]] = []
    for p in sorted(set(pages)):
        if ranges and p == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], p)
        else:
            ranges.append((p, p))
    return ranges


def format_pages(ranges: list[tuple[int, int]]) -> str:
    """[(151, 151), (152, 153)] -> "151, 152-153"."""
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in ranges)
