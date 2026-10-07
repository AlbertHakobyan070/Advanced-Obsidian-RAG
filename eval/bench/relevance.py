"""
relevance.py — is a retrieved chunk relevant to a gold locator?

Gold is a LOCATOR (a vault-relative file, optionally a page range and/or a
heading prefix), not a chunk id: chunk ids change with every re-chunk or
embedding swap — the very components the eval ablates — while a locator
survives both. A chunk satisfies a gold entry iff

  1. its source_file is the gold file (norm_file: separators, case, Unicode
     form), AND
  2. if the gold names pages, the chunk has a page range that overlaps them, AND
  3. if the gold names a heading, the chunk's heading_path starts with it,
     compared component by component.

A gold entry with no locator is satisfied by any chunk of the file.

Pageless chunks. Markdown, notebook, code and canvas chunks carry no
page_start/page_end. A gold page range cannot be shown to overlap a chunk that
has no pages, so such a chunk does NOT satisfy a paged gold entry — we never
guess "page 1". Gold written for a pageless source should use a heading, or no
locator at all.

Every judgement also has a document-level twin (file match only), reported
beside each chunk-level number: it separates "found the right file" from
"found the right passage of it".
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from eval.bench.questions import GoldSource


def norm_file(p: str) -> str:
    """Vault-relative path in one canonical spelling: NFC, '/' separators, no
    leading/trailing slash, casefolded. Canvas rows store '/', every other lane
    '\\'; folder names carry en/em dashes, which are content and stay."""
    return unicodedata.normalize("NFC", str(p)).replace("\\", "/").strip().strip("/").casefold()


def _heading_prefix(path: str, prefix: str) -> bool:
    parts = [p.strip().casefold() for p in str(path).split(">")]
    want = [p.strip().casefold() for p in str(prefix).split(">")]
    return len(parts) >= len(want) and parts[:len(want)] == want


def _same_file(meta: dict, gold: GoldSource) -> bool:
    return norm_file(meta.get("source_file", "")) == norm_file(gold.file)


def doc_matches(meta: dict, gold: GoldSource) -> bool:
    """The chunk is in the gold file or in one of its alternatives (twins)."""
    return any(_same_file(meta, s) for s in gold.sources())


def chunk_matches(meta: dict, gold: GoldSource) -> bool:
    """The chunk satisfies the gold entry's locator, or an alternative's."""
    return any(_locates(meta, s) for s in gold.sources())


def _locates(meta: dict, gold: GoldSource) -> bool:
    if not _same_file(meta, gold):
        return False
    if gold.pages is not None:
        ps = meta.get("page_start")
        if ps is None:                 # a gold page range cannot be shown to overlap a pageless chunk
            return False
        ps = int(ps)
        pe = int(meta.get("page_end") if meta.get("page_end") is not None else ps)
        if pe < gold.pages[0] or ps > gold.pages[1]:
            return False
    if gold.heading and not _heading_prefix(meta.get("heading_path") or "", gold.heading):
        return False
    return True


@dataclass
class Judged:
    matches: list[frozenset[int]]        # per rank: gold indices this chunk satisfies
    doc_matches: list[frozenset[int]]    # per rank: gold indices whose FILE it is in


def judge(metas: list[dict], gold: list[GoldSource]) -> Judged:
    return Judged(
        matches=[frozenset(i for i, g in enumerate(gold) if chunk_matches(m, g)) for m in metas],
        doc_matches=[frozenset(i for i, g in enumerate(gold) if doc_matches(m, g)) for m in metas],
    )
