"""
diversity.py — a per-file cap on the final list, after the rerank.

The reranker scores each chunk on its own, so the five best chunks of one book
can fill five of ten seats and push out the second source a multi-part
question needs. Measured on the dev split (2026-10-13): of the required gold
sources that reached the reranker but missed the final list, the lists that
dropped them spent about half their seats on extra chunks of a file already
listed. The cap keeps the best-first order and allows at most `max_per_file`
chunks per source file.

    rerank (whole pool, best first) -> CAP -> gate -> expansion -> generation

Backfill: if the cap leaves fewer than k chunks, the chunks it skipped fill the
remaining seats in score order, so a capped call never returns fewer results
than an uncapped one.
"""
from __future__ import annotations

import unicodedata

from src.retrieval.retriever import RetrievedDoc


def _file_key(doc: RetrievedDoc) -> str:
    """The normalised source file (canvas rows store '/', every other lane '\\');
    a chunk with no source file is its own key, so it is never capped."""
    src = doc.metadata.get("source_file")
    if not src:
        return "\x00" + doc.id
    return unicodedata.normalize("NFC", str(src)).replace("\\", "/").strip("/").casefold()


def cap_per_file(docs: list[RetrievedDoc], k: int, max_per_file: int) -> list[RetrievedDoc]:
    """The best-first k of `docs` with at most `max_per_file` per source file,
    backfilled from the skipped ones. `max_per_file` <= 0 means no cap."""
    if max_per_file <= 0:
        return docs[:k]
    kept, skipped, seen = [], [], {}
    for d in docs:
        key = _file_key(d)
        if seen.get(key, 0) < max_per_file:
            seen[key] = seen.get(key, 0) + 1
            kept.append(d)
            if len(kept) == k:
                return kept
        else:
            skipped.append(d)
    return kept + skipped[:k - len(kept)]
