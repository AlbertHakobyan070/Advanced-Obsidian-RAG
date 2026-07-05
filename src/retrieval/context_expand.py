"""
context_expand.py — E2 "small-to-big" context expansion (post-rerank).

Two independent, config-gated lanes (both default OFF until A/B'd):

  ParentContext   (retrieval.parent_context)
      Markdown notes only. Children carry a `parent_id` in metadata; the full
      enclosing-section text lives in a sidecar JSONL (data/parents_md.jsonl)
      that is NEVER embedded. Post-rerank, a child's text is swapped for its
      parent's text; co-retrieved siblings of the same parent are dropped
      (their content is already inside the parent). Metadata is untouched, so
      citation labels stay identical.

  NeighborContext (retrieval.neighbor_context)
      PDF corpus (re-embed forbidden), so no text/id changes: instead, fetch
      same-source chunks from ADJACENT pages (page_start ±1) via a ChromaDB
      metadata get, and append them as clearly-marked supplementary sources.
      Capped per-doc and in total; each fetch is one Chroma round-trip per
      top-k PDF doc (latency note in WORKLOG).

Both operate on the reranked top-k only — retrieval and fusion are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.retrieval.retriever import RetrievedDoc
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

_NEIGHBOR_MARK = "[supplementary context — adjacent page of the same source]\n"


def _iter_jsonl(path: Path):
    """Yield records splitting on b'\\n' ONLY (chunk/parent text may contain
    U+2028/U+2029/\\x85 — .splitlines() would shred records)."""
    buf = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            buf += block
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line, buf = buf[:nl], buf[nl + 1:]
                if line.strip():
                    yield line
    if buf.strip():
        yield buf


class ParentContext:
    """Child→parent text swap for markdown chunks (lazy sidecar load)."""

    def __init__(self, parents_file: Path, enabled: bool = False):
        self.parents_file = Path(parents_file)
        self.enabled = enabled
        self._parents: dict[str, dict] | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> "ParentContext":
        pf = (cfg.path("retrieval.parents_file")
              if cfg.get("retrieval.parents_file")
              else cfg.project_root / "data" / "parents_md.jsonl")
        return cls(parents_file=pf,
                   enabled=bool(cfg.get("retrieval.parent_context", False)))

    def _load(self) -> dict[str, dict]:
        if self._parents is None:
            self._parents = {}
            if self.parents_file.exists():
                for raw in _iter_jsonl(self.parents_file):
                    try:
                        rec = json.loads(raw.decode("utf-8", errors="replace"))
                        self._parents[str(rec["parent_id"])] = rec
                    except (json.JSONDecodeError, KeyError):
                        continue
                log.info("parent context: %d sections from %s",
                         len(self._parents), self.parents_file.name)
            else:
                log.warning("retrieval.parent_context is on but %s is missing "
                            "— swaps disabled for this process",
                            self.parents_file)
        return self._parents

    def apply(self, docs: list[RetrievedDoc]) -> tuple[list[RetrievedDoc], int, int]:
        """Returns (docs, swapped, siblings_dropped). Rank order preserved;
        a doc whose parent was already used by a higher-ranked sibling is
        dropped (its content is inside the swapped parent text)."""
        parents = self._load()
        if not parents:
            return docs, 0, 0
        out: list[RetrievedDoc] = []
        used: set[str] = set()
        swapped = dropped = 0
        for d in docs:
            pid = str((d.metadata or {}).get("parent_id") or "")
            rec = parents.get(pid) if pid else None
            if rec is None:
                out.append(d)
                continue
            if pid in used:
                dropped += 1
                continue
            used.add(pid)
            d.debug["parent_swap"] = pid
            d.debug["child_chars"] = len(d.text)
            d.text = rec["text"]
            swapped += 1
            out.append(d)
        if swapped or dropped:
            log.info("parent context: %d swapped, %d sibling(s) dropped",
                     swapped, dropped)
        return out, swapped, dropped


class NeighborContext:
    """Adjacent-page supplementary chunks for PDF hits (no re-embed)."""

    def __init__(self, enabled: bool = False, max_per_doc: int = 2,
                 max_total: int = 4, max_chars: int = 2500):
        self.enabled = enabled
        self.max_per_doc = max_per_doc
        self.max_total = max_total
        self.max_chars = max_chars

    @classmethod
    def from_config(cls, cfg: Config) -> "NeighborContext":
        return cls(
            enabled=bool(cfg.get("retrieval.neighbor_context", False)),
            max_per_doc=int(cfg.get("retrieval.neighbor_max_per_doc", 2)),
            max_total=int(cfg.get("retrieval.neighbor_max_total", 4)),
            max_chars=int(cfg.get("retrieval.neighbor_max_chars", 2500)),
        )

    def apply(self, docs: list[RetrievedDoc], collection) -> tuple[list[RetrievedDoc], int]:
        """Returns (docs + appended neighbors, added_count)."""
        have = {d.id for d in docs}
        added: list[RetrievedDoc] = []
        for d in docs:
            if len(added) >= self.max_total:
                break
            m = d.metadata or {}
            if str(m.get("file_type", "")).lower() != "pdf":
                continue
            sf = m.get("source_file")
            try:
                page = int(m.get("page_start"))
            except (TypeError, ValueError):
                continue
            if not sf:
                continue
            try:
                res = collection.get(
                    where={"$and": [{"source_file": sf},
                                    {"page_start": {"$in": [page - 1, page + 1]}}]},
                    include=["documents", "metadatas"],
                    limit=16,
                )
            except Exception as e:           # fail-soft: expansion is optional
                log.warning("neighbor fetch failed for %s p.%s: %s", sf, page, e)
                continue
            rows = sorted(
                zip(res["ids"], res["documents"], res["metadatas"]),
                key=lambda r: (abs(int(r[2].get("page_start", page)) - page),
                               int(r[2].get("page_start", page))),
            )
            taken = 0
            for nid, text, meta in rows:
                if taken >= self.max_per_doc or len(added) >= self.max_total:
                    break
                if nid in have:
                    continue
                have.add(nid)
                clipped = text[: self.max_chars]
                added.append(RetrievedDoc(
                    id=nid,
                    text=_NEIGHBOR_MARK + clipped,
                    metadata={**(meta or {}), "supplementary": "neighbor_page"},
                    score=0.0,
                    debug={"neighbor_of": d.id},
                ))
                taken += 1
        if added:
            log.info("neighbor context: +%d adjacent-page chunk(s)", len(added))
        return docs + added, len(added)
