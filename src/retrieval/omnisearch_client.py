"""
omnisearch_client.py — Live-vault lane via Obsidian's Omnisearch plugin.

The RAG index is a snapshot: anything the author wrote since the last ingest is
invisible to ChromaDB/BM25. Omnisearch (the community plugin) indexes the vault
LIVE inside Obsidian and exposes a localhost HTTP API:

    GET http://localhost:51361/search?q=your%20query
    -> [ { score, vault, path, basename, foundWords,
           matches: [{match, offset}], excerpt }, ... ]

(Enable it in Obsidian → Omnisearch settings → "HTTP Server". Localhost only;
the server stops when Obsidian closes. Docs: publish.obsidian.md/omnisearch.)

This client turns those results into the same (id, text, metadata) tuples the
retriever's other lanes emit, so live notes join the RRF fusion and the
cross-encoder ranks them against indexed chunks on equal footing. Everything is
FAIL-SOFT: Obsidian closed / server off / plugin missing → empty lane, one
warning per process, the pipeline never breaks.

Two extra jobs it does:
  * course/domain tagging via detect_course_from_path, so metadata_boost and
    scope routing treat live notes like indexed ones;
  * marks metadata["live"] = True so citations can carry a "(live)" tag —
    the answer may quote a note that is NOT yet in the index, and the author
    should be able to tell.

Usage:
    from src.retrieval.omnisearch_client import OmnisearchClient
    omni = OmnisearchClient.from_config(cfg)
    rows = omni.lane("participle punctuation")   # -> list[(id, text, meta)]
"""
from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

# Markdown-ish extensions whose excerpts are worth fusing. Omnisearch can also
# index PDFs/images via Text Extractor; those excerpts tend to be OCR soup and
# the PDFs are in our own index anyway, so they are skipped by default.
_DEFAULT_EXTS = {".md", ".markdown", ".txt", ".canvas"}


def _split_path_parts(path: str) -> list[str]:
    """Vault-relative path -> folder parts, tolerant of / and \\ separators."""
    if "\\" in path:
        return list(PureWindowsPath(path).parts)
    return list(PurePosixPath(path).parts)


class OmnisearchClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:51361",
        enabled: bool = False,
        top_k: int = 8,
        timeout: float = 1.5,
        excerpt_min_chars: int = 40,
        allowed_exts: list[str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled
        self.top_k = top_k
        self.timeout = timeout
        self.excerpt_min_chars = excerpt_min_chars
        self.allowed_exts = {
            e.lower() if e.startswith(".") else "." + e.lower()
            for e in (allowed_exts or sorted(_DEFAULT_EXTS))
        }
        self._warned_down = False   # one connection warning per process, not per query

    @classmethod
    def from_config(cls, cfg: Config) -> "OmnisearchClient":
        return cls(
            base_url=cfg.get("retrieval.omnisearch.base_url",
                             "http://127.0.0.1:51361"),
            enabled=cfg.get("retrieval.omnisearch.enabled", False),
            top_k=cfg.get("retrieval.omnisearch.top_k", 8),
            timeout=cfg.get("retrieval.omnisearch.timeout", 1.5),
            excerpt_min_chars=cfg.get("retrieval.omnisearch.excerpt_min_chars", 40),
            allowed_exts=cfg.get("retrieval.omnisearch.allowed_exts", None),
        )

    # ---- raw API ----

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Raw Omnisearch results (ResultNoteApi dicts), or [] if unreachable.
        Never raises — a closed Obsidian must not break a query.
        """
        import requests

        k = top_k or self.top_k
        try:
            resp = requests.get(
                f"{self.base_url}/search",
                params={"q": query},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            rows = resp.json()
            if not isinstance(rows, list):
                return []
            self._warned_down = False
            return rows[: max(k, 0)]
        except Exception as e:
            if not self._warned_down:
                log.warning(
                    "Omnisearch unreachable at %s (%s) — live-vault lane skipped. "
                    "Is Obsidian open with the Omnisearch HTTP server enabled?",
                    self.base_url, type(e).__name__,
                )
                self._warned_down = True
            return []

    # ---- retriever lane ----

    def lane(self, query: str, top_k: int | None = None) -> list[tuple[str, str, dict]]:
        """
        Omnisearch results as retriever lane tuples: (id, text, metadata),
        already ranked (Omnisearch's own BM25 order feeds RRF by position).
        """
        rows = self.search(query, top_k=top_k)
        if not rows:
            return []

        # Lazy import: keeps this module import-light; the parser is pure-stdlib.
        try:
            from src.ingestion.obsidian_parser import detect_course_from_path
        except Exception:                                    # pragma: no cover
            detect_course_from_path = None

        out: list[tuple[str, str, dict]] = []
        for row in rows:
            path = str(row.get("path") or "")
            basename = str(row.get("basename") or "") or (path.rsplit("/", 1)[-1])
            excerpt = str(row.get("excerpt") or "").strip()
            ext = ("." + path.rsplit(".", 1)[-1].lower()) if "." in path else ""
            if ext and self.allowed_exts and ext not in self.allowed_exts:
                continue
            if len(excerpt) < self.excerpt_min_chars:
                continue

            meta: dict[str, Any] = {
                "source_file": path,
                "filename": basename,
                "file_type": "live_note",
                "heading": basename,
                "live": True,
                "omnisearch_score": float(row.get("score") or 0.0),
            }
            if detect_course_from_path is not None:
                try:
                    course = detect_course_from_path(_split_path_parts(path))
                    if isinstance(course, dict):
                        meta.update({k: v for k, v in course.items() if v})
                except Exception:
                    pass

            # Give the reranker/generator the note name for context, then the
            # excerpt Omnisearch chose around the match.
            text = f"{basename}\n\n{excerpt}"
            out.append((f"omni::{path}", text, meta))

        return out
