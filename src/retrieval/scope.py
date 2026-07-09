"""
scope.py — Query → retrieval-scope routing.

If a query names a knowledge domain ("statistics", "BI", "NLP", "ggplot"…) or
a content type ("my homework", "the lecture files", "the tech books"…),
retrieval should *look where the user pointed*. This module detects those
hints with the same word-boundary matcher the HyDE bypass uses, and turns
them into a Scope:

    domains       -> values of the `domain` metadata field (stats, ml, nlp, …)
    path_contains -> case-insensitive substrings of `source_file`
    file_types    -> values of the `file_type` metadata field

The retriever uses a Scope to add FILTERED dense+sparse lanes to the RRF
fusion (same mechanism as the code lane): scoped chunks are guaranteed a seat
in the candidate pool, and the cross-encoder still makes the final call. Soft
routing, not hard filtering — a bad hint can't empty the results.

Matching semantics inside a Scope: OR within an aspect (either of "NLP or
GenAI"), AND between domain and content ("homework" AND "from statistics").

The keyword dictionaries live in config.yaml (retrieval.domain_signals /
retrieval.content_signals) so they can be extended without touching code.

Usage:
    router = ScopeRouter.from_config(cfg)
    scope = router.detect("bayes theorem in my statistics lectures")
    # Scope(domains=['stats'], path_contains=['lecture', 'slides'], ...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retrieval.hyde import _compile_signals
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Scope:
    domains: list[str] = field(default_factory=list)
    path_contains: list[str] = field(default_factory=list)
    file_types: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)   # human-readable, for echo/logs

    def __bool__(self) -> bool:
        return bool(self.domains or self.path_contains or self.file_types)

    def matches(self, meta: dict[str, Any]) -> bool:
        """Does this chunk's metadata fall inside the scope?"""
        meta = meta or {}
        if self.domains:
            if str(meta.get("domain", "")).lower() not in self.domains:
                return False
        if self.path_contains or self.file_types:
            path = str(meta.get("source_file", "")).lower()
            path_ok = any(s in path for s in self.path_contains)
            ft_ok = str(meta.get("file_type", "")).lower() in self.file_types
            if not (path_ok or ft_ok):
                return False
        return True


class ScopeRouter:
    def __init__(
        self,
        domain_signals: dict[str, list[str]] | None,
        content_signals: dict[str, dict] | None,
    ):
        # domain code -> compiled alias patterns
        self._domains = {
            str(dom).lower(): _compile_signals(list(aliases or []))
            for dom, aliases in (domain_signals or {}).items()
        }
        # content label -> {compiled signals, path substrings, file types}
        self._content: dict[str, dict] = {}
        for label, spec in (content_signals or {}).items():
            spec = spec or {}
            self._content[str(label)] = {
                "compiled": _compile_signals(list(spec.get("signals") or [])),
                "path_contains": [str(s).lower() for s in (spec.get("path_contains") or [])],
                "file_types": [str(s).lower() for s in (spec.get("file_types") or [])],
            }

    @classmethod
    def from_config(cls, cfg: Config) -> "ScopeRouter":
        return cls(
            cfg.get("retrieval.domain_signals", {}) or {},
            cfg.get("retrieval.content_signals", {}) or {},
        )

    def detect(self, query: str) -> Scope:
        q = query.lower()
        scope = Scope()
        for dom, compiled in self._domains.items():
            for sig, pat in compiled:
                if pat.search(q):
                    scope.domains.append(dom)
                    scope.labels.append(f"domain:{dom}({sig})")
                    break
        for label, spec in self._content.items():
            for sig, pat in spec["compiled"]:
                if pat.search(q):
                    scope.path_contains.extend(spec["path_contains"])
                    scope.file_types.extend(spec["file_types"])
                    scope.labels.append(f"content:{label}({sig})")
                    break
        if scope:
            log.info("scope routing: %s", ", ".join(scope.labels))
        return scope
