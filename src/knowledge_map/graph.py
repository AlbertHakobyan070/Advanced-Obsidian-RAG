"""The typed graph in memory: nodes, edges and their deterministic ids (spec 5.2, 5.3).

A node id is `<kind>-<slug(title)>`, an edge id a hash of its endpoints and type,
so the same input and config give the same ids on every run.
"""
import hashlib
import re
from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    kind: str
    title: str
    summary: str
    section: str
    unit: str
    importance: int
    quotes: list[str]
    located: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)   # (quote, pages)
    status: str = "candidate"
    summary_revised_from: str | None = None      # the extracted summary, when the verifier cut it down


@dataclass
class Edge:
    id: str
    source: str
    target: str
    type: str
    quote: str | None
    pages: tuple[int, ...] = ()
    status: str = "stated"          # stated | inferred | structural


def slug(s: str) -> str:
    """'Document Model (JSON)' -> 'document-model-json'. Unicode letters are kept."""
    out = re.sub(r"[\W_]+", "-", s.casefold()).strip("-")
    if not out:
        raise ValueError(f"cannot make an id from {s!r}: it has no letters or digits")
    return out


def node_id(kind: str, title: str) -> str:
    return f"{kind}-{slug(title)}"


def edge_id(source: str, target: str, type: str) -> str:
    return "e-" + hashlib.sha1(f"{source}|{type}|{target}".encode("utf-8")).hexdigest()[:10]
