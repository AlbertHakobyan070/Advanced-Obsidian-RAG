"""Extraction templates: node kinds, edge types and guidance, from templates.yaml."""
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .schema import Extraction

_PATH = Path(__file__).with_name("templates.yaml")


@dataclass(frozen=True)
class Template:
    name: str
    purpose: str
    node_kinds: tuple[str, ...]
    edge_types: tuple[str, ...]
    layer_edges: tuple[str, ...]
    guidance: str


def load_templates() -> dict[str, Template]:
    raw = yaml.safe_load(_PATH.read_text(encoding="utf-8"))
    return {
        name: Template(
            name=name, purpose=t["purpose"].strip(),
            node_kinds=tuple(t["node_kinds"]), edge_types=tuple(t["edge_types"]),
            layer_edges=tuple(t["layer_edges"]), guidance=t["guidance"].strip())
        for name, t in raw.items()
    }


def check_extraction(ext: Extraction, t: Template) -> list[str]:
    """Human-readable errors for node kinds or edge types outside the template, and for
    node titles with no letters or digits (no id can be made from them)."""
    errs = [f"node title {n.title!r} has no letters or digits" for n in ext.nodes if not re.search(r"[^\W_]", n.title)]
    errs += [f"node '{n.title}': kind '{n.kind}' is not in template '{t.name}' (allowed: {', '.join(t.node_kinds)})"
             for n in ext.nodes if n.kind not in t.node_kinds]
    errs += [f"edge '{e.source}' -> '{e.target}': type '{e.type}' is not in template '{t.name}' (allowed: {', '.join(t.edge_types)})"
             for e in ext.edges if e.type not in t.edge_types]
    return errs
