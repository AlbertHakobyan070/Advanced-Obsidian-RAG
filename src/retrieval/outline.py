"""outline.py: what the vault holds, for an agent that does not know it (route-then-cite §5.7).

build_outline is GET /outline's whole answer: the primary sources grouped by domain and
course, with chunk and page counts, and the generated files apart, as maps. :8051 builds it
once at startup from the bm25 payload the provenance check already read, so a restart
refreshes it. source_outline is GET /outline/{source}: one source's chapters with page
ranges, or one map's concepts with the primary pages each cites (the route sidecar).
Both read existing metadata only; agent-instruction files appear in neither.
"""
from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING

from src.ingestion.canvas_loader import norm_vault_path
from src.utils.provenance import provenance_of

if TYPE_CHECKING:
    from src.retrieval.route import RoutePointers


def build_outline(payload: dict, rule, pointers: "RoutePointers | None") -> dict:
    """{"groups": [{domain, course, sources: [{source_file, file_type, chunks, pages}]}],
    "maps": [{source_file, chunks}]}, from payload["metadatas"] only. Rows group, not files,
    so a daily note with lectures of two courses is a source under each. A file is one
    entry under norm_vault_path, whatever spellings it is stored under, named by the first.
    pages counts the distinct pages a source's chunks cover (None with no page metadata).
    The rows the rule calls generated are counted as maps, never as sources;
    agent-instruction files are in neither. pointers is not read: a map's targets are
    source_outline's."""
    sources: dict[tuple[str, str, str], dict] = {}
    maps: dict[str, dict] = {}
    files: dict[str, tuple[bool, str]] = {}      # stored spelling -> (agent file?, norm_vault_path)
    for m in payload["metadatas"]:
        sf = m.get("source_file")
        if not sf:
            continue
        if sf not in files:
            files[sf] = (rule.instruction_file(sf), norm_vault_path(sf))
        agent, norm = files[sf]
        if agent:
            continue
        if provenance_of(m, rule) == "generated":
            maps.setdefault(norm, {"source_file": sf, "chunks": 0})["chunks"] += 1
            continue
        # The keys and defaults GET /stats counts by.
        key = (str(m.get("domain") or "unknown"), str(m.get("course_name") or m.get("course") or "unknown"), norm)
        s = sources.setdefault(key, {"source_file": sf, "file_type": m.get("file_type"), "chunks": 0, "pages": set()})
        s["chunks"] += 1
        if m.get("page_start") is not None and m.get("page_end") is not None:
            s["pages"].update(range(m["page_start"], m["page_end"] + 1))
    groups: list[dict] = []
    for (domain, course, _), s in sorted(sources.items()):
        if not groups or (groups[-1]["domain"], groups[-1]["course"]) != (domain, course):
            groups.append({"domain": domain, "course": course, "sources": []})
        groups[-1]["sources"].append({"source_file": s["source_file"], "file_type": s["file_type"],
                                      "chunks": s["chunks"], "pages": len(s["pages"]) or None})
    return {"groups": groups, "maps": [maps[norm] for norm in sorted(maps)]}


def source_outline(payload: dict, source_file: str, rule, pointers: "RoutePointers | None") -> dict:
    """One file, found by norm_vault_path whatever its spelling, under its stored one. A file
    with primary rows is a source: {source_file, chapters: [{chapter, page_start, page_end,
    chunks}]}. A map (every row generated): {source_file, map: True, concepts: [{label,
    targets}]}, each target list the deduped union, in order, of its rows' sidecar targets
    ([] with no sidecar). KeyError (the route's 404) for an agent-instruction file or a file
    the payload does not hold."""
    if rule.instruction_file(source_file):
        raise KeyError(f"{source_file!r} is an agent-instruction file: neither evidence nor a map")
    key = norm_vault_path(source_file)
    metas = payload["metadatas"]
    # ponytail: a scan of the payload per call (~0.2 s over 160k synthetic rows); index it
    # by norm_vault_path (as HybridRetriever._file_rows does) if it shows in /stats latency.
    rows = [i for i, m in enumerate(metas) if norm_vault_path(m.get("source_file") or "") == key]
    if not rows:
        raise KeyError(f"{source_file!r} is not in the index: GET /outline lists its sources and maps")
    stored = metas[rows[0]]["source_file"]
    primary = [i for i in rows if provenance_of(metas[i], rule) != "generated"]
    if primary:
        return {"source_file": stored, "chapters": _chapters(metas, primary)}
    return {"source_file": stored, "map": True, "concepts": _concepts(payload, rows, pointers)}


def _chapters(metas: list[dict], rows: list[int]) -> list[dict]:
    """Paged rows (a PDF) in page order by their ToC `chapter` (one flattened level-1/2
    title per page; None before the first entry); other rows (a note) in document order by
    their top-level heading (_top_headings), with null pages. One entry per run, so a
    title that recurs (a "Summary" closing every chapter) keeps each run's own page range."""
    if any(metas[i].get("page_start") is not None for i in rows):
        rows = sorted(rows, key=lambda i: (metas[i].get("page_start") or 0, i))
        names = [metas[i].get("chapter") for i in rows]
    else:
        names = _top_headings([metas[i] for i in rows])
    out = []
    for name, run in groupby(zip(names, rows), key=lambda pair: pair[0]):
        run = [metas[i] for _, i in run]
        out.append({"chapter": name,
                    "page_start": min((m["page_start"] for m in run if m.get("page_start") is not None), default=None),
                    "page_end": max((m["page_end"] for m in run if m.get("page_end") is not None), default=None),
                    "chunks": len(run)})
    return out


def _top_headings(metas: list[dict]) -> list:
    """Each row's chapter, in document order: the heading of the section at the note's
    shallowest heading level (above 0) that holds it, so a note opening at H2 has one
    chapter per H2. heading_path cannot say this: the parser stacks raw levels, so there a
    note's first H2 heads every later one. A row before the first such section (a
    preamble) keeps its own heading."""
    level = lambda m: m.get("heading_level") or 0
    top = min((level(m) for m in metas if level(m) > 0), default=None)
    out, current = [], None
    for m in metas:
        if level(m) == top:
            current = m.get("heading")
        out.append(current if current is not None else m.get("heading"))
    return out


def _concepts(payload: dict, rows: list[int], pointers: "RoutePointers | None") -> list[dict]:
    """A canvas node (rows grouped by canvas_node_id, so split parts are one concept) or a
    note heading, in first-seen order."""
    metas = payload["metadatas"]
    concepts: dict = {}
    for i in rows:
        m = metas[i]
        concepts.setdefault(m.get("canvas_node_id") if m.get("file_type") == "canvas" else m.get("heading"),
                            []).append(i)
    out = []
    for members in concepts.values():
        if metas[members[0]].get("file_type") == "canvas":
            part1 = min(members, key=lambda i: metas[i].get("canvas_part") or 1)
            label = _canvas_label(payload["documents"][part1])
        else:
            label = metas[members[0]].get("heading")
        targets: list[dict] = []
        for i in members:
            for t in (pointers.targets(payload["ids"][i]) if pointers is not None else None) or []:
                if t not in targets:
                    targets.append(t)
        out.append({"label": label, "targets": targets})
    return out


def _canvas_label(text: str) -> str:
    """A node's first body line: its text cut at the Connections footer (which inlines
    neighbour text), less the "[Canvas: ...]" context header line."""
    body = text.split("\n\nConnections:", 1)[0]
    if body.startswith("[Canvas:"):
        body = body.partition("\n")[2]
    return body.strip().partition("\n")[0].strip()
