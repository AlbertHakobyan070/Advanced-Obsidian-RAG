"""
evidence_pack.py — one call for a multi-part question (POST /evidence_pack).

The calling agent is the reasoner: it splits the question into 2-4
sub-questions, and this runs RAGPipeline.graph_search once per sub-question
(the search, plus a graph walk when asked) and merges the results into one
pack. Each evidence item says where it came from (provenance), which
sub-questions it covers, and, for a walked chunk, the path that reached it.
No LLM runs here, so the pack works with the generation proxy down.

    sub-question -> graph_search -> min_score floor -> merge by evidence id
    -> coverage / gaps / generated_only / provenance_mix
"""
from __future__ import annotations

from src.generation.generator import Generator
from src.utils.provenance import PROVENANCE


def _clip(text: str, n: int) -> str | None:
    if not n:
        return None
    t = (text or "").strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def build_evidence_pack(rag, question, sub_questions, *, search_kwargs, graph, top_k,
                        min_score, include_text, max_sub_questions) -> dict:
    """Search (and walk) once per sub-question; one evidence list merged by id.
    Indexed evidence only: omnisearch is refused, because a live excerpt cannot be
    checked with GET /chunks/{id}. An evidence id is what that route serves:
    parent:<pid> for a swapped section, else the chunk id."""
    subs = [str(s).strip() for s in (sub_questions or [question])]
    if not all(subs):
        raise ValueError("an empty sub-question: send each sub_questions entry as a "
                         "non-empty question, or omit sub_questions to search q alone")
    if len(subs) > max_sub_questions:
        raise ValueError(f"{len(subs)} sub-questions, more than evidence_pack.max_sub_questions "
                         f"({max_sub_questions}): merge some, or raise that key in config.yaml")
    if search_kwargs.get("omnisearch"):
        raise ValueError("omnisearch cannot feed an evidence pack: a pack holds indexed "
                         "evidence only, and live excerpts cannot be checked with "
                         "GET /chunks/{id}. Send omnisearch false or omit it")
    kw = {**search_kwargs, "omnisearch": False, "top_k": top_k}

    items: dict[str, dict] = {}
    paths: dict[str, dict] = {}          # walked chunk id -> the path that reached it
    coverage: dict[str, list[str]] = {}
    dropped: dict[str, int] = {}
    retrieval: list[dict] = []
    for sq in subs:
        docs, info = rag.graph_search(sq, graph=graph, **kw)
        g = info["graph"]
        rows = list(zip(docs, g["provenance"]))
        if min_score is not None:
            unscored = sum(1 for d in docs if d.rerank_score is None)
            if unscored:
                raise ValueError(
                    f"min_score needs a scoring rerank mode: {unscored} doc(s) of {sq!r} carry no "
                    f"rerank_score (rerank mode {info.get('rerank_mode')!r}; 'none' scores nothing, "
                    "and neighbor_context pages are added unscored). Pick another rerank mode, "
                    "or drop min_score")
            kept = [(d, p) for d, p in rows if d.rerank_score >= min_score]
            dropped[sq] = len(rows) - len(kept)
            rows = kept
        else:
            dropped[sq] = 0
        # Only a walked chunk has a via_path; a search hit is a direct hit ([]), and
        # so is a routed page (the walk's paths end at its map): found_via is its trail.
        walked = set(g["added_ids"])
        via = {p["to"]: p for p in g["paths"] if p["to"] in walked}
        coverage[sq] = []
        for d, prov in rows:
            pid = d.debug.get("parent_swap")
            eid = f"parent:{pid}" if pid else d.id
            score = d.rerank_score if d.rerank_score is not None else d.score
            path = via.get(d.id)
            if path is not None:
                paths.setdefault(d.id, path)
            if eid in items:
                item = items[eid]
                item["score"] = max(item["score"], score)
                if sq not in item["sub_questions"]:
                    item["sub_questions"].append(sq)
                if not item["via_path"] and path is not None:
                    item["via_path"] = path["nodes"]
                seen = {f["id"] for f in item["found_via"]}     # union by id, first seen first
                item["found_via"] += [f for f in d.debug.get("found_via") or [] if f["id"] not in seen]
            else:
                m = d.metadata
                course = m.get("course_name")
                items[eid] = {
                    "id": eid, "origin_id": d.id, "text": _clip(d.text, include_text),
                    "source": Generator._source_label(m, d.source_label),
                    "source_file": m.get("source_file"), "file_type": m.get("file_type"),
                    "pages": ([m.get("page_start"), m.get("page_end")]
                              if m.get("page_start") is not None else None),
                    "heading": m.get("heading"),
                    "course": None if not course or str(course).lower() == "unknown" else course,
                    "provenance": prov, "score": score,
                    "via_path": path["nodes"] if path is not None else [],
                    "found_via": list(d.debug.get("found_via") or []),
                    "sub_questions": [sq],
                }
            if eid not in coverage[sq]:
                coverage[sq].append(eid)
        retrieval.append({"q": sq, "rerank_mode": info.get("rerank_mode"),
                          "timings": info.get("timings"), "graph_stats": g["stats"],
                          # search()'s routing, and the walk's own (graph=true): never summed
                          "route": info.get("route"), "graph_route": g.get("route")})

    evidence = list(items.values())
    return {
        "question": question,
        "sub_questions": subs,
        "evidence": evidence,
        "paths": list(paths.values()),       # only those some evidence uses
        "coverage": coverage,
        "gaps": [sq for sq in subs if not coverage[sq]],
        "generated_only": [sq for sq in subs if coverage[sq]
                           and all(items[i]["provenance"] == "generated" for i in coverage[sq])],
        "provenance_mix": {p: sum(1 for e in evidence if e["provenance"] == p) for p in PROVENANCE},
        "dropped_below_min_score": dropped,
        "graph": graph is not None,
        "retrieval": retrieval,
    }
