"""
graph_twin.py — Graph mode's own seed index: the graph lane, embedded by another model.

`graph.embedding_provider` names an embedder: an embedding.providers entry, or
`local` when the main collection's embedder is not local. Graph mode's QUERY
seeding then searches <paths.collection_name>__graph (the "twin") instead of the
main collection. The twin holds exactly the main collection's graph.file_type
chunks: ids and metadata from the main collection, text from the chunk JSONLs
(the text the sparse index has), vectors from the graph embedder. /query,
/search, seeding by chunk id and the walk itself never read it, and nothing here
writes the main collection or a chunk file.

The twin is derived data, rebuilt FROM SCRATCH by every graph_embed (main.py
graph-embed): drop it, then rebuild_into(role="graph"). A top-up never re-embeds
an id it already holds, and a canvas chunk keeps its id when its text is edited,
so a top-up after a Graph-tab re-index would keep the old vectors.

  graph_spec      the configured graph embedder, or None (null = seed from main)
  build_twin      drop + rebuild: the whole graph_embed job
  twin_id_diff    the exact id check :8051 runs at startup (graph_expand.py)
  twin_status     the console's verdict on the twin (GET /api/canvas/embedder)

Imports no torch and no chromadb at import time, like the modules it builds on.
"""
from __future__ import annotations

import os
from typing import Callable

from src.embeddings.reembed import ReembedError, rebuild_into
from src.embeddings.registry import (
    KNOWN_EMBEDDERS, RESERVED_EMBEDDING_PROVIDERS, EmbeddingSpec, resolve_embedding_spec)
from src.embeddings.sidecar import (
    GRAPH_SUFFIX, delete_sidecar, fingerprint_diff, iter_ids, read_sidecar, sidecar_label)
from src.utils.chroma_client import persistent_client

BUILD_HINT = ("run the graph_embed job (Graph tab > Graph embedder, or main.py graph-embed), "
              "then restart :8051")


def graph_collection_name(cfg) -> str:
    """<paths.collection_name>__graph: derived from the main collection, never configured."""
    return cfg.get("paths.collection_name", "obsidian_vault") + GRAPH_SUFFIX


def configured_provider(cfg):
    """graph.embedding_provider as every reader sees it: None when it is null or
    "" (empty means unset), otherwise the value as written. The one place the
    key is read, so the job, the re-index chain, the console status and :8051
    cannot disagree about whether it is set."""
    name = cfg.get("graph.embedding_provider")
    return None if name == "" else name


def graph_spec(cfg) -> EmbeddingSpec | None:
    """The embedder graph.embedding_provider names, or None when it is unset (graph
    mode then seeds from the main collection with the main embedder). ValueError
    for a name embedding.provider could not take either, and for the main
    collection's own embedder: an index built with it would repeat the main one."""
    name = configured_provider(cfg)
    if name is None:
        return None
    registry = cfg.get("embedding.providers") or {}
    known = sorted(str(k) for k in registry) if isinstance(registry, dict) else []
    if name not in (*RESERVED_EMBEDDING_PROVIDERS, *known):
        raise ValueError(f"graph.embedding_provider = {name!r} is not in embedding.providers and "
                         f"is not one of {RESERVED_EMBEDDING_PROVIDERS}. Known providers: {known}")
    spec = resolve_embedding_spec(cfg, name)
    main = resolve_embedding_spec(cfg)
    if spec.identity() == main.identity():
        raise ValueError(
            f"graph.embedding_provider = {name!r} is the main collection's embedder "
            f"({main.label()}), so a graph index would only repeat the main collection. Set it "
            f"to null to seed from the main collection, or name another embedder.")
    return spec


def embedder_choices(cfg) -> list[dict]:
    """What graph.embedding_provider can be set to: "null" (the main embedder, no
    graph index), every embedding.providers entry, and local. Each says whether a
    graph index can be built with it and, if not, why."""
    main = resolve_embedding_spec(cfg)
    rows = [{"value": "null", "label": f"main embedder ({main.label()}): no graph index",
             "available": True, "reason": None}]
    for name in [*(cfg.get("embedding.providers") or {}), "local"]:
        try:
            spec = resolve_embedding_spec(cfg, name)
        except ValueError as e:                      # a broken entry: say so, offer the rest
            rows.append({"value": name, "label": name, "available": False, "reason": str(e)})
            continue
        reason = None
        if spec.identity() == main.identity():
            reason = "this is the main collection's embedder: choose the main embedder instead"
        elif spec.api_key_env and not spec.api_key_optional and not os.environ.get(spec.api_key_env):
            reason = f"{spec.api_key_env} is not set in this environment"
        dim = spec.dimensions or KNOWN_EMBEDDERS.get(spec.model, {}).get("dimension")
        rows.append({"value": name, "label": spec.label() + (f", {dim}-d" if dim else ""),
                     "available": reason is None, "reason": reason})
    return rows


def drop_twin(chroma_dir, name: str, client) -> bool:
    """Delete graph index `name`: the collection first, then its sidecar, so a
    failure in between leaves a sidecar that still lets the next drop through.
    Refuses (ReembedError) a collection with no graph sidecar and a sidecar of
    another role: only what graph_embed built is ever deleted. True when there
    was something to drop."""
    sidecar = read_sidecar(chroma_dir, name)        # an unreadable one raises ValueError
    exists = name in {c.name for c in client.list_collections()}
    if sidecar is not None and sidecar["role"] != "graph":
        raise ReembedError(f"{name!r} is recorded as a {sidecar['role']!r} collection, not a graph "
                           f"index, so graph_embed will not drop it")
    if exists and sidecar is None:
        raise ReembedError(f"collection {name!r} exists but has no embedding fingerprint, so "
                           f"graph_embed did not build it and will not drop it. Delete or rename "
                           f"it yourself if it is yours to delete.")
    if exists:
        client.delete_collection(name)
    if sidecar is not None:
        delete_sidecar(chroma_dir, name)
    return exists or sidecar is not None


def build_twin(cfg, embedder, *, limit: int | None = None, client=None,
               progress: Callable[[str], None] = print) -> dict:
    """The graph_embed job: drop graph mode's seed index, rebuild it from scratch
    with `embedder` and return its sidecar. `limit` is the re-embed's smoke
    option: the result stays "building", so it is never served."""
    spec = embedder.spec
    if spec is None:
        raise ValueError("this embedder carries no spec; build it with Embedder.from_config")
    if spec.identity() == resolve_embedding_spec(cfg).identity():
        raise ReembedError(f"{spec.label()} is the main collection's embedder: graph mode already "
                           f"seeds from the main collection with it, so there is no index to build")
    if limit is not None and limit < 1:              # refused before anything is dropped
        raise ReembedError(f"--limit must be a positive integer, got {limit}")
    name = graph_collection_name(cfg)
    chroma_dir = cfg.path("paths.chroma_dir")
    if client is None:
        if not chroma_dir.is_dir():
            raise ReembedError(f"no Chroma store at {chroma_dir} (paths.chroma_dir)")
        client = persistent_client(chroma_dir)
    if drop_twin(chroma_dir, name, client):
        progress(f"{name}: dropped the previous build (every graph_embed starts from scratch)")
    return rebuild_into(cfg, name, embedder, limit=limit, client=client, progress=progress,
                        role="graph")


def twin_id_diff(main_col, twin_col, file_type: str) -> tuple[set[str], set[str]]:
    """(graph-lane ids of the main collection the twin lacks, twin ids the main
    collection's graph lane lacks). Both empty = the same id set. Exact and
    cheap; text edited under a kept id is not seen (only a rebuild fixes that)."""
    lane = set(iter_ids(main_col, where={"file_type": file_type}))
    twin = set(iter_ids(twin_col))
    return lane - twin, twin - lane


def twin_status(cfg, *, client=None) -> dict:
    """GET /api/canvas/embedder less `ok`, `choices` and `files_newer`: what is
    configured, the twin's fingerprint and freshness against the main
    collection's graph lane, and a verdict with its reason. Reads only: a
    missing store or collection is reported, never created. A main embedder
    that does not resolve raises ValueError (nothing here means anything without it)."""
    configured = configured_provider(cfg)
    file_type = str(cfg.get("graph.file_type", "canvas"))
    main_name = cfg.get("paths.collection_name", "obsidian_vault")
    name = graph_collection_name(cfg)
    chroma_dir = cfg.path("paths.chroma_dir")
    out = {"configured": configured, "graph_enabled": bool(cfg.get("graph.enabled", True)),
           "file_type": file_type, "collection": name,
           "main": {"collection": main_name, "embedder": resolve_embedding_spec(cfg).label(),
                    "chunks": 0},
           "twin": None, "verdict": None, "reason": None}

    def done(verdict: str, reason: str) -> dict:
        out.update(verdict=verdict, reason=reason)
        return out

    if client is None and chroma_dir.is_dir():          # PersistentClient would create it
        client = persistent_client(chroma_dir)
    names = {c.name for c in client.list_collections()} if client is not None else set()
    main_col = client.get_collection(main_name) if main_name in names else None
    twin_col = client.get_collection(name) if name in names else None
    lane = set(iter_ids(main_col, where={"file_type": file_type})) if main_col is not None else set()
    out["main"]["chunks"] = len(lane)
    try:
        sidecar, unreadable = read_sidecar(chroma_dir, name), None
    except ValueError as e:
        sidecar, unreadable = None, str(e)

    if sidecar is not None or twin_col is not None or unreadable:
        held = set(iter_ids(twin_col)) if twin_col is not None else None
        s = sidecar or {}
        out["twin"] = {
            "exists": twin_col is not None, "status": s.get("status"),
            "provider": s.get("provider"), "embedder": sidecar_label(sidecar) if sidecar else None,
            "dimension": s.get("dimension"), "count": s.get("count"), "expected": s.get("expected"),
            "completed_at": s.get("completed_at"), "diff": None,
            "only_main": len(lane - held) if held is not None else None,
            "only_twin": len(held - lane) if held is not None else None,
        }

    if configured is None:
        return done("off", f"graph.embedding_provider is null: graph mode seeds from {main_name!r} "
                           f"with its own embedder"
                    + ("; the graph index on disk is not used" if out["twin"] else ""))
    try:
        spec = graph_spec(cfg)
    except ValueError as e:
        return done("error", str(e))
    if unreadable:
        return done("error", unreadable)
    if sidecar is None and twin_col is not None:
        return done("error", f"collection {name!r} exists but has no embedding fingerprint: "
                             f"graph_embed did not build it and will not drop it. Delete or "
                             f"rename it, then {BUILD_HINT}")
    if sidecar is not None and sidecar["role"] != "graph":
        return done("error", f"{name!r} is recorded as a {sidecar['role']!r} collection, not a "
                             f"graph index")
    if sidecar is None:
        return done("missing", f"{name!r} has not been built: {BUILD_HINT}")
    diff = out["twin"]["diff"] = fingerprint_diff(spec, spec.dimensions, sidecar, role="graph")
    if sidecar["status"] == "building":
        return done("building", f"{name!r} is an unfinished build ({sidecar.get('count')} of "
                                f"{sidecar.get('expected')} chunks): graph mode refuses it until "
                                f"a graph_embed run finishes")
    if twin_col is None:
        return done("error", f"{name!r} has a complete fingerprint but no collection: {BUILD_HINT}")
    if diff:
        return done("mismatch", f"{name!r} was built by {sidecar_label(sidecar)}, but "
                                f"graph.embedding_provider is {spec.label()}; they differ in "
                                f"{', '.join(diff)}. Rebuild it: {BUILD_HINT}")
    if out["twin"]["only_main"] or out["twin"]["only_twin"]:
        return done("stale", f"{name!r} is out of step with {main_name!r}: "
                             f"{out['twin']['only_main']} {file_type} chunk(s) are not in it and "
                             f"{out['twin']['only_twin']} are no longer in the main collection. "
                             f"Rebuild it: {BUILD_HINT}")
    return done("ready", f"graph mode seeds from {name!r}: {sidecar.get('count')} {file_type} "
                         f"chunks embedded by {sidecar_label(sidecar)}")
