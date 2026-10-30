"""
canvas_refresh.py — Refresh the metadata of already-indexed canvas chunks from a
fresh canvas ingest, WITHOUT re-embedding anything.

WHY
  New canvas loader keys (canvas_edge_files, and canvas_edges directions that
  follow the arrowheads) must reach the canvas chunks that are already indexed.
  A REPLACE (delete + ingest + index --append) would re-embed every canvas chunk
  for an identical text and write fresh loader metadata over the console's
  retags. This copies only the canvas-owned keys (OWNED) from the fresh ingest:
  no re-embed, no id churn, no lost retags.

NEVER
  It never adds or deletes an id, never changes chunk text, never touches a
  non-canvas row, and never touches course, domain, tags or filename (nothing
  outside OWNED). A canvas whose text changed is a REPLACE, and the plan says so.

WRITER
  It is the store's only writer while it runs (slice B's single-writer rule), so
  it refuses while :8051, :8052 or Streamlit answers, and --apply refuses while
  any collection is mid re-embed (index_ops.refuse_if_building).

USAGE
    python main.py ingest-canvas --output <a file OUTSIDE the data folder>
    python canvas_refresh.py --new FILE            # the plan: a report, writes nothing
    python canvas_refresh.py --new FILE --apply    # + --allow-unindexed-new if the plan lists only_new

APPLY ORDER (a failure in step 2 heals by re-running --apply: merged_meta is
idempotent, so the JSONL pass then changes nothing and writes no file)
  1. every chunk JSONL's canvas rows, via index_ops.rewrite_jsonl (a
     <name>.canvas-refresh-<stamp>.bak beside each file it changes);
  2. every chunk collection (the active one, eg2 re-embeds; not HyPE, not the
     graph twin, whose metadata is never read), via collection.update(): metadata
     only, and a None value deletes an OWNED key the fresh ingest dropped;
  3. one journal line in logs/canvas_refresh.jsonl.

AFTER RUNNING
  python rebuild_bm25.py with the servers still down (the sparse pickle keeps
  its own metadata copy), then restart :8051.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import index_ops
from src.embeddings.embedder import Embedder
from src.embeddings.sidecar import role_for
from src.ingestion.canvas_loader import decode_canvas_edges, decode_canvas_files
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import load_config

BATCH = 500             # ids per Chroma get/update

# The keys the canvas loader owns: the only keys this tool may change.
OWNED = ("canvas_node_id", "canvas_neighbors", "canvas_edge_labels", "canvas_edges",
         "canvas_file_refs", "canvas_edge_files", "canvas_degree", "canvas_group",
         "canvas_part", "canvas_part_count", "provenance")


def merged_meta(base: dict, new: dict) -> dict:
    """base with its OWNED keys replaced by new's: an OWNED key new lacks is gone,
    every other key of base (a retag, the course) stays."""
    return {**{k: v for k, v in base.items() if k not in OWNED},
            **{k: new[k] for k in OWNED if k in new}}


def _owned(meta: dict) -> dict:
    return {k: meta[k] for k in OWNED if k in meta}


def _rest(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k not in OWNED}


def plan(jsonl: dict, chroma: dict, new: dict, active: str) -> dict:
    """What an apply would change, and what forbids it. Pure. jsonl and new map
    id -> (text, meta); chroma maps target -> {id: (text or None, meta)}, with the
    active collection's documents read."""
    act = chroma[active]
    both = [i for i in sorted(new) if i in act]
    files = [(i, f) for i in sorted(new) for f in decode_canvas_files(new[i][1].get("canvas_edge_files"))]
    directions = Counter(e["direction"] for _, meta in new.values()
                         for e in decode_canvas_edges(meta.get("canvas_edges")))
    missing = {t: sorted(set(jsonl) - set(rows)) for t, rows in chroma.items()}
    retags = {t: [i for i in sorted(jsonl) if i in rows
                  and _rest(rows[i][1]) != _rest(Embedder._clean_meta(jsonl[i][1]))]
              for t, rows in chroma.items()}
    return {
        "stored": len(jsonl),
        "new": len(new),
        "only_stored": sorted(set(jsonl) - set(new)),
        "only_new": sorted(set(new) - set(jsonl)),
        "missing_in_target": {t: ids for t, ids in missing.items() if ids},
        "text_changed": [i for i in both if new[i][0] != act[i][0]],
        "diverged": [i for i in sorted(jsonl) if i in act and jsonl[i][0] != act[i][0]],
        "meta_changed": [i for i in both if _owned(Embedder._clean_meta(new[i][1])) != _owned(act[i][1])],
        # the footers canvas.footer_subpath would change: file-node entries (node id set) with a subpath
        "subpaths": sorted({i for i, f in files if f["node"] and f["subpath"]}),
        "source_entries": sum(1 for _, f in files if not f["node"]),
        "directions": dict(sorted(directions.items())),
        # kept, never pushed: eg2 targets list the retags they never received
        "retags_kept": {t: ids for t, ids in retags.items() if ids},
        "collections": list(chroma),
    }


def _read_jsonl(path, keep) -> None:
    """Every parsed record of `path` to keep(rec), with rewrite_jsonl's own parsing
    (dry run: nothing is written), so the plan and the apply see the same rows."""
    def take(rec):
        keep(rec)
        return None
    index_ops.rewrite_jsonl(path, take, dry_run=True)


def _stored_canvas_rows(cfg) -> tuple[dict, list]:
    """Every canvas row of every chunk JSONL, id -> (text, meta), and the ids found on two rows."""
    rows, dup = {}, set()

    def keep(rec):
        meta = rec.get("metadata") or {}
        if meta.get("file_type") == "canvas":
            if rec.get("doc_id") in rows:
                dup.add(rec.get("doc_id"))
            rows[rec.get("doc_id")] = (rec.get("text"), meta)

    for f in index_ops.chunk_jsonls(cfg):
        _read_jsonl(f, keep)
    return rows, sorted(dup)


def _read_target(col, ids: list, documents: bool) -> dict:
    inc = ["documents", "metadatas"] if documents else ["metadatas"]
    out = {}
    for k in range(0, len(ids), BATCH):
        got = col.get(ids=ids[k:k + BATCH], include=inc)
        docs = got["documents"] if documents else [None] * len(got["ids"])
        out.update({i: (d, m or {}) for i, d, m in zip(got["ids"], docs, got["metadatas"])})
    return out


def _first(ids) -> str:
    return ", ".join(str(i) for i in ids[:5])


def _print_plan(p: dict) -> None:
    """Counts, and the first 5 ids of every non-empty list."""
    for k, v in p.items():
        if k in ("missing_in_target", "retags_kept"):
            for line in [f"{t} {len(ids)} (first: {_first(ids)})" for t, ids in v.items()] or ["0"]:
                print(f"  {k} {line}")
        elif k == "collections":
            print(f"  {k} {', '.join(v)}")
        elif isinstance(v, list):
            print(f"  {k} {len(v)}" + (f" (first: {_first(v)})" if v else ""))
        else:
            print(f"  {k} {v if v not in ('', {}) else 'none'}")


def _refusals(p: dict, active: str, allow_new: bool) -> list[str]:
    """Every reason --apply must not write, one line each."""
    out = []
    if p["only_stored"]:
        out.append(f"{len(p['only_stored'])} stored canvas row(s) are not in --new (first: "
                   f"{_first(p['only_stored'])}): applying would orphan them. A node was removed, or the "
                   f"canvas lane now skips that canvas (parser.skip_dirs / parser.skip_roots): a human call")
    if p["text_changed"]:
        out.append(f"{len(p['text_changed'])} canvas row(s) changed text (first: {_first(p['text_changed'])}): "
                   f"REPLACE those canvases first (console delete, then ingest-canvas --include-path + "
                   f"index --append with every server down), then re-plan")
    if p["diverged"]:
        out.append(f"{len(p['diverged'])} canvas row(s) differ in text between the chunk JSONLs and {active} "
                   f"(first: {_first(p['diverged'])}): pre-existing dense/sparse drift, a human call")
    if p["duplicate_stored"]:
        out.append(f"{len(p['duplicate_stored'])} canvas id(s) sit on two chunk JSONL rows (first: "
                   f"{_first(p['duplicate_stored'])}): the sparse union serves whichever file sorts first; "
                   f"keep one row per id, then re-plan")
    for t, ids in p["missing_in_target"].items():
        fix = ("dense/sparse drift in the active collection, a human call" if t == active else
               f"top it up with python main.py index --rebuild-into {t} --embedding <the provider its "
               f"sidecar records>, then re-plan")
        out.append(f"{t} lacks {len(ids)} stored canvas id(s) (first: {_first(ids)}): {fix}")
    if p["building"]:
        out.append(p["building"])
    if p["only_new"] and not allow_new:
        out.append(f"{len(p['only_new'])} row(s) of --new are not indexed (first: {_first(p['only_new'])}): "
                   f"the refresh never adds ids. Add --allow-unindexed-new to refresh the rest, then REPLACE "
                   f"those canvases after rebuild_bm25.py")
    return out


def _apply(cfg, client, p: dict, new: dict, targets: list[str], ids: list[str]) -> dict:
    """Steps 1-3 of APPLY ORDER. Returns {jsonl name: changed rows}."""
    seen = 0

    def fix(rec):
        nonlocal seen
        meta = rec.get("metadata") or {}
        if meta.get("file_type") != "canvas":
            return None
        seen += 1
        merged = merged_meta(meta, new.get(rec.get("doc_id"), (None, meta))[1])
        return {**rec, "metadata": merged} if merged != meta else None    # text untouched; never DROP

    changed = {}
    for f in index_ops.chunk_jsonls(cfg):
        n = index_ops.rewrite_jsonl(f, fix, tag="canvas-refresh")
        if n:
            changed[f] = n
    if seen != p["stored"]:
        raise RuntimeError(f"the chunk JSONLs hold {seen} canvas row(s), the plan read {p['stored']}: a JSONL "
                           f"changed between read and write. Restore its .canvas-refresh-*.bak and re-plan")

    for t in targets:
        col = client.get_collection(t)
        for k in range(0, len(ids), BATCH):
            got = col.get(ids=ids[k:k + BATCH], include=["metadatas"])
            metas = []
            for i, m in zip(got["ids"], got["metadatas"]):
                m = m or {}
                upd = merged_meta(m, Embedder._clean_meta(new[i][1]))
                upd.update({key: None for key in OWNED if key in m and key not in upd})   # None deletes
                metas.append(upd)
            if got["ids"]:
                col.update(ids=got["ids"], metadatas=metas)

    logs = cfg.project_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "plan": {k: (len(v) if isinstance(v, list) else
                          {t: len(x) for t, x in v.items()} if k in ("missing_in_target", "retags_kept") else v)
                      for k, v in p.items()},
             # escaped: a '[' or ']' in a chunk file name is a glob character class
             "jsonl": {f.name: {"changed": n,
                                "backup": sorted(f.parent.glob(glob.escape(f.name) + ".canvas-refresh-*.bak"))[-1].name}
                       for f, n in changed.items()},
             "targets": targets}
    with open(logs / "canvas_refresh.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {f.name: n for f, n in changed.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Refresh indexed canvas metadata from a fresh ingest (no re-embed).")
    ap.add_argument("--new", required=True, metavar="FILE",
                    help="A fresh `main.py ingest-canvas --output FILE`, written outside the data folder.")
    ap.add_argument("--apply", action="store_true", help="Write. Without it: the plan, and nothing is written.")
    ap.add_argument("--allow-unindexed-new", action="store_true",
                    help="Apply although --new holds ids the index lacks: they are listed, never added.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    up = index_ops.answering_ports(cfg)
    if up:
        ports = ", ".join(":" + str(p) for p in up)
        print(f"ERROR: {ports} answers. Stop :8051, :8052 and any Streamlit session first (single-writer "
              f"rule: a live reader blocks Chroma's writes)", file=sys.stderr)
        sys.exit(2)
    data_dir = cfg.path("paths.chunks_file").resolve().parent
    if os.path.normcase(str(Path(args.new).resolve().parent)) == os.path.normcase(str(data_dir)):
        print(f"ERROR: --new {args.new} sits in the data folder {data_dir}: every *_chunks.jsonl there joins "
              f"the sparse union (embedder.build_sparse_union). Write the fresh ingest outside it",
              file=sys.stderr)
        sys.exit(2)

    jsonl, duplicates = _stored_canvas_rows(cfg)
    new = {}
    _read_jsonl(args.new, lambda rec: new.update({rec.get("doc_id"): (rec.get("text"), rec.get("metadata") or {})}))

    chroma_dir = cfg.path("paths.chroma_dir")
    active = cfg.get("paths.collection_name", "obsidian_vault")
    client = persistent_client(chroma_dir)
    try:
        index_ops.refuse_if_building(chroma_dir, client)
        building = ""
    except index_ops.StoreBusy as e:
        building = str(e)
    targets = sorted(c.name for c in client.list_collections() if role_for(cfg, c.name) == "chunks")
    if active not in targets:
        print(f"ERROR: paths.collection_name {active!r} is not a chunk collection in {chroma_dir}",
              file=sys.stderr)
        sys.exit(2)
    ids = sorted(jsonl)
    chroma = {t: _read_target(client.get_collection(t), ids, documents=t == active) for t in targets}
    p = plan(jsonl, chroma, new, active)
    p["duplicate_stored"], p["building"] = duplicates, building

    print("CANVAS REFRESH " + ("(APPLY)" if args.apply else "PLAN (nothing is written; --apply writes)"))
    _print_plan(p)
    if not args.apply:
        return
    refusals = _refusals(p, active, args.allow_unindexed_new)
    if refusals:
        for line in refusals:
            print(f"ERROR: {line}", file=sys.stderr)
        sys.exit(2)

    done = _apply(cfg, client, p, new, targets, ids)
    print(f"Refreshed {p['stored']} canvas row(s): jsonl "
          + (", ".join(f"{name} {n}" for name, n in done.items()) or "unchanged")
          + f" | collections {', '.join(targets)}")
    print("NEXT: python rebuild_bm25.py (servers still down), then restart :8051.")


if __name__ == "__main__":
    main()
