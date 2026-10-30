"""
recalibrate_courses.py — Re-tag course metadata on already-indexed chunks
WITHOUT re-embedding or re-extracting anything.

WHY
  The course tags on PDF/notebook chunks were computed before FOLDER_COURSE_MAP
  learned the folder-name variants (NLP/RL/MA/GenAI abbreviations, the messy
  01-Passed leaf folders, etc.). That left ~79% of notebook chunks tagged
  "unknown", which silently disables the retriever's metadata_boost and domain
  routing for them. The same happens whenever the maps learn a new folder.

WHAT THIS DOES (fast, metadata-only)
  A row changes only when its stored label is what the PRE-slice rules produced
  for its path (the frozen comparator below: a machine made it and nobody has
  retagged it since) AND the current rule now gives something else. md rows are
  re-derived only where the PATH decides the label: frontmatter- and heading-
  derived labels and daily notes are left alone. Anything else is reported as
  "retag" and left alone, so a hand retag survives. A re-run after an apply
  reports nothing.
  Chunk text and doc_id are untouched, so vectors stay valid and nothing orphans.

  Default file set: pdf_chunks, lecture_chunks, other_chunks, ipynb_chunks.

  --all-files           every chunk JSONL (chunks.jsonl and *_chunks.jsonl)
  --include-path SUBSTR only rows whose source_file contains SUBSTR (case-
                        insensitive, backslashes read as /); repeatable
  --where-course NAME   only rows whose stored course_name is exactly NAME;
                        repeatable
  --report PATH         JSONL, one line per change/retag row. It lists exactly
                        what a run changes; never name it a chunk file
                        (*_chunks.jsonl or chunks.jsonl) or a sidecar.
  --rename-map JSONL    the rename_map.jsonl a `main.py relocate` applied: a
                        moved row's label is judged by the rules for its old_sf
                        (relocate moves the path, not the label it was made for)
  Filters AND together; rows outside them are untouched and unreported.

APPLY ORDER (a failure anywhere is healed by a re-run)
  1. refuse while :8051, :8052 or Streamlit answers, or a re-embed is building
     (a live reader blocks Chroma's writes). --dry-run skips this and never
     opens Chroma;
  2. per file: stream it and plan the changes;
  3. Chroma first: course_code / course_name / domain are merged into EVERY chunk
     collection's own metadata (eg2 re-embeds and the graph twin too; HyPE has no
     course) via collection.update(), which is metadata only;
  4. then the JSONL, rewritten with a <name>.recal-<stamp>.bak beside it.

WHAT IT DOES NOT TOUCH
  - chunk text / embeddings: unchanged. (The course is also baked into the text
    via build_context_header, but rewriting text would change every doc_id and
    orphan the old vectors — not worth it; metadata_boost reads metadata, which
    is exactly what we fix here.)

AFTER RUNNING
  Rebuild the sparse index so BM25's stored metadata matches:
      python rebuild_bm25.py
  (Dense/ChromaDB is updated in place by this script; sparse is a pickle that
  carries its own metadata copy, so it needs the rebuild.)

USAGE
    python recalibrate_courses.py            # apply changes
    python recalibrate_courses.py --dry-run  # report only, write nothing
    python recalibrate_courses.py --all-files --dry-run --report r.jsonl
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.ingestion.obsidian_parser as op
from src import index_ops
from src.embeddings.sidecar import role_for
from src.index_ops import chunk_jsonls, rewrite_jsonl
from src.ingestion.obsidian_parser import (detect_course_from_heading, detect_course_from_path,
                                           detect_md_course_from_path, resolve_course_metadata)
from src.utils import lane_guard
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

log = get_logger("recalibrate")

# Path-detected chunk files only (the default). chunks.jsonl needs --all-files.
TARGET_FILES = [
    "pdf_chunks.jsonl",
    "lecture_chunks.jsonl",
    "other_chunks.jsonl",
    "ipynb_chunks.jsonl",
]

BATCH = 500             # ids per Chroma get/update (tests lower it)

_SEP = re.compile(r"[\\/]")


def _parts(source_file: str) -> list[str]:
    """source_file is stored vault-relative by every loader; split on either
    separator so Windows-backslash paths and posix paths both work."""
    return [p for p in _SEP.split(source_file) if p]


# --- PRE-slice-B path rules (master 27ef688), frozen ON PURPOSE ---------------
# The comparator: a row is relabelled only when its stored label is one these
# rules produced, i.e. a machine made it and nobody retagged it since. They
# describe the past: never "update" them to match the parser.
_SLICE_B_KEYS = frozenset({"ai engineering", "ai engineering & llms", "data science (books)"})
UNKNOWN = ("unknown", "unknown", "general")


def _pre_map(part):                 # FOLDER_COURSE_MAP as it was at 27ef688
    key = part.lower().strip()
    return None if key in _SLICE_B_KEYS else op.FOLDER_COURSE_MAP.get(key)


def _pre_md_path_course(parts) -> str | None:
    """ObsidianParser._detect_course_from_path steps 2-4 at 27ef688: heading
    patterns on each part (root first), then the folder map, then the stem."""
    for part in parts:
        course = detect_course_from_heading(part)
        if course and course != "__SKIP__":
            return course
    for part in parts:
        match = _pre_map(part)
        if match:
            return match
    course = detect_course_from_heading(Path(parts[-1]).stem) if parts else None
    return None if course == "__SKIP__" else course


def _pre_ladder(parts) -> dict:
    """detect_course_from_path as it was at 27ef688, over the frozen map."""
    unknown = {"course_code": "unknown", "course_name": "unknown", "domain": "general"}
    if not op._DETECT_FROM_PATH:
        return unknown
    for part in parts:                                      # 1. exact folder-name match
        name = _pre_map(part)
        if name:
            return {"course_code": name, "course_name": name,
                    "domain": op.DOMAIN_MAP.get(name, "general")}
    for part in parts:                                      # 2. course-code regex
        m = re.search(r"(CS|DS|ENGS|BSDS|ECON)\s*\d{2,3}", part, re.IGNORECASE)
        if m:
            code = re.sub(r"\s+", " ",
                          re.sub(r"(\D)(\d)", r"\1 \2", m.group(0).upper())).strip()
            name = op.COURSE_MAP.get(code, code)
            return {"course_code": code, "course_name": name,
                    "domain": op.DOMAIN_MAP.get(name, "general")}
    for part in reversed(parts):                            # 3. keyword fallback, deepest first
        low = part.lower()
        for kw, name in op.COURSE_KEYWORDS:
            if kw in low:
                return {"course_code": name, "course_name": name,
                        "domain": op.DOMAIN_MAP.get(name, "general")}
    return unknown
# --- end of the frozen rules ---------------------------------------------------


def _triple(meta) -> tuple:
    """(course_code, course_name, domain); a missing key is None."""
    return (meta.get("course_code"), meta.get("course_name"), meta.get("domain"))


def _doc_id(rec: dict, meta: dict) -> str | None:
    did = rec.get("doc_id") or rec.get("id") or meta.get("id")
    return None if did is None else str(did)


def _target(meta: dict, made_for: str | None = None) -> tuple[str, tuple | None]:
    """What to do with one row: ("change", new triple), ("retag", None) or
    ("keep", None). "retag" = the stored label is not one a machine made.
    `made_for`: the source_file the label was made for when it is not the row's
    current one (a relocated file, --rename-map); the frozen rules judge that path."""
    parts = _parts(str(meta.get("source_file", "")))
    made_parts = _parts(made_for) if made_for else parts
    stored = _triple(meta)
    if meta.get("file_type") == "daily_note":
        return "keep", None                     # labels come from heading segments
    if meta.get("file_type") == "note":
        if "frontmatter_course" in meta:
            return "keep", None
        pre = _pre_md_path_course(made_parts)
        if pre:
            made = [_triple(resolve_course_metadata(pre))]
        elif stored[1] == "__SKIP__":
            # the pre-slice heading step stored it (the guard lacked the check); no human sets it
            # ponytail: a note whose LATER heading names a course stays unknown until its md REPLACE (phase H)
            made = [stored]
        elif stored == (None, None, None) and detect_md_course_from_path(parts) is not None:
            # the pre-slice parser found no course for that chunk, from the path or from a
            # heading above it; a newly mapped folder gives one. When the new rule finds
            # none either, the note is kept: an uncoursed note is never "changed" to unknown.
            made = [stored]
        else:
            return "keep", None                 # the label came from a body heading
        new = _triple(resolve_course_metadata(detect_md_course_from_path(parts)))
    else:
        # the pdf, notebook and code loaders run the ladder's steps 1-2 only
        # (pdf_loader.py), which give either the ladder's answer or unknown
        made = [_triple(_pre_ladder(made_parts)), UNKNOWN]
        new = _triple(detect_course_from_path(parts))
    if stored == new:
        return "keep", None                     # so a re-run after apply reports nothing
    return ("change", new) if stored in made else ("retag", None)


def _plan_file(fpath: Path, includes: list[str], courses: set, report, tally,
               renames: dict | None = None) -> dict:
    """Pass 1: stream one chunk file, apply the filters, write the report lines,
    count into `tally` and return {doc_id: new triple} for the rows to change.
    `renames`: {new_sf: old_sf} from --rename-map."""
    renames = renames or {}
    changes: dict[str, tuple] = {}

    def plan_fix(rec):
        meta = rec.get("metadata") or {}
        sf = str(meta.get("source_file", ""))
        if includes and not any(s in sf.replace("\\", "/").lower() for s in includes):
            return None
        if courses and meta.get("course_name") not in courses:
            return None
        kind, new = _target(meta, renames.get(sf))
        stored, did, ftype = _triple(meta), _doc_id(rec, meta), meta.get("file_type")
        name = fpath.name
        tally.examined[name] += 1
        was_unknown = meta.get("course_name", "unknown") == "unknown"
        tally.before[name] += was_unknown
        tally.after[name] += (new[1] == "unknown") if kind == "change" else was_unknown
        if kind == "change":
            if did is None:
                raise ValueError(f"{name}: a row of {sf} has no doc_id, so its change cannot be "
                                 f"keyed in Chroma or in the file")
            changes[did] = new
            tally.changed[name] += 1
            tally.moves[(ftype, stored[1], new[1])] += 1
        elif kind == "retag":
            tally.retag[ftype] += 1
        if kind != "keep" and report is not None:
            report.write(json.dumps({
                "kind": kind, "file": name, "doc_id": did, "source_file": sf, "file_type": ftype,
                "key": PurePosixPath(sf.replace("\\", "/")).parent.as_posix(),
                "old": list(stored), "new": list(new) if new else None}, ensure_ascii=False) + "\n")
        return None                             # pass 1 never rewrites

    rewrite_jsonl(fpath, plan_fix, dry_run=True)
    return changes


def _update_chroma(cols, changes: dict, tally) -> None:
    """Merge the three course keys into each collection's OWN metadata, so keys
    only Chroma holds survive under both merge and replace update semantics."""
    ids = list(changes)
    for col in cols:
        for i in range(0, len(ids), BATCH):
            batch = ids[i:i + BATCH]
            got = col.get(ids=batch, include=["metadatas"])
            tally.missing[col.name] += len(batch) - len(got["ids"])
            if not got["ids"]:
                continue
            col.update(ids=got["ids"], metadatas=[
                {**(m or {}), "course_code": changes[d][0], "course_name": changes[d][1],
                 "domain": changes[d][2]} for d, m in zip(got["ids"], got["metadatas"])])
            tally.updated[col.name] += len(got["ids"])


def _apply_fix(changes: dict):
    def apply_fix(rec):
        meta = rec.get("metadata") or {}
        new = changes.get(_doc_id(rec, meta))
        if new is None:
            return None
        return {**rec, "metadata": {**meta, "course_code": new[0], "course_name": new[1],
                                    "domain": new[2]}}
    return apply_fix


def main():
    ap = argparse.ArgumentParser(description="Re-tag course metadata in place (no re-embed).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    ap.add_argument("--all-files", action="store_true",
                    help="Every chunk JSONL (chunks.jsonl and *_chunks.jsonl), not just the "
                         "path-detected four.")
    ap.add_argument("--include-path", action="append", default=[], metavar="SUBSTR",
                    help="Only rows whose source_file contains SUBSTR (case-insensitive, "
                         "backslashes read as /). Repeatable.")
    ap.add_argument("--where-course", action="append", default=[], metavar="NAME",
                    help="Only rows whose stored course_name is exactly NAME. Repeatable.")
    ap.add_argument("--report", default=None, metavar="PATH",
                    help="JSONL: one line per change or retag row.")
    ap.add_argument("--rename-map", default=None, metavar="JSONL",
                    help="The rename_map.jsonl a main.py relocate applied: a moved row's label is "
                         "judged by the rules for the path it was made for (its old_sf).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.report:
        # Before anything is opened: the report is opened "w" below, dry run or not, so a
        # chunk file named here is truncated. Windows opens "chunks.jsonl." and
        # "chunks.jsonl " as chunks.jsonl, hence the alias check first.
        try:
            lane_guard.refuse_alias(args.report)
        except ValueError as e:
            print(f"ERROR: --report: {e}", file=sys.stderr)
            sys.exit(2)
        name, canonical = Path(args.report).name.lower(), cfg.path("paths.chunks_file").name.lower()
        if name == canonical or name.endswith("_chunks.jsonl"):
            print(f"ERROR: --report must not be named *_chunks.jsonl or {canonical}: it would "
                  f"truncate a chunk file, and a *_chunks.jsonl in the data folder would join "
                  f"the sparse union (contract 1.3)", file=sys.stderr)
            sys.exit(2)
        if name in lane_guard.sidecar_names(cfg):
            print(f"ERROR: --report must not be named {Path(args.report).name}: it would truncate "
                  f"the parents or HyPE sidecar. Name it e.g. recal_report.jsonl", file=sys.stderr)
            sys.exit(2)
    try:
        renames = ({m["new_sf"]: m["old_sf"] for m in index_ops.load_map(args.rename_map)[0]}
                   if args.rename_map else {})
    except index_ops.RelocateError as e:
        print(f"ERROR: --rename-map: {e}", file=sys.stderr)
        sys.exit(2)
    data_dir = cfg.path("paths.chunks_file").parent
    chroma_dir = cfg.path("paths.chroma_dir")
    collection_name = cfg.get("paths.collection_name", "obsidian_vault")

    cols = []
    if not args.dry_run:
        up = index_ops.answering_ports(cfg)
        if up:
            ports = ", ".join(":" + str(p) for p in up)
            print(f"ERROR: {ports} answers. Stop :8051, :8052 and Streamlit first "
                  f"(a live reader blocks Chroma writes)", file=sys.stderr)
            sys.exit(2)
        client = persistent_client(chroma_dir)
        try:
            index_ops.refuse_if_building(chroma_dir, client)
        except index_ops.StoreBusy as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)
        client.get_collection(collection_name)          # the served collection must exist: raises
        # every collection that holds chunks; HyPE metadata has no course
        cols = [client.get_collection(c.name) for c in client.list_collections()
                if role_for(cfg, c.name) != "hype"]
        log.info("ChromaDB collections to update: %s", ", ".join(c.name for c in cols))

    files = chunk_jsonls(cfg) if args.all_files else [data_dir / f for f in TARGET_FILES]
    includes = [s.replace("\\", "/").lower() for s in args.include_path]
    courses = set(args.where_course)
    tally = SimpleNamespace(examined=Counter(), changed=Counter(), before=Counter(),
                            after=Counter(), moves=Counter(), retag=Counter(),
                            updated=Counter(), missing=Counter())

    with (open(args.report, "w", encoding="utf-8") if args.report else contextlib.nullcontext()) as report:
        for fpath in files:
            if not fpath.exists():
                log.info("skip %s (not present)", fpath.name)
                continue

            changes = _plan_file(fpath, includes, courses, report, tally, renames)
            name, n_change = fpath.name, tally.changed[fpath.name]
            log.info("%s: %d/%d re-tagged (unknown %d -> %d)", name, n_change,
                     tally.examined[name], tally.before[name], tally.after[name])
            if args.dry_run or not n_change:
                continue

            _update_chroma(cols, changes, tally)                    # Chroma first
            n = rewrite_jsonl(fpath, _apply_fix(changes), tag="recal")
            if n != n_change:
                raise RuntimeError(f"{name}: rewrote {n} rows but planned {n_change}; the file "
                                   f"changed under the run, or a doc_id repeats with different "
                                   f"labels. Restore {name}.recal-*.bak and re-run.")

    print(f"\n{'=' * 60}")
    print("  COURSE RECALIBRATION " + ("(DRY RUN)" if args.dry_run else "COMPLETE"))
    print(f"{'=' * 60}")
    print(f"  Chunks examined:  {sum(tally.examined.values())}")
    print(f"  Chunks re-tagged: {sum(tally.changed.values())}")
    print(f"  Unknown before:   {sum(tally.before.values())}")
    print(f"  Unknown after:    {sum(tally.after.values())}")
    for (ftype, old, new), n in tally.moves.most_common():
        print(f"    {n:>7}  {ftype}: {old or '(none)'} -> {new}")
    if tally.retag:
        print(f"  Held back (stored label is not one a machine made; reported, not changed): "
              f"{sum(tally.retag.values())}")
        for ftype, n in tally.retag.most_common():
            print(f"    {n:>7}  {ftype}")
    if not args.dry_run:
        for col in cols:
            print(f"  Chroma {col.name}: updated {tally.updated[col.name]}, "
                  f"missing {tally.missing[col.name]}")
    print(f"{'=' * 60}")
    if args.dry_run:
        print("  Dry run — no files or indexes changed.")
    else:
        print("  Dense (ChromaDB): updated in place.")
        print("  Next: python rebuild_bm25.py   (refresh sparse index metadata)\n")


if __name__ == "__main__":
    main()
