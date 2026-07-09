"""
recalibrate_courses.py — Re-tag course metadata on already-indexed chunks
WITHOUT re-embedding or re-extracting anything.

WHY
  The course tags on PDF/notebook chunks were computed before the active
  course taxonomy learned a particular set of folder-name variants. That can
  leave a non-trivial share of notebook chunks tagged "unknown", which
  silently disables the retriever's metadata_boost and domain routing for
  them. This script re-runs the parser's course detection against the stored
  path, using the current taxonomy, and writes the result back in place.

WHAT THIS DOES (fast, metadata-only)
  For each path-detected chunk file (pdf_chunks, lecture_chunks, other_chunks,
  ipynb_chunks) it re-runs detect_course_from_path() on the stored source_file,
  rewrites course_code / course_name / domain in the JSONL, and pushes the same
  change to ChromaDB via collection.update() — which updates METADATA ONLY, no
  re-embedding. The chunk text and doc_id are untouched, so vectors stay valid
  and nothing orphans.

WHAT IT DOES NOT TOUCH
  - chunks.jsonl (markdown): course there comes from note HEADINGS, not the
    folder path, so path-based re-tagging would be wrong. Left alone.
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
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.ingestion.obsidian_parser import detect_course_from_path
from src.embeddings.embedder import Embedder

log = get_logger("recalibrate")

# Path-detected chunk files only. NOT chunks.jsonl (heading-detected markdown).
TARGET_FILES = [
    "pdf_chunks.jsonl",
    "lecture_chunks.jsonl",
    "other_chunks.jsonl",
    "ipynb_chunks.jsonl",
]

_SEP = re.compile(r"[\\/]")


def _parts(source_file: str) -> list[str]:
    """source_file is stored vault-relative by every loader; split on either
    separator so Windows-backslash paths and posix paths both work."""
    return [p for p in _SEP.split(source_file) if p]


def main():
    ap = argparse.ArgumentParser(description="Re-tag course metadata in place (no re-embed).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.path("paths.chunks_file").parent
    chroma_dir = cfg.path("paths.chroma_dir")
    collection_name = cfg.get("paths.collection_name", "obsidian_vault")

    collection = None
    present_ids: set[str] = set()
    if not args.dry_run:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(collection_name)
        # Pre-fetch existing ids so we never call update() on a missing id.
        got = collection.get(include=[])  # ids always returned
        present_ids = set(got["ids"])
        log.info("ChromaDB collection '%s' holds %d vectors.", collection_name, len(present_ids))

    grand_changed = 0
    grand_total = 0
    before_unknown = Counter()
    after_unknown = Counter()

    for fname in TARGET_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            log.info("skip %s (not present)", fname)
            continue

        records: list[dict] = []
        upd_ids: list[str] = []
        upd_metas: list[dict] = []
        changed = 0
        total = 0
        missing_in_chroma = 0

        for line in fpath.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            meta = rec.get("metadata", {}) or {}
            did = rec.get("doc_id") or rec.get("id") or meta.get("id")

            if meta.get("course_name", "unknown") == "unknown":
                before_unknown[fname] += 1

            det = detect_course_from_path(_parts(meta.get("source_file", "")))
            old = (meta.get("course_code"), meta.get("course_name"), meta.get("domain"))
            new = (det["course_code"], det["course_name"], det["domain"])

            if new != old:
                meta["course_code"] = det["course_code"]
                meta["course_name"] = det["course_name"]
                meta["domain"] = det["domain"]
                rec["metadata"] = meta
                changed += 1
                if not args.dry_run and did is not None:
                    if did in present_ids:
                        upd_ids.append(str(did))
                        upd_metas.append(Embedder._clean_meta(meta))
                    else:
                        missing_in_chroma += 1

            if meta.get("course_name", "unknown") == "unknown":
                after_unknown[fname] += 1

            records.append(rec)

        grand_total += total
        grand_changed += changed
        log.info("%s: %d/%d re-tagged (unknown %d -> %d)%s",
                 fname, changed, total,
                 before_unknown[fname], after_unknown[fname],
                 f", {missing_in_chroma} not in chroma" if missing_in_chroma else "")

        if args.dry_run:
            continue

        # Rewrite JSONL with corrected metadata.
        tmp = fpath.with_suffix(fpath.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(fpath)

        # Push metadata-only updates to ChromaDB in batches.
        for i in range(0, len(upd_ids), 500):
            collection.update(ids=upd_ids[i:i + 500], metadatas=upd_metas[i:i + 500])
        if upd_ids:
            log.info("  ChromaDB metadata updated for %d chunks.", len(upd_ids))

    print(f"\n{'=' * 60}")
    print("  COURSE RECALIBRATION " + ("(DRY RUN)" if args.dry_run else "COMPLETE"))
    print(f"{'=' * 60}")
    print(f"  Chunks examined:  {grand_total}")
    print(f"  Chunks re-tagged: {grand_changed}")
    print(f"  Unknown before:   {sum(before_unknown.values())}")
    print(f"  Unknown after:    {sum(after_unknown.values())}")
    print(f"{'=' * 60}")
    if args.dry_run:
        print("  Dry run — no files or indexes changed.")
    else:
        print("  Dense (ChromaDB): updated in place.")
        print("  Next: python rebuild_bm25.py   (refresh sparse index metadata)\n")


if __name__ == "__main__":
    main()
