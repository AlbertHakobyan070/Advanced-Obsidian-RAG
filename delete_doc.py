"""
delete_doc.py — Safely remove documents from the RAG index.

Deletes by matching a substring against either `filename` or `source_file`
metadata. ALWAYS previews what it will delete and requires confirmation.
After deleting from ChromaDB you MUST rebuild the sparse index (the script
reminds you, and can do it with --rebuild).

Usage (from project root, inside venv):
    # preview only (default — deletes NOTHING):
    python delete_doc.py "Schedule_Spring_2025"

    # match against the full path instead of the filename:
    python delete_doc.py "Other\\09 - Failed" --field source_file

    # actually delete (after reviewing the preview):
    python delete_doc.py "some_unwanted_book" --confirm

    # delete and rebuild the sparse index in one go:
    python delete_doc.py "some_unwanted_book" --confirm --rebuild

NOTE: matching is a case-insensitive substring test on the chosen field.
A broad substring can match many files — the preview is there to catch that.
"""
import argparse
import subprocess
import sys

import chromadb

COLL = "obsidian_vault"
DB = "data/chroma_db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="substring to match (case-insensitive)")
    ap.add_argument("--field", choices=["filename", "source_file"],
                    default="filename", help="metadata field to match against")
    ap.add_argument("--confirm", action="store_true",
                    help="actually delete (without this, preview only)")
    ap.add_argument("--rebuild", action="store_true",
                    help="run rebuild_bm25.py after deleting")
    args = ap.parse_args()

    c = chromadb.PersistentClient(path=DB).get_collection(COLL)
    total = c.count()
    print(f"collection: {total} chunks")

    # Pull ids + metadata in PAGES (ChromaDB errors with 'too many SQL variables'
    # if you fetch the whole collection at once). We scan client-side because
    # ChromaDB's `where` does exact match, not substring.
    pat = args.pattern.lower()
    hits = []
    offset = 0
    PAGE = 5000
    while offset < total:
        got = c.get(limit=PAGE, offset=offset, include=["metadatas"])
        for i, m in zip(got["ids"], got["metadatas"]):
            if pat in str(m.get(args.field, "")).lower():
                hits.append((i, m))
        offset += PAGE

    if not hits:
        print(f"No chunks where {args.field} contains '{args.pattern}'. Nothing to do.")
        return

    # Summarize by distinct source_file so you see FILES, not 1000s of chunks.
    by_file: dict[str, int] = {}
    for _, m in hits:
        key = str(m.get("source_file") or m.get("filename") or "?")
        by_file[key] = by_file.get(key, 0) + 1

    print(f"\nMatched {len(hits)} chunks across {len(by_file)} file(s):")
    for f, n in sorted(by_file.items(), key=lambda x: -x[1]):
        print(f"  {n:>5}  {f}")

    if not args.confirm:
        print(f"\nPREVIEW ONLY — nothing deleted. Re-run with --confirm to delete "
              f"these {len(hits)} chunks.")
        return

    # Delete by id (exact, safe — no fuzzy where clause), in batches to avoid
    # the 'too many SQL variables' limit on large deletions.
    ids_to_delete = [i for i, _ in hits]
    B = 5000
    for k in range(0, len(ids_to_delete), B):
        c.delete(ids=ids_to_delete[k:k + B])
    print(f"\nDeleted {len(ids_to_delete)} chunks from ChromaDB.")
    print(f"collection now: {c.count()} chunks")

    print("\n*** Dense index updated. The SPARSE index (bm25) is now STALE. ***")
    if args.rebuild:
        print("Running rebuild_bm25.py ...")
        subprocess.run([sys.executable, "rebuild_bm25.py"])
    else:
        print("Run:  python rebuild_bm25.py   to resync the sparse index.")
    print("\nNOTE: the chunk JSONL file(s) in data/ still contain these rows. "
          "If you re-run `index --append` on that JSONL, deleted chunks come back. "
          "To delete permanently, also remove them from the source JSONL.")


if __name__ == "__main__":
    main()
