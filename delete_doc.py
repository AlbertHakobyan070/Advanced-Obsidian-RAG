"""
delete_doc.py — Preview which documents a substring would remove from the RAG
index. PREVIEW ONLY: this script never deletes anything.

It used to delete from ChromaDB (--confirm) and rebuild the sparse index
(--rebuild), and that could not work: the chunk JSONLs in data/ still held the
rows, and the sparse index is derived FROM them. The next rebuild_bm25.py (it
unions every data/*_chunks.jsonl), or any `index --append` of that file, brought
every "deleted" chunk back. A delete has to remove the Chroma chunks AND the
JSONL rows; the console does exactly that (POST /api/documents/delete, the
Documents tab), so that is the one supported way. --confirm and --rebuild are
kept only to refuse, printing the ready-to-run call, instead of failing as
unknown arguments.

Matches a substring against either `filename` or `source_file` metadata.

Usage (from project root, inside venv):
    # what would this remove? (reads only):
    python delete_doc.py "Schedule_Spring_2025"

    # match against the full path instead of the filename:
    python delete_doc.py "Other\\09 - Failed" --field source_file

NOTE: matching is a case-insensitive substring test on the chosen field.
A broad substring can match many files — the preview is there to catch that.

WHICH STORE: the one config.yaml names — paths.chroma_dir and
paths.collection_name, located the way every other script locates the config
(./config.yaml, else the project's). That is the live index of the active
vault, and the same one the console's Documents tab deletes from. The preview
prints it, so what you read is the store the console would edit.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.chroma_client import persistent_client
from src.utils.config_loader import load_config


def _curl_example(source_files: list[str], port) -> str:
    """The console call that really deletes `source_files`, ready to paste.

    ASCII only on purpose: json.dumps writes \\uXXXX for the en dash in a vault
    path, so no console code page can mangle the body on its way to the server.
    The body is DeleteIn (manage_api.py): the exact source_file values from the
    preview, and whether to queue the sparse rebuild."""
    body = json.dumps({"source_files": source_files, "rebuild": True})
    quoted = body.replace('"', '\\"')            # inside -d "..." a quote is written \"
    return (f'curl.exe -s -X POST http://127.0.0.1:{port}/api/documents/delete '
            f'-H "Content-Type: application/json" -d "{quoted}"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="substring to match (case-insensitive)")
    ap.add_argument("--field", choices=["filename", "source_file"],
                    default="filename", help="metadata field to match against")
    ap.add_argument("--confirm", action="store_true",
                    help="REFUSED: this script is preview-only; the message it "
                         "prints names the supported delete")
    ap.add_argument("--rebuild", action="store_true",
                    help="REFUSED: see --confirm")
    args = ap.parse_args()

    cfg = load_config()
    chroma_dir = cfg.path("paths.chroma_dir")
    coll_name = cfg.get("paths.collection_name", "obsidian_vault")
    c = persistent_client(chroma_dir).get_collection(coll_name)
    total = c.count()
    print(f"collection '{coll_name}' at {chroma_dir}: {total} chunks")

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

    how = (f"To delete these {len(by_file)} file(s) from the index use the console's "
           f"Documents tab, or POST /api/documents/delete: it removes the Chroma "
           f"chunks AND the JSONL rows (a delete that leaves the rows is undone by "
           f"the next rebuild) and queues the sparse rebuild. Ready to run, with the "
           f"console up:\n  {_curl_example(sorted(by_file), cfg.get('webui.port', 8052))}")
    if args.confirm or args.rebuild:
        sys.exit(f"\nREFUSED: delete_doc.py is preview-only and deleted nothing. {how}")
    print(f"\nPREVIEW ONLY — nothing deleted. {how}")


if __name__ == "__main__":
    main()
