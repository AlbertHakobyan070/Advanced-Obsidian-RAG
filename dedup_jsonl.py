"""
dedup_jsonl.py — Remove duplicate doc_id lines from a chunks JSONL file, in place.

ChromaDB's upsert rejects a batch containing two records with the same id, so a
chunk file must have unique doc_ids before `index --append`. This keeps the first
occurrence of each doc_id and drops the rest.

    python dedup_jsonl.py data/lecture_chunks.jsonl
"""
import json
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("usage: python dedup_jsonl.py <path-to.jsonl>")
        sys.exit(1)
    path = Path(sys.argv[1])
    lines = path.read_text(encoding="utf-8").split("\n")
    seen, kept, dupes = set(), [], 0
    for line in lines:
        if not line.strip():
            continue
        rec = json.loads(line)
        did = rec.get("doc_id") or rec.get("id")
        if did is None:
            # no stable id -> can't judge duplication; keep the record
            kept.append(line)
            continue
        if did in seen:
            dupes += 1
            continue
        seen.add(did)
        kept.append(line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"{path.name}: kept {len(kept)}, removed {dupes} duplicate(s)")

if __name__ == "__main__":
    main()
