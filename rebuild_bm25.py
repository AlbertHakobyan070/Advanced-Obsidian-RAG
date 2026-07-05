"""
rebuild_bm25.py — Rebuild ONLY the BM25 sparse index, low-memory.

The sparse index is DERIVED: it can always be rebuilt from the JSONLs (the
source of truth). This is the standalone entry point — it loads no embedding
model and never touches ChromaDB, so it is also the low-RAM recovery path
when `index --append` finishes its dense half but dies during the sparse
rebuild (e.g. the 2026-07-03 MemoryError).

The actual union/dedupe/build logic lives in ONE place —
src.embeddings.embedder.build_sparse_union — shared with `index --append`,
so the two paths cannot drift apart again. It streams every JSONL
record-by-record (no whole-file read_text) and unions chunks.jsonl + every
data/*_chunks.jsonl, deduped by doc_id.

Run from the project root:

    python rebuild_bm25.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.embeddings.embedder import build_sparse_union

log = get_logger("rebuild_bm25")


def main():
    cfg = load_config(None)
    chunks_file = cfg.path("paths.chunks_file")
    bm25_index = cfg.path("paths.bm25_index")
    n = build_sparse_union(chunks_file, bm25_index)
    if n == 0:
        log.error("No chunks found under %s — nothing indexed.", Path(chunks_file).parent)
        sys.exit(3)
    log.info("✅ Sparse index rebuilt: %d docs -> %s", n, bm25_index)


if __name__ == "__main__":
    main()
