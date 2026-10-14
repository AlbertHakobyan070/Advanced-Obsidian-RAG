"""
build_tantivy.py — (re)build the tantivy index beside the BM25 pickle.

The index is DERIVED from the pickle, row for row (see
src/retrieval/sparse_tantivy.py), so it can always be rebuilt and it touches
neither ChromaDB nor the JSONLs. Run it once before switching
`retrieval.sparse_backend` to tantivy, then restart :8051; while the backend
is tantivy, every pickle rebuild (rebuild_bm25, index --append, a full index)
rebuilds it as well.

Run from the project root:

    python build_tantivy.py
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.retrieval import sparse_tantivy
from src.utils.config_loader import load_config
from src.utils.console import force_utf8_console
from src.utils.logger import get_logger

log = get_logger("build_tantivy")


def main():
    force_utf8_console()
    cfg = load_config(None)
    bm25_index = cfg.path("paths.bm25_index")
    if not Path(bm25_index).is_file():
        log.error("No BM25 pickle at %s — build the sparse index first (rebuild_bm25.py).",
                  bm25_index)
        sys.exit(3)
    t0 = time.time()
    with open(bm25_index, "rb") as f:
        payload = pickle.load(f)
    n = sparse_tantivy.build(bm25_index, payload["ids"], payload["documents"])
    log.info("✅ Tantivy index built: %d rows -> %s (%.0fs)", n,
             sparse_tantivy.index_dir(bm25_index), time.time() - t0)


if __name__ == "__main__":
    main()
