"""
chroma_client.py — the one place this repo opens a Chroma store.

Chroma's anonymised product telemetry stays OFF: the corpus is personal (lecture
notes, homework, notebooks) and Chroma reports usage unless a client is built with
Settings(anonymized_telemetry=False).

It has to be ONE place, not a setting repeated at each call site. Chroma caches a
client system per store path and raises ValueError ("An instance of Chroma already
exists for <path> with different settings") when a second client on that path
disagrees with the first, so a single site that forgot the setting would turn the
next client opened in the same process into a crash, not just a telemetry leak.
tests/test_chroma_telemetry.py fails if any module builds a client another way.

Usage:
    from src.utils.chroma_client import persistent_client
    client = persistent_client(cfg.path("paths.chroma_dir"))
"""
from __future__ import annotations

import sys

# chromadb 1.5.9 keeps each loaded HNSW index in an in-memory pool that, on Windows, it
# sizes from msvcrt.dll's stdio limit: 512 // 5 = 102 slots, split across cache shards,
# so a store of only a few collections can have one evicted. An evicted index reloads
# from disk as of its last persist (every hnsw:sync_threshold adds, never on close) and
# misses every add since: a small collection fails "Error creating hnsw segment reader:
# Nothing found on disk", a large one loses its newest rows ("Error finding id"). Which
# indexes collide is fixed by the store's segment ids, so one store fails every time and
# another never. 2048 is the most msvcrt.dll accepts (409 slots: no eviction measured at
# 20 collections). The UCRT, whose streams the HNSW files really use (4 per loaded
# index), gets the same limit, so the slots Chroma sizes from it have streams behind them.
WINDOWS_STDIO_LIMIT = 2048


def persistent_client(path):
    """chromadb.PersistentClient(path) with anonymised telemetry off. chromadb is
    imported here, on first use, as every call site always did, so importing this
    module costs nothing. On Windows it first raises the stdio limit Chroma sizes its
    HNSW pool from (WINDOWS_STDIO_LIMIT): Chroma reads it as it builds a store's system."""
    import chromadb
    from chromadb.config import Settings
    if sys.platform == "win32":
        import ctypes
        for name, crt in (("msvcrt", ctypes.windll.msvcrt), ("ucrtbase", ctypes.CDLL("ucrtbase"))):
            if crt._getmaxstdio() < WINDOWS_STDIO_LIMIT and crt._setmaxstdio(WINDOWS_STDIO_LIMIT) == -1:
                raise OSError(f"{name}._setmaxstdio({WINDOWS_STDIO_LIMIT}) failed: Chroma would size its "
                              f"HNSW pool from {crt._getmaxstdio()} and can evict an index it cannot reload")
    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False))
