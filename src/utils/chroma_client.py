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


def persistent_client(path):
    """chromadb.PersistentClient(path) with anonymised telemetry off. chromadb is
    imported here, on first use, as every call site always did, so importing this
    module costs nothing."""
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False))
