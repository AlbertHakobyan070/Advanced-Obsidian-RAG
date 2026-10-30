"""src/utils/chroma_client.py: every collection of a store stays readable on Windows.

Run:  python -m pytest tests/test_chroma_client.py -q

chromadb 1.5.9 sizes its in-memory HNSW pool from msvcrt.dll's stdio limit, and at the
default 512 it evicts an index even from a store of a few collections. A small collection
written in this process was never persisted, so once evicted every read of its vectors
fails "Error creating hnsw segment reader: Nothing found on disk". With 20 collections
every fresh store lost one (15 of 15 measured) before persistent_client raised the limit.
A real store under tmp_path, explicit vectors, no model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.utils.chroma_client import WINDOWS_STDIO_LIMIT, persistent_client

pytest.importorskip("chromadb")


@pytest.fixture
def release_chroma():
    """Stop the store this test opened: its loaded collections hold C-runtime file
    streams (4 per HNSW segment) until its System stops."""
    from chromadb.api.shared_system_client import SharedSystemClient
    before = set(SharedSystemClient._identifier_to_system)
    yield
    for key in set(SharedSystemClient._identifier_to_system) - before:
        SharedSystemClient._identifier_to_refcount.pop(key, None)
        SharedSystemClient._identifier_to_system.pop(key).stop()


def test_no_collection_of_a_store_is_evicted_beyond_reloading(tmp_path, release_chroma):
    client = persistent_client(tmp_path / "chroma")
    if sys.platform == "win32":                       # the pool Chroma sized from the limit
        assert client._server.hnsw_cache_size >= WINDOWS_STDIO_LIMIT // 5
    cols = []
    for k in range(20):
        col = client.get_or_create_collection(f"col{k:02d}", metadata={"hnsw:space": "cosine"})
        col.add(ids=["a", "b"], embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], documents=["a", "b"])
        col.get(include=["embeddings"])               # loads it into the pool
        cols.append(col)
    for col in cols:                                  # an evicted one: "Nothing found on disk"
        assert len(col.get(include=["embeddings"])["embeddings"]) == 2, col.name
