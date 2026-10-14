"""The tantivy sparse backend: same rows, same tokenizer, refused when stale.

The contracts worth pinning:

  1. It ranks rows of the BM25 pickle, with embedder._tokenize's split, so a
     token bm25s indexes is a token tantivy indexes.
  2. An index that does not match the pickle (another row count or other ids)
     is REFUSED, not searched: its rows are positions in the pickle, so a stale
     one returns the wrong chunks while looking fine. A missing one names the
     job that builds it, and fails the retriever's build, not the first query.
  3. While the backend is tantivy, the pickle's writer rebuilds the index from
     the same rows, so the two cannot drift.
  4. A filtered lane (scope/code) filters tantivy's hits and still returns at
     most top_k.

Measured parity with bm25s on the real corpus lives in the PR that added this,
not here: these tests pin behaviour, not ranking quality.
"""
import json
import pickle

import pytest

pytest.importorskip("tantivy")

from src.embeddings.embedder import _tokenize, build_sparse_union      # noqa: E402
from src.retrieval import sparse_tantivy                               # noqa: E402
from src.retrieval.retriever import HybridRetriever                    # noqa: E402

DOCS = [
    ("a1", "ARIMA models difference a series to remove trend", {"file_type": "pdf"}),
    ("b2", "Gradient boosting fits trees to residuals", {"file_type": "py"}),
    ("c3", "Café ARIMA(1,1,0)-model in statsmodels", {"file_type": "py"}),
    ("d4", "Seasonal differencing removes yearly patterns", {"file_type": "md"}),
]


def _payload():
    return {"bm25": None, "ids": [d[0] for d in DOCS], "documents": [d[1] for d in DOCS],
            "metadatas": [d[2] for d in DOCS]}


@pytest.fixture
def built(tmp_path):
    pkl = tmp_path / "bm25_index.pkl"
    p = _payload()
    sparse_tantivy.build(pkl, p["ids"], p["documents"])
    return pkl, p


def _retriever(tmp_path, pkl, **kw):
    return HybridRetriever(embedder=None, chroma_dir=tmp_path, bm25_index=pkl,
                           collection_name="c", sparse_backend="tantivy", **kw)


def test_it_ranks_the_matching_rows_with_the_bm25_tokenizer(built):
    pkl, p = built
    s = sparse_tantivy.TantivySparse(pkl, p["ids"])
    assert {r for r, _ in s.search(_tokenize("arima"), 4)} == {0, 2}
    # No stemming, as in bm25s: "differencing" does not match row 0's "difference".
    assert [r for r, _ in s.search(_tokenize("differencing"), 4)] == [3]
    # "Café" splits to "caf" under _tokenize; tantivy's analyzer must agree.
    assert [r for r, _ in s.search(_tokenize("Café"), 4)] == [2]


def test_a_stale_index_is_refused_not_searched(built):
    pkl, p = built
    with pytest.raises(sparse_tantivy.TantivyIndexError, match="another BM25 pickle"):
        sparse_tantivy.TantivySparse(pkl, p["ids"][:-1])
    with pytest.raises(sparse_tantivy.TantivyIndexError, match="another BM25 pickle"):
        sparse_tantivy.TantivySparse(pkl, list(reversed(p["ids"])))


def test_a_missing_index_names_the_job(tmp_path):
    with pytest.raises(sparse_tantivy.TantivyIndexError, match="build_tantivy"):
        sparse_tantivy.TantivySparse(tmp_path / "bm25_index.pkl", ["x"])


def test_a_rebuild_replaces_the_index_and_leaves_no_staging_folder(built):
    pkl, p = built
    sparse_tantivy.build(pkl, p["ids"][:2], p["documents"][:2])
    meta = json.loads((sparse_tantivy.index_dir(pkl) / "noetrix_meta.json").read_text())
    assert meta["count"] == 2
    assert not sparse_tantivy.index_dir(pkl).with_name(
        sparse_tantivy.index_dir(pkl).name + ".building").exists()


def test_an_unknown_backend_or_pool_is_refused(tmp_path):
    with pytest.raises(ValueError, match="sparse_backend"):
        HybridRetriever(embedder=None, chroma_dir=tmp_path, bm25_index=tmp_path / "x",
                        collection_name="c", sparse_backend="lucene")
    with pytest.raises(ValueError, match="tantivy_filter_pool"):
        _retriever(tmp_path, tmp_path / "x", tantivy_filter_pool=0)


def test_the_sparse_lane_returns_pickle_rows_and_filters_a_lane(tmp_path, built):
    pkl, p = built
    r = _retriever(tmp_path, pkl)
    r._bm25_payload = p
    hits = r._sparse_search("arima", 5)
    assert {h[0] for h in hits} == {"a1", "c3"}
    assert all(h[1] == p["documents"][p["ids"].index(h[0])] for h in hits)
    code = r._sparse_search("arima differencing", 5, predicate=lambda m: m["file_type"] == "py")
    assert [h[0] for h in code] == ["c3"]
    assert len(r._sparse_search("arima differencing removes", 1)) == 1


def test_the_retriever_build_fails_without_an_index(tmp_path):
    from src.utils.config_loader import Config
    cfg = Config({"paths": {"chroma_dir": str(tmp_path), "bm25_index": str(tmp_path / "b.pkl")},
                  "retrieval": {"sparse_backend": "tantivy"}}, tmp_path)
    with pytest.raises(sparse_tantivy.TantivyIndexError, match="build_tantivy"):
        HybridRetriever.from_config(cfg, embedder=None)


def test_the_pickle_writer_rebuilds_a_matching_index(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("".join(json.dumps({"doc_id": i, "text": t, "metadata": m}) + "\n"
                              for i, t, m in DOCS), encoding="utf-8")
    pkl = tmp_path / "bm25_index.pkl"
    assert build_sparse_union(chunks, pkl, tantivy=True) == len(DOCS)
    ids = pickle.load(open(pkl, "rb"))["ids"]
    sparse_tantivy.TantivySparse(pkl, ids)           # matches: does not raise
    # Without the flag, no index is written (the default backend never pays for it).
    other = tmp_path / "other.pkl"
    build_sparse_union(chunks, other)
    assert not sparse_tantivy.index_dir(other).exists()
