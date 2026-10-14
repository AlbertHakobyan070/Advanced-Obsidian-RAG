"""
sparse_tantivy.py — the BM25 lane ranked by tantivy instead of bm25s.

EXPERIMENTAL and OPTIONAL, behind `retrieval.sparse_backend: tantivy` (the
default stays `bm25s`). `tantivy` is a Rust search engine with Python bindings;
it is deliberately NOT in requirements.txt (pip install tantivy==0.26.2), and
nothing here is imported or loaded while the backend is bm25s.

WHAT IT REPLACES, AND WHAT IT DOES NOT. Only the ranking. The BM25 pickle
stays the source of every row's text and metadata (the console, /stats and
graph mode read it), so the index stores no text: one row number per document,
in the pickle's order. Measured on the dev split against the same corpus and
tokenizer: no metric moved significantly, and the lane went from ~176 ms to
~7 ms at the median (bm25s scores every document and the retriever sorted
them all in Python).

THE INDEX LIVES BESIDE THE PICKLE, at `<paths.bm25_index>.tantivy/`, so a
vault switch, which carries bm25_index, carries it too. It is built from the
pickle's payload by the `build_tantivy` job, and rebuilt whenever the pickle
is (build_sparse_union, the full index) while the backend is tantivy.

A STALE INDEX IS REFUSED, NOT SEARCHED. Its rows are positions in the pickle,
so an index built from another pickle would hand back the wrong chunks while
looking fine. noetrix_meta.json records the row count and a digest of the ids;
a mismatch raises TantivyIndexError naming the job that fixes it.

Tokenizer: the same split as embedder._tokenize (runs of ASCII letters and
digits, lowercased), so both backends see an identical vocabulary. Scoring is
tantivy's BM25 (k1 1.2, b 0.75) against bm25s's lucene variant (k1 1.5), which
is why the two rankings agree closely but not exactly.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

TANTIVY_VERSION = "0.26.2"
SPARSE_BACKENDS = ("bm25s", "tantivy")
_TOKENIZER = "noetrix_ascii_alnum"


class TantivyIndexError(RuntimeError):
    """The tantivy index is missing, stale, or the package is not installed."""


def check_backend(name: str) -> str:
    if name not in SPARSE_BACKENDS:
        raise ValueError(f"retrieval.sparse_backend must be one of {SPARSE_BACKENDS}, "
                         f"got {name!r}")
    return name


def index_dir(bm25_index: Path) -> Path:
    return Path(str(bm25_index) + ".tantivy")


def ids_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()[:16]


def _tantivy():
    try:
        import tantivy
    except ImportError as e:
        raise TantivyIndexError(
            f"retrieval.sparse_backend is tantivy but the package is not installed "
            f"({e}): pip install tantivy=={TANTIVY_VERSION}") from e
    return tantivy


def _register(index, tantivy) -> None:
    index.register_tokenizer(_TOKENIZER, tantivy.TextAnalyzerBuilder(
        tantivy.Tokenizer.regex(r"[A-Za-z0-9]+")).filter(tantivy.Filter.lowercase()).build())


def build(bm25_index: Path, ids: list[str], texts: list[str]) -> int:
    """(Re)build the index for this pickle's rows. Written to a sibling folder
    and swapped in at the end, so a failed build leaves the old index whole."""
    tantivy = _tantivy()
    final = index_dir(bm25_index)
    tmp = final.with_name(final.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("text", stored=False, tokenizer_name=_TOKENIZER, index_option="freq")
    sb.add_unsigned_field("row", stored=True, indexed=False)
    index = tantivy.Index(sb.build(), path=str(tmp))
    _register(index, tantivy)
    writer = index.writer(heap_size=300_000_000)
    for row, text in enumerate(texts):
        writer.add_document(tantivy.Document(text=text or "", row=row))
    writer.commit()
    writer.wait_merging_threads()
    (tmp / "noetrix_meta.json").write_text(json.dumps({
        "count": len(ids), "ids_sha256_16": ids_digest(ids),
        "tantivy_version": TANTIVY_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    return len(ids)


class TantivySparse:
    """Top-k rows of the pickle for a token list, scored by tantivy's BM25."""

    def __init__(self, bm25_index: Path, ids: list[str]):
        tantivy = _tantivy()
        d = index_dir(bm25_index)
        meta_p = d / "noetrix_meta.json"
        if not meta_p.is_file():
            raise TantivyIndexError(
                f"no tantivy index at {d}: run the build_tantivy job "
                "(python build_tantivy.py) or set retrieval.sparse_backend: bm25s")
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("count") != len(ids) or meta.get("ids_sha256_16") != ids_digest(ids):
            raise TantivyIndexError(
                f"the tantivy index at {d} was built from another BM25 pickle "
                f"({meta.get('count')} rows, the pickle has {len(ids)}): rebuild it "
                "with the build_tantivy job before searching")
        self._tantivy = tantivy
        self._index = tantivy.Index.open(str(d))
        _register(self._index, tantivy)
        self._searcher = self._index.searcher()

    def search(self, tokens: list[str], k: int) -> list[tuple[int, float]]:
        """[(row, score)] best first, at most k, only rows that matched a term."""
        if not tokens or k <= 0:
            return []
        t = self._tantivy
        schema = self._index.schema
        q = t.Query.boolean_query(
            [(t.Occur.Should, t.Query.term_query(schema, "text", tok)) for tok in tokens])
        hits = self._searcher.search(q, limit=k).hits
        return [(self._searcher.doc(addr)["row"][0], score) for score, addr in hits]
