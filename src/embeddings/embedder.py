"""
embedder.py — Turn chunks.jsonl into searchable indexes.

Builds two indexes from the parser's JSONL output:
  1. Dense  : ChromaDB collection (vector similarity)
  2. Sparse : BM25 index (exact-term / keyword matching), pickled to disk

Embedding provider is swappable (mirrors llm_client):
  openai -> text-embedding-3-small via API (1536-dim, cheap)
  local  -> sentence-transformers model on-device (free, no API)

Usage:
    from src.embeddings.embedder import Embedder
    emb = Embedder.from_config(cfg)
    emb.build_indexes()          # reads chunks.jsonl, writes chroma + bm25
"""
from __future__ import annotations

import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Cheap whitespace+alnum tokenizer for BM25."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """
    Drop-in replacement for rank_bm25.BM25Okapi, backed by bm25s (scipy-sparse,
    ~10x less memory — rank_bm25 builds pure-Python dicts that OOM at ~90K
    technical docs). Exposes the same methods the retriever may call:
    get_scores / get_batch_scores / get_top_n. Picklable (bm25s stores numpy +
    scipy-sparse arrays), so the existing pickle payload shape is unchanged.

    Ranking is standard BM25; absolute scores differ slightly from Okapi but
    feed into RRF by rank position, so fusion output is effectively identical.
    """

    def __init__(self, tokenized_corpus: list[list[str]], method: str = "lucene"):
        import bm25s
        self._bm = bm25s.BM25(method=method)
        # bm25s needs at least one non-empty doc to build a vocabulary.
        self._bm.index(tokenized_corpus or [[""]])

    def get_scores(self, query_tokens):
        import numpy as np
        toks = list(query_tokens)
        if not toks:
            return np.zeros(self._bm.scores["num_docs"], dtype=float)
        return self._bm.get_scores(toks)

    def get_batch_scores(self, query_tokens, doc_ids):
        scores = self.get_scores(query_tokens)
        return [float(scores[i]) for i in doc_ids]

    def get_top_n(self, query_tokens, documents, n: int = 5):
        import numpy as np
        scores = self.get_scores(query_tokens)
        top = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top]


@dataclass
class Chunk:
    """One retrievable unit, as emitted by obsidian_parser.py."""
    id: str
    text: str
    metadata: dict[str, Any]

    @classmethod
    def from_jsonl_record(cls, rec: dict, idx: int) -> "Chunk":
        # The parser writes {text, metadata: {...}}. Be tolerant of shape.
        text = rec.get("text") or rec.get("content") or ""
        meta = rec.get("metadata", {}) or {}
        cid = rec.get("doc_id") or rec.get("id") or meta.get("id") or f"chunk_{idx:06d}"
        return cls(id=str(cid), text=text, metadata=meta)


# ---------------------------------------------------------------------------
#  Streaming JSONL + shared sparse-union rebuild (memory-bounded)
# ---------------------------------------------------------------------------

def iter_jsonl_records(path: Path) -> Iterator[dict]:
    """Stream records without loading the file: bytes split on b'\\n' ONLY
    (chunk text contains U+2028/U+2029/\\x85), 1MB blocks. read_text() on the
    255MB pdf_chunks.jsonl is what MemoryError'd the 2026-07-03 20:38 append
    on this 16GB machine."""
    buf = b""
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            buf += block
            while True:
                nl = buf.find(b"\n")
                if nl < 0:
                    break
                line, buf = buf[:nl], buf[nl + 1:]
                if line.strip():
                    try:
                        yield json.loads(line.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        continue
    if buf.strip():
        try:
            yield json.loads(buf.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            pass


def build_sparse_union(chunks_file: Path, bm25_index: Path,
                       extra: Path | None = None) -> int:
    """
    THE sparse rebuild (used by both `index --append` and rebuild_bm25.py so
    the two can't diverge again): BM25 over chunks.jsonl + every
    data/*_chunks.jsonl (+ the just-appended file), deduped by doc_id,
    written as the same pickle payload shape the retriever loads.

    Memory: streams each file record-by-record and keeps only the payload
    lists (ids/texts/metas) — no whole-file strings, no Chunk objects. The
    token corpus + bm25s matrices are the irreducible footprint.
    """
    chunks_file = Path(chunks_file)
    data_dir = chunks_file.parent
    candidates = [chunks_file] + ([Path(extra)] if extra else [])
    candidates += sorted(data_dir.glob("*_chunks.jsonl"))

    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for src in candidates:
        src = Path(src)
        key = str(src.resolve())
        if key in seen_paths or not src.exists():
            continue
        seen_paths.add(key)
        n_before = len(ids)
        for i, rec in enumerate(iter_jsonl_records(src)):
            c = Chunk.from_jsonl_record(rec, i)
            if c.id in seen_ids:
                continue
            seen_ids.add(c.id)
            ids.append(c.id)
            texts.append(c.text)
            metas.append(c.metadata)
        log.info("  %s: +%d chunks (total %d)", src.name, len(ids) - n_before, len(ids))
    del seen_ids

    log.info("Tokenizing %d documents...", len(ids))
    tokenized = [_tokenize(t) for t in texts]
    log.info("Building bm25s index (low-memory)...")
    bm25 = BM25Index(tokenized)
    del tokenized

    payload = {"bm25": bm25, "ids": ids, "documents": texts, "metadatas": metas}
    bm25_index = Path(bm25_index)
    bm25_index.parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_index, "wb") as f:
        pickle.dump(payload, f)
    write_sparse_meta(bm25_index, len(ids))
    log.info("Sparse index rebuilt over union: %d docs", len(ids))
    return len(ids)


def write_sparse_meta(bm25_index: Path, count: int) -> Path:
    """Sidecar next to the pickle with the doc count + build time, so health
    checks can report the sparse count WITHOUT unpickling the multi-GB payload
    (which would spike RAM on the 16 GB box)."""
    meta_path = Path(str(bm25_index) + ".meta.json")
    meta_path.write_text(json.dumps({
        "count": count,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }), encoding="utf-8")
    return meta_path


# ---------------------------------------------------------------------------
#  Embedding backends
# ---------------------------------------------------------------------------

class _OpenAIEmbedding:
    def __init__(self, model: str, api_key: str, dimensions: int | None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self.client.embeddings.create(**kwargs)
        return [d.embedding for d in resp.data]


class _LocalEmbedding:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        log.info("Loading local embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


# ---------------------------------------------------------------------------
#  Embedder
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(
        self,
        backend,
        chunks_file: Path,
        chroma_dir: Path,
        bm25_index: Path,
        collection_name: str,
        batch_size: int = 100,
    ):
        self.backend = backend
        self.chunks_file = chunks_file
        self.chroma_dir = chroma_dir
        self.bm25_index = bm25_index
        self.collection_name = collection_name
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, cfg: Config) -> "Embedder":
        provider = cfg.get("embedding.provider", "openai")
        if provider == "openai":
            backend = _OpenAIEmbedding(
                model=cfg.get("embedding.model", "text-embedding-3-small"),
                api_key=cfg.require_secret("OPENAI_API_KEY"),
                dimensions=cfg.get("embedding.dimensions"),
            )
        elif provider == "local":
            backend = _LocalEmbedding(cfg.get("embedding.local_model", "BAAI/bge-small-en-v1.5"))
        else:
            raise ValueError(f"Unknown embedding provider: {provider!r}")

        return cls(
            backend=backend,
            chunks_file=cfg.path("paths.chunks_file"),
            chroma_dir=cfg.path("paths.chroma_dir"),
            bm25_index=cfg.path("paths.bm25_index"),
            collection_name=cfg.get("paths.collection_name", "obsidian_vault"),
            batch_size=cfg.get("embedding.batch_size", 100),
        )

    # ---- chunk loading ----

    def _iter_chunks(self) -> Iterator[Chunk]:
        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_file}. Run the parser first."
            )
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                yield Chunk.from_jsonl_record(json.loads(line), i)

    def load_chunks(self) -> list[Chunk]:
        chunks = list(self._iter_chunks())
        log.info("Loaded %d chunks from %s", len(chunks), self.chunks_file.name)
        return chunks

    # ---- index building ----

    def build_indexes(self) -> dict[str, int]:
        chunks = self.load_chunks()
        if not chunks:
            raise ValueError("No chunks to index.")
        self._build_dense(chunks)
        self._build_sparse(chunks)
        return {"chunks": len(chunks)}

    def append_indexes(self, extra_file: Path) -> dict[str, int]:
        """
        Add chunks from `extra_file` (e.g. data/pdf_chunks.jsonl) to the EXISTING
        indexes without wiping them.

        Dense: ChromaDB upsert (deterministic doc_id => re-running is idempotent).
        Sparse: BM25 has no incremental add, so we rebuild it from the union of
                the original chunks_file + extra_file.
        """
        extra_file = Path(extra_file)
        if not extra_file.exists():
            raise FileNotFoundError(f"Append source not found: {extra_file}")

        # Load the new chunks
        new_chunks = [
            Chunk.from_jsonl_record(json.loads(line), i)
            for i, line in enumerate(extra_file.read_text(encoding="utf-8").split("\n"))
            if line.strip()
        ]
        if not new_chunks:
            log.warning("No chunks found in %s", extra_file.name)
            return {"appended": 0}

        # Dedupe by id BEFORE upserting: a file can legitimately carry repeated
        # doc_ids (same source + same first-500 chars — e.g. chunks.jsonl has
        # 25), and ChromaDB rejects duplicate ids WITHIN one upsert payload
        # (DuplicateIDError mid-run = partial append). Keep first occurrence,
        # matching what the sparse union rebuild does.
        seen_ids: set[str] = set()
        unique_chunks = []
        for c in new_chunks:
            if c.id in seen_ids:
                continue
            seen_ids.add(c.id)
            unique_chunks.append(c)
        if len(unique_chunks) < len(new_chunks):
            log.info("append: %d duplicate-id row(s) in %s skipped (kept first)",
                     len(new_chunks) - len(unique_chunks), extra_file.name)
        new_chunks = unique_chunks

        self._append_dense(new_chunks)
        # Rebuild sparse from union (original vault chunks + everything appended)
        try:
            self._rebuild_sparse_from_union(extra_file)
        except MemoryError:
            log.error(
                "Sparse rebuild ran out of RAM. The DENSE half of this append "
                "IS committed (idempotent upsert) — nothing is lost or "
                "duplicated. Free memory (stop :8051/:8100/eval runs) and "
                "RETRY this job, or run rebuild_bm25.py standalone (it loads "
                "no embedding model, so it needs much less RAM)."
            )
            raise
        log.info("Appended %d chunks from %s", len(new_chunks), extra_file.name)
        return {"appended": len(new_chunks)}

    def _append_dense(self, chunks: list["Chunk"]) -> None:
        import chromadb
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        collection = client.get_collection(self.collection_name)
        total = len(chunks)
        for start in range(0, total, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            embeddings = self.backend.embed([c.text for c in batch])
            # upsert => safe to re-run; deterministic ids overwrite, not duplicate
            collection.upsert(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[self._clean_meta(c.metadata) for c in batch],
            )
            log.info("  dense append: %d/%d", min(start + self.batch_size, total), total)

    def _rebuild_sparse_from_union(self, extra_file: Path) -> None:
        """Delegates to build_sparse_union (module-level, streaming) — see it
        for the memory story. Kept as a method for call-site compatibility."""
        build_sparse_union(self.chunks_file, self.bm25_index, extra=extra_file)

    def _build_dense(self, chunks: list[Chunk]) -> None:
        import chromadb

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.chroma_dir))

        # Fresh collection each build (idempotent re-index)
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        total = len(chunks)
        for start in range(0, total, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            embeddings = self.backend.embed([c.text for c in batch])
            collection.add(
                ids=[c.id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[self._clean_meta(c.metadata) for c in batch],
            )
            log.info("  dense: embedded %d/%d", min(start + self.batch_size, total), total)
        log.info("Dense index built: %d vectors in '%s'", total, self.collection_name)

    def _build_sparse(self, chunks: list[Chunk]) -> None:
        tokenized = [_tokenize(c.text) for c in chunks]
        bm25 = BM25Index(tokenized)
        payload = {
            "bm25": bm25,
            "ids": [c.id for c in chunks],
            "documents": [c.text for c in chunks],
            "metadatas": [c.metadata for c in chunks],
        }
        self.bm25_index.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bm25_index, "wb") as f:
            pickle.dump(payload, f)
        log.info("Sparse index built: %d docs -> %s", len(chunks), self.bm25_index.name)

    @staticmethod
    def _clean_meta(meta: dict[str, Any]) -> dict[str, Any]:
        """ChromaDB metadata values must be str/int/float/bool. Coerce lists/None."""
        clean: dict[str, Any] = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, tuple)):
                clean[k] = ", ".join(str(x) for x in v)
            else:
                clean[k] = str(v)
        return clean

    # ---- query-time embedding (used by retriever) ----

    def embed_query(self, text: str) -> list[float]:
        return self.backend.embed([text])[0]
