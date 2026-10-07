"""
embedder.py — Turn chunk JSONL files into the two searchable indexes.

Builds and maintains:
  1. Dense  : ChromaDB collection (vector similarity, cosine space)
  2. Sparse : BM25 via bm25s, pickled to disk together with the ids, texts and
              metadata it was built over (the retriever loads it whole)

Two ways in, with very different reach:
  build_indexes()     FULL rebuild from paths.chunks_file ALONE (`main.py
                      index`). The Chroma collection is deleted first, so rows
                      appended from any other *_chunks.jsonl are dropped.
  append_indexes(f)   adds one more JSONL to the existing indexes
                      (`main.py index --append f`): dense upsert, then a sparse
                      rebuild over the union of every chunk file on disk.

The invariant the rest of this file protects: the dense and sparse indexes
hold the SAME set of doc_ids. Nothing errors when they drift apart — each lane
just quietly searches a different corpus — which is why malformed lines are
counted and reported rather than skipped silently.

Embedding provider is swappable (mirrors llm_client; resolved in registry.py):
  local  -> sentence-transformers model on-device (free, no API)
  openai -> LEGACY: OpenAI's own API, text-embedding-3-small (1536-dim, cheap)
  <name> -> an embedding.providers entry: any OpenAI-compatible /v1/embeddings
            endpoint (Ollama, llama.cpp, a gateway) with its own base_url

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

from src.embeddings.registry import EmbeddingSpec, resolve_embedding_spec
from src.embeddings.sidecar import (
    check_collection, delete_sidecar, ids_digest, iter_ids, make_sidecar, sidecar_path,
    stored_dimension, unstamped_warning, write_sidecar)
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """BM25 tokenizer: lowercased runs of ASCII letters and digits; every
    other character is a separator. The retriever imports this same function
    for the query side, so index and query always split identically.

    The ASCII class has a cost: text in any other script (Armenian, Cyrillic,
    Greek symbols, CJK) yields NO tokens — only the dense lane can find it —
    and an accented Latin letter splits the word it sits in."""
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

def iter_jsonl_records(path: Path, *, strict: bool = False) -> Iterator[dict]:
    """Stream records without loading the file: bytes split on b'\\n' ONLY
    (chunk text contains U+2028/U+2029/\\x85), 1MB blocks. read_text() on the
    255MB pdf_chunks.jsonl is what MemoryError'd the 2026-07-03 20:38 append
    on this 16GB machine.

    `strict` is the MALFORMED-LINE POLICY, and the two callers genuinely want
    different ones:

      strict=True   raise. Used by the APPEND path, where a bad line means the
                    file being indexed is damaged and half-indexing it would
                    put rows in the dense index that the sparse rebuild later
                    derives differently. Fail before anything is written.
      strict=False  skip, but COUNT and warn. Used by the sparse rebuild, which
                    reads every JSONL on disk: one damaged legacy file must not
                    make the index unrebuildable. The warning is the point —
                    silently dropping rows here is how the dense and sparse
                    halves drift apart, which is this project's worst failure
                    mode precisely because nothing errors.

    Either way a malformed line is now VISIBLE. It previously vanished.
    """
    skipped = 0

    def _parse(raw: bytes):
        nonlocal skipped
        try:
            return json.loads(raw.decode("utf-8", errors="replace")), True
        except json.JSONDecodeError as e:
            if strict:
                raise ValueError(
                    f"{Path(path).name}: malformed JSON on a line "
                    f"({e}). Refusing to index a damaged file — re-run the "
                    f"ingest that produced it."
                ) from e
            skipped += 1
            return None, False

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
                    rec, ok = _parse(line)
                    if ok:
                        yield rec
    if buf.strip():
        rec, ok = _parse(buf)
        if ok:
            yield rec
    if skipped:
        log.warning("%s: skipped %d malformed line(s). The sparse index will "
                    "hold fewer rows than the dense one until that file is "
                    "re-ingested.", Path(path).name, skipped)


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
    def __init__(self, model: str, api_key: str, dimensions: int | None,
                 base_url: str | None = None):
        from openai import OpenAI
        # base_url only when set: a plain OpenAI config builds the client with
        # exactly the arguments it always did.
        self.client = (OpenAI(api_key=api_key, base_url=base_url) if base_url
                       else OpenAI(api_key=api_key))
        self.model = model
        self.dimensions = dimensions

    @property
    def dimension(self) -> int | None:
        return self.dimensions or None

    def embed(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        resp = self.client.embeddings.create(**kwargs)
        return [d.embedding for d in resp.data]


class _LocalEmbedding:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        log.info("Loading local embedding model: %s", model_name)
        # No device kwarg unless one was asked for: `auto` must stay exactly
        # the call this made before the device setting existed.
        self.model = (SentenceTransformer(model_name) if device is None
                      else SentenceTransformer(model_name, device=device))

    @property
    def dimension(self) -> int | None:
        return self.model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


# ---------------------------------------------------------------------------
#  Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """Builds the indexes for one vault: the embedding backend, the Chroma
    collection and the BM25 pickle. The retriever borrows embed_query() and
    reads the same two stores. This class is not their only writer: the
    console's delete and retag, delete_doc.py, recalibrate_courses.py
    (metadata-only updates) and rebuild_bm25.py write them too."""

    # Class-level defaults, so an Embedder built without __init__ (the tests do:
    # Embedder.__new__ plus the few attributes a method needs) still has them.
    # No spec means no prefixes: the embed_* methods then hand the backend
    # exactly the texts they were given.
    spec: EmbeddingSpec | None = None
    # The HyPE question collection paired with `collection_name`, carried so the
    # fingerprint written next to a collection can record the pair.
    hype_collection_name: str | None = None

    def __init__(
        self,
        backend,
        chunks_file: Path,
        chroma_dir: Path,
        bm25_index: Path,
        collection_name: str,
        batch_size: int = 100,
        spec: EmbeddingSpec | None = None,
        hype_collection_name: str | None = None,
    ):
        self.backend = backend
        self.chunks_file = chunks_file
        self.chroma_dir = chroma_dir
        self.bm25_index = bm25_index
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.spec = spec
        self.hype_collection_name = hype_collection_name

    @property
    def query_prefix(self) -> str:
        return self.spec.query_prefix if self.spec else ""

    @property
    def doc_prefix(self) -> str:
        return self.spec.doc_prefix if self.spec else ""

    @classmethod
    def from_config(cls, cfg: Config, spec: EmbeddingSpec | None = None) -> "Embedder":
        """`spec` defaults to the embedder `cfg` configures; pass one to build
        a different embedder against the same paths and collection settings."""
        if spec is None:
            spec = resolve_embedding_spec(cfg)
        if spec.kind == "local":
            backend = _LocalEmbedding(spec.model, None if spec.device == "auto" else spec.device)
        elif spec.kind == "openai":
            # Keys come from the environment, never from config.yaml. An entry
            # with no api_key_env is a keyless local endpoint (Ollama, llama.cpp)
            # and the OpenAI SDK still wants a non-empty string; same rules as
            # the LLM registry. Legacy `openai` names OPENAI_API_KEY and
            # requires it, exactly as before.
            if spec.api_key_env:
                raw_key = (cfg.secret(spec.api_key_env) if spec.api_key_optional
                           else cfg.require_secret(spec.api_key_env))
                api_key = raw_key or "not-needed"
            else:
                api_key = "not-needed"
            backend = _OpenAIEmbedding(
                model=spec.model,
                api_key=api_key,
                dimensions=spec.dimensions,
                base_url=spec.base_url,
            )
        else:
            raise ValueError(f"Unknown embedding kind: {spec.kind!r}")

        return cls(
            backend=backend,
            chunks_file=cfg.path("paths.chunks_file"),
            chroma_dir=cfg.path("paths.chroma_dir"),
            bm25_index=cfg.path("paths.bm25_index"),
            collection_name=cfg.get("paths.collection_name", "obsidian_vault"),
            batch_size=cfg.get("embedding.batch_size", 100),
            spec=spec,
            hype_collection_name=cfg.get("retrieval.hype.collection", "hype_questions"),
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
        """Full rebuild of BOTH indexes from chunks_file alone.

        Destructive by design: _build_dense deletes the whole collection first,
        and _build_sparse pickles only these chunks. Rows that came in through
        append_indexes() from any other *_chunks.jsonl — PDFs, notebooks, code,
        canvases — are therefore gone from both indexes afterwards, and putting
        them back means re-appending (re-embedding) each of those files. Use
        append_indexes() to add a file, and build_sparse_union() /
        rebuild_bm25.py to re-sync BM25 over everything without touching the
        dense side.

        The collection's embedding fingerprint goes with its vectors: the old
        sidecar is deleted before they are, and a fresh one is written for the
        ones just built."""
        chunks = self.load_chunks()
        if not chunks:
            raise ValueError("No chunks to index.")
        delete_sidecar(self.chroma_dir, self.collection_name)
        self._build_dense(chunks)
        self._record_built_collection("index", ids=[c.id for c in chunks])
        self._build_sparse(chunks)
        return {"chunks": len(chunks)}

    def append_indexes(self, extra_file: Path) -> dict[str, int]:
        """
        Add chunks from `extra_file` (e.g. data/pdf_chunks.jsonl) to the EXISTING
        indexes without wiping them.

        Dense: ChromaDB upsert (deterministic doc_id => re-running is idempotent).
        Sparse: BM25 has no incremental add, so we rebuild it from the union of
                the original chunks_file + extra_file.

        A collection fingerprinted by another embedder (or left unfinished by
        a re-embed) refuses the append: its vectors and these would be mixed.
        """
        extra_file = Path(extra_file)
        if not extra_file.exists():
            raise FileNotFoundError(f"Append source not found: {extra_file}")

        # Stream, and dedupe in the SAME pass.
        #
        # This used to be read_text().split("\n") — the exact pattern
        # iter_jsonl_records was written to replace, and which MemoryError'd on
        # the 255MB pdf_chunks.jsonl. It held four copies of the file at once:
        # the whole-file string, the list of line strings, the Chunk list, and
        # then a second deduplicated Chunk list. Only the last of those is
        # actually needed, and it is the one the upsert consumes.
        #
        # strict=True because a malformed line here means the file being
        # indexed is damaged: better to fail before any row is written than to
        # commit a dense half the sparse rebuild will not reproduce.
        #
        # Dedupe BEFORE upserting: a file can legitimately carry repeated
        # doc_ids (same source + same first-500 chars — e.g. chunks.jsonl has
        # 25), and ChromaDB rejects duplicate ids WITHIN one upsert payload
        # (DuplicateIDError mid-run = partial append). Keep the first
        # occurrence, matching what the sparse union rebuild does.
        seen_ids: set[str] = set()
        new_chunks: list[Chunk] = []
        duplicates = 0
        for i, rec in enumerate(iter_jsonl_records(extra_file, strict=True)):
            c = Chunk.from_jsonl_record(rec, i)
            if c.id in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(c.id)
            new_chunks.append(c)
        del seen_ids

        if not new_chunks:
            log.warning("No chunks found in %s", extra_file.name)
            return {"appended": 0}
        if duplicates:
            log.info("append: %d duplicate-id row(s) in %s skipped (kept first)",
                     duplicates, extra_file.name)

        state = self._prepare_dense_write()
        self._append_dense(new_chunks)
        if state == "new":
            self._record_built_collection("append")
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

    def _prepare_dense_write(self) -> str:
        """Decide, before any vector is written, whether this embedder may add
        to the collection. "stamped": it carries a fingerprint and ours matches
        (a mismatch or an unfinished re-embed raises: two embedders' vectors
        in one collection is the state the fingerprint exists to forbid).
        "new": no fingerprint and nothing stored, so this write creates the
        collection and the caller records who built it. "unstamped": a legacy
        collection that already holds vectors, appended to as it always was,
        with a warning; only `stamp`, after verifying, may vouch for it."""
        if check_collection(self, self.chroma_dir, self.collection_name) is not None:
            return "stamped"
        if self.chroma_dir.is_dir():                 # PersistentClient would create a missing one
            from chromadb.errors import NotFoundError
            client = persistent_client(self.chroma_dir)
            try:
                existing = client.get_collection(self.collection_name)
            except NotFoundError:
                return "new"
            if existing.count() > 0:
                log.warning(unstamped_warning(
                    self.collection_name,
                    sidecar_path(self.chroma_dir, self.collection_name), self.spec))
                return "unstamped"
        return "new"

    def _record_built_collection(self, written_by: str, ids: list[str] | None = None) -> None:
        """Write the fingerprint for the collection this embedder just filled.
        The dimension is measured from a stored vector, not taken from the
        model's word for it. `ids` is the id set when the caller already holds
        it; otherwise it is read back from the collection."""
        if self.spec is None:
            raise ValueError("this embedder carries no spec, so it cannot record which "
                             "embedder built the collection; build it with Embedder.from_config")
        client = persistent_client(self.chroma_dir)
        col = client.get_collection(self.collection_name)
        path = write_sidecar(self.chroma_dir, self.collection_name, make_sidecar(
            self.spec, collection=self.collection_name, role="chunks", status="complete",
            dimension=stored_dimension(col), count=col.count(), written_by=written_by,
            source_digest=ids_digest(ids if ids is not None else iter_ids(col)),
            hype_collection=self.hype_collection_name))
        log.info("Embedding fingerprint written: %s", path)

    def _append_dense(self, chunks: list["Chunk"]) -> None:
        # mkdir + get_or_create, NOT get_collection: a vault whose indexes have
        # never been built has no collection yet, and appending into it is a
        # completely normal first move — it is what the vault switcher sets a
        # brand-new vault up to do, and what every "ingest, then append" job
        # chain does. get_collection raised NotFoundError there, which read as
        # a broken install rather than an empty one.
        #
        # The metadata MUST match _build_dense: Chroma only applies it at
        # creation time, so a collection born here without hnsw:space=cosine
        # would silently use L2 and score every query differently from one
        # built by `main.py index`.
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = persistent_client(self.chroma_dir)
        collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        total = len(chunks)
        for start in range(0, total, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            embeddings = self.embed_documents([c.text for c in batch])
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
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        client = persistent_client(self.chroma_dir)

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
            embeddings = self.embed_documents([c.text for c in batch])
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
        """ChromaDB metadata values must be str/int/float/bool. Coerce lists/None.

        Lists are joined with ", ", and readers split them back on exactly
        that delimiter (the retriever's tag boost, the canvas edge decoder).
        That round-trip is only lossless while no element contains ", " —
        which is why tags are normalised comma-free where they are stamped.
        The BM25 pickle keeps metadata verbatim, so the same field can be a
        list in one lane and a joined string in the other."""
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

    # ---- embedding with the configured prefixes (embed_query is the retriever's) ----
    # A prefix only ever enters the text the model sees: what Chroma stores and
    # what BM25 indexes stays the raw chunk text. With no prefix the backend is
    # handed the very list it was given.

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = self.doc_prefix
        return self.backend.embed([prefix + t for t in texts] if prefix else texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        prefix = self.query_prefix
        return self.backend.embed([prefix + t for t in texts] if prefix else texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]

    def dimension(self) -> int | None:
        """Vector size as far as the backend knows it: the loaded model's own
        for local, the configured `dimensions` for hosted (None when unset)."""
        return self.backend.dimension
