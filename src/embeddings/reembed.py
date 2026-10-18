"""
reembed.py -- Re-embed the corpus into a NEW Chroma collection, resumably.

`main.py index --rebuild-into NAME [--embedding PROVIDER]` embeds with another
embedder (a hosted one, or the GPU embed-sidecar) without touching the live
collection: the active `paths.collection_name` is only ever READ.

What it copies and from where:
  * MEMBERSHIP (which doc_ids) from the active collection, so the new one holds
    exactly the same ids. A re-embed can add a collection; it cannot churn an id.
  * TEXT from the chunk JSONLs (the primary data), in build_sparse_union's file
    order with the first occurrence of an id winning, so the dense text equals
    the sparse text. An id the JSONLs cannot supply is an error, raised before
    any vector is written.
  * METADATA from the active collection (console retags live there, not in the
    JSONLs).

A re-embed can take hours or days, so progress is on disk. The fingerprint
sidecar is written with status "building" BEFORE the collection exists and the
startup guard refuses a building collection, so a half-built one can never be
served. A rerun of the same command reads the ids already stored, embeds only the
rest, and tops up (prunes ids the active collection lost, re-syncs metadata).
Only a finished, verified run rewrites the sidecar as "complete".

Deferred: the HyPE question collection is NOT re-embedded. Its twin name
(<target>__hype) is recorded in the sidecar and the sidecar says so.
"""
from __future__ import annotations

import copy
import re
import time
from pathlib import Path
from typing import Callable, Iterator

from src.embeddings.registry import EmbeddingSpec, resolve_embedding_spec
from src.embeddings.sidecar import (
    GRAPH_SUFFIX, HYPE_SUFFIX, fingerprint_diff, ids_digest, iter_ids, make_sidecar,
    read_sidecar, stored_dimension, verify_sample, write_sidecar)
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import Config

PAGE = 5000                 # ids per paged Chroma read (SQLite's variable limit)
CHECKPOINT_BATCHES = 20     # rewrite the sidecar's count / print progress this often
COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$")


class ReembedError(RuntimeError):
    """A mistake in the request or in the data (bad target, ids the JSONLs cannot
    supply, ...). The CLI prints it as `ERROR:` and exits 2."""


def chunk_union_files(chunks_file: Path) -> list[Path]:
    """The chunk files build_sparse_union reads, in its order (chunks_file first,
    then every sibling *_chunks.jsonl sorted), de-duplicated, missing ones skipped."""
    chunks_file = Path(chunks_file)
    files, seen = [], set()
    for f in [chunks_file] + sorted(chunks_file.parent.glob("*_chunks.jsonl")):
        key = str(f.resolve())
        if key not in seen and f.exists():
            seen.add(key)
            files.append(f)
    return files


def _iter_union(files: list[Path]) -> Iterator[tuple[str, str]]:
    """(doc_id, text) over the union, malformed lines skipped (as the sparse
    rebuild does). Duplicates are NOT filtered here."""
    from src.embeddings.embedder import Chunk, iter_jsonl_records
    for path in files:
        for i, rec in enumerate(iter_jsonl_records(path)):
            c = Chunk.from_jsonl_record(rec, i)
            yield c.id, c.text


def _pages(items: list[str]) -> Iterator[list[str]]:
    for i in range(0, len(items), PAGE):
        yield items[i:i + PAGE]


def _check_target(cfg: Config, target: str, role: str = "chunks") -> None:
    if not COLLECTION_NAME_RE.fullmatch(target):
        raise ReembedError(f"{target!r} is not a collection name: 3-512 characters of "
                           f"letters, digits, '.', '_' or '-', alphanumeric at both ends")
    if target.endswith(HYPE_SUFFIX):
        raise ReembedError(f"{target!r}: the suffix {HYPE_SUFFIX!r} is reserved for HyPE collections")
    if role == "graph":
        derived = cfg.get("paths.collection_name", "obsidian_vault") + GRAPH_SUFFIX
        if target != derived:
            raise ReembedError(f"{target!r}: a graph index is always {derived!r} "
                               f"(paths.collection_name + {GRAPH_SUFFIX!r})")
    elif target.endswith(GRAPH_SUFFIX):
        raise ReembedError(f"{target!r}: the suffix {GRAPH_SUFFIX!r} is reserved for graph mode's "
                           f"seed index (the graph_embed job builds it)")
    for key, default in (("paths.collection_name", "obsidian_vault"),
                         ("retrieval.hype.collection", "hype_questions")):
        if target == cfg.get(key, default):
            raise ReembedError(f"{target!r} is {key}: the active collections are never written. "
                               f"Pick a new name.")


def rebuild_into(cfg: Config, target: str, embedder, *, limit: int | None = None,
                 client=None, progress: Callable[[str], None] = print,
                 role: str = "chunks") -> dict:
    """Embed the active collection's corpus into `target` with `embedder` and
    return the target's sidecar. `limit` embeds only that many ids (a smoke run):
    the result is verified but stays "building", so it can never be served.
    role="graph" embeds only the graph.file_type chunks, into graph mode's seed
    index (graph_twin.py), which pairs with no HyPE collection."""
    spec: EmbeddingSpec | None = embedder.spec
    if spec is None:
        raise ValueError("this embedder carries no spec; build it with Embedder.from_config")
    _check_target(cfg, target, role)
    if limit is not None and limit < 1:
        raise ReembedError(f"--limit must be a positive integer, got {limit}")

    chroma_dir = cfg.path("paths.chroma_dir")
    source_name = cfg.get("paths.collection_name", "obsidian_vault")
    if client is None:
        if not chroma_dir.is_dir():
            raise ReembedError(f"no Chroma store at {chroma_dir} (paths.chroma_dir)")
        client = persistent_client(chroma_dir)
    from chromadb.errors import NotFoundError
    try:
        source = client.get_collection(source_name)
    except NotFoundError as e:
        raise ReembedError(f"the active collection {source_name!r} does not exist in {chroma_dir}") from e

    # ---- the target: new, or an earlier run of this same embedder to resume ----
    sidecar = read_sidecar(chroma_dir, target)
    exists = target in {c.name for c in client.list_collections()}
    if exists and sidecar is None:
        raise ReembedError(f"collection {target!r} exists but has no embedding fingerprint, so it "
                           f"is not a re-embed target. Pick another name.")
    if sidecar is not None:
        diff = fingerprint_diff(spec, embedder.dimension(), sidecar, role=role)
        if diff:
            raise ReembedError(
                f"{target!r} was started with {sidecar['provider']} · {sidecar['model']}, but this "
                f"run uses {spec.label()}; they differ in {', '.join(diff)}. Pick another name or "
                f"delete the old collection.")
        if sidecar["status"] == "complete" and not exists:
            raise ReembedError(f"{target!r} has a 'complete' fingerprint but no collection: delete "
                               f"the stale file {target}.embedding.json")
        if limit is not None:
            raise ReembedError(f"--limit is a smoke option for a fresh collection; {target!r} "
                               f"already exists")

    # ---- membership, and proof the JSONLs can supply every id, BEFORE any write ----
    where = {"file_type": str(cfg.get("graph.file_type", "canvas"))} if role == "graph" else None
    all_ids = set(iter_ids(source, PAGE, where=where))
    if not all_ids:
        raise ReembedError(f"the active collection {source_name!r} is empty" if where is None else
                           f"the active collection {source_name!r} holds no chunks with file_type "
                           f"{where['file_type']!r} (graph.file_type)")
    members = set(sorted(all_ids)[:limit]) if limit else all_ids
    files = chunk_union_files(cfg.path("paths.chunks_file"))
    supplied = {cid for cid, _ in _iter_union(files) if cid in members}
    if members - supplied:
        lost = sorted(members - supplied)
        raise ReembedError(
            f"{len(lost)} of {len(members)} ids in {source_name!r} are in none of the chunk files "
            f"({', '.join(f.name for f in files)}), so their text cannot be re-embedded. First "
            f"{min(50, len(lost))}: {lost[:50]}. Nothing was written.")
    del supplied

    present = set(iter_ids(client.get_collection(target), PAGE)) if exists else set()
    stale = sorted(present - members)
    todo = members - present
    dimension = (sidecar["dimension"] if sidecar else
                 embedder.dimension() or len(embedder.embed_documents(["dimension probe"])[0]))
    twin = target + HYPE_SUFFIX if role == "chunks" else None
    created_at = sidecar["created_at"] if sidecar else time.strftime("%Y-%m-%dT%H:%M:%S")
    expected = len(all_ids)
    started = time.perf_counter()

    def write(status: str, count: int, **extra) -> dict:
        data = make_sidecar(
            spec, collection=target, role=role, status=status, dimension=dimension,
            count=count, written_by="rebuild-into", hype_collection=twin, created_at=created_at,
            expected=expected, source_collection=source_name, **extra)
        write_sidecar(chroma_dir, target, data)
        return data

    write("building", len(present) - len(stale))          # BEFORE the collection exists
    col = client.get_or_create_collection(target, metadata={"hnsw:space": "cosine"})
    for page in _pages(stale):
        col.delete(ids=page)
    progress(f"{target}: {len(present) - len(stale)} already embedded, {len(todo)} to go"
             + (f", {len(stale)} pruned (no longer in {source_name})" if stale else ""))

    # ---- pass 2: embed what is missing, a batch at a time ----
    done = text_drift = batches = 0
    batch_size = embedder.batch_size
    emitted: set[str] = set()
    batch: list[tuple[str, str]] = []

    def flush() -> None:
        nonlocal done, text_drift, batches
        ids, texts = [b[0] for b in batch], [b[1] for b in batch]
        got = source.get(ids=ids, include=["metadatas", "documents"])
        by_id = {i: (m, d) for i, m, d in zip(got["ids"], got["metadatas"], got["documents"])}
        lost = [i for i in ids if i not in by_id]
        if lost:
            raise ReembedError(f"{source_name!r} lost {len(lost)} id(s) during the build "
                               f"(e.g. {lost[0]!r}); run the same command again to prune them")
        text_drift += sum(by_id[i][1] != t for i, t in zip(ids, texts))
        metas = [by_id[i][0] or None for i in ids]
        col.upsert(ids=ids, embeddings=embedder.embed_documents(texts), documents=texts,
                   metadatas=metas if any(metas) else None)
        done += len(ids)
        batches += 1
        batch.clear()
        if batches % CHECKPOINT_BATCHES == 0:
            write("building", len(present) - len(stale) + done)
            rate = done / (time.perf_counter() - started)
            progress(f"embedded {done}/{len(todo)} ({100 * done // len(todo)}%) · "
                     f"{rate:.1f} chunks/s · ETA {(len(todo) - done) / rate / 60:.0f} min")

    for cid, text in (_iter_union(files) if todo else ()):
        if cid in todo and cid not in emitted:
            emitted.add(cid)
            batch.append((cid, text))
            if len(batch) >= batch_size:
                flush()
    if batch:
        flush()

    # ---- resumed run: bring already-stored rows' metadata up to date ----
    synced = 0
    for page in _pages(sorted(present - set(stale))):
        want = source.get(ids=page, include=["metadatas"])
        have = col.get(ids=page, include=["metadatas"])
        have_by_id = dict(zip(have["ids"], have["metadatas"]))
        changed = [(i, m) for i, m in zip(want["ids"], want["metadatas"])
                   if m and have_by_id.get(i) != m]
        if changed:
            col.update(ids=[i for i, _ in changed], metadatas=[m for _, m in changed])
            synced += len(changed)

    # ---- never "complete" on faith ----
    stored = set(iter_ids(col, PAGE))
    if stored != members:
        raise RuntimeError(f"{target!r} holds {len(stored)} ids but {len(members)} were expected "
                           f"({len(members - stored)} missing, {len(stored - members)} extra); "
                           f"left as 'building'. Run the same command again.")
    verified = verify_sample(embedder, col, role=role,
                             sample_size=int(cfg.get("embedding.verify.sample_size", 32)),
                             min_cosine=float(cfg.get("embedding.verify.min_cosine", 0.99)))
    seconds = time.perf_counter() - started
    build = {"seconds": round(seconds, 1), "chunks_per_s": round(done / seconds, 2) if done else 0,
             "embedded": done, "skipped_present": len(present) - len(stale), "pruned": len(stale),
             "metadata_synced": synced, "text_drift": text_drift, "limited_to": limit}
    hype = {"mode": "none", "collection": twin, "count": 0} if twin else None   # deferred: not re-embedded
    dimension = stored_dimension(col)
    if limit:                                  # a smoke subset must stay unservable
        return write("building", col.count(), hype=hype, verified=verified, build=build,
                     source_digest=ids_digest(members))
    return write("complete", col.count(), hype=hype, verified=verified, build=build,
                 source_digest=ids_digest(members))


def config_for_collection(cfg: Config, collection: str) -> Config:
    """A copy of `cfg` that points at `collection` and the embedder its sidecar
    records, for `bench run --collection`. Refuses a collection that is not a
    complete chunk collection, whose registry entry has changed since it was
    built, or that does not hold exactly the active collection's ids (a bench
    over different corpora would not be paired). `cfg` itself is not mutated."""
    chroma_dir = cfg.path("paths.chroma_dir")
    sidecar = read_sidecar(chroma_dir, collection)
    if sidecar is None:
        raise ValueError(f"collection {collection!r} has no embedding fingerprint "
                         f"({chroma_dir}\\{collection}.embedding.json); `rag stamp --collection "
                         f"{collection}` or `index --rebuild-into` creates one")
    if sidecar["role"] != "chunks" or sidecar["status"] != "complete":
        raise ValueError(f"collection {collection!r} is a {sidecar['status']} {sidecar['role']} "
                         f"collection; only a complete chunk collection can be scored")

    data = copy.deepcopy(cfg.as_dict())
    data.setdefault("paths", {})["collection_name"] = collection
    emb = data.setdefault("embedding", {})
    emb["provider"] = sidecar["provider"]
    if sidecar["provider"] == "local":
        emb.update(local_model=sidecar["model"], query_prefix=sidecar["query_prefix"],
                   doc_prefix=sidecar["doc_prefix"])
    derived = Config(data, cfg.project_root)
    spec = resolve_embedding_spec(derived)
    diff = fingerprint_diff(spec, spec.dimensions, sidecar, role="chunks")
    if diff:
        raise ValueError(f"collection {collection!r} was built with {sidecar['provider']} · "
                         f"{sidecar['model']}, but the configuration for that provider now differs "
                         f"in {', '.join(diff)}; restore it or rebuild the collection")

    active = cfg.get("paths.collection_name", "obsidian_vault")
    if collection != active:
        client = persistent_client(chroma_dir)
        if ids_digest(iter_ids(client.get_collection(active), PAGE)) != sidecar["source_digest"]:
            raise ValueError(f"collection {collection!r} does not hold the same ids as the active "
                             f"collection {active!r}; top it up first: main.py index "
                             f"--rebuild-into {collection} --embedding {sidecar['provider']}")
    return derived
