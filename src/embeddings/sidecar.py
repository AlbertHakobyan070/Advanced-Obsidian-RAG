"""
sidecar.py — Which embedder built each Chroma collection, recorded and enforced.

A collection's vectors only mean something to the embedder that made them. Two
models of the same width yield vectors Chroma compares without complaint and
ranks into nonsense, so the fact is written down next to the store and checked
before anything searches it. The record is a JSON sidecar,
`<paths.chroma_dir>/<collection>.embedding.json`: Chroma will not change a
collection's metadata after creation, and rewriting the live collection is the
very risk this avoids. One atomic file per collection (chunks and HyPE
questions alike), written by whatever built the vectors.

  check_collection   the guard: the configured embedder against the recorded
                     one. No sidecar -> None, and the caller warns (a store that
                     predates this keeps serving). A mismatch RAISES.
  stamp_collection   record an existing collection nobody fingerprinted: re-embed
                     a sample of its stored chunks and require each to match its
                     stored vector, so a stamp is a proof and not a claim.

What has to match is the vector space: kind, model, normalisation, both
prefixes and, when the configured side knows it, the dimension. The provider
NAME, base_url, device and batch size do not change the vectors and are not
compared.

Imports registry only (stdlib): no torch, no openai, and not embedder.py, which
imports THIS module for its write paths. chromadb and numpy load where used.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable, Iterator

from src.embeddings.registry import EmbeddingSpec

SIDECAR_SUFFIX = ".embedding.json"
SIDECAR_VERSION = 1
ROLES = ("chunks", "hype")
STATUSES = ("building", "complete")
# A re-embed target X pairs with the question collection X__hype: reserved.
HYPE_SUFFIX = "__hype"

# A sidecar missing any of these cannot say which vectors it describes. That is
# an error, never "no sidecar": a half-written file must not unlock serving.
REQUIRED_FIELDS = ("sidecar_version", "collection", "role", "status", "provider",
                   "kind", "model", "dimension", "normalize", "query_prefix", "doc_prefix")
# Written by the re-embed only; make_sidecar takes them as keyword extras.
_EXTRA_FIELDS = ("expected", "source_collection", "hype", "verified", "build")


class EmbeddingMismatchError(RuntimeError):
    """The configured embedder is not the one that built a collection, the
    collection is of the wrong kind for how it is used, or it is an unfinished
    re-embed. Raised, never logged and served: a search against vectors from
    another embedder returns confident nonsense."""


class StampError(RuntimeError):
    """`stamp` refused, or its sample check failed. Nothing was written."""


# --------------------------------------------------------------- the file ----

def sidecar_path(chroma_dir: Path, collection: str) -> Path:
    return Path(chroma_dir) / f"{collection}{SIDECAR_SUFFIX}"


def _check(data, collection: str, where: str) -> None:
    """Shared by read and write: a sidecar that is not a complete description of
    ONE named collection is refused wherever it is met."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a JSON object, got {type(data).__name__}")
    missing = [k for k in REQUIRED_FIELDS if k not in data]
    if missing:
        raise ValueError(f"{where}: missing required field(s) {missing}")
    if data["sidecar_version"] != SIDECAR_VERSION:
        raise ValueError(f"{where}: sidecar_version {data['sidecar_version']!r}, but this "
                         f"build reads version {SIDECAR_VERSION}")
    if data["role"] not in ROLES:
        raise ValueError(f"{where}: role {data['role']!r} is not one of {ROLES}")
    if data["status"] not in STATUSES:
        raise ValueError(f"{where}: status {data['status']!r} is not one of {STATUSES}")
    if data["collection"] != collection:
        raise ValueError(f"{where}: describes collection {data['collection']!r} but is "
                         f"named for {collection!r} (a renamed or copied file)")


def read_sidecar(chroma_dir: Path, collection: str) -> dict | None:
    """The recorded fingerprint, or None ONLY when the file is absent. A file
    that exists but cannot be trusted raises ValueError naming it."""
    path = sidecar_path(chroma_dir, collection)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError as e:                  # JSONDecodeError and UnicodeDecodeError both
        raise ValueError(f"{path} is not readable JSON ({e}). Delete it and run "
                         f"`rag stamp` to record the collection again.") from e
    _check(data, collection, str(path))
    return data


def write_sidecar(chroma_dir: Path, collection: str, data: dict) -> Path:
    """Validate, then write atomically: a reader (the :8051 startup guard) sees
    the old file or the new one, never half of either."""
    path = sidecar_path(chroma_dir, collection)
    _check(data, collection, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def delete_sidecar(chroma_dir: Path, collection: str) -> bool:
    try:
        sidecar_path(chroma_dir, collection).unlink()
    except FileNotFoundError:
        return False
    return True


def make_sidecar(spec: EmbeddingSpec, *, collection: str, role: str, status: str,
                 dimension: int, count: int, written_by: str,
                 source_digest: str | None = None, hype_collection: str | None = None,
                 created_at: str | None = None, **extra) -> dict:
    """Every field of a sidecar. `extra` carries the re-embed's own fields
    (expected, source_collection, hype, verified, build); anything else is a
    typo and an error."""
    unknown = sorted(set(extra) - set(_EXTRA_FIELDS))
    if unknown:
        raise ValueError(f"make_sidecar: unknown field(s) {unknown}; "
                         f"optional fields are {list(_EXTRA_FIELDS)}")
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    data = {
        "sidecar_version": SIDECAR_VERSION,
        "collection": collection,
        "role": role,
        "status": status,
        "provider": spec.provider,
        "kind": spec.kind,
        "model": spec.model,
        "base_url": spec.base_url,
        "dimension": dimension,
        "normalize": spec.normalize,
        "query_prefix": spec.query_prefix,
        "doc_prefix": spec.doc_prefix,
        "created_at": created_at or now,
        "completed_at": now if status == "complete" else None,
        "count": count,
        "expected": None,
        "source_digest": source_digest,
        "source_collection": None,
        "hype_collection": hype_collection,
        "hype": None,
        "written_by": written_by,
        "verified": None,
        "build": None,
    }
    data.update(extra)
    return data


# --------------------------------------------------- reading the collection ----

def ids_digest(ids: Iterable[str]) -> str:
    """sha256 of the sorted, newline-joined id SET. Membership, not text: the
    dense = sparse invariant is about which ids exist, and a digest of ids is
    cheap to recompute where one of 161K chunk texts is not."""
    joined = "\n".join(sorted(set(ids)))
    return "sha256:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()


def iter_ids(col, page: int = 5000) -> Iterator[str]:
    """Every id in the collection, a page at a time: one get() over all of
    them fails on SQLite's variable limit (delete_doc.py pages for the same
    reason)."""
    for offset in range(0, col.count(), page):
        yield from col.get(limit=page, offset=offset, include=[])["ids"]


def stored_dimension(col) -> int:
    """The vector size as STORED: measured from one row, not taken from the
    model's word for it."""
    got = col.get(limit=1, include=["embeddings"])
    if len(got["ids"]) == 0:
        raise ValueError(f"collection {col.name!r} holds no vectors, so its dimension "
                         f"cannot be measured")
    return int(len(got["embeddings"][0]))


def role_for(cfg, collection: str) -> str:
    """A collection named like retrieval.hype.collection, or ending in __hype, is
    a HyPE question collection (embedded query-side); anything else holds chunks."""
    if collection == cfg.get("retrieval.hype.collection", "hype_questions") \
            or collection.endswith(HYPE_SUFFIX):
        return "hype"
    return "chunks"


# ------------------------------------------------------------- the guard ----

def sidecar_label(sidecar: dict) -> str:
    """`provider · model`, spelled as EmbeddingSpec.label() spells it."""
    return f"{sidecar['provider']} · {sidecar['model']}"


def _describe(label: str, dimension: int | None, doc_prefix: str, query_prefix: str) -> str:
    dim = f"{dimension}-d, " if dimension else ""
    return f"{label} ({dim}doc_prefix {doc_prefix!r}, query_prefix {query_prefix!r})"


def fingerprint_diff(spec: EmbeddingSpec, dimension: int | None, sidecar: dict, *,
                     role: str) -> list[str]:
    """Names of the fields in which the configured embedder differs from a
    sidecar, in a fixed order; empty means the same vector space. `dimension` is
    the configured side's, or None when it does not know (a hosted endpoint
    with no `dimensions`): then it is not compared."""
    diff = []
    if sidecar["role"] != role:
        diff.append("role")
    for field, ours in (("kind", spec.kind), ("model", spec.model),
                        ("normalize", spec.normalize)):
        if sidecar[field] != ours:
            diff.append(field)
    if dimension is not None and sidecar["dimension"] != dimension:
        diff.append("dimension")
    for field, ours in (("doc_prefix", spec.doc_prefix), ("query_prefix", spec.query_prefix)):
        if sidecar[field] != ours:
            diff.append(field)
    return diff


def check_collection(embedder, chroma_dir: Path, collection: str, *,
                     role: str = "chunks") -> dict | None:
    """The guard. None when the collection has no sidecar (the caller warns and
    carries on: the live store predates fingerprints). The sidecar when the
    configured embedder matches it. Anything else raises EmbeddingMismatchError:
    another embedder, the wrong kind of collection, or an unfinished re-embed."""
    spec = embedder.spec
    if spec is None:
        raise ValueError("this embedder carries no spec, so it cannot be compared with a "
                         "collection's fingerprint; build it with Embedder.from_config")
    sidecar = read_sidecar(chroma_dir, collection)
    if sidecar is None:
        return None

    dimension = embedder.dimension()
    diff = fingerprint_diff(spec, dimension, sidecar, role=role)
    if "role" in diff:
        raise EmbeddingMismatchError(
            f"collection {collection!r} is recorded as a {sidecar['role']!r} collection but is "
            f"used as the {role!r} one (paths.collection_name names the chunk collection, "
            f"retrieval.hype.collection the HyPE question collection). Point the setting at a "
            f"collection of the right kind."
        )
    if diff:
        raise EmbeddingMismatchError(
            f"collection {collection!r} was embedded by "
            f"{_describe(sidecar_label(sidecar), sidecar['dimension'], sidecar['doc_prefix'], sidecar['query_prefix'])}"
            f", but the configured embedder is "
            f"{_describe(spec.label(), dimension, spec.doc_prefix, spec.query_prefix)}"
            f"; they differ in {', '.join(diff)}. Searching would compare vectors from two "
            f"different embedders. Either set embedding.* back to the collection's embedder, or "
            f"point paths.collection_name at a collection built with the configured one."
        )
    if sidecar["status"] == "building":
        expected = sidecar.get("expected")
        raise EmbeddingMismatchError(
            f"collection {collection!r} is an unfinished re-embed "
            f"({sidecar.get('count')} of {expected if expected is not None else '?'} chunks so far, "
            f"by {sidecar_label(sidecar)}); serving it would search a partial corpus. Finish the "
            f"re-embed (run it again) or use a complete collection."
        )
    return sidecar


def unstamped_warning(collection: str, path: Path, spec: EmbeddingSpec, *,
                      command: str = "rag stamp") -> str:
    """What to log when a collection has no sidecar. `command` is the stamp
    invocation that would fix it: a HyPE collection needs `--collection`."""
    return (
        f"Collection {collection!r} has no embedding fingerprint ({path} is missing), so "
        f"nothing checks that its vectors came from the configured embedder ({spec.label()}). "
        f"Run `{command}` once: it re-embeds a sample of stored chunks, compares them with the "
        f"stored vectors and records the model. Serving it unchecked until then."
    )


# ---------------------------------------------------------------- stamping ----

def verify_sample(embedder, col, *, role: str, sample_size: int, min_cosine: float) -> dict:
    """Re-embed `sample_size` evenly spaced stored documents with the configured
    embedder and require each to match its stored vector at >= min_cosine. The
    same model on the same text scores ~1.0; another model or prefix lands far
    below, so a pass proves which embedder built the collection. Chunk
    collections are re-embedded document-side, HyPE questions query-side (that
    is how each was built). Raises StampError on any failure."""
    import numpy as np

    if sample_size < 1:
        raise ValueError(f"embedding.verify.sample_size = {sample_size!r} must be a positive integer")
    total = col.count()
    if total == 0:
        raise StampError(f"collection {col.name!r} is empty: there is nothing to verify")

    n = min(sample_size, total)
    texts, stored = [], []
    for i in range(n):
        got = col.get(limit=1, offset=i * total // n, include=["documents", "embeddings"])
        if len(got["ids"]) == 0 or not got["documents"][0]:
            continue                     # no text to re-embed: not evidence either way
        texts.append(got["documents"][0])
        stored.append(got["embeddings"][0])
    if not texts:
        raise StampError(f"collection {col.name!r} has no stored documents among the {n} sampled "
                         f"rows, so there is nothing to re-embed and compare")

    spec = embedder.spec
    embed = embedder.embed_documents if role == "chunks" else embedder.embed_queries
    new = np.asarray(embed(texts), dtype=float)
    old = np.asarray(stored, dtype=float)
    if new.shape[1] != old.shape[1]:
        raise StampError(
            f"collection {col.name!r} stores {old.shape[1]}-d vectors but the configured "
            f"embedder ({spec.label()}) produces {new.shape[1]}-d ones, so it did not build "
            f"this collection. Nothing was written."
        )
    norms = np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1)
    if not norms.all():
        raise StampError(f"collection {col.name!r}: a sampled vector has zero length, so "
                         f"it cannot be compared. Nothing was written.")
    cos = (old * new).sum(axis=1) / norms
    worst = float(cos.min())
    if not (cos >= min_cosine).all():
        raise StampError(
            f"{int((cos < min_cosine).sum())} of {len(cos)} sampled chunks do not match their "
            f"stored vectors (min cosine {worst:.5f}, threshold {min_cosine}): collection "
            f"{col.name!r} was not built by "
            f"{_describe(spec.label(), None, spec.doc_prefix, spec.query_prefix)}. Point "
            f"embedding.* at the embedder that built it, or re-embed into a new collection with "
            f"the configured one. Nothing was written."
        )
    return {"sample": len(cos), "min_cosine": worst, "threshold": min_cosine}


def stamp_collection(cfg, embedder, collection: str | None = None, *, force: bool = False,
                     client=None) -> tuple[dict, str]:
    """Record which embedder built `collection` (default paths.collection_name),
    AFTER verifying it. Returns (sidecar, "written") or, when the collection is
    already recorded as built by the configured embedder, (sidecar, "already")
    with nothing touched. Refuses (StampError, nothing written) a missing
    collection, an unfinished re-embed, a failed sample check, and a differing
    existing sidecar unless `force` — which still verifies: forcing replaces the
    record, never the proof."""
    spec = embedder.spec
    if spec is None:
        raise ValueError("this embedder carries no spec; build it with Embedder.from_config")
    name = collection or cfg.get("paths.collection_name", "obsidian_vault")
    role = role_for(cfg, name)
    chroma_dir = cfg.path("paths.chroma_dir")

    if client is None:
        if not chroma_dir.is_dir():
            raise StampError(f"no Chroma store at {chroma_dir} (paths.chroma_dir): nothing to stamp")
        from src.utils.chroma_client import persistent_client
        client = persistent_client(chroma_dir)
    from chromadb.errors import NotFoundError
    try:
        col = client.get_collection(name)
    except NotFoundError as e:
        raise StampError(f"collection {name!r} does not exist in {chroma_dir}") from e

    existing = read_sidecar(chroma_dir, name)
    if existing is not None:
        if existing["status"] == "building":
            raise StampError(
                f"collection {name!r} is an unfinished re-embed ({existing.get('count')} of "
                f"{existing.get('expected')} chunks): finish it, don't stamp it."
            )
        diff = fingerprint_diff(spec, embedder.dimension(), existing, role=role)
        if not diff:
            return existing, "already"
        if not force:
            raise StampError(
                f"collection {name!r} is already recorded as built by "
                f"{sidecar_label(existing)}, but the configured embedder is {spec.label()}; they "
                f"differ in {', '.join(diff)}. --force verifies the collection against the "
                f"configured embedder and replaces the record."
            )

    verified = verify_sample(
        embedder, col, role=role,
        sample_size=int(cfg.get("embedding.verify.sample_size", 32)),
        min_cosine=float(cfg.get("embedding.verify.min_cosine", 0.99)))
    sidecar = make_sidecar(
        spec, collection=name, role=role, status="complete",
        dimension=stored_dimension(col), count=col.count(), written_by="stamp",
        source_digest=ids_digest(iter_ids(col)),
        hype_collection=(cfg.get("retrieval.hype.collection", "hype_questions")
                         if role == "chunks" else None),
        verified=verified)
    write_sidecar(chroma_dir, name, sidecar)
    return sidecar, "written"
