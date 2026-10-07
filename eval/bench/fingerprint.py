"""
fingerprint.py — what a run actually measured.

A run record that only says "nDCG 0.71" can be neither compared nor explained:
was the working tree dirty, did the config change, was the index rebuilt in
between? These helpers capture exactly the inputs that move a retrieval number
— git state, the effective config, the index (collection count, BM25 sidecar,
embedding and reranker models, chunk-file names/sizes/mtimes) — as small
values two runs can be diffed on. Nothing is guessed: when git or the index
cannot be read these RAISE, because a run stamped with an invented sha is worse
than no run at all.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=str(repo), capture_output=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo} (exit {r.returncode}): "
                           f"{r.stderr.strip() or 'no output'}")
    return r.stdout


def git_state(repo: Path) -> dict:
    """{"sha": HEAD commit, "dirty": bool} for the checkout at `repo`. Dirty is
    ANY `git status --porcelain` output, untracked files included: an untracked
    helper can change what a run does as surely as an edit to a tracked one."""
    sha = _git(repo, "rev-parse", "HEAD").strip()
    dirty = bool(_git(repo, "status", "--porcelain").strip())
    return {"sha": sha, "dirty": dirty}


def config_digest(cfg) -> str:
    """Short digest of the effective config tree; key order does not matter."""
    blob = json.dumps(cfg.as_dict(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def chunk_files_digest(data_dir: Path) -> str:
    """Short digest of the sorted (name, size, mtime_ns) of every *chunks.jsonl
    in `data_dir`: the cheap way to see that the corpus changed between runs
    without hashing hundreds of MB of text."""
    entries = []
    for p in sorted(Path(data_dir).glob("*chunks.jsonl")):
        st = p.stat()
        entries.append([p.name, st.st_size, st.st_mtime_ns])
    return hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()[:16]


def index_fingerprint(cfg, rag) -> dict:
    """The index and models a run measured, from the live pipeline `rag`."""
    # The BM25 count and build time come from the sidecar written at build time,
    # never from unpickling the payload (a multi-GB RAM spike on a 16 GB box).
    meta_path = Path(str(cfg.path("paths.bm25_index")) + ".meta.json")
    bm25 = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    return {
        "dense_count": rag.retriever._get_collection().count(),
        "bm25": bm25,
        "embedding": {"provider": cfg.get("embedding.provider"),
                      "model": cfg.get("embedding.local_model")},
        "reranker": cfg.get("retrieval.cross_encoder_model"),
        "chunk_files": chunk_files_digest(cfg.path("paths.chunks_file").parent),
    }
