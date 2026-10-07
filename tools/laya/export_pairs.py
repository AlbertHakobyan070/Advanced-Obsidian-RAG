"""
export_pairs.py — export the passages and hard negatives the Laya reranker fine-tune starts from.

Run from the project root with the project venv's python:

    python tools/laya/export_pairs.py [--n-passages 4000] [--negatives 15] [--seed 0]
                                      [--out data/laya/pairs.jsonl] [--sets DIR] [--smoke DIR]

Local only: no LLM call, and this code uploads nothing. READ-ONLY on the store: the chunk JSONLs are
read and the Chroma collection is queried, never written, and a store or collection that is not
there is an error, not something this tool creates. What it WRITES is vault text, so the file stays
on this machine until the author moves it himself; uploading it anywhere sends vault passages to a third
party.

One row per passage: {pid, passage, negatives: [{id, text}], meta: {source_file, file_type}}.

How the rows are chosen (deterministic per --seed; no random module):
  1. LEAKAGE GUARD first. Every chunk of every gold file that a question in the sets directory or the
     smoke directory names is out, as a passage AND as a negative. The guard is file-level because a
     gold locator is: it survives a re-chunk, so the guard has to. Rejected questions count too (one
     can be un-rejected, or have its gold re-pointed, in review: keeping a few more files out of the
     training set costs little, a contaminated eval costs the eval). Files are compared with the
     eval's own rule, relevance.norm_file (NFC, casefold, "/" separators).
  2. PASSAGES come from the chunk JSONLs through the seed sampler's own reader and skip rules (too
     short, a contents page, an unassigned course), kept only if the store has a vector for them: a
     chunk without one cannot get neighbours. Those are counted and listed.
  3. THE DRAW orders candidates by sha256(f"{seed}:{doc_id}"), as the sampler does, half from prose
     and half from code (see _split_budget).
  4. HARD NEGATIVES: the passage's STORED embedding is the query, so nothing is re-embedded. The
     nearest neighbours are walked nearest first and the first --negatives are kept that are of a
     different file (normalised), not gold, and not the passage itself. Negatives are not held to
     the passage skip rules: the index holds contents pages and stubs too, so they can turn up in a
     real candidate pool, where the reranker should rank them down.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple

# Run as a script, sys.path[0] is this folder rather than the project: put the project first so
# `src` and `eval` import here as they do for delete_doc.py and rebuild_bm25.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.bench.questions import load_sets
from eval.bench.relevance import norm_file
from eval.bench.sample import CODE_TYPES, _did, _fetch, _hash, _meta, _skipped, _text, iter_chunks
from src.utils.chroma_client import persistent_client
from src.utils.config_loader import load_config
from src.utils.console import force_utf8_console

# The plan's defaults for the flags (Task 13 of the eval-content-and-laya plan). They belong to this
# tool, not to the pipeline, so they are not config keys.
DEFAULT_N_PASSAGES = 4000
DEFAULT_NEGATIVES = 15
DEFAULT_OUT = "data/laya/pairs.jsonl"       # under the project root, as paths.* resolve
DEFAULT_SMOKE = "eval/smoke"                # no config key names the smoke set

# Neighbours fetched per negative wanted. A chunk's nearest neighbours are often the rest of its own
# file and copies of itself, which the filters drop; a multiple of what is wanted leaves enough
# survivors (the retriever's path-scoped dense search over-fetches and filters client-side for the
# same reason). A passage that still comes up short keeps what it found and is counted.
OVERFETCH = 8
BATCH = 200         # passages per get() of embeddings and per query(): a few round trips, not thousands
ID_PAGE = 5000      # ids per page when listing the store; delete_doc.py pages its scans at 5000 too


class ExportError(RuntimeError):
    """The export cannot be made as asked: a store, folder or corpus that is not there. The message
    names what to look at."""


@dataclass
class Stats:
    """Everything the run counted; render() prints it."""
    data_dir: Path
    chroma_dir: Path
    collection: str
    out: Path
    n_passages: int                     # asked for
    negatives: int
    seed: int
    store_count: int = 0
    rows: int = 0                       # distinct chunk rows read
    gold_sources: list = field(default_factory=list)    # (directory, questions or None, gold files)
    gold_named: int = 0                 # distinct gold files the question sets name
    gold_chunks: int = 0                # chunk rows in those files, kept out of the passages
    gold_files: int = 0                 # of the named files, how many have a chunk row
    skipped: int = 0                    # the sampler's skip rules
    eligible_prose: int = 0
    eligible_code: int = 0
    missing: list = field(default_factory=list)         # ids with a chunk row and no stored vector
    exported_prose: int = 0
    exported_code: int = 0
    fetched: int = 0                    # neighbours fetched per passage
    short: int = 0                      # passages that found fewer negatives than asked
    passed_same_file: int = 0           # neighbours walked past: the passage's own file...
    passed_gold: int = 0                # ...or a gold file's chunk
    out_bytes: int = 0


class _Cand(NamedTuple):
    key: str        # sha256 order
    doc_id: str
    lane: str       # the chunk file it lives in, to find it again in the second pass
    code: bool      # in the code half of the draw


# --- the leakage guard ----------------------------------------------------------------

def _gold_files(dirs: Iterable[Path], stats: Stats) -> frozenset[str]:
    """Normalised gold file names of every question in `dirs`. A directory that does not exist has
    none, and the report says so; a malformed set file raises (SchemaError, naming file and
    question) rather than quietly narrowing the guard."""
    named: set[str] = set()
    for d in map(Path, dirs):
        if not d.is_dir():
            stats.gold_sources.append((d, None, 0))
            continue
        questions = load_sets(d, include_rejected=True)
        files = {norm_file(g.file) for q in questions for g in q.gold}
        named |= files
        stats.gold_sources.append((d, len(questions), len(files)))
    stats.gold_named = len(named)
    return frozenset(named)


# --- reading the corpus and the store ---------------------------------------------------

def _scan(data_dir: Path, gold: frozenset[str], seed: int, stats: Stats) -> list[_Cand]:
    """One streaming pass over the chunk files: the leakage guard, then the sampler's skip rules.
    Keeps a small tuple per candidate, not its text (holding every text is what MemoryError'd the
    index builder here; the chosen few are read in a second pass)."""
    seen: set[str] = set()
    hit_files: set[str] = set()
    cands: list[_Cand] = []
    for lane, rec in iter_chunks(data_dir):
        did = _did(rec)
        if did in seen:                 # first occurrence wins, as in the index
            continue
        seen.add(did)
        stats.rows += 1
        meta = _meta(rec)
        file = norm_file(meta.get("source_file") or "")
        if file in gold:
            stats.gold_chunks += 1
            hit_files.add(file)
            continue
        if _skipped(meta, _text(rec)):
            stats.skipped += 1
            continue
        cands.append(_Cand(_hash(seed, did), did, lane, meta.get("file_type") in CODE_TYPES))
    stats.gold_files = len(hit_files)
    return cands


def _open_collection(chroma_dir: Path, name: str):
    """The collection, opened as the retriever opens it. The directory is checked first because
    PersistentClient would create a missing one, and this tool reads a store or stops."""
    if not chroma_dir.is_dir():
        raise ExportError(f"no Chroma store at {chroma_dir} (paths.chroma_dir in config.yaml); "
                          f"this tool only reads a store, it never creates one")
    from chromadb.errors import NotFoundError
    client = persistent_client(chroma_dir)
    try:
        return client.get_collection(name)
    except NotFoundError as e:
        raise ExportError(f"collection {name!r} not found in {chroma_dir} (paths.collection_name in "
                          f"config.yaml); this tool only reads a store, it never creates one") from e


def _stored_ids(col) -> set[str]:
    """Every id in the store, listed a page at a time (one get() over the whole collection fails on
    SQL variable limits; delete_doc.py pages for the same reason)."""
    ids: set[str] = set()
    for offset in range(0, col.count(), ID_PAGE):
        ids.update(col.get(limit=ID_PAGE, offset=offset, include=[])["ids"])
    return ids


# --- the draw ---------------------------------------------------------------------------

def _split_budget(n: int, n_prose: int, n_code: int) -> tuple[int, int]:
    """(prose, code) counts for a draw of n: half and half, the odd one to prose, and a side that
    cannot fill its half hands the slack to the other.

    Not proportional, because a uniform draw would follow the corpus mix and crowd code out: the
    retriever puts scripts and notebooks at about 2.4% of the corpus (see its code lane), so 4,000
    uniform passages would hold fewer than a hundred of them, and the reranker has to rank the
    code lane's results too. Prose and code are told apart by file_type, with the sampler's own
    CODE_TYPES."""
    code = min(n_code, n // 2)
    prose = min(n_prose, n - code)
    code = min(n_code, n - prose)
    return prose, code


def _draw(cands: list[_Cand], n: int) -> list[_Cand]:
    """The first of each side by hash, then merged back into one hash order, so the file is mixed
    from its first row (a notebook that reads only the head still sees both)."""
    prose = sorted((c for c in cands if not c.code), key=lambda c: c.key)
    code = sorted((c for c in cands if c.code), key=lambda c: c.key)
    n_prose, n_code = _split_budget(n, len(prose), len(code))
    return sorted(prose[:n_prose] + code[:n_code], key=lambda c: c.key)


# --- negatives and output -----------------------------------------------------------------

def _rows(col, chosen: list[_Cand], found: dict, gold: frozenset[str], negatives: int, k: int,
          stats: Stats) -> Iterator[dict]:
    """The output rows, one per passage, in draw order. Runs lazily, so the texts of at most one
    batch of neighbours are in memory at a time."""
    for start in range(0, len(chosen), BATCH):
        batch = chosen[start:start + BATCH]
        got = col.get(ids=[c.doc_id for c in batch], include=["embeddings"])
        # get() returns rows in the store's own order and silently omits an id it does not have,
        # so match vectors to passages by id.
        vec = dict(zip(got["ids"], got["embeddings"]))
        gone = [c.doc_id for c in batch if c.doc_id not in vec]
        if gone:
            raise ExportError(f"{len(gone)} passage id(s) left the store during the export "
                              f"(e.g. {gone[0]}): it changed underneath this run; run it again")
        res = col.query(query_embeddings=[vec[c.doc_id] for c in batch], n_results=k,
                        include=["documents", "metadatas"])
        for c, ids, docs, metas in zip(batch, res["ids"], res["documents"], res["metadatas"]):
            rec = found[(c.lane, c.doc_id)]
            meta = _meta(rec)
            mine = norm_file(meta.get("source_file") or "")
            kept: list[dict] = []
            for nid, ntext, nmeta in zip(ids, docs, metas):
                if len(kept) == negatives:
                    break
                theirs = norm_file((nmeta or {}).get("source_file") or "")
                if nid == c.doc_id:
                    continue
                # The gold test reads the NEIGHBOUR's own stored metadata, so it also catches a
                # gold chunk that has a vector but no JSONL row.
                if theirs == mine:
                    stats.passed_same_file += 1
                elif theirs in gold:
                    stats.passed_gold += 1
                else:
                    kept.append({"id": nid, "text": ntext})
            if len(kept) < negatives:
                stats.short += 1
            yield {"pid": c.doc_id, "passage": rec["text"], "negatives": kept,
                   "meta": {"source_file": meta.get("source_file"), "file_type": meta.get("file_type")}}


def _write_atomic(out: Path, rows: Iterable[dict]) -> None:
    """One JSON object per line, UTF-8 (not \\u escapes), LF endings. Written beside the target and
    moved into place only when complete: a run that dies midway must not leave a truncated file
    that looks like a finished export, and must not destroy the previous one."""
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    try:
        with open(partial, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(partial, out)
    finally:
        partial.unlink(missing_ok=True)     # already gone after a successful replace


def export(*, data_dir: Path, chroma_dir: Path, collection: str, out: Path, sets_dirs: Iterable[Path],
           n_passages: int, negatives: int, seed: int) -> Stats:
    """Write `out` and return what was counted along the way."""
    data_dir, chroma_dir, out = Path(data_dir), Path(chroma_dir), Path(out)
    if not data_dir.is_dir():
        raise ExportError(f"data directory {data_dir} does not exist "
                          f"(the folder of paths.chunks_file in config.yaml)")
    stats = Stats(data_dir=data_dir, chroma_dir=chroma_dir, collection=collection, out=out,
                  n_passages=n_passages, negatives=negatives, seed=seed)
    # The guard first: a malformed set file stops the run before any slow work.
    gold = _gold_files(sets_dirs, stats)
    col = _open_collection(chroma_dir, collection)
    stats.store_count = col.count()
    stored = _stored_ids(col)

    cands = _scan(data_dir, gold, seed, stats)
    present = [c for c in cands if c.doc_id in stored]
    stats.missing = [c.doc_id for c in cands if c.doc_id not in stored]
    stats.eligible_code = sum(c.code for c in present)
    stats.eligible_prose = len(present) - stats.eligible_code
    if not present:
        raise ExportError(
            f"no passage qualifies: {stats.rows} distinct chunk row(s) read from {data_dir}, "
            f"{stats.gold_chunks} in gold files, {stats.skipped} skipped by the sampler's rules, "
            f"{len(stats.missing)} with no vector in collection {collection!r} at {chroma_dir}")

    chosen = _draw(present, n_passages)
    stats.exported_code = sum(c.code for c in chosen)
    stats.exported_prose = len(chosen) - stats.exported_code
    found = _fetch(data_dir, {(c.lane, c.doc_id) for c in chosen})
    stats.fetched = min(stats.store_count, negatives * OVERFETCH)
    _write_atomic(out, _rows(col, chosen, found, gold, negatives, stats.fetched, stats))
    stats.out_bytes = out.stat().st_size
    return stats


# --- report and command line --------------------------------------------------------------

def render(s: Stats) -> str:
    def row(label: str, text: str) -> str:
        return f"  {label:<18}{text}"

    eligible = s.eligible_prose + s.eligible_code
    exported = s.exported_prose + s.exported_code
    lines = [
        "Laya pair export (read-only: the chunk files are read and the store is queried, never written)",
        row("store", f"{s.chroma_dir}  collection {s.collection!r}: {s.store_count} vector(s)"),
        row("chunk files", f"{s.data_dir}: {s.rows} distinct chunk row(s)"),
        "",
        "Leakage guard: every chunk of a gold file is kept out of the passages and the negatives",
    ]
    for path, n_questions, n_files in s.gold_sources:
        what = ("does not exist, so no gold files from it" if n_questions is None
                else f"{n_questions} question(s) naming {n_files} gold file(s)")
        lines.append(row("question set", f"{path}: {what}"))
    lines.append(row("excluded", f"{s.gold_chunks} chunk(s) in {s.gold_files} file(s)"))
    if s.gold_files < s.gold_named:
        lines.append(row("", f"{s.gold_named - s.gold_files} of the {s.gold_named} gold file(s) named "
                             f"match no chunk row"))
    lines += ["", "Passages",
              row("eligible", f"{eligible} (prose {s.eligible_prose}, code {s.eligible_code}) after the "
                              f"guard, the sampler's skip rules ({s.skipped} skipped) and the vector check")]
    if s.missing:
        shown = ", ".join(s.missing[:5]) + (", ..." if len(s.missing) > 5 else "")
        lines.append(row("no stored vector", f"{len(s.missing)} chunk(s) have a row but no vector, "
                                             f"skipped: {shown}"))
    fewer = f"; asked for {s.n_passages}, only {eligible} qualify" if exported < s.n_passages else ""
    lines.append(row("exported", f"{exported} (prose {s.exported_prose}, code {s.exported_code}), "
                                 f"seed {s.seed}{fewer}"))
    lines += ["", "Negatives",
              row("per passage", f"{s.negatives} wanted, from the {s.fetched} nearest chunks"),
              row("short", f"{s.short} of {exported} passage(s) got fewer than {s.negatives}"),
              row("passed over", f"{s.passed_same_file} neighbour(s) of the passage's own file and "
                                 f"{s.passed_gold} of a gold file, over all passages"),
              "", f"Wrote {s.out} ({s.out_bytes} bytes, {s.out_bytes / 2**20:.1f} MiB)"]
    return "\n".join(lines)


def _positive(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"{n} is not positive")
    return n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Export passages and hard negatives for the Laya reranker fine-tune "
                    "(read-only on the store; no LLM).")
    ap.add_argument("--n-passages", type=_positive, default=DEFAULT_N_PASSAGES, metavar="N",
                    help=f"passages to export (default: {DEFAULT_N_PASSAGES})")
    ap.add_argument("--negatives", type=_positive, default=DEFAULT_NEGATIVES, metavar="K",
                    help=f"hard negatives per passage, from other files (default: {DEFAULT_NEGATIVES})")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    ap.add_argument("--out", default=None, metavar="JSONL",
                    help=f"where to write the rows (default: {DEFAULT_OUT} under the project root)")
    ap.add_argument("--sets", default=None, metavar="DIR",
                    help="question sets whose gold files are kept out of the export (default: "
                         "eval.sets_dir from config.yaml, i.e. eval/sets)")
    ap.add_argument("--smoke", default=None, metavar="DIR",
                    help=f"the smoke set, guarded the same way (default: {DEFAULT_SMOKE} under the "
                         f"project root)")
    return ap


def main(argv: list[str] | None = None) -> None:
    force_utf8_console()
    args = build_parser().parse_args(argv)
    cfg = load_config()
    # Paths come from where the rest of the repo reads them: the chunk folder as the bench CLI
    # derives it, the store as the retriever opens it, the sets directory as the review queue
    # resolves it. A path typed on the command line is used as typed, as main.py does.
    chroma_dir = cfg.path("paths.chroma_dir")
    data_dir = cfg.path("paths.chunks_file").parent
    sets = Path(args.sets) if args.sets else cfg.path("eval.sets_dir", "eval/sets")
    smoke = Path(args.smoke) if args.smoke else cfg.project_root / DEFAULT_SMOKE
    out = Path(args.out) if args.out else cfg.project_root / DEFAULT_OUT
    print(f"Reading {data_dir} and the vectors in {chroma_dir} ...", flush=True)
    stats = export(data_dir=data_dir, chroma_dir=chroma_dir,
                   collection=cfg.get("paths.collection_name", "obsidian_vault"), out=out,
                   sets_dirs=[sets, smoke], n_passages=args.n_passages, negatives=args.negatives,
                   seed=args.seed)
    print(render(stats))


if __name__ == "__main__":
    main()
