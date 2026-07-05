"""
build_hype.py — HyPE (Hypothetical Prompt Embeddings) index builder.

HyDE's index-side twin: instead of expanding the QUERY into a hypothetical
answer at search time, HyPE expands each CHUNK into the hypothetical
questions it answers at INGEST time. The questions (not the chunk) are
embedded into a separate ChromaDB collection pointing back at the chunk's
doc_id; at query time an extra dense lane matches query→question
(question-to-question similarity is tighter than question-to-passage) and
maps hits back to the parent chunks for RRF fusion.

COST REALITY (why this is scoped, cached, and opt-in): one LLM call per
chunk. The full 172K corpus is months of free-tier quota — DON'T. Scope to
one book / one course with --include-path, or markdown notes with
--file-types note,daily_note. Runs are resumable: chunks whose questions
already exist in the collection are skipped (keyed by doc_id, which changes
with the text — stale questions never survive a corpus swap).

    python build_hype.py --include-path "Albada" --dry-run
    python build_hype.py --include-path "Albada"
    python build_hype.py --file-types note,daily_note --max-chunks 2000

Query side: set retrieval.hype.enabled: true (config.yaml) or per-call
{"hype": true} on /search /query. Fail-soft: no collection -> no lane.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config_loader import load_config
from src.utils.logger import configure_logging, get_logger
from src.embeddings.embedder import Embedder, iter_jsonl_records

log = get_logger("build_hype")

_PROMPT_SYS = ("You generate search queries. Given a passage from study "
               "materials, you write the questions a student would type to "
               "find exactly this passage.")
_PROMPT_USER = ("Passage:\n---\n{chunk}\n---\n\nWrite {n} short, distinct, "
                "standalone questions that this passage directly answers. "
                "One question per line. No numbering, no commentary.")


def _questions_from(text: str, n: int) -> list[str]:
    lines = [l.strip(" -•\t") for l in text.split("\n")]
    qs = [l for l in lines if len(l) > 10 and "?" in l]
    return qs[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HyPE question embeddings.")
    ap.add_argument("--include-path", default=None,
                    help="Only chunks whose source_file contains this substring")
    ap.add_argument("--file-types", default=None,
                    help="Comma list of metadata file_type values (e.g. note,daily_note)")
    ap.add_argument("--questions", type=int, default=None,
                    help="Questions per chunk (default: retrieval.hype.questions_per_chunk or 3)")
    ap.add_argument("--max-chunks", type=int, default=2000,
                    help="Refuse to process more than this many chunks (cost guard; "
                         "raise it EXPLICITLY for bigger scopes)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count matching chunks + estimate calls, change nothing")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    configure_logging(level=cfg.get("logging.level", "INFO"), console=True)
    hype_cfg = cfg.get("retrieval.hype") or {}
    n_q = args.questions or int(hype_cfg.get("questions_per_chunk", 3))
    coll_name = hype_cfg.get("collection", "hype_questions")
    inc = args.include_path.lower() if args.include_path else None
    fts = ({t.strip().lower() for t in args.file_types.split(",") if t.strip()}
           if args.file_types else None)

    # ---- gather matching chunks (streaming) ----
    data_dir = cfg.path("paths.chunks_file").parent
    files = [cfg.path("paths.chunks_file")] + sorted(data_dir.glob("*_chunks.jsonl"))
    seen_paths, seen_ids = set(), set()
    todo: list[tuple[str, str, str]] = []          # (doc_id, text, source_file)
    for f in files:
        f = Path(f)
        key = str(f.resolve())
        if key in seen_paths or not f.exists():
            continue
        seen_paths.add(key)
        for rec in iter_jsonl_records(f):
            m = rec.get("metadata") or {}
            sf = str(m.get("source_file", ""))
            did = str(rec.get("doc_id", ""))
            if not did or did in seen_ids:
                continue
            if inc and inc not in sf.lower():
                continue
            if fts and str(m.get("file_type", "")).lower() not in fts:
                continue
            seen_ids.add(did)
            todo.append((did, str(rec.get("text", "")), sf))

    print(f"Matched {len(todo)} chunks "
          f"(include_path={args.include_path!r}, file_types={args.file_types!r})")
    if not todo:
        sys.exit(3)
    if args.dry_run:
        print(f"DRY RUN: would make up to {len(todo)} LLM calls "
              f"({n_q} questions each) into collection '{coll_name}'.")
        return
    if len(todo) > args.max_chunks:
        print(f"❌ {len(todo)} chunks exceeds --max-chunks {args.max_chunks}. "
              f"One LLM call per chunk — narrow the scope or raise the cap "
              f"explicitly if you really mean it.")
        sys.exit(3)

    # ---- lazily build the heavy deps only past the guards ----
    import chromadb
    from src.llm.llm_client import LLMClient
    llm = LLMClient.from_config(cfg, role="generation")
    emb = Embedder.from_config(cfg)
    client = chromadb.PersistentClient(path=str(cfg.path("paths.chroma_dir")))
    col = client.get_or_create_collection(coll_name,
                                          metadata={"hnsw:space": "cosine"})

    # resume: skip chunks whose q0 is already there
    existing: set[str] = set()
    for k in range(0, len(todo), 500):
        probe = [f"{d}::q0" for d, _, _ in todo[k:k + 500]]
        existing.update(col.get(ids=probe, include=[])["ids"])
    pending = [t for t in todo if f"{t[0]}::q0" not in existing]
    print(f"{len(todo) - len(pending)} already built (resume), {len(pending)} to do.")

    sidecar = data_dir / "hype_questions.jsonl"
    done = failed = 0
    with open(sidecar, "a", encoding="utf-8") as side:
        for i, (did, text, sf) in enumerate(pending, 1):
            try:
                resp = llm.complete(system=_PROMPT_SYS,
                                    user=_PROMPT_USER.format(chunk=text[:3000], n=n_q))
                qs = _questions_from(resp.text, n_q)
                if not qs:
                    raise ValueError("no questions parsed from LLM output")
                vecs = emb.backend.embed(qs)
                col.upsert(
                    ids=[f"{did}::q{j}" for j in range(len(qs))],
                    embeddings=vecs,
                    documents=qs,
                    metadatas=[{"doc_id": did, "source_file": sf}] * len(qs),
                )
                side.write(json.dumps({"doc_id": did, "source_file": sf,
                                       "questions": qs}, ensure_ascii=False) + "\n")
                done += 1
            except Exception as e:
                failed += 1
                log.warning("chunk %s: %s (continuing)", did, e)
                time.sleep(2)                     # give a rate-limited API air
            if i % 25 == 0 or i == len(pending):
                print(f"  [{i}/{len(pending)}] ok={done} failed={failed}")

    print(f"\n✅ HyPE: {done} chunks -> '{coll_name}' "
          f"({col.count()} question vectors total), {failed} failed "
          f"(re-run to resume). Sidecar: {sidecar.name}")
    print("   Enable with retrieval.hype.enabled: true (or per-call "
          "{\"hype\": true}); restart :8051.")
    if done == 0 and failed:
        sys.exit(3)


if __name__ == "__main__":
    main()
