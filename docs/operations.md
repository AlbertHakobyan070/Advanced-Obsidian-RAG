# Operations

The invariants that keep a large, evolving index correct and cheap to maintain.

## JSONL is the source of truth

The `data/*.jsonl` chunk files are authoritative. The **ChromaDB vector store** and the
**BM25 pickle** are *derived* artifacts — both can be rebuilt from the JSONLs at any
time:

```bash
python main.py index          # rebuild dense (and sparse) from data/*.jsonl
python rebuild_bm25.py        # rebuild ONLY the sparse index (unions every data/*_chunks.jsonl)
```

Readers stream the JSONLs and split on `"\n"` only (never a generic line-splitter), so
exotic Unicode inside chunk text can't shred a record.

## Content-addressed IDs

```
doc_id = sha256(source_file + "::" + text[:500])[:16]
```

Consequences you operate by:

- **Re-ingesting is idempotent** — identical text yields the same `doc_id`.
- **Query evidence is dereferenceable** — `/search`, `/query`, and `/compare` expose
  an evidence id as `sources[].id`; `GET /chunks/{id}` returns the current evidence
  text and metadata for follow-up inspection. Parent-context results use
  `parent:<parent_id>` and retain the child Chroma id as `origin_id`.
- **Changing chunk *text* changes the `doc_id`**, which orphans the old vector. Text
  changes therefore go through the **swap playbook** (below), not an in-place edit.
- **Metadata-only changes keep the `doc_id`** — so retagging (course, domain, tags)
  updates records in place with **no re-embedding**.

## Metadata changes: retag, don't re-embed

Fixing a course label, domain, or tag is a metadata update. It rewrites the record's
metadata in both the vector store and the sparse index and never touches embeddings —
fast and safe. Do it from the console's Documents tab or the retag endpoint; the sparse
index carries its own metadata copy, so **rebuild the sparse index after any metadata
change or deletion**.

## Text changes: the swap playbook

When chunk text changes (a re-chunk, a re-OCR), the new chunks get new `doc_id`s and the
old vectors become orphans. The safe sequence:

1. Produce the new chunks in a fresh JSONL.
2. Append/index the new chunks.
3. Delete the superseded `doc_id`s from the index.
4. Rebuild the sparse index.
5. Restart the warm query service so it reloads.

Keep a backup of any deleted vectors (id + embedding + document + metadata) so the
operation is reversible.

## Paged maintenance

At this corpus size, every scan / update / delete pages the vector store in bounded
batches so maintenance stays within a modest RAM budget. The maintenance scripts already
do this — follow the same pattern for any new one.

## After any index change

1. `index --append` (auto-rebuilds sparse) **or** `rebuild_bm25.py`.
2. **Restart the query service** (`:8051`) — it loads indexes once at startup and stays
   warm on the pre-restart version until restarted.

## Utilities

| Script | Purpose |
|---|---|
| `rebuild_bm25.py` | Rebuild only the sparse index after metadata changes / deletions. |
| `recalibrate_courses.py` | Re-tag course metadata in place without re-embedding. |
| `delete_doc.py` | Preview-then-confirm removal of documents from the index. |
| `dedup_jsonl.py` | Drop duplicate `doc_id`s from a chunk file. |
| `build_hype.py` | Build hypothetical-prompt embeddings (HyPE) for a scoped set. |

## Testing

```bash
python -m pytest tests/ -q
```

The suite guards chunking (including a past bug where a splitter silently dropped text
past a size threshold in paragraph-less prose) plus job and loader behaviour.
