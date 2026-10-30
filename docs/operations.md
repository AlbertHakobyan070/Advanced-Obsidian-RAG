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
- **Metadata-only changes keep the `doc_id`** — so retagging (group label, domain,
  tags) updates records in place with **no re-embedding**.

## Metadata changes: retag, don't re-embed

Fixing a group label, domain, or tag is a metadata update. It rewrites the record's
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

1. `index --append` (auto-rebuilds sparse) **or** `rebuild_bm25.py`. `index --append` refuses a file that would leave rows a REPLACE removes (an edited or relabelled file): delete the source with `POST /api/documents/delete` and `"keep_jsonls": ["<that file>"]` (without it the delete empties the file waiting to be appended too), then run the same append again. A different document that only shares the stored path is renamed and re-ingested instead: deleting would drop the indexed one. A file renamed or moved in the vault is a different source and passes that check: move its rows with `main.py relocate` before re-ingesting it, then `recalibrate_courses.py --rename-map <the same map>` judges each moved label by the path it was made for.
2. **Restart the query service** (`:8051`) — it loads indexes once at startup and stays
   warm on the pre-restart version until restarted.

## `/health` is failed with `UnstampedProvenanceError`

Under `retrieval.generated_mode: drop` (also what a config without the key means) or `route`, the
query service refuses to start while stored rows the provenance rule calls generated are
not stamped `provenance: generated`. An upgraded install whose index holds
agent-instruction files starts this way. `error` names the files. The stamp is
metadata-only: it never re-embeds and never changes an id.

1. Stop the servers (`:8051`, `:8052` and Streamlit); `--yes` refuses while one answers.
2. Back up `chroma_db` and the bm25 index (with its `.meta.json`): the Chroma write keeps
   no `.bak`.
3. `python main.py stamp-provenance --dry-run`, read its plan, then
   `python main.py stamp-provenance --yes`.
4. `python rebuild_bm25.py`.
5. Restart and poll `/health`. If it is still failed, `error` names each row left and
   its fix.

## `/health` is failed with `RouteUnavailableError`

Under `retrieval.generated_mode: route`, the query service refuses to start while the
store holds generated rows and no route pointer sidecar (`retrieval.route.pointers_file`)
exists. With zero generated rows it starts without one. Under `drop` or `off` the same
gap surfaces per call instead: a `generated_mode: "route"` request answers `409`. The fix is
the same. The sidecar is built from the chunk JSONLs and the vault, and never writes a
store or re-embeds:

1. `python main.py route-pointers`, or the console's `route_pointers` job. It prints how
   many generated rows it routed, how many it could not, and the commonest reasons.
2. Restart `:8051` and poll `/health`. `route.sidecar` names the loaded file. The
   console job restarts it for you when `webui.auto_restart_rag` is on.

Rebuild the sidecar, then restart, whenever the set of generated rows changes:

- after an ingest that adds or changes generated content (maps, summaries);
- after a `provenance.generated_paths` edit and the `stamp-provenance` run that follows it;
- after switching to a vault that has never had one built (each vault has its own).

Until then, a generated row the sidecar has not seen is counted as `unrouted` and routes
nothing.

## Utilities

| Script | Purpose |
|---|---|
| `rebuild_bm25.py` | Rebuild only the sparse index after metadata changes / deletions. |
| `build_tantivy.py` | EXPERIMENTAL. Build the tantivy index beside the BM25 pickle (optional `tantivy` package), so `retrieval.sparse_backend: tantivy` can rank the BM25 lanes; restart the query API after switching. While that backend is set, every sparse rebuild rebuilds it too, and a stale index is refused rather than searched. |
| `recalibrate_courses.py` | Re-tag the grouping label in place without re-embedding; changes only labels a machine made (a hand retag is reported, not changed); `--all-files`, `--include-path`, `--where-course`, `--report` list exactly which chunks change. |
| `main.py relocate` | Metadata-only move of indexed files (`rename_map.jsonl`): rewrites path metadata in every JSONL, parents, HyPE, every collection and OCR Studio; `--dry-run` / `--verify`; refuses canvas files (REPLACE them); servers down, then `rebuild_bm25.py`. |
| `delete_doc.py` | Preview only: lists what a removal would touch and prints the console's `POST /api/documents/delete` call, which removes it from both indexes. |
| `dedup_jsonl.py` | Drop duplicate `doc_id`s from a chunk file. |
| `build_hype.py` | Build hypothetical-prompt embeddings (HyPE) for a scoped set. |

## Testing

```bash
python -m pytest tests/ -q
```

The suite covers chunking (a splitter must not drop text, even from a paragraph-less
wall of prose longer than a chunk), the loaders and jobs, both APIs' contracts, the
bench, the MCP server, the deploy assets and archives, and the versioning rule.
