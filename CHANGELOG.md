# Changelog

What changed in each Noetrix release, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version
numbers follow [Semantic Versioning](https://semver.org/).

The version itself lives in `VERSION`. A release bumps it and adds an entry
here in the same commit, and the `Noetrix version:` line in each agent skill
(`SKILL.md`, `SKILL_RAGOPS.md`) moves with it; `tests/test_versioning.py`
fails when any of them disagree. The console shows this file under the
version chip in its header, beside what each part of the install is running.

Versions before 0.27.0 were never numbered at the time. Their entries were
reconstructed on 2026-10-14 from the commit history and the session notes,
one per shipped milestone, so read them as a map of when things landed rather
than as release-day notes.

## [0.30.0] - 2026-10-30

### Added
- Generated content (maps and summaries an LLM wrote, agent-instruction files,
  anything under `provenance.generated_paths`) no longer grounds an answer.
  `retrieval.generated_mode` decides what it does, and the shipped default is
  `drop`: generated chunks are kept out of every evidence path, so a citation
  is always authored or source text. `off` lets them compete as evidence and is
  config only; a per-call `"off"` is a `422`. Every returned source carries its
  `provenance`, `/answer` refuses generated ids with a `400`, and `/health` and
  `/schema` report the mode. In the console's graph fork, generated map nodes
  show disabled and unticked.
- `main.py stamp-provenance (--dry-run | --yes)` stamps the rows already stored
  with the same rule (metadata only, no re-embed, a `.bak` of each JSONL), and
  every ingest lane now stamps new rows. Under `drop` or `route` the query
  service refuses to start while generated rows are unstamped: `/health` is
  `failed` and names the fix, so run the stamp once on an existing store.
- Route mode, opt in with `generated_mode: route` in the config or `"route"` on
  a call to `/search`, `/query`, `/compare` and `/evidence_pack`. It keeps
  generated chunks out of the evidence as `drop` does, and lets them find the
  primary pages they cite: those pages are ranked and cited, and a source a map
  brought in carries `found_via`. It reads a pointer sidecar built by
  `main.py route-pointers` or the `route_pointers` job
  (`retrieval.route.pointers_file`), and its caps are `retrieval.route.router_k`,
  `per_target`, `max_targets` and `max_new`. `/health` and `/schema` report
  route coverage, the `/answer` `400` carries `routes_to`, and `/graph/expand`
  returns `routed_ids`. Route needs `retrieval.sparse_backend: bm25s`, and a
  call it cannot serve is a `409`. The example config and the archives stay on
  `drop`; route is available, not the default.
- `GET /outline` lists the primary sources by domain and course, with file
  type, chunk count and page count, and the generated maps apart.
  `GET /outline/{source}` gives one source's chapters with page ranges, or one
  map's concepts with the primary pages each cites. The MCP server gains
  `noetrix_outline` for it, so an agent that does not know the vault can look
  before it searches.
- `source_file` and `pages: [first, last]` on `/search`, `/query`, `/compare`
  and `/evidence_pack` restrict every lane to one file's primary chunks, or to
  a page range of it. The echo's `file_filter` says whether anything matched.
  They are refused with `graph: true` and need `sparse_backend: bm25s`.
- A graph engine for multi-hop questions, all off by default (the `graph.*`
  switches, `canvas.footer_subpath`, `provenance.label_generated` and
  `provenance.generated_only_confidence`). A canvas file node or a `Source:`
  line can resolve to the note section or PDF page it cites (`graph.bridges`),
  a walk can hop between canvases (`graph.cross_canvas`) and follow outgoing
  wikilinks (`graph.wikilinks`), and it can be limited by `graph.edge_types`
  and `graph.direction`. Every node reached carries the labelled path from its
  seed. `POST /evidence_pack` takes several sub-questions and returns evidence
  with provenance, coverage, gaps and the walk paths, and the MCP server gains
  `noetrix_evidence_pack` and `noetrix_graph_expand`.
- `SKILL_NAVIGATOR.md`, a new agent skill that turns a question into the right
  calls: an intent table with graph-on and graph-off rows, how to read an
  evidence pack, and an answer contract with claim-level citations, a
  confidence rubric and "not in the vault" decided from the items returned.
- Index hygiene tools for reorganising a vault. `main.py relocate --map
  rename_map.jsonl [--dry-run|--verify]` moves indexed files across every
  JSONL, sidecar and collection without re-embedding, after checking for
  blockers first. `recalibrate_courses.py` re-derives course labels and leaves
  hand retags alone. `canvas_refresh.py` refreshes the metadata of indexed
  canvas chunks, plan first. `parser.skip_roots` keeps whole top-level folders
  (agent workspaces, for one) out of every lane unless a run scopes them back
  in with `--include-path`.
- `rag map <pdf> --chapter ...` (or `--pages a-b`) turns a PDF chapter or page
  range into a generated Obsidian canvas and a typed graph (`map.json`) in
  which every node and edge carries verbatim quotes, located on real PDF pages
  and checked by a pinned verifier. It writes to a staging folder outside the
  vault and never ingests; a `map` job kind and a `knowledge_map:` config
  section go with it, and `--export-prompts` / `--import-extract` let another
  agent do the extract step. See `docs/knowledge-maps.md`.
- The bench gains arms for the generated modes (`GM-off`, `GM-drop`,
  `GM-route`) and for the graph engine (`--configs graph`), a per-hop recall
  report, and `hop` and `strategy` fields on gold records. `bench validate`
  refuses gold that names a generated source.

### Changed
- Generated chunks are no longer evidence by default (see `generated_mode`
  above). The dense lanes drop them by over-fetching and checking the stored
  `provenance` key rather than by a Chroma `$ne` filter, which slowed the main
  dense lane from milliseconds to hundreds of milliseconds.
- One lane guard, shared by the console, `main.py` and the loaders, refuses a
  scoped md run that would write its lane's canonical file, any run that would
  write another lane's file or a sidecar, and the Windows spellings of those
  paths. `parser.skip_dirs`, which did nothing before, is now pruned by every
  lane.
- Delete reaches every collection, the HyPE sidecar and the parents file, keeps
  a `.bak` of every JSONL it rewrites, and is refused while a re-embed target
  is building. `POST /api/documents/delete` also returns `eval_gold`, the ids
  of the eval questions whose gold names a deleted source; it warns and never
  refuses, and the Documents tab shows it. A sets file that cannot be loaded
  now fails the delete with a `500` before anything is removed.
- Appending to an index refuses rows a replace would remove (stale ids of a
  re-ingested source, or a page whose text changed) and prints the
  delete-then-append recipe that keeps the incoming rows.
- Web imports and md and PDF ingest stamp `provenance`. The parser reads an
  `AI Engineering` course and `Data Science (Books)`, prefers exact folder
  names to abbreviations, accepts frontmatter tags as a scalar, a list or
  `#`-prefixed values, and no longer reads CSS hex colours as tags.
- The archives leave out the working question sets (`eval/graph_sets`,
  `eval/route_sets`) and `eval/test_ledger.jsonl`; git is where the ledger is
  versioned.

### Fixed
- A Windows-only store corruption: chromadb sizes its pool of loaded indexes
  from the C runtime's open-file limit, so a store with only a few
  collections could evict an index, and a small one then failed with "Nothing
  found on disk" while a large one came back without its newest rows. The
  Chroma client now raises that limit before it starts, and raises an error if
  it cannot.
- Console retag updated only the active collection and left the others, and
  the graph twin, on the old metadata until a later rebuild. It now updates
  every chunk collection (the active one last), keeps a `.retag-*.bak` of each
  JSONL, and answers `409` while a collection is building. The response keeps
  its keys and gains `chroma_updated_by_collection`.
- The vault registry is saved whole or not at all, so a crash, a kill or a full
  disk mid-save no longer leaves a truncated file.
- Graph walks were slow on PDF page bridges because the page range was sent to
  Chroma, which compared the pages of every PDF chunk in the store. The page is
  now tested in Python after a lookup by file alone: a walk that took about a
  minute takes a couple of seconds with the same output, and a graph-on
  evidence pack now fits the MCP bridge's timeouts.

## [0.29.1] - 2026-10-19

### Fixed
- `GET /api/vaults` answered 500 on a fresh checkout, before `data/` existed:
  saving the vault registry now creates its folder.
- The console no longer overwrites a vault registry it cannot read. Only a
  missing file means "no vaults yet"; an unreadable one is an error, where it
  used to be read as empty and saved back over every other vault.
- The lane guards refuse other spellings of a chunk file that Windows opens as
  the same file: a case variant (`CHUNKS.JSONL`), a trailing dot
  (`chunks.jsonl.`), an alternate data stream (`chunks.jsonl::$DATA`) and an
  empty path segment (`data//x`, which resolves to the drive root). `Data/`
  is anchored to the data folder like `data/`.

### Changed
- The FreeLLMAPI proxy needs its own key (`FREELLMAPI_API_KEY`, from the Keys
  page of its dashboard) for chat. Without it the query API still starts and
  searches, but every generated answer fails with 401; the example config,
  `.env` templates and setup guides now say so.

## [0.29.0] - 2026-10-19

### Added
- An MCP server (`mcp_server.py`, `rag mcp`) over stdio, or streamable HTTP on
  :8053 bound to loopback, which the container starts. Its tools are built
  from both services' OpenAPI and permission maps, a risky call asks the user
  first, and a call retries a service that is not up yet.
- A request guard on both services: Host and Origin checks, and an optional
  bearer token (`RAG_API_TOKEN`) that the console asks for on a 401.
- Permission tiers (read, mutating, destructive) for every route of both
  services, in `/schema` and `/api/schema`.
- An engines map: `GET /api/engines` shows the backend of each component, and
  `POST /api/engines/set` switches one through the Settings writer where that
  can be done reliably.
- Graph tab label suggestions built from the canvas as it is open, unsaved
  edits included: both nodes (long ones summarised first), the arrow's
  direction, the canvas title, the groups and the other edges at each node,
  on a backend and model picked per request (`canvas.augment.provider`,
  `canvas.augment.model`). The reply carries the label, its relation read
  along the arrow, tags, alternatives, confidence, the reason and the routed
  model; nothing is written until Save. `GET /api/canvas/suggest_options`
  lists the choices.
- `providers.<name>.vision: true` marks a backend whose model reads images;
  suggestions send image nodes only to such a backend.
- `#tags` in edge labels; with `canvas.label_tags: true` they become tags of
  both nodes' chunks the next time that canvas is ingested.
- Removing a canvas from the index from the Graph tab. The vault file stays.
- A graph-mode seed embedder (`graph.embedding_provider`): graph mode seeds
  from a canvas-only collection, `<collection>__graph`, built by
  `main.py graph-embed` or the `graph_embed` job. `GET /api/canvas/embedder`
  says whether it is ready.
- Taking a file out of the OCR queue: `POST /api/ocrstudio/remove` moves an
  import or an inbox document into the vault's `.trash`, and moving it back
  restores it with its OCR Studio work.
- A release-notes page on the docs site, rendered from this file.

### Changed
- HyDE is off by default and in every preset (`retrieval.use_hyde: false`):
  two drafting models measured no gain on the dev split. `"hyde": true` on a
  call still turns it on.
- The cold archive no longer carries the question sets (`eval/sets`,
  `eval/smoke`); only the with-data archive does.
- One design system across the console's tabs and themes: themed dialogs for
  every confirmation and prompt, an icon set in place of emoji, a kicker
  heading on every panel, flat fills, one focus ring, and a theme designer with
  a row per token, radii included, that starts from the theme in use.
- The example config's `synthesis` preset turns on parent and neighbour
  context, as the live config and the docs describe.

### Fixed
- Suggest label failed with a 502 on a model that reasons before answering:
  the reply had too little room, a greedy match took the wrong braces, and the
  console dropped the reply. The cap is now `canvas.augment.max_tokens`, the
  parser takes the last JSON answer, and an error carries the stage that
  failed and the model's raw reply.
- The Ingest tab's × on a staged conversion could permanently delete a
  same-named inbox original. `POST /api/inbox/delete` takes `folder` and
  refuses a name both folders hold without it, it is tiered destructive, and
  that × asks first.

### Removed
- The legacy golden-set eval: `main.py eval`, its runner, metrics, golden set
  and judge prompt, the console's `eval` job kind, and the `eval.judge` and
  `eval.ngram_n` settings. The bench (`main.py bench`) is the evaluation.

## [0.28.0] - 2026-10-15

### Added
- OCR tab imports: upload one or many PDFs or images, keep a working page set
  typed as a range (pages keep their real numbers, so citations stay right),
  see a modified/untouched badge per imported document, and export the
  corrected markdown with page markers and figures as one zip.
- An optional tantivy backend for the BM25 lanes (`retrieval.sparse_backend`,
  default `bm25s`) and the `build_tantivy` job that builds its index. On the
  dev split it changes no metric and makes the sparse lane much faster.
- Install from an archive on any OS: `start.sh` / `start.ps1` check Docker,
  fill `.env`, start the stack, wait until it is ready and install both agent
  skills; `smoke.py` boots an archive end to end; each archive carries
  `BUILD_INFO.txt` and a `.sha256`.

### Changed
- Every archive stages the app from the repo by one allowlist (the personal
  archive no longer ships a hand-kept copy) and carries the agent skills,
  `VERSION` and `CHANGELOG.md`; a root script left off the list fails the tests.
- The agent skills are OS-neutral: `curl` instead of `curl.exe`, and a start
  section per install (archive launcher, Windows launcher, plain uvicorn).
- Inside the container the servers listen on the published port numbers, so a
  remapped port keeps the console's Query tab on its own API; a Windows home
  mounts through `RAG_CONTAINER_HOME`.

## [0.27.0] - 2026-10-14

### Added
- Versions and release notes: `VERSION`, this changelog, a version chip in the
  console header that opens the release notes and shows what each part of the
  install runs against this checkout (console, query API, installed skills,
  packaged bundle), `GET /api/releases`, and `version`/`commit` on `/health`.
- Agent controls: a job record carries `elapsed_s`, `queue_position`, the
  current phase's `progress` read from its log, and an ETA from past runs of
  the same kind; `/api/jobs/{id}/log` takes `tail` and `wait` (a long poll);
  `/stats` reports p50/p95 latency per endpoint and per stage.
- The OCR Studio tab (per-region OCR, clean-up, page store, ingest) and the
  Graph tab (open, edit and save one `.canvas`, then re-index it).
- The container stack versioned in the repo: `deploy/docker`, `deploy/package`,
  Kubernetes manifests, and a CI image build.
- Coverage-aware retrieval switches (a per-file cap, query decomposition) and
  a judged answer tier for the bench, with an LLM pinned by a replay store.
- EmbeddingGemma 2 on a GPU sidecar with a resumable re-embed into a new
  collection.
- Bench ablations for routing and HyDE; the Laya step-0 teacher check served
  over http.

### Changed
- The question sets were reviewed by their owner, 61 hand-written questions
  added, and the 400 verified questions split into dev and a sealed test half.
- Routing is two switches; the scope lanes and the metadata boost ship OFF, on
  the dev evidence.
- The Info tab was rebuilt around the current pipelines.

### Fixed
- Lane counts, a dangling-citation case and wording found in the Atlas
  read-through; the bench's routing factor opens the scope lanes again.

## [0.26.0] - 2026-10-07

### Added
- A component-level eval bench: gold sources with locators, per-source recall,
  configurations generated from six factors, paired bootstrap confidence
  intervals, per-stage timings, `bench validate`, and immutable run records.
- The eval review API and the console's Eval tab; a seeded, stratified sampler
  and `bench draft`, a resumable drafter of validated questions.
- A relevance gate that lets weak evidence abstain; the experimental Laya
  scorer and `laya` rerank mode behind a flag, with a Kaggle fine-tune notebook.
- An embedder provider registry, and a record of which embedder built each
  collection, so a mismatch is refused instead of served.
- A HyDE draft cache on disk, so repeated searches retrieve identically.

### Changed
- Graph mode is gated as an experimental feature; generation goes through
  FreeLLMAPI.

### Fixed
- `delete_doc.py --confirm` deleted only half; it is now preview-only.
- Ingest refuses an output that is another lane's chunk file.

## [0.25.0] - 2026-10-04

### Added
- An experimental-features registry with a Settings panel; graph querying is
  experimental and off by default.
- Console job records: every job's argv and config are persisted beside its
  log, and `/api/jobs/provenance` reports which chunk files have a recorded
  command.

### Fixed
- Five loaders judged files by their absolute path, so a vault under a `.git`
  ancestor found nothing; they now judge vault-relative paths.
- The console served stale HTML; it now sends `Cache-Control`.

## [0.24.0] - 2026-09-17

### Added
- Graph RAG mode (`/graph/expand`, `/answer`) with a human-in-the-loop fork,
  and its console panel.
- Per-lane fusion weights; canvas chunk size and node depth as ingestion knobs;
  a Graph section in the Ingest tab.

### Fixed
- Answers written with `**CONFIDENCE:**` or fullwidth citation brackets parsed
  as UNKNOWN or as uncited; the API and the eval now share one parser.

## [0.23.0] - 2026-09-16

### Added
- The canvas lane: Obsidian canvases ingested with their edges, flattened at
  ingest time, with a command, a config block, a job kind and a console entry.

## [0.22.0] - 2026-07-29

### Added
- A per-request choice of PaddleOCR engine (`ocr` or `vl`) in the OCR sidecar,
  which also names the engine it runs.

## [0.21.0] - 2026-07-29

### Added
- A GPU PaddleOCR sidecar; the sidecar became a repo-owned component.

## [0.20.0] - 2026-07-29

### Fixed
- The rerank criterion was dropped by `/search` and retrieve-only `/query`; it
  is forwarded from every retrieval endpoint.
- The public archive shipped the app folder unsanitized; it is now staged from
  the sanitized checkout.

## [0.19.0] - 2026-07-27

### Added
- Rerank instructions, `/compare/options`, a project site, and the API and
  agent guides.

### Changed
- The product was renamed to Noetrix.

## [0.18.0] - 2026-07-23

### Added
- Query comparison trees; hardened agent APIs and release contracts.

### Fixed
- A second vault works end to end; the Settings panels are wired up; the Vault
  tree is no longer empty for a root-level setting.

## [0.17.0] - 2026-07-22

### Added
- A tiered eval suite, an LLM provider registry and a 100-question golden set.
- A swappable cross-encoder reranker, including an out-of-process http mode.
- An OCR page scanner (which PDF pages need OCR) and a VLM-OCR preset registry.

### Changed
- The console runs cross-platform; the taxonomy is configurable.

## [0.16.0] - 2026-07-21

### Added
- Per-file ingest settings, a fetch-to-PDF lane, a vault switcher, `/history`,
  destination folders, notebook and script routing, and per-file domain and
  tags for every kind of file.

## [0.15.0] - 2026-07-20

### Added
- A Settings tab that edits `config.yaml`, more themes, a description mode,
  one-click and automatic restarts of the query API, a dense/sparse health
  verdict and a query history.

### Changed
- The runtime moved to its own venv and data drive.

## [0.14.0] - 2026-07-06

### Added
- A rewritten README and the MkDocs documentation site.

### Changed
- The repository was cleaned up and a sanitized GitHub snapshot published; the
  Mac bundle was refreshed.

## [0.13.0] - 2026-07-05

Everything from the first commit up to here, in one entry.

### Added
- Hybrid retrieval: dense and BM25 lanes fused with reciprocal rank fusion and
  reranked by a cross-encoder, with code-intent and scope routing, presets and
  per-query knobs.
- Cited answers with a confidence line; `/search` for retrieval without an LLM.
- The management console (Corpus Ledger) with a job queue, the inbox lane,
  small-to-big retrieval, a tag lane, the HyPE lane, the Vault tab, the Info
  tab and its diagrams, and `/api/schema` with the rag-ops skill.
- PDF page ranges, raw code ingestion, chunking-method selection, a VLM OCR
  engine, the Omnisearch lane, a pytest suite and an exam-grounded eval.
