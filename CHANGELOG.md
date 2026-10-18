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
