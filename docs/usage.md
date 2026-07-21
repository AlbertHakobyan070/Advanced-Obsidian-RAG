# Usage

Three ways to drive the system: the **CLI**, the warm **HTTP API**, and the **console**.
They share one pipeline and one config.

## CLI

```bash
python main.py index                       # build indexes from data/*.jsonl
python main.py index --append <file>.jsonl # add a source family (auto-rebuilds sparse)
python main.py ingest-pdfs [--pages "1-50,60"] [--chunking heading|fixed|document|none] [--include-files "a.pdf,b.pdf"]
python main.py ingest-notebooks [--include-path "<scope>"] [--include-files "a.py,b.ipynb"] [--force-domain ml] [--force-tags "a,b"]
python main.py ingest-code --include-path "<subtree>" [--include-files "x.sql"] [--force-domain swe] [--force-tags "a,b"]
python main.py ingest-md --include-path "<scope>" --output data/<name>.jsonl [--force-domain nlp] [--force-tags "a,b"]  # scoped md parse (guarded)
python main.py fetch-web --urls "https://…" [--backend auto|requests|crawl4ai|scrapling] [--format md|pdf]
python main.py convert-files --files "report.docx" [--ocr-pages "1-4,9"]       # markitdown → .md
python main.py query "<question>" [--preset code|concept|synthesis] [--top-k N] [--max-tokens N]
python main.py chat                        # interactive REPL
python main.py eval [--retrieval-only]     # score the golden suite
```

**Chunking strategies** (`--chunking`, also selectable per run in the console): `heading` —
paragraph packing that follows the text's structure (default); `fixed` — strict sliding
window, for OCR'd / wall-of-text sources; `document` — element-aware packing that never
cuts inside a fenced code block, table, or list; `none` — no splitting at all (one chunk
per section; embedding models truncate very long inputs, so use deliberately). Small
sections chunk identically under every strategy.

## HTTP API (`:8051`)

The warm endpoint keeps the indexes and models hot, so queries never pay startup cost.

=== "Ask (grounded answer)"

    ```bash
    curl -s -X POST http://127.0.0.1:8051/query \
      -H "Content-Type: application/json" \
      -d '{"q": "Explain conjugate priors from my notes", "preset": "concept"}'
    ```

=== "Search (retrieval only, no LLM)"

    ```bash
    curl -s -X POST http://127.0.0.1:8051/search \
      -H "Content-Type: application/json" \
      -d '{"q": "docker networking", "top_k": 7}'
    ```

=== "Change defaults live"

    ```bash
    # persist: true also rewrites config.yaml, comment-preserving
    curl -s -X POST http://127.0.0.1:8051/config \
      -d '{"rerank_top_k": 10, "persist": true}'
    ```

Every `/query` response echoes what actually ran:

```json
"retrieval": { "preset": "code", "rerank_top_k": 10, "hyde_used": false,
               "dense_top_k": 40, "sparse_top_k": 40, "scopes": ["code"] }
```

`GET /history` returns the last `/search` + `/query` calls (newest first, in-memory):
the question, the knobs the caller explicitly set, the full retrieval echo of what
actually ran, confidence and timing — so an agent tuning hyperparameters can see what
it already tried instead of re-deriving it.

`fetch-web --format pdf` prints the fully rendered page through headless Chromium
(LaTeX, tables and highlighted code exactly as the site shows them) instead of
converting to markdown — the right lane for math- or code-heavy sources, and the
output ingests through the PDF lane with real page numbers. It needs Playwright's
Chromium once: `python -m playwright install chromium`.

## Presets and per-query knobs

Named override bundles in `config.yaml`, selectable per query with no restart — the warm
pipeline is never mutated:

```yaml
retrieval:
  presets:
    code:      {rerank_top_k: 10, use_hyde: false, dense_top_k: 40, sparse_top_k: 40, boost_code: true}
    concept:   {rerank_top_k: 5,  use_hyde: true}
    synthesis: {rerank_top_k: 10, use_hyde: true, dense_top_k: 30, sparse_top_k: 30}
```

| Knob | Effect |
|---|---|
| `preset` | Apply a named bundle (`code` / `concept` / `synthesis`). |
| `top_k` / `rerank_top_k` | How many reranked chunks reach the generator. |
| `dense_top_k` / `sparse_top_k` | Candidate-pool width per lane before fusion. |
| `use_hyde` / `hype` | Toggle query expansion. |
| `rerank` | Rerank method for this call: `cross_encoder` (semantic scoring, the config default), `lexical` (model-free query-term coverage — exact-keyword hunts), `none` (raw fused order). Config default: `retrieval.rerank_mode`. |
| `parent_context` / `neighbor_context` | E2 small-to-big, post-rerank: swap note chunks for their full section / append a PDF hit's adjacent pages. Carried by the `synthesis` preset; per-call override beats preset beats config. |
| `max_tokens` | Cap the answer length. |

!!! warning "`max_tokens` and citations"
    A very small `max_tokens` can truncate the citation footer and drop the answer's
    confidence to `UNKNOWN`. Leave enough room (a few hundred tokens) for a cited answer.

## The Corpus Ledger console (`:8052`)

Open **http://127.0.0.1:8052**. Tabs:

- **Query** — Ask / Search with every knob in labeled groups (pool sizes, extra lanes
  HyDE/HyPE/Omnisearch, rerank method, E2 parents/neighbors), plus copy and `.md` export
  (Obsidian-ready). The Markdown + LaTeX **Preview** toggle renders the answer *and every
  source chunk* — math, tables and fenced code included.
- **Documents** — search, `#tag` filter, **retag** (domain / course / tags —
  metadata-only, no re-embed), and **delete** from the index.
- **Vault** — browse the mounted vault tree read-only.
- **Ingest** — a three-step flow: **1 · Add documents** (upload to the inbox; **every
  routable type — pdf, md, code, notebook — rides a `default | custom` lane**, and every
  file can carry its **own ⚙ settings** — domain and tags for all kinds, chunking for
  pdf/md, OCR engine and page subset for PDFs — plus a **destination folder**: the file
  *moves to its vault home when the job is queued, before parsing*, so the index records
  the final path (blank = stays in the inbox, archives to `_ingested`). The **batch
  defaults** (domain / tags / chunking / destination, each with a 📁 picker) fill in for
  any file without its own ⚙ value. 👁 previews any md/PDF, with PDF page numbers visible
  for OCR-range picking); **2 · Fetch & convert** (pull web links as `.md` via markitdown
  **or as a printed `.pdf`** of the rendered page via headless Chromium —
  LaTeX/tables/code intact; convert any upload — pdf/docx/pptx/xlsx/html — to `.md`, with
  optional per-page OCR; outputs stage in `_converted` with preview until promoted);
  **3 · Jobs designer** — one **card per queued job** (kind badge, lane, file chips,
  effective-settings line): files sharing identical effective settings batch together,
  differing ⚙ settings split automatically; then **Queue the plan** (everything shown) or
  one lane at a time (*custom only* / *default only*). Vault-wide passes live under
  **Advanced** in the same panel. Routing is kind-aware: `.py`/`.R`/`.ipynb`/`.Rmd` go to
  the **notebook** lane (`ingest-notebooks` owns them — it has the Python
  `ast`/`# %%` cell splitter), while `.js`/`.ts`/`.sql`/`.go`/… go to the **code** lane.
- **Jobs** — watch long-running ingest / maintenance jobs.
- **Settings** — appearance with **dark and light preset shelves**: the built-in themes
  (the warm Ledger pair + **Material mint** dark/light) plus your own **saved presets** —
  export the active theme as CSS variables, tweak, save under a name, rename or delete
  from its chip; **font pickers** for headings / body / mono (system serif and sans
  choices like Times New Roman, Georgia, Arial — or any installed font by name; applied
  over every theme, browser-local); an Obsidian-style **vault switcher** that remembers
  every vault ever opened together with its own path/index settings and swaps the whole
  set atomically; and the editable config surface with **📁 folder pickers**: vault root,
  Chroma / BM25 / chunks paths, embedding + cross-encoder models, default rerank &
  chunking, generation endpoint. Saves rewrite `config.yaml` in place (comments
  preserved); nothing hot-applies — the response says which service to restart.
  Two panels sit under the fields: **Generation backends**, listing the
  `providers:` registry with each backend's endpoint/model and whether its API-key
  environment variable is set (never the value); and **Reranker**, which suggests
  known-good cross-encoders with their measured cost and states what PyTorch can
  actually reach — a `cuda` device on a CPU-only torch build would otherwise fail
  silently into a much slower path.
- **Info** — an in-app diagram of the whole pipeline with a query/ingestion toggle and a
  per-knob influence table.

Ops niceties: the header's **⟳ restart** button relaunches the warm query API in one
click (and `webui.auto_restart_rag` in Settings does it automatically after every
successful index-changing job); the Ledger shows an **Index health** card — ✓ in sync
when the dense (Chroma) and sparse (BM25) indexes hold the same corpus, ✗ drifted with
the fix spelled out (the BM25 count comes from a sidecar written at build time — no
giant unpickle); and the Query tab keeps a restorable **history** of your last 20
queries with their knobs.

## For agents

`manage_api` exposes `GET /api/schema` — a machine-readable, permission-tiered map of
every management operation (read / mutating / destructive) so an agent can discover what
it may call before it calls it. Pair it with the query API: **ask** → `:8051`, **change
the corpus** → `:8052`.
