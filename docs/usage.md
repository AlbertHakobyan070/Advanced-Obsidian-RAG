# Usage

Three ways to drive the system: the **CLI**, the warm **HTTP API**, and the **console**.
They share one pipeline and one config.

!!! info "Agent-facing query surface"
    The warm API exposes live capability and provider discovery, stable source lookup,
    and bounded comparison trees in addition to single `/search` and `/query` calls.
    Fetch `/schema` and `/providers` at runtime instead of copying preset or backend
    lists into an agent prompt.

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

=== "Compare branches"

    ```bash
    curl -s -X POST http://127.0.0.1:8051/compare \
      -H "Content-Type: application/json" \
      -d '{
        "q": "Explain conjugate priors from my notes",
        "mode": "search",
        "branches": [
          {"id": "baseline", "label": "Config baseline", "auto_preset": false},
          {"id": "concept", "preset": "concept"},
          {"id": "lexical", "rerank": "lexical"}
        ]
      }'
    ```

=== "Discover providers"

    ```bash
    curl -s http://127.0.0.1:8051/providers
    ```

=== "Change defaults live"

    ```bash
    # Live-only is the safe default and reverts on restart.
    curl -s -X POST http://127.0.0.1:8051/config \
      -d '{"rerank_top_k": 10}'
    ```

    Set `persist:true` only with operator authorization; it rewrites
    `config.yaml`. Prefer the management Settings API for deliberate persistent
    changes.

Every `/query` response identifies what actually ran. Every source has a stable
evidence id, and generated answers report the resolved backend/model rather than
leaving the caller to infer them from config:

```json
{
  "sources": [
    {
      "id": "<stable-evidence-id>",
      "origin_id": "<retrieval-id>",
      "lookup_available": true,
      "n": 1,
      "label": "<source label>",
      "cited": true
    }
  ],
  "retrieval": {
    "preset": "code",
    "rerank_top_k": 10,
    "hyde_used": false,
    "reranker_model": "<resolved-reranker>"
  },
  "generation": {
    "backend": "<provider-registry-name>",
    "protocol": "<wire-protocol>",
    "model": "<resolved-model>",
    "usage": {}
  }
}
```

`GET /history` returns the last `/search` + `/query` calls (newest first, in-memory):
the question, the knobs the caller explicitly set, the full retrieval echo of what
actually ran, confidence and timing — so an agent tuning hyperparameters can see what
it already tried instead of re-deriving it.

`GET /chunks/{chunk_id}` fetches the current evidence record behind a stable id returned
from `/search`, `/query`, or `/compare` when `lookup_available` is true.
Parent-expanded sections use a `parent:<id>` evidence id and also report the indexed
child as `origin_id`, so overlap is computed from the text each branch actually
received. Live Omnisearch excerpts use a content-derived `live:<hash>` evidence id and
set `lookup_available` to false because they are not stored index records.
`GET /schema` advertises the current request fields, branch limits, preset registry,
and endpoint map.

### Comparison-tree semantics

`POST /compare` accepts a question, `mode: search|query`, and a bounded list of named
branches. Read the live `/schema` for the current limits. A branch can override the
same retrieval controls as `/search`; query-mode branches may additionally override
the configured `provider` and `model`.

The response contains every branch plus `comparison`: common and branch-unique source
ids, per-branch ranks, rank spread, and pairwise overlap. It intentionally does not
compare raw scores, because cross-encoder logits, lexical scores, HTTP reranker scores,
and fused RRF values are on different scales.

Branches that differ only by provider/model reuse one exact evidence set. This makes a
generation-backend comparison about answer behavior instead of retrieval noise. A
generation failure stays on that branch; a retrieval/reranker failure marks the branch
as a retrieval error without discarding successful siblings.

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
| `auto_preset` | `false` suppresses implicit code-intent preset selection for a config-only baseline. Explicit `preset` still wins. |
| `top_k` / `rerank_top_k` | How many reranked chunks reach the generator. |
| `dense_top_k` / `sparse_top_k` | Candidate-pool width per lane before fusion. |
| `use_hyde` / `hype` | Toggle query expansion. |
| `rerank` | Rerank method for this call: `cross_encoder` (in-process semantic scoring), `http` (configured external `/v1/rerank` service), `lexical` (model-free query-term coverage), or `none` (raw fused order). Config default: `retrieval.rerank_mode`. |
| `parent_context` / `neighbor_context` | E2 small-to-big, post-rerank: swap note chunks for their full section / append a PDF hit's adjacent pages. Carried by the `synthesis` preset; per-call override beats preset beats config. |
| `max_tokens` | Cap the answer length. |
| `provider` / `model` | `/query` and query-mode `/compare` only: select a configured provider and optionally override its default model. Endpoints and secrets cannot be supplied per request. |

!!! warning "`max_tokens` and citations"
    A very small `max_tokens` can truncate the citation footer and drop the answer's
    confidence to `UNKNOWN`. Leave enough room (a few hundred tokens) for a cited answer.

!!! warning "Reranker failures are retrieval failures"
    A response beginning `Reranking failed:` did not reach generation. Preserve the
    underlying model error and inspect `GET /config` for the active reranker. In
    particular, `BAAI/bge-reranker-base` has a 512-token input limit; configuring it
    above that limit is invalid. Do not relabel this as a generation-provider outage.

## Generation providers

Provider definitions live in the `providers:` registry in `config.yaml`. A request can
name a configured backend, but it cannot inject a URL or secret. `GET /providers` returns
the active backend plus each configured backend's protocol, endpoint, default model,
secret environment-variable name, and readiness flags; it never returns secret values.
`available` means the required key is present and type-compatible, not that a remote
endpoint has been contacted.

The MiniMax entry is for MiniMax M3 through the Token Plan's Anthropic-compatible API.
Its subscription key begins `sk-cp-`. A MiniMax pay-as-you-go key begins `sk-api-` and
does not consume Token Plan quota. Put the subscription key only in the environment
variable declared by the registry, or store it through the console/provider-key API;
the configured prefix check rejects the wrong credential type. Restart the query
service after changing a provider secret.

## The Corpus Ledger console (`:8052`)

Open **http://127.0.0.1:8052**. Tabs:

- **Query** — Ask / Search with every knob in labeled groups (pool sizes, extra lanes
  HyDE/HyPE/Omnisearch, rerank method, E2 parents/neighbors), plus copy and `.md` export
  (Obsidian-ready). The Markdown + LaTeX **Preview** toggle renders the answer *and every
  source chunk* — math, tables and fenced code included. Its **Query comparison tree**
  fans the current question across live-config presets and generation backends or the
  built-in rerank baselines, with evidence-only and generated-answer modes.
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
  environment variable is set and type-compatible (never the value), plus controls
  to activate a backend or write its declared key to the gitignored `.env`; and
  **Reranker**, which suggests known-good cross-encoders with their measured cost and
  states what PyTorch can actually reach — a `cuda` device on a CPU-only torch build
  would otherwise fail silently into a much slower path.
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
it may call before it calls it. The query service has its own `GET /schema`: pair the
two maps rather than assuming a copied capability list. **Ask/search/compare/inspect
evidence or change warm retrieval defaults** → `:8051`; **manage the corpus,
persistent install settings, or provider secrets** → `:8052`.
