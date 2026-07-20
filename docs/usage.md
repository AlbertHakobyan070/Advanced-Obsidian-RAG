# Usage

Three ways to drive the system: the **CLI**, the warm **HTTP API**, and the **console**.
They share one pipeline and one config.

## CLI

```bash
python main.py index                       # build indexes from data/*.jsonl
python main.py index --append <file>.jsonl # add a source family (auto-rebuilds sparse)
python main.py ingest-pdfs [--pages "1-50,60"] [--chunking heading|fixed|document|none] [--include-files "a.pdf,b.pdf"]
python main.py ingest-notebooks
python main.py ingest-code --include-path "<subtree>" [--include-files "x.sql"]
python main.py ingest-md --include-path "<scope>" --output data/<name>.jsonl   # scoped md parse (guarded)
python main.py fetch-web --urls "https://…" [--backend auto|requests|crawl4ai|scrapling]
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

- **Query** — Ask / Search with every knob (pools, HyDE/HyPE, rerank method, E2
  parents/neighbors), plus copy, `.md` export (Obsidian-ready), and a Markdown + LaTeX
  preview.
- **Documents** — search, `#tag` filter, **retag** (domain / course / tags —
  metadata-only, no re-embed), and **delete** from the index.
- **Vault** — browse the mounted vault tree read-only.
- **Ingest** — a three-step flow: **1 · Add documents** (upload to the inbox, remove
  mistakes with ✕); **2 · Fetch & convert** (pull web links as markdown, or convert any
  upload — pdf/docx/pptx/xlsx/html — to `.md` via markitdown, with optional per-page OCR;
  outputs stage in `_converted` until promoted); **3 · Route & ingest** — the
  custom-jobs designer: route each file `default | custom`, custom files sort into
  PDF / code / md job groups with per-group knobs and a global chunking pick, review the
  live plan preview, then *Ingest custom jobs* / *Ingest all* / *Ingest inbox only*.
  Vault-wide passes live under *Advanced · Custom job*.
- **Jobs** — watch long-running ingest / maintenance jobs.
- **Settings** — four themes (the warm Ledger pair + **Material mint** dark/light), a
  description mode that overlays inline explainers on every Ingest/Query section, and the
  editable config surface: vault root, Chroma / BM25 / chunks paths, embedding +
  cross-encoder models, default rerank & chunking, generation endpoint. Saves rewrite
  `config.yaml` in place (comments preserved); nothing hot-applies — the response says
  which service to restart. Point the paths at another vault + index trio to run the same
  console against a different corpus.
- **Info** — an in-app diagram of the whole pipeline with a query/ingestion toggle and a
  per-knob influence table.

Ops niceties: the header's **⟳ restart** button relaunches the warm query API in one
click (and `webui.auto_restart_rag` in Settings does it automatically after every
successful index-changing job); the Ledger shows an **Index health** card — ✓ in sync
when the dense (Chroma) and sparse (BM25) indexes hold the same corpus, ✗ drifted with
the fix spelled out (the BM25 count comes from a sidecar written at build time — no
giant unpickle); and the Query tab keeps a restorable **history** of your last 20
queries with their knobs. Themes are also **importable/exportable** as CSS-variable
blocks from Settings → Appearance — export the active theme as a template, edit the
values, paste it back as the `custom` theme.

## For agents

`manage_api` exposes `GET /api/schema` — a machine-readable, permission-tiered map of
every management operation (read / mutating / destructive) so an agent can discover what
it may call before it calls it. Pair it with the query API: **ask** → `:8051`, **change
the corpus** → `:8052`.
