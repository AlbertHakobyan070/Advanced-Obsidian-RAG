# Usage

Three ways to drive the system: the **CLI**, the warm **HTTP API**, and the **console**.
They share one pipeline and one config.

## CLI

```bash
python main.py index                       # build indexes from data/*.jsonl
python main.py index --append <file>.jsonl # add a source family (auto-rebuilds sparse)
python main.py ingest-pdfs [--pages "1-50,60"] [--chunking heading|fixed]
python main.py ingest-notebooks
python main.py ingest-code --include-path "<subtree>"
python main.py query "<question>" [--preset code|concept|synthesis] [--top-k N] [--max-tokens N]
python main.py chat                        # interactive REPL
python main.py eval [--retrieval-only]     # score the golden suite
```

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
| `max_tokens` | Cap the answer length. |

!!! warning "`max_tokens` and citations"
    A very small `max_tokens` can truncate the citation footer and drop the answer's
    confidence to `UNKNOWN`. Leave enough room (a few hundred tokens) for a cited answer.

## The Corpus Ledger console (`:8052`)

Open **http://127.0.0.1:8052**. Tabs:

- **Query** — Ask / Search with every knob, plus copy, `.md` export (Obsidian-ready), and
  a Markdown + LaTeX preview.
- **Documents** — search, `#tag` filter, **retag** (metadata-only, no re-embed), and
  **delete** from the index.
- **Vault** — browse the mounted vault tree read-only.
- **Ingest** — add material (PDFs / notebooks / inbox uploads), pick a chunking strategy.
- **Jobs** — watch long-running ingest / maintenance jobs.
- **Info** — an in-app diagram of the whole pipeline with a query/ingestion toggle and a
  per-knob influence table.

## For agents

`manage_api` exposes `GET /api/schema` — a machine-readable, permission-tiered map of
every management operation (read / mutating / destructive) so an agent can discover what
it may call before it calls it. Pair it with the query API: **ask** → `:8051`, **change
the corpus** → `:8052`.
