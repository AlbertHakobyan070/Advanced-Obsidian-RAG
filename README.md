# Advanced Obsidian RAG

**Grounded, cited question-answering over a large personal Obsidian vault** — built
for real interview and exam preparation, where a wrong-but-confident answer is worse
than none. Ask a question in your own words and get an answer assembled *only* from
your own notes, textbooks, homework and notebooks, with inline `[n]` citations and a
per-answer confidence line.

<p>
<img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
<img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
<img alt="ChromaDB" src="https://img.shields.io/badge/Vectors-ChromaDB-FF6C37">
<img alt="bm25s" src="https://img.shields.io/badge/Sparse-bm25s-4B8BBE">
<img alt="Docker" src="https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED?logo=docker&logoColor=white">
<img alt="Local first" src="https://img.shields.io/badge/Runs-Free%20%2F%20Local-2ea44f">
</p>

> **Scale:** ~170,000 retrieval chunks across **4,400+ documents** — markdown lecture
> notes, 280+ textbooks, lecture PDFs, passed homework, Jupyter/R notebooks and
> scripts, and OCR'd scanned books — spanning **24+ courses** and **9 knowledge
> domains**, all served from **~4 GB** of prebuilt indexes on a laptop.
>
> **Runs entirely on free / local infrastructure:** CPU embeddings, on-disk vector +
> sparse indexes, and *any* OpenAI-compatible endpoint for generation (a local model
> server, a free-tier proxy, or a cloud API — one config line).

---

## Why this exists

General-purpose chatbots answer from the open web; they can't tell you what *your*
professor emphasised, how *your* homework solved a problem, or which page of *your*
textbook covers a proof. This system indexes a personal knowledge vault and answers
strictly from it — so every claim is traceable to a source you already trust, and the
answer says *"I don't have this"* instead of inventing when the vault is silent.

It is a full pipeline, not a wrapper: hybrid retrieval, query expansion, intent-aware
scope routing, a dedicated code lane, cross-encoder reranking, grounded generation
with a citation-audit pass, a management console, an agent-facing API, and a
reproducible evaluation suite.

## How a document ingestion takes place

<img width="1017" height="787" alt="SCR-20260706-eygn" src="https://github.com/user-attachments/assets/2b05a9b2-a494-4041-8ce8-23bff68a3ae6" />


## How a query flows

<img width="1146" height="1034" alt="SCR-20260706-eutn" src="https://github.com/user-attachments/assets/2c2881e0-2764-44b2-a879-f3618a841e52" />


Every stage is swappable from `config.yaml`. Solid path = always on; the rest are
optional lanes that open only when the query calls for them.

## What makes retrieval good here

- **Hybrid dense + sparse, fused by RRF.** Dense embeddings (bge-small-en-v1.5) catch
  paraphrase and meaning; BM25 catches exact terminology, symbols, and rare names.
  Reciprocal Rank Fusion combines them with a downstream cross-encoder deciding final
  order — no fragile score normalisation.
- **Intent-aware scope routing.** Queries that name a domain or content type ("in my
  statistics lectures", "in the tech books", "my NLP homework") open *filtered* lanes
  toward the right material. Routing is *soft*: scoped chunks get guaranteed seats in
  the candidate pool, but the reranker still makes the final call — a bad hint can
  never empty your results. Both dictionaries are config-only; extend them without
  touching code.
- **A dedicated code lane.** Code/notebook chunks are a tiny fraction of the corpus, so
  for a query like *"show me a complex ggplot from my code"* a prose-oriented pipeline
  buries them under textbook pages that merely mention the keyword. Detecting code
  intent, skipping HyDE, widening the pool, and reserving a filtered lane for
  script/notebook chunks brings the user's own code back to the top.
- **Cross-encoder reranking** re-scores the fused candidates against the actual query
  for precision at the top.
- **Grounded, cited generation** answers from the retrieved excerpts only, emits
  inline `[n]` citations and a confidence line, and can run a second pass that verifies
  each citation actually supports its sentence.

## Tuning without restarts

Named retrieval **presets** live in `config.yaml` and are selectable per query — the
warm pipeline is never mutated:

```yaml
retrieval:
  presets:
    code:      {rerank_top_k: 10, use_hyde: false, dense_top_k: 40, sparse_top_k: 40, boost_code: true}
    concept:   {rerank_top_k: 5,  use_hyde: true}
    synthesis: {rerank_top_k: 10, use_hyde: true, dense_top_k: 30, sparse_top_k: 30}
```

```bash
# CLI
python main.py query "Explain conjugate priors from my notes" --preset concept

# Warm HTTP endpoint (no restart, hot pipeline)
curl -s -X POST http://127.0.0.1:8051/query \
     -H "Content-Type: application/json" \
     -d '{"q": "show me a complex ggplot from my code", "preset": "code", "top_k": 10}'
```

Every response echoes exactly what ran (`retrieval: {preset, rerank_top_k, hyde_used, …}`),
so results are always explainable.

## Two services + a console

| Surface | Port | What it's for |
|---|---|---|
| **Query API** (`serve_api`) | `:8051` | Warm FastAPI endpoint — `/query`, `/search`, `/config`. For agents, scripts, bots. |
| **Corpus Ledger console** (`manage_api`) | `:8052` | Visual management: Query, Documents (search / filter / retag / delete), Vault browser, Ingest, Jobs, and an **Info** tab that diagrams the whole pipeline in-app. |

## Evaluation — honest by design

A 94-question, exam-grounded suite (`eval/golden_queries.yaml`) scored automatically:

| Metric | Result |
|---|---|
| Retrieval hit-rate (expected domain in top-k) | **~96%** |
| Keyword recall (expected terms in the answer) | **~96%** |
| Course-routing accuracy | **~78%** |

These are deliberately framed as *automatic proxy metrics*. Keyword recall checks that
expected terms appear, not that the explanation is correct; retrieval hit checks the
domain, not the exact passage. They exist to **catch regressions**, not to certify
faithfulness — that job belongs to the second-pass citation auditor, the per-answer
confidence line, and ultimately reading the cited source. A `--retrieval-only` mode
runs the whole regression offline in minutes, no LLM required.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env                 # add your generation API key (or point at a local server)
# edit config.yaml: parser.vault_path -> your Obsidian vault

# 1. Parse markdown notes -> data/chunks.jsonl
python -m src.ingestion.obsidian_parser "path/to/vault" -o data/chunks.jsonl

# 2. Build the dense + sparse indexes
python main.py index

# 3. (optional) add PDFs, notebooks, and code, then append
python main.py ingest-pdfs                       # -> data/pdf_chunks.jsonl
python main.py ingest-notebooks                  # -> data/ipynb_chunks.jsonl
python main.py ingest-code --include-path "src"  # -> data/code_chunks.jsonl
python main.py index --append data/pdf_chunks.jsonl

# 4. Ask
python main.py query "How did I implement knowledge distillation in my capstone?"
python main.py chat                              # interactive REPL

# 5. Serve
python -m uvicorn serve_api:app --host 127.0.0.1 --port 8051   # warm JSON API
python -m uvicorn manage_api:app --host 127.0.0.1 --port 8052  # Corpus Ledger console

# 6. Measure
python main.py eval --retrieval-only             # fast offline regression
```

Full local-only setup (no cloud key) is in [`RUN_LOCAL.md`](RUN_LOCAL.md); day-to-day
usage is in [`MANUAL.md`](MANUAL.md); the **documentation site** (architecture, API,
operations, Docker deployment) is under [`docs/`](docs/) and builds with MkDocs:

```bash
pip install mkdocs-material
mkdocs serve        # http://127.0.0.1:8000
```

## Run it anywhere with Docker

The project ships a self-contained Docker bundle that runs both services plus a
generation backend with **two commands** — no Python, no venv, no path surgery — and
mounts your vault so the full console (including the Vault browser and Ingest) works on
a second machine:

```bash
docker compose up --build -d     # query API :8051 · console :8052 · generation :3001
```

See [`docs/deployment-docker.md`](docs/deployment-docker.md) for the full walkthrough.

## Repository layout

```
config.yaml / config.example.yaml   # every tunable: providers, top-k, presets, paths, routing
.env.example                        # secrets template (real .env is gitignored)
main.py                             # CLI: index | ingest-* | query | chat | eval | serve
serve_api.py                        # warm query API (:8051)
manage_api.py                       # Corpus Ledger console backend (:8052)
webui/index.html                    # the console front-end
app.py                              # optional Streamlit interface
src/
  ingestion/   # obsidian_parser, pdf_loader (OCR-capable), ipynb_loader, code_loader, ocr_vlm
  embeddings/  # embedder — builds ChromaDB + bm25s from chunk JSONL
  retrieval/   # retriever (hybrid + RRF + code lane), reranker, hyde, scope, context_expand
  generation/  # generator — grounded answers + citation verification
  llm/         # unified OpenAI-compatible / local client
  prompts/     # versioned YAML prompt templates + loader
  utils/       # config_loader (comment-preserving persistence), logger
  pipeline.py  # wires the query path together
eval/          # 94-question golden suite + runner (retrieval hit-rate, keyword recall)
tests/         # pytest suite (chunking, jobs, loaders)
docs/          # MkDocs documentation site
```

## Design notes

- **Content-addressed IDs.** `doc_id = sha256(source_file + text[:500])[:16]` — re-ingesting
  is idempotent; changing chunk *text* orphans vectors (there's a swap playbook for that),
  while metadata-only fixes go through an in-place retag with no re-embedding.
- **JSONL is the source of truth**; the vector DB and the sparse pickle are *derived*
  and rebuildable from it. Readers stream and split on `"\n"` only, so exotic Unicode
  inside chunk text can never shred a record.
- **Paged maintenance.** At this corpus size every scan/update/delete pages the vector
  store in bounded batches, so maintenance scripts stay within a modest RAM budget.
- **Graceful degradation.** If the generation endpoint is down, the API returns a
  readable error object (not a 500) so callers can relay the cause, and retrieval-only
  still works.

Deliberately **not** implemented: weighted RRF. It was evaluated and skipped — the
downstream cross-encoder already absorbs the benefit once the right chunks are in the
pool, and per-lane weights only re-introduce a tuning burden for gains within noise.
