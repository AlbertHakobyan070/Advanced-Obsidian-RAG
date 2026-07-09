# Advanced Obsidian RAG

**Grounded, cited question-answering over a personal knowledge base of markdown
notes, textbooks, lecture PDFs, and notebooks** — built for the case where a
wrong-but-confident answer is worse than none. Ask a question in your own words
and get an answer assembled *only* from your own materials, with inline `[n]`
citations and a per-answer confidence line.

<p>
<img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
<img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
<img alt="ChromaDB" src="https://img.shields.io/badge/Vectors-ChromaDB-FF6C37">
<img alt="bm25s" src="https://img.shields.io/badge/Sparse-bm25s-4B8BBE">
<img alt="Docker" src="https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED?logo=docker&logoColor=white">
<img alt="Local first" src="https://img.shields.io/badge/Runs-Free%20%2F%20Local-2ea44f">
</p>

> **Index it once, query it forever.** A full retrieval-augmented pipeline that
> turns a folder of markdown notes, PDFs, scanned books, and notebooks into a
> search system you can talk to. Every claim is traceable to a source you
> already trust, and the answer says *"I don't have this"* instead of inventing
> when your materials are silent on a topic.

---

## Why this exists

A general chatbot answers from the open web. This system indexes *your*
documents — your notes, your textbooks, your scanned books, your notebooks —
and answers strictly from them, so every claim traces back to a source you
already trust and a wrong-but-confident answer is impossible.

It is a full pipeline, not a wrapper: hybrid retrieval, query expansion,
intent-aware scope routing, a dedicated code lane, cross-encoder reranking,
grounded generation with a citation-audit pass, a management console, an
agent-facing API, and a reproducible evaluation suite.

## How a document ingestion takes place

<img width="1017" height="787" alt="Ingestion pipeline diagram" src="https://github.com/user-attachments/assets/2b05a9b2-a494-4041-8ce8-23bff68a3ae6" />


## How a query flows

<img width="1146" height="1034" alt="Query pipeline diagram" src="https://github.com/user-attachments/assets/2c2881e0-2764-44b2-a879-f3618a841e52" />


Every stage is swappable from `config.yaml`. Solid path = always on; the rest
are optional lanes that open only when the query calls for them.

## What makes retrieval good here

- **Hybrid dense + sparse, fused by RRF.** Dense embeddings (bge-small-en-v1.5)
  catch paraphrase and meaning; BM25 catches exact terminology, symbols, and
  rare names. Reciprocal Rank Fusion combines them with a downstream
  cross-encoder deciding final order — no fragile score normalisation.
- **Intent-aware scope routing.** Queries that name a domain, content type, or
  library ("in my statistics lectures", "in the tech books") open *filtered*
  lanes toward the right material. Routing is *soft*: scoped chunks get
  guaranteed seats in the candidate pool, but the reranker still makes the
  final call — a bad hint can never empty your results. Both dictionaries are
  config-only; extend them without touching code.
- **A dedicated code lane.** Code/notebook chunks are a tiny fraction of a
  typical corpus, so for a query like *"show me a complex ggplot from my
  code"* a prose-oriented pipeline buries them under textbook pages that
  merely mention the keyword. Detecting code intent, skipping HyDE, widening
  the pool, and reserving a filtered lane for script/notebook chunks brings
  the user's own code back to the top.
- **Cross-encoder reranking** re-scores the fused candidates against the
  actual query for precision at the top.
- **Grounded, cited generation** answers from the retrieved excerpts only,
  emits inline `[n]` citations and a confidence line, and can run a second
  pass that verifies each citation actually supports its sentence.

## Tuning without restarts

Named retrieval **presets** live in `config.yaml` and are selectable per query
— the warm pipeline is never mutated:

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

## Evaluation — example setup

The repo ships a **small illustrative golden suite** (`eval/golden_queries.yaml`,
six queries across three categories — conceptual, exam-like, code) so the
runner is exercised out of the box:

```bash
python main.py eval --retrieval-only      # offline regression, no LLM, runs in minutes
```

| Metric | What it measures |
|---|---|
| **Keyword recall** | Fraction of `expect_keywords` that appear in the top-k chunks (or in the answer, full run). |
| **Retrieval hit** | Whether the expected `domain` metadata value landed in the top-k. |
| **Course hit** *(optional)* | Whether the #1 source's `course_name` metadata matches `expect_course` (only if you populated `parser.course_taxonomy` in `config.yaml`). |
| **Citation support** *(full run)* | Fraction of cited sources the second-pass auditor marked supported. |
| **Confidence / Answered** *(full run)* | Distribution of HIGH/MEDIUM/LOW and the fraction of non-punting answers. |

These are deliberately framed as **automatic proxy metrics**. Keyword recall
checks that expected terms *appear* — not that the explanation is correct;
retrieval hit checks the domain, not the exact passage. They exist to **catch
regressions**, not to certify faithfulness — that job belongs to the
second-pass citation auditor, the per-answer confidence line, and ultimately
reading the cited source. Replace the shipped example with your own golden
set (30-100 queries written from your corpus) for serious regression work.

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
python main.py index --append data/ipynb_chunks.jsonl
python main.py index --append data/code_chunks.jsonl

# 4. Ask
python main.py query "Explain the central limit theorem from my notes"
python main.py chat                              # interactive REPL

# 5. Serve
python -m uvicorn serve_api:app --host 127.0.0.1 --port 8051   # warm JSON API
python -m uvicorn manage_api:app --host 127.0.0.1 --port 8052  # Corpus Ledger console

# 6. Measure
python main.py eval --retrieval-only             # fast offline regression
```

Full local-only setup (no cloud key) is in [`RUN_LOCAL.md`](RUN_LOCAL.md); the
day-to-day usage cookbook is in [`MANUAL.md`](MANUAL.md); the **documentation
site** (architecture, API, operations, Docker deployment) is under [`docs/`](docs/)
and builds with MkDocs:

```bash
pip install mkdocs-material
mkdocs serve        # http://127.0.0.1:8000
```

## Run it anywhere with Docker

The project ships a self-contained Docker bundle that runs both services plus a
generation backend with **two commands** — no Python, no venv, no path surgery —
and mounts your vault so the full console (including the Vault browser and
Ingest) works on a second machine:

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
  embeddings/  # embedder - builds ChromaDB + bm25s from chunk JSONL
  retrieval/   # retriever (hybrid + RRF + code lane), reranker, hyde, scope, context_expand
  generation/  # generator - grounded answers + citation verification
  llm/         # unified OpenAI-compatible / local client
  prompts/     # versioned YAML prompt templates + loader
  utils/       # config_loader (comment-preserving persistence), logger
  pipeline.py  # wires the query path together
eval/          # golden suite (illustrative example) + runner
tests/         # pytest suite (chunking, jobs, loaders)
docs/          # MkDocs documentation site
```

## Design notes

- **Content-addressed IDs.** `doc_id = sha256(source_file + text[:500])[:16]` —
  re-ingesting is idempotent; changing chunk *text* orphans vectors (there's a
  swap playbook for that), while metadata-only fixes go through an in-place
  retag with no re-embedding.
- **JSONL is the source of truth**; the vector DB and the sparse pickle are
  *derived* and rebuildable from it. Readers stream and split on `"\n"` only,
  so exotic Unicode inside chunk text can never shred a record.
- **Paged maintenance.** At this corpus size every scan/update/delete pages the
  vector store in bounded batches, so maintenance scripts stay within a modest
  RAM budget.
- **Graceful degradation.** If the generation endpoint is down, the API returns
  a readable error object (not a 500) so callers can relay the cause, and
  retrieval-only still works.
- **Configurable course taxonomy.** All course-name and domain mappings
  (course codes, folder names, abbreviations, daily-note conventions) live in
  `parser.course_taxonomy` in `config.yaml` — the pipeline ships with sensible
  generic defaults; you populate the entries for your own institution / corpus.

Deliberately **not** implemented: weighted RRF. It was evaluated and skipped —
the downstream cross-encoder already absorbs the benefit once the right chunks
are in the pool, and per-lane weights only re-introduce a tuning burden for
gains within noise.
