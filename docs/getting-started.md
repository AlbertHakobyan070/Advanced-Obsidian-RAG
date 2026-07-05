# Getting started

## Requirements

- **Python 3.11**
- ~4 GB of disk for prebuilt indexes (or build your own from source notes)
- A generation endpoint — **any** OpenAI-compatible API. That can be a cloud provider,
  a free-tier proxy, or a fully local model server. Retrieval works with no LLM at all.

CPU is enough: embeddings (bge-small) and the cross-encoder reranker both run on CPU.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env      # add your generation key, or point at a local server
```

## Configure

All tunables live in `config.yaml` (copy `config.example.yaml` if you're starting
fresh). The two things to set first:

```yaml
parser:
  vault_path: "/path/to/your/obsidian/vault"   # where your notes live

generation:
  base_url: "http://127.0.0.1:3001/v1"          # any OpenAI-compatible endpoint
  model: "your-model-id"
```

Secrets stay in `.env` (gitignored); everything else is in `config.yaml`, which the
system can also rewrite in place (comment-preserving) when you change defaults live.

!!! tip "Fully local"
    To run with no cloud dependency at all — local embeddings plus a local model server
    for generation — follow `RUN_LOCAL.md`. Point `generation.base_url` at your local
    server and you're done.

## Build the indexes

```bash
# 1. Parse markdown notes -> data/chunks.jsonl
python -m src.ingestion.obsidian_parser "path/to/vault" -o data/chunks.jsonl

# 2. Build the dense (ChromaDB) + sparse (bm25s) indexes
python main.py index
```

Add other source families and append them (each writes its own JSONL, and `index
--append` rebuilds the sparse half automatically):

```bash
python main.py ingest-pdfs                        # -> data/pdf_chunks.jsonl
python main.py ingest-notebooks                   # -> data/ipynb_chunks.jsonl
python main.py ingest-code --include-path "src"   # -> data/code_chunks.jsonl

python main.py index --append data/pdf_chunks.jsonl
python main.py index --append data/ipynb_chunks.jsonl
python main.py index --append data/code_chunks.jsonl
```

!!! note "Chunking strategy"
    Pass `--chunking heading` (default) for structured documents or `--chunking fixed`
    for OCR walls of text. See [Architecture](architecture.md#ingestion).

## Ask your first question

```bash
python main.py query "How did I implement knowledge distillation in my capstone?"
python main.py chat        # interactive REPL
```

## Serve

```bash
# Warm query API (agents / scripts / bots)
python -m uvicorn serve_api:app --host 127.0.0.1 --port 8051

# Corpus Ledger console (visual management + Query tab)
python -m uvicorn manage_api:app --host 127.0.0.1 --port 8052
```

Then open **http://127.0.0.1:8052** for the console, or POST to **:8051** from code.
Next: [Usage](usage.md).

## Prefer containers?

Skip the local Python setup entirely and run everything — both services plus a
generation backend — with Docker Compose. See [Docker deployment](deployment-docker.md).
