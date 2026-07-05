# Docker deployment

Run the whole system on another machine with **two commands** — no Python, no venv, no
path surgery. Docker Compose brings up both services plus a generation backend, all
bound to `127.0.0.1` (nothing is exposed to your network).

- **Query API** → `http://127.0.0.1:8051`
- **Corpus Ledger console** → `http://127.0.0.1:8052`
- **Generation backend** (OpenAI-compatible) → `http://127.0.0.1:3001`

## What's in the bundle

```
docker-compose.yml        # the two services + generation backend, 127.0.0.1-bound
Dockerfile                # CPU-only PyTorch, both APIs in one container
config.docker.yaml        # container config (Linux paths); copied to config.yaml in the image
docker-entrypoint.sh      # runs serve_api (:8051) + manage_api (:8052) together
.env.example              # generation-backend settings template
app/                      # the application source
data/                     # JSONL chunk files (the source of truth)
rag_data/                 # prebuilt ChromaDB + BM25 indexes
```

The image installs **CPU-only PyTorch first**, so it never drags in multi-gigabyte GPU
wheels — the embedder and cross-encoder run on CPU. The data and indexes are mounted as
volumes, not baked into the image, so `docker compose build` stays fast.

## Prerequisite

Install **Docker Desktop** and start it once.

## Start it

```bash
docker compose up --build -d
docker compose logs -f rag      # watch startup
```

First run builds the image and, on the first query, downloads the embedding + rerank
models (~150 MB, cached forever after). Wait until:

```bash
curl -fsS http://127.0.0.1:8051/health   # -> {"ready": true}
```

## Point it at a model

The generation backend needs a model provider. Open **http://127.0.0.1:3001** and add
your key once (it's stored encrypted in the backend's own volume). Prefer a cloud key
instead? Edit `config.docker.yaml` → `generation.base_url` (and the key) and drop the
generation service. Retrieval-only (`/search`) needs no model at all.

## Mount your vault (optional but recommended)

To use the **Vault browser** and the **Ingest** tab on the target machine, mount your
Obsidian vault into the container. In `docker-compose.yml`, under the `rag` service:

```yaml
    volumes:
      - "/path/to/your/vault:/vault"     # host vault -> container /vault
```

The container config points `parser.vault_path` at `/vault`, and the console's Vault tab
browses `webui.vault_tree_root` beneath it. With the vault mounted, everything works —
Query, Documents, the Vault tree, and Ingest. Without it, Query / Documents / Jobs still
work fully against the shipped corpus; only the vault-dependent tabs are inert.

!!! tip "Read-only vs read-write"
    Mount read-only (`:ro`) for a safe demo where nothing can alter the vault, or
    read-write if you want the Ingest tab and inbox uploads to write new material into
    the mounted vault.

## Day-to-day

```bash
docker compose stop      # stop, keep data + models
docker compose start     # fast start, no rebuild
docker compose down      # remove containers (named volumes persist)
```

## Re-packaging after corpus changes

New material is added on the machine that owns the full vault, then the bundle is
re-packaged (the packaging script copies the current JSONLs + prebuilt indexes and
archives everything). Ship the archive, unpack, and `docker compose up` on the target.

## Troubleshooting

- **Console loads but the query dot is red** — the container is still loading the index
  or downloading models. Give it a minute; check `docker compose logs -f rag` for
  `Application startup complete`.
- **Ask returns a generation error** — the generation backend has no provider key yet, or
  a free quota is exhausted. `Search only` still works.
- **Port already in use** — change the left-hand host port in `docker-compose.yml`
  (e.g. `"127.0.0.1:9051:8051"`).
