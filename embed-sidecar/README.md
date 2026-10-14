# Embedding sidecar

EmbeddingGemma 2 (`google/embeddinggemma-2`, 768-d) on a GPU, behind an
OpenAI-compatible `POST /v1/embeddings` and `GET /health`, in its own venv.

It is a sidecar because EmbeddingGemma 2 needs sentence-transformers >= 6.1.0,
transformers >= 5.19 and torchvision, while the project venv
(`<runtime-dir>\.venv`) pins an older stack with a CPU-only torch. The project
reaches it through the `embedding.providers` registry (`kind: openai`), so
nothing in the project venv is upgraded and `main.py` never imports torch's GPU
build.

One file (`server.py`, standard-library HTTP server), one `requirements.txt`.

## What it does and does not do

- Loads the model **text-only** (vision and audio towers dropped, 271M params),
  **fp32** (Pascal has no bf16; fp16 is NaN or degraded on this model), on the
  CUDA device you choose.
- **Applies no prefixes.** The client sends already-prefixed text: the registry
  entry's `query_prefix` / `doc_prefix` (`task: search result | query: ` and
  `title: none | text: `).
- **`max_seq_length` is always bounded** (`EMBED_MAX_SEQ_LENGTH`, default 512).
  sentence-transformers' own default for this model is effectively unbounded;
  500 real chunks reached 7.6 GB RSS. The server reports the value it applies in
  every response (extra field `max_seq_length`) and in `/health`, and the client
  refuses a server whose value differs from the entry's `max_seq_length`.
- Optional **Matryoshka truncation**: a request's `dimensions` (or the server
  default `EMBED_TRUNCATE_DIM`) keeps the first N values and re-normalises.
- Never substitutes: another model id, a bad `dimensions` or non-string input is
  HTTP 400; asking for CUDA without a visible CUDA device stops startup.

## Create the venv (once)

The measured stack (torch 2.14.1+cu126, sentence-transformers 6.1.0,
transformers 5.19.0, torchvision 0.29.1+cu126) is what `G:\eg2_bench\gpu_venv`
ran. Use a venv of its own, never the project venv:

```powershell
uv venv <runtime-dir>\.venv-embed --python 3.11
uv pip install --python <runtime-dir>\.venv-embed\Scripts\python.exe -r embed-sidecar\requirements.txt
```

`uv pip compile embed-sidecar/requirements.txt` resolves to exactly those
versions. (This README's commands were run against `G:\eg2_bench\gpu_venv`; the
fresh install above was resolved but not performed.)

## Launch

Pick the physical card with `CUDA_VISIBLE_DEVICES` (inside the process it is
always `cuda:0`). The weights already sit in the project's Hugging Face cache;
pointing `HF_HUB_CACHE` at it avoids a second 1.5 GB download. The server also
drops `SENTENCE_TRANSFORMERS_HOME`, which this machine sets user-wide to a
folder that does not hold them.

```powershell
$env:CUDA_VISIBLE_DEVICES = "1"            # the idle card
$env:EMBED_MAX_SEQ_LENGTH = "1024"         # must equal the registry entry you will use
$env:EMBED_PORT           = "8104"         # must equal the entry's base_url port
$env:HF_HOME      = "<runtime-dir>\.cache\huggingface"
$env:HF_HUB_CACHE = "<runtime-dir>\.cache\huggingface\hub"
<runtime-dir>\.venv-embed\Scripts\python.exe embed-sidecar\server.py
```

| variable | default | meaning |
|---|---|---|
| `EMBED_MODEL` | `google/embeddinggemma-2` | model id served (clients must ask for exactly this) |
| `EMBED_MAX_SEQ_LENGTH` | `512` | token cap; positive integer |
| `EMBED_TRUNCATE_DIM` | unset | server default Matryoshka width (1..768) |
| `EMBED_BATCH_SIZE` | `16` | GPU batch inside one request (32 doubled VRAM for no speed) |
| `EMBED_DEVICE` | `cuda` | `cuda`, `cuda:N` or `cpu` (explicit; there is no silent fallback) |
| `EMBED_HOST` / `EMBED_PORT` | `127.0.0.1` / `8104` | listen address |

Check it: `curl.exe http://127.0.0.1:8104/health`.

The two shipped registry entries (`config.yaml`, `config.example.yaml`):

| entry | `max_seq_length` | `base_url` |
|---|---|---|
| `eg2` | 1024 | `http://127.0.0.1:8104/v1` |
| `eg2_l512` | 512 | `http://127.0.0.1:8105/v1` |

Run one sidecar per cap (two ports), on two cards at once if you like, or one
after the other on the same port with the other `EMBED_MAX_SEQ_LENGTH`.

## In a container

`Dockerfile` here builds the sidecar as an image: **GPU-only and linux/amd64 only**
(the CUDA 12.6 torch wheels are x86_64 builds, and an Apple Silicon host has no
NVIDIA runtime to run them on). It runs as a non-root user, listens on
`0.0.0.0:8104`, keeps the weights in `/hf_cache` (a volume), and reads the same
variables as the table above. `HF_TOKEN` is passed through for a gated model.

```bash
docker build -t noetrix-embed:local embed-sidecar/
docker run --rm --gpus all -p 127.0.0.1:8104:8104 -v embed-hf:/hf_cache noetrix-embed:local
```

Inside the compose stack and the Kubernetes manifests it answers to the service
name `embed` (and `embed-l512` for the 512-token entry), which is what
`config.example.yaml`'s container rewrite points `eg2` / `eg2_l512` at. See
`deploy/README.md`.

## Building a collection with it

The sidecar only embeds. Collections are built by the project, into a NEW
collection, resumably (run the same command again after an interruption):

```powershell
.\rag.bat index --rebuild-into obsidian_vault_eg2_l1024 --embedding eg2
.\rag.bat index --rebuild-into obsidian_vault_eg2_l512  --embedding eg2_l512
```

Smoke test first (200 chunks into a throwaway collection, left unservable):
add `--limit 200`.

## Contract

```
GET  /health          -> {"ok", "model", "device", "dtype", "max_seq_length",
                          "dimension", "truncate_dim", "batch_size", "versions"}
POST /v1/embeddings   {"model", "input": str | [str], "dimensions"?, "encoding_format"?: "float"|"base64"}
                      -> OpenAI embeddings response + "max_seq_length"
```

`usage` token counts are not computed (always 0).

## Rerank mode

`SIDECAR_TASK=rerank` makes the same server load a cross-encoder instead
(`RERANK_MODEL`, default `BAAI/bge-reranker-v2-m3`, fp32, checked after the load)
and answer `POST /v1/rerank` in the shape the project's `rerank_mode: http` client
reads, so the bench and `/search` can use a big reranker on a GPU while the project
venv stays CPU-only. `config.yaml`'s `retrieval.rerank_http` already points at
`http://127.0.0.1:8101/v1` with model `bge-reranker-v2-m3`, which the server accepts
as the bare name of its id.

```powershell
$env:SIDECAR_TASK         = "rerank"
$env:CUDA_VISIBLE_DEVICES = "1"            # the headless card
$env:EMBED_PORT           = "8101"         # retrieval.rerank_http.base_url
$env:HF_HOME      = "<runtime-dir>\.cache\huggingface"
$env:HF_HUB_CACHE = "<runtime-dir>\.cache\huggingface\hub"
G:\eg2_bench\gpu_venv\Scripts\python.exe embed-sidecar\server.py
```

```
GET  /health      -> {"ok", "task": "rerank", "model", "device", "dtype", "max_seq_length", "batch_size", "versions"}
POST /v1/rerank   {"model", "query", "documents": [str], "top_n"?}
                  -> {"model", "results": [{"index", "relevance_score"}]}, best first
```

`relevance_score` is the raw logit. `EMBED_MAX_SEQ_LENGTH` (default 512, as the
in-process `cross_encoder_max_length`) caps each (query, document) pair, and
`EMBED_BATCH_SIZE`, `EMBED_DEVICE`, `EMBED_HOST` work as in embed mode.
