# Docker deployment

Run the whole system on another machine from the packaged `rag-docker-bundle`
tree with **two commands** — no Python, no venv, no path surgery. A plain source
clone does not contain the Compose/Dockerfile scaffold.

Everything binds to `127.0.0.1`. Nothing is exposed to your network.

| Service | Port | Starts by default | Stop it when |
|---|---|---|---|
| `rag` — query API + console | `8051`, `8052` | yes | never; it *is* the RAG |
| `freellmapi` — generation proxy | `3001` | yes | you generate through a direct cloud backend |
| `paddleocr` — OCR for scanned PDFs | `8103` | **no** (`--profile ocr`) | you are not ingesting scanned material |
| `rerank` — external reranker | `8102` | **no** (`--profile rerank`) | you are not using an external reranker |

## What's in the bundle

```
docker-compose.yml        # the services, 127.0.0.1-bound, with opt-in profiles
Dockerfile                # CPU-only PyTorch, both APIs in one container
config.docker.yaml        # container config (Linux paths); copied to config.yaml in the image
docker-entrypoint.sh      # supervises serve_api (:8051) alongside manage_api (:8052)
paddleocr/                # the OCR sidecar image + its tiny HTTP service
                          # (a shipping copy of the repo's ocr-sidecar/)
.env.example              # settings + provider-key template
app/                      # the application source
data/                     # JSONL chunk files (the source of truth)
rag_data/                 # prebuilt ChromaDB + BM25 indexes
```

The image installs **CPU-only PyTorch first**, so it never drags in
multi-gigabyte GPU wheels. Data and indexes are mounted as volumes rather than
baked into the image, so `docker compose build` stays fast.

## Prerequisite

Install **Docker Desktop**, start it once, and unpack a bundle produced by the
release packaging workflow.

## Start it

```bash
docker compose up --build -d
docker compose logs -f rag      # watch startup
```

First run builds the image and, on the first query, downloads the configured
embedding and rerank models into a persistent cache volume. Wait until:

```bash
curl -fsS http://127.0.0.1:8051/health   # -> {"ready":true,"state":"ready",...}
```

## One path to set

Set `RAG_HOST_HOME` in `.env` to your home folder:

```bash
RAG_HOST_HOME=/Users/yourname        # Linux: /home/yourname
```

Compose bind-mounts that folder **at the same absolute path inside the
container**, which is the whole trick: `/Users/yourname/Notes` means the same
thing to your file manager and to the RAG, so there is no translation layer to
drift. Every document folder under your home directory becomes browsable and
selectable from the console's vault switcher with no further edits.

!!! note "Folders outside your home directory"
    The console can only offer paths the **container** can see. A collection on
    an external drive needs its own entry in the `volumes:` list of the `rag`
    service.

## Point it at a model

Generation resolves through the `providers:` registry in `config.docker.yaml`.
Select one with `generation.provider`, keep its secret in the environment
variable named by `api_key_env`, and restart the query service. `GET /providers`
shows the active backend and whether each configured credential is present and
type-compatible, without returning secret values.

The console's **Settings → Generation backends** panel does all of this with
buttons, including writing a key into the mounted `.env` so it survives a
rebuild. Never put a key in the YAML.

The bundled proxy has its own dashboard at **http://127.0.0.1:3001**.
Retrieval-only (`/search`) needs no generation provider at all.

!!! warning "`[decrypt failed]` in the proxy dashboard"
    The proxy encrypts the keys you add through its dashboard with
    `ENCRYPTION_KEY` from `.env`, and stores them in a **named volume** that
    outlives that file. If `.env` is replaced with one carrying a different key
    — for instance by unpacking a newer archive — every stored key becomes
    unreadable. Restore the previous `ENCRYPTION_KEY`, or reset the store and
    re-enter the keys:

    ```bash
    docker compose down
    docker volume rm rag-docker-bundle_freellmapi-data
    docker compose up -d
    ```

## OCR for scanned PDFs

The container image ships **Tesseract** ready to use (`pdf.ocr_engine: auto`).
For scanned books it is the weakest of the three lanes; the **PaddleOCR
sidecar** is the one worth turning on:

```bash
docker compose --profile ocr up -d --build paddleocr
# then set pdf.ocr_engine: paddle  (Settings, or --ocr-engine paddle per job)
docker compose stop paddleocr        # when the scanned material is indexed
```

It is a separate image because its dependency tree is larger than the rest of
the stack put together and it is only needed while ingesting. Recognition
weights download once into the `paddle-cache` volume. The console's
**Settings → OCR** panel distinguishes *not configured* from *container
stopped*, so a deliberately-stopped sidecar never looks like a fault — and
`/health` returns **503** when the engine cannot import, so a container that is
up but unusable is reported unhealthy rather than accepting pages and failing
every one.

### On a machine with an NVIDIA GPU

`ocr-sidecar/` holds a GPU build of the same sidecar, as its **own compose
project**:

```bash
cd ocr-sidecar
docker compose -f docker-compose.gpu.yml up -d --build
docker compose -f docker-compose.gpu.yml down
```

It binds the same `127.0.0.1:8103` and answers the same contract, so
`pdf.paddle_ocr.base_url` is unchanged and the two are simply mutually
exclusive. Two things make it worth the separate project rather than another
profile:

- the recognition models sit in **VRAM**, so on a memory-tight host OCR stops
  competing with the query pipeline for system RAM;
- its lifecycle is independent — bringing the RAG stack down does not take the
  OCR engine with it, and the Mac bundle stays buildable on a host with no card.

`GET /health` reports `device` and `paddleocr_version`, so a GPU build that
quietly fell back to CPU is visible rather than merely slow. See
`ocr-sidecar/README.md`, in particular why the CUDA pin is 11.8 — a wheel
without kernels for your card still imports and still reports the right device
count, so that pin was chosen by running a real GPU op, not by reading a
support matrix.

### Two engines: `ocr` and `vl`

"PaddleOCR" names two different products, and the GPU sidecar serves both.
Choose per request with `"pipeline"`, or set the container default with
`PADDLE_OCR_PIPELINE`:

| | `ocr` (default) | `vl` |
|---|---|---|
| models | `PP-OCRv6_medium_det` + `_rec` | `PP-DocLayoutV3` + `PaddleOCR-VL-1.6-0.9B` |
| how | detect boxes, recognise each | layout detection, then a document VLM reads each block |
| output | plain text, one line per detected line | **markdown**: reflowed paragraphs, `#` headings, `$…$` LaTeX, HTML tables |

`ocr` is the default on measured evidence. Benchmarked over dense textbook
pages scored against each PDF's own text layer, `vl` costs roughly **30x the
time and 7x the VRAM while transcribing prose slightly worse**. On a 400-page
scanned book that is the difference between minutes and most of a day.

`vl` earns its cost only where the page's *structure* is the information. It
reconstructs a table as real HTML with its row and column headers, where `ocr`
returns the same digits with every relationship gone; and it emits
`$$p(x)=a_{0}+\cdots+a_{n}x^{n}$$` where `ocr` returns the fragments the
equation was printed as. That is a per-book judgement, which is why it is a
per-request field and not a new default:

```yaml
pdf:
  paddle_ocr:
    pipeline: vl        # null (sidecar default) | ocr | vl
    timeout: 600        # vl is ~90s/page — the 120s default will time out
```

Two things to know before pointing an ingest at it. `vl` needs the
`paddlex[ocr]` extras, which only the GPU image installs — a sidecar that
cannot serve the requested engine answers **501** rather than silently
returning the other engine's output under the wrong name. And engines are
cached for the life of the container, so one `vl` request pins its ~7.9 GB of
VRAM until you restart the sidecar.

`GET /health` reports `pipelines`, including the **model names** behind each.
That matters more than it looks: `PaddleOCR()` resolves to whatever the
installed PaddleX pins as current, so the same call yields a different model on
a different image, with no visible symptom until you compare transcripts.

The third lane, a document **vision model**, is reached over HTTP and
deliberately lives in no image here: on container CPU it is minutes per page.
Run one on the host and the container reaches it at `host.docker.internal`.

## An external reranker

`retrieval.rerank_mode: http` scores candidates against a `/v1/rerank` endpoint,
so a large reranker can run outside the query API's memory budget:

```bash
# put a reranker GGUF in ./models and name it in .env (RERANK_MODEL_FILE)
docker compose --profile rerank up -d rerank
# console: Settings -> Reranker -> "External rerank server"
# set retrieval.rerank_http.base_url to http://rerank:8102/v1
```

## Day-to-day

```bash
docker compose ps                  # what is actually running
docker compose stop               # stop everything, keep data + models
docker compose start              # fast start, no rebuild
docker compose stop paddleocr     # stop just one service
docker compose down               # remove containers (named volumes persist)
```

## Restarting the query API

After changing settings or switching document folders, press **⟳ restart :8051**
in the console header. This works *inside* the container: the entrypoint
supervises the query API, so the button restarts that process rather than
needing `docker compose restart`.

If it does not come back, the console now prints the query API's startup log
instead of spinning. The cause is almost always the setting that was just
changed — a reranker id that cannot be downloaded, an embedding model with a
typo, an index path that moved:

```bash
curl -s http://127.0.0.1:8051/health                    # {"state":"failed","error":...}
curl -s "http://127.0.0.1:8052/api/service/log?lines=60"
```

Before switching cross-encoder, press **Check** on the model row — it reports
whether the id resolves and whether the weights are already cached, which are
the two things that make the following restart fail.

## Troubleshooting

- **Console loads but the query dot is red** — still loading the index or
  downloading models. Check `docker compose logs -f rag` for
  `Application startup complete`.
- **`docker compose up` fails with "set RAG_HOST_HOME in .env"** — Compose
  refuses to guess where your home folder is. See [One path to set](#one-path-to-set).
- **Ask returns a generation error** — the backend has no key yet, or a free
  quota is exhausted. `/search` and evidence comparison trees still work.
- **Port already in use** — change the left-hand host port in
  `docker-compose.yml` (e.g. `"127.0.0.1:9051:8051"`).
- **Model download is slow the first time** — expected; it is cached in a named
  volume afterwards.
