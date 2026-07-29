# OCR sidecar

PaddleOCR behind a two-endpoint HTTP contract, in its own container. This is
the middle OCR lane: better than Tesseract on skewed, multi-column and
non-Latin scans, and far cheaper than asking a vision-language model to parse
the page.

It is a sidecar rather than an import so its dependency tree never enters the
RAG image, and so you can stop it and get the memory back when you are not
ingesting.

## Two images, one `server.py`

| | CPU (`Dockerfile`) | GPU (`Dockerfile.gpu`) |
|---|---|---|
| base | `python:3.11-slim-bookworm` | `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` |
| paddle | `paddlepaddle==2.6.2` | `paddlepaddle-gpu==3.3.1` (cu118) |
| PaddleOCR | `2.9.1` | `3.7.0` |
| runs where | anywhere, including the Mac bundle | a host with an NVIDIA GPU |
| models live in | system RAM | VRAM |

`server.py` is shared. The two PaddleOCR lines disagree about the constructor
keywords (`use_angle_cls` + `show_log` versus `use_textline_orientation` +
`device`) and about the result shape (nested `[box, (text, score)]` tuples
versus a dict carrying `rec_texts`), so it detects the installed major and
speaks both. Forking it would have produced two files that drift.

## Running the GPU sidecar

Its own compose project, so the RAG stack's lifecycle and this one are
independent:

```bash
docker compose -f docker-compose.gpu.yml up -d --build
docker compose -f docker-compose.gpu.yml logs -f
docker compose -f docker-compose.gpu.yml down
```

It binds `127.0.0.1:8103` — the same port the CPU sidecar uses — so
`pdf.paddle_ocr.base_url` needs no change and the two are simply mutually
exclusive. Turn the lane on with `pdf.ocr_engine: paddle`.

Requires an NVIDIA runtime. Check first:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

`docker-compose.gpu.yml` reserves one card by host index. Edit `device_ids` for
your machine, or swap it for `count: 1` to let Docker choose.

## Why CUDA 11.8

Read this before "modernising" the pin.

The card this was built for is a GTX 1070 Ti — Pascal, compute capability 6.1.
Modern PyTorch wheels dropped Pascal, which is why this project's VLM OCR lane
runs through llama.cpp instead. PaddlePaddle is the exception: its **cu118**
build still ships `sm_61` kernels. That was measured before the pin was chosen
— `paddlepaddle-gpu 3.3.1` reported capability `(6, 1)` and ran a real matmul
on the card.

A wheel with no matching architecture still imports cleanly and still reports
the right device count. Only a real kernel launch tells you. So if you move to
cu126/cu129, re-run that check on the actual hardware rather than trusting that
it built.

## Contract

```
GET  /health -> {"ok": true, "engine": "paddleocr", "engine_importable": true,
                 "engine_import_error": null, "device": "gpu:0",
                 "paddleocr_version": "3.7.0", "loaded": ["en"], "langs": [...]}
             -> 503, same body, when the engine cannot import
POST /ocr    {"image_b64": "<base64 png>", "lang": "en"}
             -> {"text": "...", "lines": 12, "ms": 1830, "device": "gpu:0"}
```

`/health` answers the question "can this container OCR", not "did the HTTP
server start". It runs a real import in a subprocess and returns **503** when
that fails, so a container whose engine is broken is reported unhealthy instead
of accepting pages and failing every one of them.

`loaded` lists languages whose engine has actually been built — empty is the
normal state of a freshly started, healthy sidecar, since engines are built on
the first page.

`device` is reported on both endpoints so a GPU build that quietly fell back to
CPU is visible, rather than just mysteriously slow.

## Client

`src/ingestion/ocr_paddle.py` subclasses the VLM OCR driver and replaces only
the wire call, so page rendering, DPI, page selection and endpoint-down
handling cannot drift between the two lanes. A page that fails stays sparse;
one bad page never costs the whole ingest.
