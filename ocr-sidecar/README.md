# OCR sidecar

PaddleOCR behind a two-endpoint HTTP contract, in its own container. This is
the middle OCR lane: better than Tesseract on skewed, multi-column and
non-Latin scans, and far cheaper than asking a vision-language model to parse
the page.

It is a sidecar rather than an import so its dependency tree never enters the
RAG image, and so you can stop it and get the memory back when you are not
ingesting.

## Two engines — `ocr` and `vl`

"PaddleOCR" names two different products, and choosing between them is the main
quality lever here. Both are served from the same container; pick one per
request with `"pipeline"`, or set the container default with
`PADDLE_OCR_PIPELINE`.

| | `ocr` | `vl` |
|---|---|---|
| models | `PP-OCRv6_medium_det` + `_rec` | `PP-DocLayoutV3` + `PaddleOCR-VL-1.6-0.9B` |
| how | detect boxes, recognise each | layout detection, then a document VLM reads each block |
| output | plain text, one line per detected line | **markdown**: reflowed paragraphs, `#` headings, `$…$` LaTeX, HTML tables |
| needs | nothing extra | `paddlex[ocr]` extras, GPU image only |

`ocr` is the default. `vl` is worth choosing when the page's STRUCTURE is the
information — see the measurements below before switching anything global.

### Measured, on this box

Two GTX 1070 Ti (Pascal, 8GB). Seven dense pages from Goodfellow's *Deep
Learning*, Prince's *Understanding Deep Learning*, and Jurafsky & Martin,
rendered at the lane's own 200 dpi / 2000px cap and scored against **each PDF's
own text layer** — an objective reference rather than a human reading two
transcripts.

The metric is **word recall**, not character accuracy, because this OCR feeds a
retrieval index: a page whose words are all present but whose punctuation is
mangled retrieves fine, and one that drops a third of its terms does not.

| | `ocr` | `vl` |
|---|---|---|
| mean word recall | **0.964** | 0.933 |
| mean seconds/page | **3.06** | 91.2 |
| engine build (first page) | 6.6 s | 184.6 s |
| peak VRAM | ~1.1 GB | **~7.9 GB** |

Both ran their own pipeline defaults, which is the honest comparison — what you
actually get from each. (`ocr` therefore paid for document-orientation and
UVDoc unwarping that `vl` skips, and `vl` paid for a layout-detection pass that
`ocr` has no equivalent of.)

**`vl` costs ~30x the time, 7x the VRAM, and transcribes prose slightly WORSE.**
On a 400-page scanned book that is the difference between ~20 minutes and
~10 hours. A faster card changes the ratio, not the shape of the trade.

### Where `vl` does win

The aggregate hides the one case that matters. On the Jurafsky page carrying
the minimum-edit-distance dynamic-programming matrix, `vl` scored its *highest*
recall of the run (0.973 against `ocr`'s 0.917) because it reconstructed the
10x11 table as real HTML with its row and column headers intact. `ocr` returns
those same numbers as an undifferentiated stream of digits — every value
present, every relationship gone.

The same holds for formulas: `ocr` shatters a displayed equation into the
fragments it was printed as, while `vl` emits `$p(\boldsymbol{x}, y)$` inline
and keeps the paragraph around it whole. It also drops running headers and
footers, which `ocr` faithfully repeats into every page's chunk.

So: **`ocr` for scanned prose, `vl` for pages where tables and formulas ARE the
content.** That is a per-book decision, which is why it is a per-request field.

### Two caveats before pointing an ingest at `vl`

- On figure pages it emits `<div style="text-align: center;">` wrappers and
  `<img src="imgs/…">` references to crop files **the sidecar never writes** —
  about 14% of one such page here. Dead references and layout noise in a chunk.
  Text and table pages are clean.
- ~7.9 GB of VRAM means it effectively owns an 8 GB card. Serving both engines
  at once from one container wants more headroom than that — and engines are
  cached for the **life of the container**, so a single exploratory `vl`
  request pins that memory until you restart the sidecar. On an 8 GB card,
  comparing the two engines and then continuing to ingest with `ocr` leaves the
  VL weights resident the whole time. Restart it when you are done comparing:

  ```bash
  docker compose -f docker-compose.gpu.yml restart
  ```

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

Pick the card with `OCR_GPU_INDEX` (host index, default `1`; use `0` on a
single-GPU machine):

```bash
OCR_GPU_INDEX=0 docker compose -f docker-compose.gpu.yml up -d
```

It sets `CUDA_VISIBLE_DEVICES`, **not** compose's `device_ids`, and that is
deliberate. With `device_ids: ["1"]` and nothing else, `nvidia-smi` *inside* the
container still listed both cards and Paddle put its weights on host card 0 —
the one driving the display, the exact opposite of the intent. Docker
Desktop/WSL2 ignores per-device reservations and exposes every GPU;
`CUDA_VISIBLE_DEVICES` is read by the driver in-process and works on both. The
reservation is therefore a plain `count: 1`, so it stops documenting a selection
that was not happening.

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
                 "paddleocr_version": "3.7.0", "loaded": ["ocr:en"],
                 "pipeline": "ocr", "langs": [...],
                 "pipelines": {"ocr": {"available": true, "reason": null,
                                       "models": ["PP-OCRv6_medium_det", ...]},
                               "vl":  {"available": true, "reason": null,
                                       "models": ["PaddleOCR-VL-1.6-0.9B", ...]}}}
             -> 503, same body, when the engine cannot import
POST /ocr    {"image_b64": "<base64 png>", "lang": "en", "pipeline": "ocr"}
             -> {"text": "...", "lines": 12, "ms": 1830, "device": "gpu:0",
                 "pipeline": "ocr"}
             -> 501 when this image cannot serve the requested pipeline
```

`/health` answers the question "can this container OCR", not "did the HTTP
server start". It runs a real import in a subprocess and returns **503** when
that fails, so a container whose engine is broken is reported unhealthy instead
of accepting pages and failing every one of them.

`loaded` lists the engines actually built, as `pipeline:lang` — empty is the
normal state of a freshly started, healthy sidecar, since engines are built on
the first page.

`device` is reported on both endpoints so a GPU build that quietly fell back to
CPU is visible, rather than just mysteriously slow.

`pipelines[*].models` names the MODELS behind each engine, because "PaddleOCR"
is a moving target: the same `PaddleOCR()` call resolves to PP-OCRv4 on the 2.x
CPU image and PP-OCRv6_medium on the 3.x GPU one, and a version bump can change
it again with no other visible symptom. `available: false` always carries a
`reason` — a 2.x image has no `PaddleOCRVL` at all, while a 3.x image built
without the `paddlex[ocr]` extras does; those want different fixes.

A request for a pipeline this image cannot serve gets **501**, never a silent
substitution. Returning PP-OCRv6 text to a caller who asked for `vl` would
relabel plain lines as markdown, and the only symptom would be a corpus that
mysteriously has no LaTeX in it.

## Client

`src/ingestion/ocr_paddle.py` subclasses the VLM OCR driver and replaces only
the wire call, so page rendering, DPI, page selection and endpoint-down
handling cannot drift between the two lanes. A page that fails stays sparse;
one bad page never costs the whole ingest.
