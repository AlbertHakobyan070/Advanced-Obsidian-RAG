# RAG 2.0 — User Manual

Your Obsidian vault (~169K chunks: courses, textbooks, notebooks) behind a local
search-and-answer API. Ask a question → get a grounded, cited answer from YOUR
materials. This manual covers day-to-day use; internals live in WORKLOG.md and
the session handoffs.

**Every command below runs inside the project venv, from the project root:**

```cmd
%LOCALAPPDATA%\rag\venv\Scripts\activate.bat
cd /d "A:\DS_Vault\DS Main Vault\rag_project"
```

---

## 1. The three servers — you almost never need all of them

| Port | What it is | Start it when… | RAM/VRAM |
|---|---|---|---|
| **:8051** | Query API — answers questions, searches chunks | you (or the agent/agents) want to ASK things | ~2.5–3 GB RAM |
| **:8052** | Management console — browse/ingest/delete docs in a browser | you're ADDING or REMOVING documents | ~0.3 GB (+ ingest jobs it spawns) |
| **:8100** | DeepSeek-OCR vision model — reads scanned pages | an ingest pass needs `--ocr-engine vlm` | ~1 GB RAM + 3.5 GB VRAM (GPU1) |

**Practical combos on 16 GB RAM:**
- Just asking questions → **:8051 only** (+ FreeLLMAPI container for generated answers).
- Corpus housekeeping day → **:8052 only** (queries can wait; jobs spawn their own workers).
- OCR-ingesting a scanned book → **:8100 + the ingest command**, nothing else.
- Avoid: running `main.py eval` while :8051 is up (two full pipelines = ~5–6 GB),
  or heavy ingest while FreeLLMAPI + browser + Obsidian are all open. When in
  doubt, close what you're not using — every server restarts warm in <1 min.

### Start commands

```cmd
:: :8051 — query API
python -m uvicorn serve_api:app --host 127.0.0.1 --port 8051

:: :8052 — management console (then open http://127.0.0.1:8052 in a browser)
python -m uvicorn manage_api:app --host 127.0.0.1 --port 8052

:: :8100 — OCR model (one line; every flag is load-bearing, don't trim it)
A:\Llamacpp\llama-b9860-bin-win-vulkan-x64\llama-server.exe -m A:\Llamacpp\models\deepseek-ocr\DeepSeek-OCR-Q8_0.gguf --mmproj A:\Llamacpp\models\deepseek-ocr\mmproj-DeepSeek-OCR-Q8_0.gguf -ngl 999 -dev Vulkan1 -c 8192 -np 1 --host 127.0.0.1 --port 8100 --jinja --chat-template-file A:\Llamacpp\deepseek-ocr-passthrough.jinja --flash-attn off
```

Stop any of them with **Ctrl+C** in its window. Each runs in its own terminal.

---

## 2. Asking questions (:8051)

Full answer with citations (needs FreeLLMAPI on :3001):

```cmd
curl.exe -s -X POST http://127.0.0.1:8051/query -H "Content-Type: application/json" -d "{\"q\": \"Where did I use conjugate priors in my coursework?\"}"
```

Chunks only, **no LLM needed** — works even when FreeLLMAPI is down:

```cmd
curl.exe -s -X POST http://127.0.0.1:8051/search -H "Content-Type: application/json" -d "{\"q\": \"ARIMA stationarity\", \"include_text\": 500}"
```

Optional knobs (add to either request):

| Knob | Example | Effect |
|---|---|---|
| `preset` | `"preset": "code"` | `code` = find MY scripts/notebooks · `concept` = tight definition lookup · `synthesis` = wide cross-course pull |
| `top_k` | `"top_k": 10` | how many chunks reach the answer (default 7) |
| `hyde` | `"hyde": false` | skip the query-expansion LLM call (faster, needs no LLM) |
| `hype` | `"hype": true` | match the query against pre-generated hypothetical questions (needs a `build_hype.py` run; fails soft) |
| `max_tokens` | `"max_tokens": 800` | `/query` only: cap the ANSWER's length (default 2500). Small = quick fact, large = full synthesis |
| `retrieve_only` | `"retrieve_only": true` | `/query` behaves like `/search` |
| `include_text` | `"include_text": 800` | attach N chars of each source chunk |

In the **console Query tab**, these are checkboxes/fields (HyDE, HyPE, top_k,
max tokens, preset, Omnisearch) — plus a **light/dark theme toggle** in the
header (top-right; your choice is remembered).

You usually don't need any of them — code-looking questions auto-apply the
`code` preset, and naming a domain ("…in my statistics lectures") automatically
scopes retrieval. Other endpoints: `GET /stats` (corpus overview),
`GET /schema` (full API self-description), `GET /health`.

The Streamlit UI (`rag web`, port 8501) still works as before.

---

## 3. Adding documents (console, easiest)

1. Start :8052, open **http://127.0.0.1:8052**.
2. **Ingest tab** → drag-drop the PDF → it lands in the vault inbox
   (`Other/Inbox`) → click **Ingest inbox now**.
3. Watch the job log on the **Jobs tab**. When it's done: **restart :8051**.

What the button does (server-side, `POST /api/ingest_inbox`): refuses an
empty inbox; warns if a filename already looks indexed (force to override);
queues ingest → append with a per-batch `data\inbox_<timestamp>_chunks.jsonl`;
archives each processed PDF to `Inbox\_ingested\` so the next batch never
re-processes or clobbers an earlier one. A run that ingests **zero** documents
now exits non-zero — a "done" ingest job means chunks actually landed.

Or by CLI, for a file already in the vault (example — a new tech book):

```cmd
python main.py ingest-pdfs --include-path "SomeBookName" --output data\somebook_chunks.jsonl
python main.py index --append data\somebook_chunks.jsonl
```

(`index --append` rebuilds the sparse index by itself — no separate rebuild
needed. Then restart :8051.)

**Only part of a book?** Trim front-matter/index/back-matter with a 1-based page
subset (ranges + singletons). Takes precedence over `--max-pages`; kept pages
keep stable doc_ids, so re-ingesting with a wider range only ADDS pages:

```cmd
python main.py ingest-pdfs --include-path "Wackerly" --pages "20-810" --output data\wack_chunks.jsonl
```

In the console: the Custom-job **Pages (subset)** field does the same.

**Chunking strategy (session 11).** How OVERSIZED sections get split is now a
per-run choice: `heading` (paragraph packing that respects the text's
structure — the long-standing default, and the A/B winner on structured PDFs)
or `fixed` (strict sliding window with even sizes — pick it for OCR'd scans
and wall-of-text sources whose paragraph structure is noise). Small sections
chunk identically either way, so doc_ids only change where it matters:

```cmd
python main.py ingest-pdfs --include-path "ScannedBook" --chunking fixed --output data\scan_chunks.jsonl
```

Console: the **Chunking** select in the Custom job form, or the `chunking:`
select next to the inbox button. Re-chunking an already-indexed document is a
swap: delete it from the index first, then re-ingest (new text → new doc_ids).

**Code files** (`.js/.ts/.sql/.go/.java/.c/.cpp/.rs/.sh/…` — every language the
notebook loader doesn't handle; `.py/.R/.Rmd/.ipynb` stay on `ingest-notebooks`):

```cmd
python main.py ingest-code --include-path "Capstone" --output data\code_chunks.jsonl
python main.py index --append data\code_chunks.jsonl
```

It auto-skips `node_modules`/`venv`/`dist`/`_Backups`/minified files, and skips
your agent-project roots (Workspace1, Workspace2, …) unless `--include-path` names one.
Console: the **ingest-code** job kind.

Notes that keep biting: `--include-path` matches any part of the vault path;
"Tech Books" counts as a Books folder for `--only-books`/`--skip-books`.
**Never combine `--only-books` with an inbox/include ingest** — the inbox is
not a book folder, so `--only-books` silently filtered all 9 uploads to zero
on 2026-07-03 (that's what the non-zero-exit + filter-attribution summary now
catches).

### Failed jobs: retry, not redo

Every console job is idempotent (ingest archives processed PDFs; append
upserts deterministic doc_ids; rebuilds are derived from the JSONLs), so the
**retry** button on a failed job re-queues the same params and never
duplicates committed work. Typical case: `index --append` OOMs during the
sparse rebuild — its dense half is already committed; retry (or run
`rebuild_bm25.py`, which loads no embedding model) finishes the job.

### Tags, domains, and the Vault tab

- **At upload:** set optional batch domain + tags in the Ingest tab before
  clicking Ingest inbox now — stamped on every chunk (inbox files have no
  course path, so they'd land `general` otherwise).
- **Later:** Documents tab → *Retag selected…* (domain, course, and tags —
  course fixes coursework that lost its course on ingest), or the **Vault
  tab**: browse the tree from `00 – AUA_DS` (✓ = indexed, with chunk counts),
  stage domain/tag edits, Apply (one BM25 rebuild); the search box below
  covers the whole vault, including books outside the tree root and archived
  `Inbox\_ingested` files. Metadata-only — doc_ids/embeddings never change.
- Tags matter at query time: a tag word appearing in the query boosts tagged
  documents (same 1.15× family as the course/domain boost).

### HyPE (optional question index)

`python build_hype.py --include-path "SomeBook" --dry-run` then without
`--dry-run` — ONE LLM call per chunk, so scope it (guard refuses >2000 chunks
unless you raise `--max-chunks`). Resumable; questions live in the separate
`hype_questions` Chroma collection. Query with `{"hype": true}` per call or
`retrieval.hype.enabled: true`. Also available as console job `build_hype`.

### What happens to images in a PDF

Short version: **figures are pulled out to disk and *linked* to the chunk on
their page — they are never embedded and never searched.** Retrieval is
text-only (the embedding model reads text, not pictures). During ingest
(`pdf.extract_images: true`):

- Every page's raster images are enumerated; an image is **kept only if it
  covers ≥ 8 % of the page** (`pdf.image_min_frac`) — logos, icons, bullets, and
  rules are dropped. Repeated images (e.g. a header logo) are saved once.
- Kept images are written to `data\figures\<book>\pNNNN_xXREF.ext`. The chunk
  text itself is extracted with images **off** (`write_images=False`) — no image
  bytes enter the index.
- The chunk on that page gets metadata `has_figure: true`, `figure_count`, and
  `figure_images` (a `;`-joined path string). So a retrieved chunk can say "a
  diagram lives here → this file," and the Streamlit UI renders it beside the
  text. The picture rode along on its page's words; it was never a search target.
- **`--no-images`** (checked by default in the console, and always on for the
  inbox lane) **skips figure extraction** — faster, no `data\figures` writes.
  Uncheck it only when you want the figures-beside-chunks affordance.

A *scanned* page (no extractable text at all) is a different case — that's OCR
(§5), not figure extraction. Figure extraction is about raster images inside an
otherwise-text PDF.

## 4. Deleting documents

Console → **Documents tab** → search → select → Delete. It removes the chunks
from the index AND the JSONLs (vault files are never touched), then queues the
sparse rebuild automatically. When the job finishes: **restart :8051**.

## 5. OCR-ingesting a scanned book

A page counts as **"scanned"** when it has under 50 chars of extractable text
(`pdf.skip_scanned_threshold`). What happens to those pages is set by the OCR
engine — chosen with `pdf.ocr_engine`, the CLI `--ocr-engine`, or the **Corpus
Ledger's OCR-engine dropdown** in the Custom-job form:

| engine | what runs |
|---|---|
| `auto` (default) | Tesseract on the pages MuPDF flags — no GPU, safe |
| `tesseract` | pins the Tesseract path |
| `vlm` | the vision model re-parses each scanned page → clean markdown + real LaTeX/tables |
| `none` | scanned pages stay sparse (text-only pass) |

Tesseract (`auto`) handles ordinary scans fine. For **math-heavy scans** where
you want real LaTeX/tables, use the vision model:

1. **Start the OCR server:** `rag ocr` (launches DeepSeek-OCR on GPU1, port
   8100 — the exact llama-server line lives in `rag.bat` and config.yaml
   `pdf.vlm_ocr`). It holds ~3.5 GB VRAM and blocks its window; leave it open
   only for the pass. Check it: `curl.exe http://127.0.0.1:8100/v1/models`.
2. **Run the pass** — either:
   - **Corpus Ledger (:8052) → Ingest tab → "OCR pass (VLM)…"** prefills the
     Custom-job form with the `vlm` engine; set an Include path to the ONE
     scanned book and an Output JSONL, then queue. (Or just set the OCR-engine
     dropdown to `vlm` in a normal Custom ingest.) Watch it on the Jobs tab.
   - **CLI**, e.g. the Bertsekas book:
     ```cmd
     python main.py ingest-pdfs --only-books --include-path "Optimal Control -- Dimitri" --ocr-engine vlm --output data\ocr_bertsekas_vlm_chunks.jsonl
     ```
3. Skim the output JSONL, `index --append` it, restart :8051. ~12 s/page ≈ 3 h
   for a 389-page book (a GPU job). If the OCR server dies mid-run the pass keeps
   going — those pages just stay sparse (rerun later). **Stop :8100 when done**
   (frees the GPU + VRAM). `dpi: 150` / `max_edge_px: 1200` in `pdf.vlm_ocr` are
   tuned to the 8 GB card — raising either OOMs the encoder.

**Do not** re-index a book that's already in the index without deleting the old
version first (console → Documents → delete), or you'll have both copies. To
*replace* a Tesseract-era OCR file with a fresh VLM pass, use the corpus-swap
playbook (`RAG_2.0\CLAUDE_CODE_HANDOFF_SESSION7.md §6`) — the text changes, so
doc_ids change.

## 6. Eval (health check after changes)

```cmd
python main.py eval --retrieval-only     :: minutes, offline — the default check
python main.py eval                      :: full mode, needs FreeLLMAPI up
```

The suite (`eval/golden_queries.yaml`) is 94 questions as of session 11: the
original 30 conceptual seeds plus exam-like questions mined from your real
midterms/PSS/quizzes across courses, plus own-code problem situations.
`--retrieval-only` skips generation — keyword recall is measured over the
retrieved chunks, so it runs offline in minutes and is the right regression
check after index changes. Fresh baselines live in `eval/*.json` next to the
historical ones. For an apples-to-apples FULL-mode comparison, pin
`generation.model` in config.yaml first, and revert to `"auto"` after.
Don't run the full mode while :8051 is up if RAM is tight.

---

## 7. Rules of thumb / troubleshooting

| Symptom / situation | Do this |
|---|---|
| Added, deleted, or re-ingested anything | **Restart :8051** — it loads indexes once and stays warm |
| `/query` says "Generation backend unreachable" | FreeLLMAPI (:3001) is down. Start it, or use `/search` / `retrieve_only` meanwhile |
| Queries suddenly slow / machine thrashing | Too much loaded at once — close servers you're not using (see §1 combos) |
| OCR requests return 400 "media markers" | :8100 was started without `--jinja --chat-template-file …` — use the full launch line |
| OCR server crashes mid-batch | It was started without `-np 1`, or a request slipped past the size cap — full line, and keep `pdf.vlm_ocr.max_edge_px: 1200` |
| Everything on G: lost/corrupted | Indexes are DERIVED — rebuild: `python main.py index` (full re-embed, ask first — hours) + `python rebuild_bm25.py`. JSONLs on A:\ are the source of truth |
| Console job fails with weird encoding error | Should be fixed (UTF-8 forced); if you run main.py yourself with output redirected, `set PYTHONIOENCODING=utf-8` first |

**Where things live:** project + JSONLs + git → `A:\...\rag_project` · hot
indexes → `G:\rag_data\` (`data\chroma_db` is a junction pointing there) ·
OCR model + llama.cpp → `A:\Llamacpp\` · secrets → `.env` (never committed).
