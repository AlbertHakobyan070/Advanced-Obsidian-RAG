# Advanced Obsidian RAG — User Manual

A practical cookbook for day-to-day use. The repo serves a personal knowledge
base of markdown notes, textbooks, lecture PDFs, and notebooks behind a local
search-and-answer API. Ask a question, get a grounded, cited answer from your
materials.

**All commands below run from the project root, inside your Python venv.**

---

## 1. The three servers — you almost never need all of them

| Port | What it is | Start it when… | RAM/VRAM |
|---|---|---|---|
| **:8051** | Query API — answers questions, searches chunks | you (or a tool/agent) want to ASK things | ~2.5-3 GB RAM |
| **:8052** | Management console — browse/ingest/delete docs in a browser | you're ADDING or REMOVING documents | ~0.3 GB (+ ingest jobs it spawns) |
| **:8100** *(optional)* | Vision-model OCR server — reads scanned pages | an ingest pass needs `--ocr-engine vlm` | ~1 GB RAM + GPU VRAM |

**Practical combos on a 16 GB RAM box:**
- Just asking questions -> **:8051 only** (+ your generation endpoint for
  generated answers).
- Corpus housekeeping day -> **:8052 only** (queries can wait; jobs spawn
  their own workers).
- OCR-ingesting a scanned book -> **:8100 + the ingest command**, nothing
  else.
- Avoid: running `main.py eval` while :8051 is up (two full pipelines =
  ~5-6 GB). When in doubt, close what you're not using — every server
  restarts warm in <1 min.

### Start commands

```bash
# :8051 - query API
python -m uvicorn serve_api:app --host 127.0.0.1 --port 8051

# :8052 - management console (then open http://127.0.0.1:8052 in a browser)
python -m uvicorn manage_api:app --host 127.0.0.1 --port 8052

# :8100 - vision-model OCR (start only when running a vlm ingest pass)
# Configure the launch line in config.yaml under pdf.vlm_ocr; the
# `unlimited-ocr` reference recipe is documented inline.
```

Stop any of them with **Ctrl+C** in its window. Each runs in its own
terminal.

---

## 2. Asking questions (:8051)

Full answer with citations (needs your generation endpoint up):

```bash
curl.exe -s -X POST http://127.0.0.1:8051/query -H "Content-Type: application/json" \
  -d "{\"q\": \"Where do my notes discuss conjugate priors?\"}"
```

Chunks only, **no LLM needed** — works even when the generation endpoint
is down:

```bash
curl.exe -s -X POST http://127.0.0.1:8051/search -H "Content-Type: application/json" \
  -d "{\"q\": \"ARIMA stationarity\", \"include_text\": 500}"
```

Optional knobs (add to either request):

| Knob | Example | Effect |
|---|---|---|
| `preset` | `"preset": "code"` | `code` = find YOUR scripts/notebooks · `concept` = tight definition lookup · `synthesis` = wide cross-material pull |
| `top_k` | `"top_k": 10` | how many chunks reach the answer (default 7) |
| `hyde` | `"hyde": false` | skip the query-expansion LLM call (faster, needs no LLM) |
| `hype` | `"hype": true` | match the query against pre-generated hypothetical questions (needs a `build_hype.py` run; fails soft) |
| `max_tokens` | `"max_tokens": 800` | `/query` only: cap the ANSWER's length (default 2500). Small = quick fact, large = full synthesis |
| `retrieve_only` | `"retrieve_only": true` | `/query` behaves like `/search` |
| `include_text` | `"include_text": 800` | attach N chars of each source chunk |

In the **console Query tab**, these are checkboxes/fields (HyDE, HyPE,
top_k, max tokens, preset, Omnisearch) — plus a light/dark theme toggle in
the header (top-right; your choice is remembered).

You usually don't need any of them — code-looking questions auto-apply
the `code` preset, and naming a domain ("…in my statistics lectures")
automatically scopes retrieval. Other endpoints: `GET /stats` (corpus
overview), `GET /schema` (full API self-description), `GET /health`.

The Streamlit UI (`python main.py serve`, port 8501) is also available if
you prefer a visual interface over curl.

---

## 3. Adding documents (console, easiest)

1. Start :8052, open **http://127.0.0.1:8052**.
2. **Ingest tab** -> drag-drop the PDF -> it lands in the vault inbox
   -> click **Ingest inbox now**.
3. Watch the job log on the **Jobs tab**. When it's done: **restart :8051**.

What the button does (server-side, `POST /api/ingest_inbox`): refuses an
empty inbox; warns if a filename already looks indexed (force to
override); queues ingest -> append with a per-batch
`data/inbox_<timestamp>_chunks.jsonl`; archives each processed PDF to
`Inbox/_ingested/` so the next batch never re-processes or clobbers an
earlier one. A run that ingests **zero** documents is treated as a no-op
success.

Or by CLI, for a file already in the vault (example — a new tech book):

```bash
python main.py ingest-pdfs --include-path "MyBook" --pages "20-810" --output data/mybook_chunks.jsonl
python main.py index --append data/mybook_chunks.jsonl
# then: restart :8051
```

Or scripts:

```bash
python main.py ingest-code --include-path "MyProject" --output data/code_chunks.jsonl
python main.py index --append data/code_chunks.jsonl
# then: restart :8051
```

## 4. Daily workflow

| Task | Use |
|---|---|
| Ask a one-shot question | `python main.py query "..."` |
| Interactive REPL | `python main.py chat` |
| Ask from a script | `curl.exe ... :8051/query` or `python main.py query` |
| Ask an agent (HTTP) | the same `:8051/query` endpoint |
| Retag a mis-tagged domain | Console -> **Documents** tab -> filter -> Retag selected |
| Browse what's in the index | Console -> **Vault** tab (filename, not chunk) |
| Delete a document | Console -> **Documents** tab -> select rows -> Delete |
| Track what's running | Console -> **Jobs** tab |
| Re-run the regression check | `python main.py eval --retrieval-only` |

## 5. OCR (scanned PDFs)

OCR is a per-PDF flag in the ingest command. Options:

| Engine | Behaviour |
|---|---|
| `auto` *(default)* | Tesseract-first probe; the historical default |
| `tesseract` | pin the classic path |
| `vlm` | the vision model re-parses each scanned page -> clean markdown + real LaTeX/tables |
| `none` | scanned pages stay sparse (text-only pass) |

Tesseract (`auto`) handles ordinary scans fine. For **math-heavy scans**
where you want real LaTeX/tables, use the vision model:

1. **Start the OCR server** on port 8100. Configure the launch command in
   `config.yaml` under `pdf.vlm_ocr`; the reference recipe is documented
   inline (a vision model served over an OpenAI-compatible `/v1`
   endpoint). It holds ~3.5 GB VRAM and blocks its window; leave it open
   only for the pass. Check it: `curl.exe http://127.0.0.1:8100/v1/models`.
2. **Run the pass** — either:
   - **Corpus Ledger (:8052) -> Ingest tab -> "OCR pass (VLM)…"** prefills
     the Custom-job form with the `vlm` engine; set an Include path to
     the ONE scanned book and an Output JSONL, then queue. Watch it on
     the Jobs tab.
   - **CLI**, e.g.:
     ```bash
     python main.py ingest-pdfs --only-books --include-path "MyScannedBook" \
       --ocr-engine vlm --output data/ocr_book_vlm_chunks.jsonl
     ```
3. Skim the output JSONL, `index --append` it, restart :8051. ~12 s/page
   on a typical book. If the OCR server dies mid-run the pass keeps
   going — those pages just stay sparse (rerun later). **Stop :8100
   when done** (frees the GPU + VRAM). `dpi: 200` / `dpi: 300` in
   `pdf.vlm_ocr` are tuned for ordinary and math-heavy scans
   respectively.

**Do not** re-index a book that's already in the index without deleting
the old version first (console -> Documents -> delete), or you'll have
both copies. To *replace* an older OCR file with a fresh pass, use the
corpus-swap playbook (see [Operations](docs/operations.md)) — the text
changes, so doc_ids change.

## 6. Eval (health check after changes)

```bash
python main.py eval --retrieval-only     # minutes, offline - the default check
python main.py eval                      # full mode, needs generation endpoint up
```

The shipped suite (`eval/golden_queries.yaml`) is a small illustrative
example (six queries across three categories). `--retrieval-only` skips
generation — keyword recall is measured over the retrieved chunks, so it
runs offline in minutes and is the right regression check after index
changes. For an apples-to-apples full-mode comparison, pin
`generation.model` in `config.yaml` first, and revert to `"auto"` after.
Don't run the full mode while :8051 is up if RAM is tight.

For real regression work, write your own golden set (30-100 queries
drawn from your actual corpus) — see [Evaluation](docs/evaluation.md)
for the schema.

---

## 7. Rules of thumb / troubleshooting

| Symptom / situation | Do this |
|---|---|
| Added, deleted, or re-ingested anything | **Restart :8051** — it loads indexes once and stays warm |
| `/query` says "Generation backend unreachable" | Your generation endpoint is down. Start it, or use `/search` / `retrieve_only` meanwhile |
| Queries suddenly slow / machine thrashing | Too much loaded at once — close servers you're not using (see §1 combos) |
| OCR requests return a chat-template error | The OCR server was started without its required template flag — use the full launch line from `config.yaml` |
| OCR server crashes mid-batch | It was started without a connection cap, or a request slipped past the size cap — full line, and keep `pdf.vlm_ocr.max_tokens` bounded |
| Indexes are corrupted / lost | Indexes are *derived* — rebuild: `python main.py index` (full re-embed, ask first — hours) + `python rebuild_bm25.py`. Your JSONLs are the source of truth; restore those first. |
| Console job fails with weird encoding error | Set `PYTHONIOENCODING=utf-8` in the environment before running `main.py` directly. |

**Where things live:** project + JSONLs + git -> the project root · hot
indexes -> wherever `config.yaml` points `paths.chroma_dir` /
`paths.bm25_index` · generation endpoint + model files -> wherever your
local server points · secrets -> `.env` (never committed).
