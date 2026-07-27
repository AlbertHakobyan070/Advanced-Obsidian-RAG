# API reference

Two HTTP services, and **every capability of this system is reachable through
one of them**. Nothing requires the browser UI: the console's web app is itself
just a client of the management API.

| Service | Port | Module | Live contract |
|---|---|---|---|
| **Query API** | `:8051` | `serve_api.py` | `GET /schema` |
| **Management console API** | `:8052` | `manage_api.py` | `GET /api/schema` |

Both also expose FastAPI's own `/docs` (Swagger), `/redoc` and `/openapi.json`.

!!! tip "Read the schema, don't copy this page"
    Presets, branch limits, provider names and job kinds all come from *your*
    config. `GET /schema` and `GET /api/schema` return the live values. This
    page tells you what exists and why; the schema endpoints tell you what your
    install currently accepts.

!!! warning "There is no authentication"
    Both services bind to `127.0.0.1` and assume a single trusted operator.
    Permission tiers (below) are a **policy for callers to enforce**, not an
    access-control mechanism. Do not expose either port to a network.

---

## Query API — `:8051`

Thirteen endpoints. The pipeline is loaded once at startup and stays warm, so
no call pays index or model load cost.

### Ask and search

| Endpoint | Purpose |
|---|---|
| `POST /search` | **Retrieval only** — chunks, labels, scores, text. No LLM, no tokens, no generation backend required. This is the primary call for agents. |
| `POST /query` | Full RAG: retrieve, then a grounded answer with `[n]` citations and a confidence line. |
| `POST /compare` | Run one question down a bounded tree of named branches (presets, rerank methods, generation backends) and report how they differ. |
| `GET /compare/options` | **Start here for `/compare`.** Ready-to-post branch sets per dimension, the real branch caps, and the reason any backend is unavailable. |

Both `/search` and `/query` accept the same retrieval controls:

| Field | Effect |
|---|---|
| `q` | The question. Required. |
| `preset` | Apply a named bundle from `retrieval.presets`. |
| `auto_preset` | `false` suppresses implicit code-intent preset selection, for a config-only baseline. An explicit `preset` still wins. |
| `top_k` | How many reranked chunks reach the answer (1–50). |
| `dense_top_k` / `sparse_top_k` | Candidate-pool width per lane *before* fusion (1–200). The higher-leverage knobs. |
| `hyde` / `hype` | Toggle query expansion. |
| `omnisearch` | Add a live-vault lane (results are not stored index records). |
| `parent_context` / `neighbor_context` | Small-to-big expansion after reranking: swap a chunk for its full section, or append a PDF hit's adjacent pages. |
| `rerank` | Method for this call: `cross_encoder`, `http`, `lexical`, or `none`. |
| `rerank_instruction` | **Ranking criterion** in plain language (max 2000 chars). See [Rerank instructions](#rerank-instructions). |
| `include_text` | Characters of chunk text to return (0–6000). |
| `max_sources` | Cap the number of sources returned. |

`/query` additionally accepts `provider` and `model` (a **configured** backend
alias — endpoints and secrets can never be supplied per request),
`max_tokens` (64–8192), and `retrieve_only`.

!!! warning "`max_tokens` and citations"
    A very small `max_tokens` can truncate the citation footer and drop the
    answer's confidence to `UNKNOWN`. Leave a few hundred tokens of room.

### Inspect and discover

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{ready, state, uptime_s, error?}`. `state` is `ready`, `loading`, or `failed` — see [Readiness](#readiness-and-failure). |
| `GET /schema` | Machine-readable capability map: request fields, branch limits, the live preset registry, and the endpoint map. |
| `GET /config` | Live retrieval defaults, the preset registry, the active reranker, and generation provenance. |
| `POST /config` | Change retrieval defaults on the **warm** pipeline with no restart. `persist: true` also rewrites `config.yaml` and requires operator authorization. |
| `GET /providers` | Configured generation backends: protocol, endpoint, default model, the *name* of the environment variable holding each key, and readiness flags. Never returns a secret value. |
| `GET /chunks/{chunk_id}` | Fetch the current evidence record behind a stable id returned by `/search`, `/query` or `/compare`. |
| `GET /stats` | Corpus size with per-domain and per-file-type breakdown. |
| `GET /history` | The last `/search` and `/query` calls (newest first, in-memory): question, the knobs the caller set, the full retrieval echo, confidence, timing. |
| `GET /omnisearch` | Raw live-vault results, bypassing the index. |

### Evidence ids

Every source carries a **stable evidence id** so results can be dereferenced:

```json
{
  "sources": [
    {
      "id": "<stable-evidence-id>",
      "origin_id": "<indexed-chunk-id>",
      "lookup_available": true,
      "n": 1,
      "label": "<source label>",
      "cited": true
    }
  ],
  "retrieval": {"preset": "code", "rerank_top_k": 10, "hyde_used": false,
                "reranker_model": "<resolved-reranker>"},
  "generation": {"backend": "<provider-name>", "protocol": "<wire-protocol>",
                 "model": "<resolved-model>", "usage": {}}
}
```

- `lookup_available: true` → `GET /chunks/{id}` resolves it.
- Parent-expanded sections use `parent:<id>` and report the indexed child as
  `origin_id`, so overlap is computed from the text each branch actually received.
- Live Omnisearch excerpts use a content-derived `live:<hash>` id and set
  `lookup_available: false`, because they are not stored index records.

### Comparison trees

`POST /compare` takes a question, `mode: search|query`, and a bounded list of
named branches. Read the live `/schema` for current limits.

The response contains every branch plus `comparison`: common and branch-unique
source ids, per-branch ranks, rank spread, and pairwise overlap. It
**intentionally does not compare raw scores**, because cross-encoder logits,
lexical scores, HTTP reranker scores and fused RRF values are on different
scales and a shared axis would be meaningless.

Branches differing only by provider/model reuse one exact evidence set, which
makes a backend comparison about answer behaviour rather than retrieval noise. A
generation failure stays on its branch; a retrieval or reranker failure marks
that branch as a retrieval error without discarding successful siblings.

```bash
curl -s -X POST http://127.0.0.1:8051/compare \
  -H "Content-Type: application/json" \
  -d '{
    "q": "What does the retention policy say about backups?",
    "mode": "search",
    "branches": [
      {"id": "baseline", "label": "Config baseline", "auto_preset": false},
      {"id": "concept",  "preset": "concept"},
      {"id": "lexical",  "rerank": "lexical"}
    ]
  }'
```

### Comparison options

Building a valid `/compare` call used to mean reading `presets` from `/schema`,
`rerank_modes` from `/schema`, and the backends from `/providers`, then joining
the three by hand and re-deriving the branch caps from a sentence of English.
Every caller was reimplementing the console's sidebar.

`GET /compare/options` returns the joined result: per dimension, a list of
branch objects that can be posted to `/compare` unchanged.

```bash
curl -s http://127.0.0.1:8051/compare/options
```

```json
{
  "branch_limits": {"min": 2, "max": 6, "query": 3},
  "dimensions": {
    "preset":   {"mode": "search", "branches": [{"id": "baseline", "auto_preset": false}, ...]},
    "reranker": {"mode": "search", "branches": [{"id": "rerank_lexical", "rerank": "lexical"}, ...]},
    "provider": {"mode": "query",  "branches": [{"id": "provider_x", "provider": "x",
                                                 "available": false,
                                                 "unavailable_reason": "..."}]}
  }
}
```

Unavailable backends are **included and marked** rather than dropped, so a
caller can explain why one is missing instead of silently omitting it. The
`baseline` branch (config defaults, `auto_preset: false`) is first in the preset
dimension because a comparison without it has nothing to be measured against.

### Rerank instructions

A cross-encoder answers "how relevant is this passage to this query" — but
*relevant for what* is left implicit, and the model's answer comes from whatever
it was trained on. Two passages can be equally on-topic while only one is the
kind of thing you wanted: a worked procedure rather than a definition, a primary
source rather than a summary, code rather than prose about code.

`rerank_instruction` states that criterion explicitly and applies it **at
ranking time**:

```bash
curl -s -X POST http://127.0.0.1:8051/search \
  -H "Content-Type: application/json" \
  -d '{"q": "how do I rotate credentials",
       "rerank_instruction": "prefer worked procedures and runnable examples over definitions"}'
```

Two properties worth relying on:

- **It reweights, it never filters.** Retrieval already built the candidate pool
  against your real question. The instruction only changes the text the reranker
  scores against, so it reorders that pool and cannot remove anything from it.
- **It is routed per mode, and the echo tells you which happened.**

| Mode | Effect |
|---|---|
| `cross_encoder` | Joined to the question; the model reads the pair together. This is the lane it is for. |
| `http` | The same joined text goes to the external service — the only channel the `/v1/rerank` shape offers. |
| `lexical` | **Ignored by design.** Lexical scoring is query-term coverage; folding a sentence of instruction into the term set would dilute every real query term. |
| `none` | Nothing is scored at all. |

The retrieval echo reports `rerank_instruction` and `rerank_instruction_applied`,
so a no-op is visible rather than looking like a setting that took effect. Send
`""` to disable a configured instruction for one call.

Config: `retrieval.rerank_instruction` (blank = off) and
`retrieval.rerank_instruction_format` — `prefix` (the instruction, a newline,
then the question; works with any cross-encoder because it simply reframes the
scored text) or `instruct` (the labelled `<Instruct>/<Query>` form, only for a
reranker documented to expect it).

### Readiness and failure

`GET /health` distinguishes three states, because "not ready" used to conflate
two situations that call for opposite responses:

| `state` | Meaning | What to do |
|---|---|---|
| `ready` | Serving. | — |
| `loading` | Indexes and models are still loading. | Wait. |
| `failed` | The pipeline could **not** be built from the current config. | Act — waiting cannot fix it. `error` names the cause. |

A `failed` service still answers: every retrieval endpoint returns **503** with
the same reason attached, rather than refusing the connection. The usual causes
are a model id that cannot be downloaded, an embedding model with a typo, or an
index path that moved — normally the setting that was changed most recently.
`GET /api/service/log` on `:8052` returns the startup log.

!!! warning "Reranker failures are retrieval failures"
    A response beginning `Reranking failed:` never reached generation. Preserve
    the underlying model error and check `GET /config` for the active reranker.
    Do not relabel it as a generation-provider outage.

---

## Management console API — `:8052`

Thirty-eight endpoints covering the whole corpus lifecycle. `GET /api/schema`
returns them with a **permission tier** on every operation.

### Permission tiers

The local API has no authentication, so these tiers are a contract the **caller**
honours. Any agent toolkit driving this API should gate on them.

| Tier | Meaning |
|---|---|
| `read` | Safe. No confirmation needed. |
| `mutating` | Changes local state, or invokes a potentially billed external action. Confirm with the operator first. |
| `destructive` | Removes content. Always confirm, echo exactly what will be deleted, never run unprompted. |

### Inspect the corpus — `read`

| Endpoint | Purpose |
|---|---|
| `GET /api/schema` | This capability map, with tiers. |
| `GET /api/overview` | Corpus summary: chunk/doc counts, per-domain and per-JSONL breakdown, vector count, disk use, query-API health, recent jobs. |
| `GET /api/documents` | Search / filter indexed documents. |
| `GET /api/facets` | Available domains, group labels, tags, file types. |
| `GET /api/documents/preview` | Preview a document's indexed chunks. |
| `GET /api/vault/tree` | Browse the vault tree, with in-index status per file. |
| `GET /api/vault/search` | Find files by name in the vault. |
| `GET /api/browse` | Filesystem folder picker (for choosing paths the container can see). |
| `GET /api/settings` | The editable config surface, provider registry status, reranker suggestions, device capabilities, taxonomy. |
| `GET /api/vaults` | Registered vaults and their per-vault index settings. |
| `GET /api/ocr/status` | What OCR this install can actually do right now, per engine. |
| `GET /api/service/log` | Tail the query API's startup log. |
| `GET /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/log` | Job queue state and output. |
| `GET /api/inbox`, `GET /api/import/converted`, `GET /api/import/file` | Staged-file listings and raw fetch. |
| `GET /api/import/ocr_scan` | Which pages of a staged PDF carry no extractable text — run this before OCRing anything. |
| `POST /api/rerank/check` | Preflight a cross-encoder: reachable? already cached (and where)? within its context limit? Downloads nothing. |

### Change things — `mutating`

| Endpoint | Purpose |
|---|---|
| `POST /api/settings` | Persist whitelisted config values into `config.yaml` in place, comments preserved. Nothing hot-applies; the response says which service to restart. |
| `POST /api/providers/key` | Set or clear one provider key in `.env`. Only variable names the registry already declares are accepted. Never returns the value; rejects a declared credential-prefix mismatch. |
| `POST /api/service/restart` | Restart the warm query API. Returns the relaunch generation counter; fails loudly if the supervisor does not actually relaunch. |
| `POST /api/jobs`, `.../retry`, `.../cancel` | Queue, retry and cancel jobs. |
| `POST /api/upload`, `POST /api/ingest_inbox`, `POST /api/ingest_custom` | Stage files and run ingest passes. |
| `POST /api/import/fetch`, `/convert`, `/promote` | Pull a URL as markdown or printed PDF, convert any upload to markdown, promote a staged result into ingest. |
| `POST /api/documents/retag` | Metadata-only update (domain / group label / tags). **No re-embedding.** |
| `POST /api/ocr/warm` | Load a vision-OCR model now rather than stalling the first ingest page. May bill an external endpoint. |
| `POST /api/vaults/switch`, `POST /api/vaults/forget` | Swap the whole per-vault path set atomically; drop a registration. |

### Remove things — `destructive`

| Endpoint | Purpose |
|---|---|
| `POST /api/documents/delete` | Remove documents from the index. |
| `POST /api/inbox/delete` | Remove staged files. |

### Which service do I call?

- **Ask, search, compare, inspect evidence, or change warm retrieval defaults**
  → `:8051`.
- **Manage the corpus, persistent settings, jobs, OCR, or provider secrets**
  → `:8052`.

Pair the two schema maps rather than assuming a copied capability list.

---

## Worked example: an agent's first three calls

```bash
# 1. Is the query API up, and what does this install accept?
curl -s http://127.0.0.1:8051/health
curl -s http://127.0.0.1:8051/schema

# 2. Ground a question — retrieval only, no generation backend needed
curl -s -X POST http://127.0.0.1:8051/search \
  -H "Content-Type: application/json" \
  -d '{"q": "how do we rotate service credentials", "top_k": 6, "include_text": 1200}'

# 3. Pull the full text behind the best hit
curl -s "http://127.0.0.1:8051/chunks/<id-from-step-2>?include_text=4000"
```

Next: [Agent integration](agents.md) for the patterns that make this
token-efficient in a loop.
