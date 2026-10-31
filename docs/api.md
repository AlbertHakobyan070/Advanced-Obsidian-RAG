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

## Security

Both services bind to `127.0.0.1` and assume a single trusted operator. Every
request passes three checks, in this order:

1. **Host, always on.** The `Host` header must name a loopback address
   (`localhost`, `127.0.0.1`, `::1`) or a name listed in
   `security.allowed_hosts`; anything else gets `400`. This is what stops DNS
   rebinding, which CORS does not.
2. **Origin, always on, for `POST`, `PUT`, `PATCH` and `DELETE`.** A request
   that carries an `Origin` passes only from the console's own origin
   (`http://localhost:<webui.port>` or `http://127.0.0.1:<webui.port>`), the
   service's own, or one listed in `security.allowed_origins`; anything else
   gets `403`. A request with no `Origin` (curl, scripts, agents) passes.
3. **Token, only when `RAG_API_TOKEN` is set** (in `.env`; blank means unset).
   Every request then needs `Authorization: Bearer <token>`, else `401`.
   Exempt: `GET /health`, CORS preflights (`OPTIONS`) and the console's page
   shell `GET /`. `/docs` and `/openapi.json` need the token like any other
   route.

Every refusal is JSON naming the setting to change, with `guard` set to
`host`, `origin` or `token`.

Reaching a service through a LAN address or a Kubernetes Service name needs
that name in `security.allowed_hosts`. `kubectl port-forward` arrives as
`localhost` and needs nothing. The console asks for the token once when a
request returns `401`, and keeps it in the browser's local storage. Do not
expose either port to a network: the permission tiers (below) are still a
**policy for callers to enforce**, and the token says who may call, not what
they may do.

**A known limit of the console's token.** The console keeps the token in the
browser's local storage, and it renders note content, imported pages and
generated answers as Markdown without sanitising the HTML, so a crafted note
could run script in the page and read the token. Use a token with a vault you
trust. The MCP server and other non-browser callers keep the token in their
own environment and are unaffected.

---

## Query API — `:8051`

The pipeline is loaded once at startup and stays warm, so no call pays index or
model load cost. `GET /schema` returns the live endpoint map — treat that, not
this page, as the authoritative list for a running install.

### Ask and search

| Endpoint | Purpose |
|---|---|
| `POST /search` | **Retrieval only** — chunks, labels, scores, text. No LLM, no tokens, no generation backend required. This is the primary call for agents. |
| `POST /query` | Full RAG: retrieve, then a grounded answer with `[n]` citations and a confidence line. |
| `POST /compare` | Run one question down a bounded tree of named branches (presets, rerank methods, generation backends) and report how they differ. |
| `GET /compare/options` | **Start here for `/compare`.** Ready-to-post branch sets per dimension, the real branch caps, and the reason any backend is unavailable. |
| `POST /graph/expand` | **Graph mode** — walk the hand-drawn edges of the graph lane. No LLM. **Experimental, off by default.** See [Graph mode](#graph-mode). |
| `POST /answer` | Generate over a **caller-supplied** document set. No retrieval runs. See [Graph mode](#graph-mode). |
| `POST /evidence_pack` | **Evidence pack**: one search per sub-question, plus an optional graph walk, merged into one evidence list with provenance, coverage and gaps. Writes no answer. See [Evidence pack](#evidence-pack). |

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
| `rerank` | Method for this call: `cross_encoder`, `http`, `lexical`, `none`, or `laya` — **experimental**: refused unless `retrieval.laya.enabled`, never swapped for another method. See [Laya reranker](#laya-reranker-experimental). |
| `rerank_instruction` | **Ranking criterion** in plain language (max 2000 chars). See [Rerank instructions](#rerank-instructions). |
| `lane_weights` | Per-lane RRF fusion weights, e.g. `{"sparse": 1.5}`. See [Lane weights](#lane-weights). |
| `lanes` | Restrict the call to a subset of the lanes (names under [Lane weights](#lane-weights)), e.g. `["sparse"]`: a lane outside the list is not run at all. It only restricts — a conditional lane (code, scope, `omnisearch`, `hype`, and `routed` under `generated_mode: route`) still needs its own trigger. An empty list or an unknown name is an error ([below](#when-a-knob-is-wrong)); the echo's `lanes_requested` and `lanes_run` show what ran. |
| `metadata_boost` | `true` / `false` forces the course/domain/tag metadata boost on or off for this call; unset follows `retrieval.metadata_boost`. |
| `scope_lanes` | `true` / `false` forces the scope lanes (`dense_scope` / `sparse_scope`) on or off for this call. The scope is detected and echoed either way; unset follows `retrieval.scope_lanes`. |
| `max_per_file` | At most this many chunks of one source file in the final list (0–50, `0` = no cap), best first. When the cap leaves fewer than `top_k`, the skipped chunks fill the tail. Unset follows `retrieval.max_per_file`. |
| `decompose` | `true` / `false` forces **query decomposition**: one generation-model call (on `/search` too) splits a multi-part question into sub-queries, each retrieved and reranked against its own text, and the branches merge by reciprocal rank so every part keeps a seat. An atomic question runs unchanged. A failed or unparseable split is the call's error, never a quiet fallback to one search. Unset follows `retrieval.decompose.enabled`; the echo's `decompose` says what ran. |
| `gate` / `gate_threshold` / `gate_scorer` | The post-rerank **relevance gate**: drop chunks scoring below a cutoff before generation, and abstain when nothing is left. Off by default; a `gate_threshold` alone turns it on for the call. See [Relevance gate](#relevance-gate). |
| `generated_mode` | `"drop"` keeps generated chunks out of this call's evidence. `"route"` does too, and lets generated hits bring the primary pages they cite into the pool, each such source carrying `found_via` ([Route mode](#route-mode)). `off` is config-only (`retrieval.generated_mode`), so a per-call `"off"` is a `422`. Unset follows the configured mode. `/compare` branches and `/evidence_pack` take it too. |
| `source_file` / `pages` | Restrict every lane to one file's primary chunks, by any stored spelling of its path (`GET /outline` lists the sources), and with `pages: [first, last]` to the chunks whose pages overlap that range. The pages `neighbor_context` adds stay inside them too, and Omnisearch does not run. Under `route`, the routed lane takes only the cited spans inside them, clipped to them; a span outside counts in no `route` field. The echo's `file_filter` repeats them, with `matched: false` when the store holds no primary chunk there: an unknown file, pages outside it, or a map. `pages` needs `source_file`. Needs `retrieval.sparse_backend: bm25s`. `/compare` branches and `/evidence_pack` take them too; `/evidence_pack` refuses them with `graph: true`. |
| `include_text` | Characters of chunk text to return (0–6000). |
| `max_sources` | Cap the number of sources returned. |

`/query` additionally accepts `provider` and `model` (a **configured** backend
alias — endpoints and secrets can never be supplied per request),
`max_tokens` (64–8192), and `retrieve_only`.

!!! warning "`max_tokens` and citations"
    A very small `max_tokens` can truncate the citation footer and drop the
    answer's confidence to `UNKNOWN`. Leave a few hundred tokens of room.

### When a knob is wrong

A per-call retrieval knob with a bad **value** is never a silent no-op, and it
is not a `400` either: `/search`, `/query` and `/compare` answer **HTTP 200**
with the reason in the body, so a caller handles it where it handles any other
retrieval failure.

| Endpoint | How the error appears |
|---|---|
| `POST /search` | A top-level `error` string, with `results: []` and `retrieval: {}`. |
| `POST /query` (generated and `retrieve_only` alike) | `confidence: "ERROR"` and the message in `answer`, with no citations or sources. `Bad request: …` for a bad value, `Reranking failed: …` for a reranker that failed or refused, `Retrieval failed (<ErrorType>: …)` for anything else. |
| `POST /compare` | That branch's `error` — `Bad branch configuration: …`, `Reranking failed: …` or `Retrieval failed (…)` — with `retrieval_error: true`. The other branches still run and report normally. |

A bad value is: an unknown `preset`; an empty or unknown entry in `lanes`; an
unknown lane or a negative weight in `lane_weights`; an unknown `rerank` mode,
or one that is refused (`laya` while `retrieval.laya.enabled` is off); a gate
that cannot work — no threshold anywhere, `rerank: none` under the
`rerank_score` scorer, an unknown `gate_scorer`, or a scorer other than the
configured one without its own `gate_threshold`; `decompose` together with
the gate, which would judge chunks scored against sub-queries;
`generated_mode: "route"` on a server with no `retrieval.route` block or with
the tantivy sparse backend; or `pages` without `source_file`, a `pages` range
that is not `1 <= first <= last`, or `source_file` on the tantivy sparse
backend.

A wrong **type** is different: `top_k: "ten"`, `lanes: "sparse"` (a string, not
a list), a non-numeric `gate_threshold` — or a number outside a field's
documented range — is rejected by request validation before retrieval runs, as
FastAPI's usual `422` with a `detail` list.

Two habits follow. Read `error` / `confidence` before reading `results` or
`sources`: an empty list that comes with an `error` is a rejected request, not
a gap in the corpus. And change the knob rather than resending the same body.

The rest keep their real status codes. [`/graph/expand` and `/answer`](#graph-mode)
answer `400` for a bad seed or document list, and `/compare` answers `400` for
duplicate branch ids or too many query-mode branches. A `route` the server
cannot serve is a `409` ([Route mode](#route-mode)). A generation failure on
`/query` is also reported in the body (`confidence: "ERROR"`), but keeps the
`sources` and `retrieval` that were already retrieved.

### Route mode

Under `generated_mode: route`, generated chunks are kept out of the evidence as
under `drop`, and they become routers. The routed lane searches the generated
rows (a dense pass and a sparse pass, plus the generated parents the `hype`
lane found), looks up the primary pages each hit cites, and
brings those pages into the pool, where they fuse and rerank like any other
candidate. Its fusion weight is the `routed` entry of `retrieval.lane_weights`
or a per-call `lane_weights` ([Lane weights](#lane-weights)); in search, a `lanes` list
without `routed` runs route as `drop` (the routed lane does not run), but a graph walk
under route still swaps the walked maps for their routed primaries. The block
`retrieval.route` holds the sidecar
path and the lane's caps: `router_k`, `per_target`, `max_targets` and
`max_new`. Read them from `config.yaml`.

- **`found_via`.** Each source the lane brought in carries
  `[{id, label}]`: the generated chunks whose cites found it. Every other
  source, and every source in another mode, has `[]`. Cite the source, never
  the map.
- **The sidecar.** `retrieval.route.pointers_file` maps each generated chunk id
  to the spans it cites, `{source_file, page_start, page_end}` (null pages for a
  whole-file cite), and lists the unrouted ids with a reason. Build it with
  `python main.py route-pointers` or the console's `route_pointers` job. It
  reads the chunk JSONLs and the vault and writes no store. `:8051` loads it
  once at startup, in every mode, whenever the `retrieval.route` block is set
  and the file exists, so restart `:8051` after a build. Each vault has its
  own.
- **The `409`.** While the store held generated rows at startup and no sidecar
  is loaded, a per-call `"route"` is HTTP **409** with
  `{"detail": {"error": "…"}}`, and the error names the build. This is a real
  status, unlike [a bad knob](#when-a-knob-is-wrong). It applies on `/search`,
  `/query` (with and without `retrieve_only`), `/evidence_pack` and `/compare`.
  On `/compare`, one `route` branch fails the whole request, after the earlier
  branches ran, so their results are lost too. With zero generated rows, route
  runs without a sidecar. Route also needs `retrieval.sparse_backend: bm25s`,
  and tantivy is refused first: on tantivy a per-call route is the knob error
  above, never the `409`, whose sidecar build could not make it route.
- **A configured route** refuses to start under the same condition:
  `/health` is `failed` with `RouteUnavailableError`, and its `hint` names the
  build. A configured route with no `retrieval.route` block, or with the
  tantivy sparse backend, fails to start too.
- **Coverage.** A ready `GET /health` and `GET /schema` (under
  `provenance.generated_mode`) carry `route`:
  `{generated_rows, with_pointers, sidecar}`. These are the rows stamped
  generated, how many of them the sidecar routes, and the loaded sidecar's path
  (null when none is). Agent files and other rows the sidecar cannot read stay
  unrouted, so `with_pointers` below `generated_rows` is normal. The key's
  presence is the capability probe: a server without it predates route mode.
  It shows that the server knows route mode and whether a sidecar is loaded,
  not that a call can route: with no `retrieval.route` block, or on tantivy, a
  per-call route still fails.
- **What routed.** Each call's echo carries `retrieval.route`; see
  [the retrieval echo](#the-retrieval-echo). Generated rows indexed after the
  sidecar was built count as `unrouted` until it is rebuilt and `:8051`
  restarted.

### Inspect and discover

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{ready, state, uptime_s, version, commit, generated_mode?, route?, error?}`. `version` is the product version; `commit` the git commit the process started from (null outside a git checkout). `generated_mode`, present once `ready`, is `retrieval.generated_mode` as the pipeline resolved it. `route`, present once `ready`, is route coverage, `{generated_rows, with_pointers, sidecar}` ([Route mode](#route-mode)). `state` is `ready`, `loading`, or `failed` — see [Readiness](#readiness-and-failure). |
| `GET /schema` | Machine-readable capability map: request fields, branch limits, the live preset registry, and the endpoint map. `provenance.generated_mode` describes each mode and carries the same `route` coverage. |
| `GET /config` | Live retrieval defaults, the preset registry, the active reranker, and generation provenance. |
| `POST /config` | Change retrieval defaults on the **warm** pipeline with no restart. `persist: true` also rewrites `config.yaml` and requires operator authorization. |
| `GET /providers` | Configured generation backends: protocol, endpoint, default model, the *name* of the environment variable holding each key, and readiness flags. Never returns a secret value. |
| `GET /chunks/{chunk_id}` | Fetch the current evidence record behind a stable id returned by `/search`, `/query` or `/compare`. |
| `GET /stats` | Corpus size with per-domain and per-file-type breakdown, plus `latency`: p50/p95 per endpoint, total and per stage, over the recent in-memory calls, so a caller can price a call before making it. |
| `GET /outline` | What the vault holds, for an agent that does not know it: `groups` of the primary sources by `domain` and `course`, each with `file_type`, `chunks` and `pages`, paged over sources (`limit`, `offset`; `total` counts the matches; `domain` filters exactly, `course` as a substring), and `maps`, the generated files, whole and apart. Built when `:8051` starts. See [Start from the outline](agents.md#start-from-the-outline-when-you-do-not-know-the-vault). |
| `GET /outline/{source}` | One file's outline: a source's `chapters` (`chapter`, `page_start`, `page_end`, `chunks`), or a map's `concepts`, each with the primary spans it cites. `404` for a file the index does not hold and for an agent-instruction file; `503` while no bm25 index is built. |
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
      "cited": true,
      "provenance": "source"
    }
  ],
  "retrieval": {"preset": "code", "rerank_top_k": 10, "hyde_used": false,
                "reranker_model": "<resolved-reranker>"},
  "generation": {"backend": "<provider-name>", "protocol": "<wire-protocol>",
                 "model": "<resolved-model>", "usage": {}}
}
```

- `lookup_available: true` → `GET /chunks/{id}` resolves it.
- `provenance` is `authored`, `source` or `generated`, by the rule in
  [Evidence pack](#evidence-pack).
- `found_via` is `[{id, label}]` for a source that the
  [routed lane](#route-mode) brought in: the generated chunks whose cites
  found it. It is `[]` for every other source.
- Parent-expanded sections use `parent:<id>` and report the indexed child as
  `origin_id`, so overlap is computed from the text each branch actually received.
- Live Omnisearch excerpts use a content-derived `live:<hash>` id and set
  `lookup_available: false`, because they are not stored index records.

### The retrieval echo

Every `/search` and `/query` response carries `retrieval`: what actually ran, as
opposed to what was asked for. Most keys restate an effective setting
(`preset`, `rerank_mode`, `lane_weights`, `metadata_boost`, …). These report how
the call went:

| Key | Meaning |
|---|---|
| `timings` | Milliseconds per stage of this call: `preset`, `scope`, `hyde`, `embed`, one `lane.<name>` per lane that ran, `fuse`, `boost`, `rerank`, `gate` (only on a call that gates), `parent`, `neighbor` and `total`. A stage that did not run is absent, not zero. On `/query` the generation stages (`generate`, `parse`, `verify`) are in `generation.timings`. |
| `lanes_requested` | The `lanes` list the caller sent, sorted; `null` when none was sent. |
| `lanes_run` | `{lane: candidates it returned}` for every lane that ran. A lane missing here did not run — it was not asked for, or its trigger did not fire. |
| `cold` | `true` when this call loaded the Chroma collection, the BM25 payload, the cross-encoder or the Laya checkpoint. When a BM25 index file exists, the service loads the collection and the payload at startup, before `/health` reports `ready`; the cross-encoder and the Laya checkpoint load on the first search that uses them, so that search's `timings` are a one-off spike, not steady state. |
| `hyde_cache` | Why the HyDE text is what it is: `off` (HyDE not used on this call), `bypass` (a code-intent signal skipped it), `hit` (draft replayed from the cache, no LLM call), `miss` (written fresh and stored), `nocache` (written fresh, no cache attached), `error` (the LLM failed, so the raw query was used). |
| `gate` | What the relevance gate did: `{"enabled": false}` when it did not run, otherwise `{enabled, scorer, threshold, kept, dropped, abstained}`. See [Relevance gate](#relevance-gate). |
| `route` | Under `generated_mode: route` only, and absent otherwise: what the [routed lane](#route-mode) did, summed over every retrieval of the call (a decomposed search runs one for the question plus one per sub-query). `router_hits` counts the generated hits, `routed` those the sidecar has targets for, and `unrouted` those it has none for (or every hit, with no sidecar). `targets` counts the distinct cited spans taken (at most `retrieval.route.max_targets` per retrieval), `targets_missing` the cited spans not in the store: the file is not indexed, or no primary row lies in its pages (such a span takes no `max_targets` slot), and `new` the routed chunks no other lane had found (at most `retrieval.route.max_new` per retrieval). The caps hold per retrieval, so a decomposed call can exceed them, and a span two retrievals took counts in both. Each `/compare` branch's `retrieval` carries its own. On `/evidence_pack` it is `retrieval[i].route`, beside `retrieval[i].graph_route` for the walk. |

`GET /history` keeps each recent call's echo, with its `timings` repeated at the
top level of the entry.

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

Building a valid `/compare` call by hand means reading `presets` and
`rerank_modes` from `/schema` and the backends from `/providers`, joining the
three, and re-deriving the branch caps from a sentence of English.

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

The `reranker` dimension is built from the rerank modes themselves and, unlike the provider
dimension, carries no `available` flag. It lists the experimental `laya` mode whether or not that
is enabled; while it is off, that branch fails with its `Reranking failed:` error (the siblings are
unaffected), so leave it out unless the feature is on.

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
| `laya` | **Ignored by design.** The query goes into Laya's own question, the exact wording its checkpoint was fine-tuned on; a prepended instruction would change the input it was trained on. |

The retrieval echo reports `rerank_instruction` and `rerank_instruction_applied`,
so a no-op is visible rather than looking like a setting that took effect. Send
`""` to disable a configured instruction for one call.

Config: `retrieval.rerank_instruction` (blank = off) and
`retrieval.rerank_instruction_format` — `prefix` (the instruction, a newline,
then the question; works with any cross-encoder because it simply reframes the
scored text) or `instruct` (the labelled `<Instruct>/<Query>` form, only for a
reranker documented to expect it).

### Lane weights

Fusion is Reciprocal Rank Fusion over every lane that ran. The lanes:

| Lane | When it opens |
|---|---|
| `dense` / `sparse` | Always. Vector and BM25. |
| `dense_code` / `sparse_code` | On code-intent queries, filtered to the configured code file types. |
| `dense_scope` / `sparse_scope` | When `retrieval.scope_lanes` (or a call's `scope_lanes`) is on and the query names a domain or a content type, filtered to that scope. The lanes ship off; the scope is detected and echoed either way. |
| `omnisearch` | When the live-vault lane is enabled and reachable. |
| `hype` | When the hypothetical-question collection exists and the lane is on. |
| `routed` | Under `generated_mode: route`: the primary pages that generated hits cite. See [Route mode](#route-mode). |

A document's fused score is the sum over lanes of
`weight(lane) / (rrf_k + rank_in_that_lane)`. Every weight defaults to `1.0`,
so an untouched config is plain unweighted RRF.

Set defaults in `retrieval.lane_weights`, per preset inside
`retrieval.presets.*`, or per call. All three **merge lane by lane**, so setting
one lane leaves every other lane where it was.

A weight of `0` is not an off switch: the lane still contributes candidates for
the reranker to consider, it just stops influencing the fused order. To drop a
lane, turn the lane off.

An unknown lane name or a negative weight is an error, never a setting that
silently does nothing: it comes back as HTTP 200 with an `error` field (see
[When a knob is wrong](#when-a-knob-is-wrong)), and a non-number is a `422`. The
retrieval echo reports every lane's effective weight, so the ones you did not
override are visible too.

!!! tip "Tune these against an eval, not by feel"
    `main.py bench run` scores labelled questions per named configuration; a
    configuration in `eval/configs.yaml` can set `lane_weights` like any other
    search override. See [Evaluation](evaluation.md).

### Relevance gate

Reranking orders the candidate pool, but it never says "none of these answers
the question": the top-k always reaches the generator, however weak. The
relevance gate is the decision step in between. After the rerank it scores each
chunk against the **original** question (never the HyDE text, which is a
hypothetical answer), drops the ones below a threshold, and, on `/query`,
abstains when nothing is left. It runs before parent/neighbor expansion, so
expansion never pads the context of a chunk the gate rejected.

It is **off by default**: configure it under `retrieval.relevance_gate`, or
drive it per call.

| Field | Effect |
|---|---|
| `gate` | `true` / `false` forces the gate on or off for this call; unset follows `retrieval.relevance_gate.enabled`. |
| `gate_threshold` | The cutoff: a chunk scoring *below* it is dropped. A finite number in the **scorer's own units**. It beats the configured threshold, and sent alone it turns the gate on for this call. |
| `gate_scorer` | `rerank_score` or `laya`; unset follows `retrieval.relevance_gate.scorer`. An unknown name is an error even while the gate is off. |

The scorers are not interchangeable:

| Scorer | Reads | Notes |
|---|---|---|
| `rerank_score` | The active reranker's own score, already on each chunk. | Free, and the baseline any learned gate has to beat. Its unit is that reranker's scale (a cross-encoder emits logits, `lexical` a coverage score), so a threshold means something for **one** reranker only. It needs a scoring rerank mode: `none` scores nothing, and a gate over it is an error, never a pass-through. |
| `laya` | P(relevant) in [0, 1] from the experimental Laya scorer. | **Experimental.** Needs `retrieval.laya.enabled`, the `laya` package and a checkpoint (see [Laya reranker](#laya-reranker-experimental)), and is never quietly replaced by `rerank_score`. |

A scorer other than the configured one must come with its own `gate_threshold`,
because the configured threshold is in the configured scorer's units.

When nothing passes, `POST /query` **abstains**: the fixed "nothing relevant"
answer at `LOW` confidence, no citations, no sources, and **no LLM call**.
`POST /search` returns whatever survived, possibly nothing. Either way the echo's
`gate` key says what happened: `{enabled, scorer, threshold, kept, dropped,
abstained}`, or `{"enabled": false}` when the gate did not run. `abstained` is
`kept == 0`, also when retrieval itself found nothing.

Misconfiguration is an error, never a pass-through. In config, an enabled gate
with no threshold, or an unknown scorer, fails the pipeline build, so
[`/health` reports `failed`](#readiness-and-failure) instead of the service
coming up half-configured. Per call, a gate with no threshold anywhere, a
`rerank_score` gate over `rerank: none`, an unknown `gate_scorer`, or a different
scorer without its own threshold is
[answered with an `error`](#when-a-knob-is-wrong). `gate_threshold` must be
finite, since a non-finite cutoff would keep or drop everything.

`/compare` branches that differ only by the gate do not share an evidence set.
There is no console switch for the gate. Choose its threshold on the dev split of
the [eval bench](evaluation.md#the-relevance-gate-and-the-laya-reranker) first;
the starting values in `eval/configs.yaml` are starting points, not tuned ones.

### Graph mode

!!! warning "Experimental — off by default"
    Graph mode ships behind `graph.enabled: false`. It works and is covered by
    tests, but it is experimental, so it is not part of the default workflow. While the flag is off,
    `POST /graph/expand` answers `503` naming the key to set; `/query` and
    `/search` are unaffected, and the graph lane still competes in ordinary
    retrieval. Turn it on in the management console under **Settings >
    Experimental features** (or set the key directly), then restart the API.

Some corpora carry a graph the author drew by hand — Obsidian canvases, where
nodes are connected by labelled edges. The canvas lane flattens each node's
edges into its chunk at ingest time, and graph mode walks them at query time.

`GET /schema` reports whether the endpoint is available and which indexed lane
holds the graph. Its top-level `experimental.features` block names the flagged
features that own an endpoint, with each one's config key and whether it is
currently on — so an agent can plan around a disabled mode instead of
discovering it through a `503`. An experimental feature that is an option on an
ordinary request, like the [Laya reranker](#laya-reranker-experimental), is
reported by the console API instead: `GET :8052/api/settings`, under
`experimental`.

**`POST /graph/expand`** — traversal, no LLM. Works with the generation backend
down.

| Field | Effect |
|---|---|
| `q` | Seed by searching the graph lane. |
| `seeds` | Seed from chunk ids of a previous result. Supplying both `q` and `seeds`, or neither, is a `400` — not a precedence rule. |
| `depth` | Hops from the seed set (0–5). `0` returns the seeds untouched. |
| `max_per_hop` | Neighbours followed per node (1–50). |
| `max_total` | Nodes in one traversal, seeds included (1–500). |
| `seed_top_k` | Seeds taken from a query-seeded run (1–50). |
| `rerank` | Orders query-seeded seeds. Ignored when `seeds` is given — those already carry the order the caller saw. |
| `include_text` | Characters of node text to return (0–6000). |
| `bridges` | `true` also attaches, to each canvas node, the chunks of the note or PDF page it cites ([below](#widening-and-typing-the-walk)). Null = `graph.bridges`. |
| `cross_canvas` | `true` also hops between canvases. Null = `graph.cross_canvas`. |
| `wikilinks` | `true` also follows the `[[wikilinks]]` of each note in the walk. Null = `graph.wikilinks`. |
| `direction` | `both`, `down` (along the arrowhead) or `up` (against it). Null = `graph.direction`. |
| `edge_types` | 1–20 edge labels; only edges with these labels are followed. Labels are compared ignoring case, with spaces and underscores read as `-`, and an unlabelled edge never passes. `same-as` hops and bridges always pass. Null = `graph.edge_types` (every edge). |

The response carries `nodes` (each with its hop depth and graph metadata),
`tree` (parent → child with each edge's label and direction), `cross_edges`
(edges back onto nodes already reached), and `stats`. A `direction` is the
edge as drawn, seen from the parent: `->` means an arrowhead points away from
it, `<-` one points at it, `--` there are no arrowheads, and `<>` there are
two. A canvas chunk indexed before the loader read arrowheads stores only `->`
or `<-`, taken from the end the edge starts at, until `canvas_refresh.py`
rewrites its metadata. A chunk indexed before edge labels were stored reports
`--`.

With a [route sidecar](#route-mode) loaded, in any `generated_mode`, each node
whose `provenance` is `generated` also carries `routed_ids`. These are the
first `retrieval.route.per_target` primary chunks of each span it cites, in
page order, for at most `retrieval.route.max_targets` spans. A cite with no
pages gives the file's first chunks, in document order. A map the sidecar does not route gets `[]`. Send them to `/answer` in the map's place. Ticking a generated node in the
console's graph view does that. Such a node starts unticked, and one with no
`routed_ids` cannot be ticked.

Two details matter when reading it:

- **Canvas graphs are cyclic.** A knowledge map that loops A → B → C → A is
  ordinary, not pathological. The walk tracks visited nodes keyed on
  `(source_file, canvas_node_id)` — a node id is unique only inside its own
  canvas file — and reports the loops as `cross_edges` instead of dropping
  them, which would draw a cyclic graph as though it were a tree.
- **Caps are reported.** `stats.truncated_at` names the first cap that bit and
  `stats.caps_hit` lists every one, so a truncated traversal never passes for a
  complete graph. `stats.dangling_edges` counts edges pointing at nodes that
  never became chunks — `file` and `group` nodes never do — which is normal,
  not an error.

**A seed embedder of its own.** By default (`graph.embedding_provider: null`)
a query-seeded walk searches the main collection with the main embedder. Set
the key to an `embedding.providers` name (or to `local`, when the main embedder
is not local) and query seeding searches a separate collection,
`<paths.collection_name>__graph`, instead. It holds exactly the main
collection's `graph.file_type` chunks: their ids and metadata, embedded by the
named model. Every seed found there is read back from the main collection, so
the walk and the node text are unchanged, and `/query`, `/search` and seeding by
chunk id never touch it.

- **Building it.** `python main.py graph-embed`, or the console's `graph_embed`
  job. `--limit N` (the job's `limit`) makes a smoke build that is never served.
  `POST /api/jobs` refuses the job with a `400` while the key is null, unknown,
  or names the main embedder. Every build drops the collection and rebuilds it
  from scratch, because a canvas chunk keeps its id when its text is edited. It
  never writes the main collection or a chunk file, and
  `POST /api/canvas/reindex` queues it after its append when the key is set.
  Restart `:8051` after a build.
- **The startup check.** `:8051` checks that the collection exists, was built by
  the configured embedder, is complete, and holds exactly the main collection's
  graph-lane ids. When a check fails, `:8051` still starts, and `/query`,
  `/search` and seeding by id keep working; only a query-seeded
  `POST /graph/expand` fails, answering with an `error` (`Seeding failed (…)`)
  that names the reason.
- **What the check cannot see** is text edited under a kept id. So
  `GET :8052/api/canvas/embedder` also lists the chunk files written since the
  build, beside its verdict (`off`, `missing`, `building`, `mismatch`, `stale`,
  `ready` or `error`) and the reason.

**`POST /answer`** — generation over a document set the caller chose. No
retrieval runs.

| Field | Effect |
|---|---|
| `q` | The question these documents should answer. |
| `docs` | 1–100 chunk ids. An unknown id is a `400`, never a quietly smaller context: an answer grounded on fewer documents than you chose is a different answer. |
| `provider` / `model` / `max_tokens` / `include_text` / `max_sources` | As on `/query`. |

Evidence ids that are not indexed records — `live:` excerpts and `parent:`
sections — cannot ground an answer and are rejected by name.

Under any `generated_mode` but `off`, ids whose `provenance` is `generated` are
a `400` and nothing is generated. `detail` is `{"error": "generated chunks are
maps, not evidence: answer from the primary pages they cite", "ids": [...]}`,
and `ids` names only the generated ones. An unknown id is reported first.

With a [route sidecar](#route-mode) loaded, in any mode, `detail` also carries
`routes_to`: `{id: [{source_file, page_start, page_end}, …]}` for each
generated id the sidecar routes, leaving out any it does not. It lists only
the spans the store serves: a span whose file is not indexed, or with no
primary row in its pages (what `targets_missing` counts), is left out, and so
is an id left with none. These are the
full spans each one cites, as pages, not chunk ids. The ids to send in a map's
place are a `/graph/expand` node's `routed_ids` (above), which hold at most
`retrieval.route.per_target` chunks of each span. With no walk at hand, `POST /graph/expand` with
`seeds: [<that id>]` and `depth: 0` returns the node with its `routed_ids`
while graph mode is on.

This shape is what makes the human-in-the-loop fork work without server
session state: *answer from the graph alone* and *merge with the previous
result* are the same call with a different id list, so "what counts as the
previous stack" stays with the caller.

#### Widening and typing the walk

`POST /graph/expand` and the [evidence pack](#evidence-pack) take five walk
switches: `bridges`, `cross_canvas`, `wikilinks`, `direction` and `edge_types`.
Each falls back to its `graph.*` key, and those keys ship switched off
(`false`, `both`, `null`). With every switch at that default, the walk and its output are exactly the
canvas-edge walk described above. The caps named below
(`graph.target_max_chunks`, `graph.bridge_max_total`,
`graph.same_as_min_chars`, `graph.same_as_max_canvases`) are config keys only,
not request fields. Setting any switch off its default also extends the output:

- every node carries `kind` (`canvas`, `note` or `bridge`), `file_type`,
  `heading`, `page_start` and `page_end`. `nodes` lists the walk's nodes first,
  then the bridges, and `stats.nodes` counts only the walk's nodes;
- every `tree` and `cross_edges` entry carries `kind`: `canvas` for a drawn
  edge, otherwise `bridge`, `same-as`, `shares-bridge`, `opens-canvas` or
  `links-to`;
- `paths` says how each node was reached, one entry per node:
  `{to, nodes: [seed, …, to], edges: [{label, kind, direction}, …]}`, along the
  first (shortest) route the walk found;
- `stats` adds `bridges` (bridge chunks attached), `filtered_edges` (edges that
  `edge_types` or `direction` rejected; such an edge is neither a hop nor a
  cross edge), `virtual_edges` (hops followed, per kind),
  `virtual_edges_dropped`, the miss counters named below, `options` (the
  switches in effect) and `caps.bridge_max_total`.

**Bridges** (`bridges`) attach to a canvas node the chunks it cites:

- The targets are the node's file nodes (except those on a `.canvas` file,
  which are cross-canvas hops) and its `Source:` lines. A `Source:`
  line is a text-node line `Source: <vault path>.pdf` or `.md`, optionally
  followed by `p. N` or `pp. a-b, c`. At ingest, each line keeps its first two
  page groups and a node keeps six in all, and a page range bridges its first
  page and the page after it.
- A PDF bridges only to a page: a file node with a `#page=N` subpath, or a
  `Source:` line with pages. Any other PDF target bridges nothing, because a
  PDF's first pages are front matter, never evidence, and it is counted in
  `stats.bridge_pdf_without_page`.
- A note bridges its chunks, or with a `#Heading` subpath only that section's
  chunks. A heading the note does not have bridges nothing and is counted in
  `bridge_heading_missing`; the walk never widens to the whole note. A
  `#^block` subpath takes the whole note.
- A target with no indexed chunk (an image, a missing or unindexed file, a path
  that differs only in case) is counted in `bridge_targets_without_chunks`. A
  path that differs only in Unicode form (NFC against NFD, which macOS can
  produce) still matches, and is counted in `bridges_matched_after_normalised`.
- Bridges end the walk at that point. They are never walked further, never
  counted in `max_total`, and never filtered by `edge_types` or `direction`.
  `graph.target_max_chunks` caps the chunks per target and
  `graph.bridge_max_total` the bridges per walk. A bridge's tree entry carries
  the edge's own label, or `source` for a `Source:` line. An edge onto one of
  the node's file nodes is now its bridge, so it no longer counts in
  `dangling_edges`. A cited chunk the walk already holds (a seed, a walk node)
  gets a `cross_edges` entry instead, and a chunk that is both is listed once,
  as a walk node.
- A canvas chunk indexed before `canvas_edge_files` existed bridges nothing.
  The walk counts it in `stats.bridges_unavailable` and never guesses from
  older metadata, and the response's `note` names the fix: run
  `canvas_refresh.py`, then `rebuild_bm25.py`, then restart this API.

**Cross-canvas hops** (`cross_canvas`) join a canvas node to nodes of other
canvases. They come after the node's drawn edges and share its `max_per_hop`
budget:

- `same-as`: a node whose title (its first line of text, ignoring case and
  punctuation) is the same. A title shorter than `graph.same_as_min_chars`
  joins nothing, and neither does a generic title found in more than
  `graph.same_as_max_canvases` canvases.
- `opens-canvas`: a file node on a `.canvas` file leads to that canvas's most
  connected nodes first.
- `shares-bridge`: a node of another canvas with a file node on the same file.
  `Source:` lines do not count here: a book cited on every node would join
  every canvas that cites it.
- `same-as` passes every `direction` and `edge_types` filter. `opens-canvas`
  and `shares-bridge` have no direction, so they are followed only under
  `both`, and under `edge_types` only when the list names them. Both read
  `canvas_edge_files`, so an older chunk takes only `same-as` hops (counted in
  `bridges_unavailable`).
- A hop over the budget is counted in `virtual_edges_dropped`, never as the
  `max_per_hop` cap. The canvases are indexed once per API process, so a canvas
  appended later joins at the restart that the append needs anyway.

**Wikilinks** (`wikilinks`) follow the outgoing `[[links]]` of each note in the
walk. Query seeding (`q`) starts from canvas chunks, so on this route a note
joins the walk only as a seed passed by id. An [evidence pack](#evidence-pack)
also seeds the notes its search returned.

- A link's name must equal a note's file name exactly, case included. Obsidian
  ignores case here and this walk does not: `[[other]]` does not reach
  `Other.md`, and is counted in `wikilinks_unresolved`. A name two notes share
  is counted in `wikilinks_ambiguous` and never guessed. Links to attachments
  (`.png`, `.pdf` and the like) are ignored.
- A link with a folder (`[[A/Note]]`, which Obsidian writes when two notes share
  a name) reaches only the note under that folder. With none indexed there it is
  unresolved, never another folder's note.
- `[[Note#Heading]]` takes only that section, and a heading the note lacks is
  unresolved. A note is one node, taken as up to `graph.target_max_chunks`
  chunks per link: links to two sections of a note reach both, and a link into
  a note the walk already holds adds the section it names.
- A link is followed along its arrow: under `down` or `both`, and under
  `edge_types` only when the list names `links-to`; otherwise each link counts
  in `filtered_edges`. Backlinks are not followed.
- Chroma stores a note's links joined by `, `, which a name holding `, ` itself
  breaks, so a joined list is read back from the sparse index. A note missing
  from it is dense/sparse drift: the walk refuses, and the error names
  `rebuild_bm25.py`.
- The parser stores a note's first `MAX_WIKILINKS` links
  (`src/ingestion/obsidian_parser.py`); `wikilinks_truncated` counts the notes
  that reach that cap.

### Evidence pack

**`POST /evidence_pack`** is one call for a multi-part question. The calling
agent splits the question into sub-questions, and the pack runs the `/search`
pipeline once for each, merges the results by evidence id, and reports which
sub-questions came back empty. It writes no answer. As on `/search`, the
generation model is called only for `hyde` or `decompose`. `GET /schema` lists
it under the experimental Graph RAG feature, but it works with graph mode off:
only `graph: true` needs graph mode. The [agent guide](agents.md#multi-hop-questions-evidence-packs-and-graph-walks)
shows a full request and response.

With `graph: true`, each sub-question's search is followed by a walk. It is
seeded by that search's graph-lane (canvas) hits, its note hits when
`wikilinks` is on,
plus a graph-lane query (`seed_top_k`), and the
[walk switches](#widening-and-typing-the-walk) apply. The search hits and the
walked chunks are then reranked together down to the search's own `top_k`, so a
walk never wins by having more slots. A walk that adds nothing leaves the search
result as it was. `parent_context` and `neighbor_context` run after that
rerank, so it scores the child chunks on both sides. `max_per_file` and the
relevance gate are not applied again after it.

| Field | Effect |
|---|---|
| `q` | The question. It is searched alone when `sub_questions` is null. |
| `sub_questions` | The parts, each searched on its own. At most `evidence_pack.max_sub_questions` entries; an empty entry is a `400`. |
| `top_k` | Evidence per sub-question (1–50). Null = `evidence_pack.top_k`. |
| `include_text` | Characters of text per evidence item (0–6000; `0` sends none). Null = `evidence_pack.include_text`. |
| `min_score` | A floor in the reranker's own units: items below it are dropped and counted. Null = `evidence_pack.min_score`, and when that is null too, everything retrieved counts as coverage. It needs scored items: a floor under `rerank: none` (which scores nothing), or with `neighbor_context` pages (added unscored) in the result, is a `400`. |
| `graph` | `true` walks too, and needs graph mode: when it is off, a `503` names `graph.enabled`. Default `false`. |
| `depth` / `max_per_hop` / `max_total` / `seed_top_k` / the five walk switches | The walk, as on `/graph/expand`, read only with `graph: true`. Null = the `graph.*` key. |
| every other `/search` field | As on `/search` (`preset`, `rerank`, `lanes`, …). These are refused with a `400`: `omnisearch: true`, because a live excerpt is not indexed evidence and `GET /chunks/{id}` could not check it; `max_sources`, because `top_k` is already per sub-question; and `source_file` or `pages` with `graph: true`, because a graph walk leaves the file. |

| Response key | Meaning |
|---|---|
| `evidence` | One item per distinct evidence id, merged across sub-questions: their `sub_questions` are joined and the best `score` is kept. Each item has `id`, `origin_id`, `text`, `source` (the label), `source_file`, `file_type`, `pages` (`[start, end]`, or null), `heading`, `course` (null when unknown), `provenance`, `score`, `via_path`, `found_via` and `sub_questions`. |
| `paths` | The walk paths some evidence uses, in `/graph/expand`'s `paths` shape. |
| `coverage` | `{sub-question: [evidence ids]}`. |
| `gaps` | The sub-questions with no evidence. |
| `generated_only` | The sub-questions whose every evidence item is `generated`. |
| `provenance_mix` | `{authored, source, generated}`: counts over `evidence`. |
| `dropped_below_min_score` | `{sub-question: n}`, `0` for every sub-question when no floor is set. |
| `graph` | Whether a walk ran. |
| `retrieval` | Per sub-question: `q`, `rerank_mode`, `timings` (the `/search` stages, plus `graph_seed`, `graph_expand`, and `graph_rerank` when the walk added chunks), `graph_stats` (the walk's `stats`, null without a walk), `route` (the search's [route summary](#the-retrieval-echo), null outside `route`) and `graph_route` (the same keys for the walk's own routing, where `new` counts the routed chunks neither the search nor the walk had found; null unless a `graph: true` walk under `route` reached generated nodes while a sidecar is loaded). The two are never summed. |
| `question` / `sub_questions` / `ms` | The question and sub-questions as searched (stripped), and the call's time in milliseconds. |

How to read an item:

- **`id`** is what `GET /chunks/{id}` serves: `parent:<id>` when
  `parent_context` swapped in the whole section, otherwise the chunk id, which
  is also `origin_id`. Every pack holds indexed evidence only, so every id
  resolves.
- **`provenance`** is `authored`, `source` or `generated`, decided in this
  order. A stored `provenance: generated` is `generated`. Next, a
  `source_file` the path rule names is `generated` whatever the chunk stores:
  one under a `provenance.generated_paths` prefix, an agent-instruction file
  (its name is in `provenance.agent_instruction_files`, or a directory on its
  path is in `provenance.agent_instruction_dirs`), or a `.canvas` path holding
  one of `canvas.generated_markers`. Then any other stored `provenance` value,
  then the file type's value from `provenance.by_file_type` (which maps `pdf`
  to `source` as shipped), and `authored` otherwise. The note lane stores the
  value from the frontmatter: an explicit `provenance:`, else `generated` for
  `generated: true`, else `authored`. The PDF lane stores `source`. An unknown
  stored value is an error, never a guess.
- **`score`** is the rerank score, or the fusion score under a rerank mode
  that scores nothing.
- **`via_path`** is the node ids of the walk path that reached a chunk the walk
  added, and `[]` for a chunk the search itself returned. It is `[]` for a
  routed page too.
- **`found_via`** is a routed page's trail: `[{id, label}]`, the maps whose
  cites brought it in, joined by id across sub-questions. It is `[]` for every
  other item. Cite the page, never the map.

A bad request is a `400` naming the problem. That covers a bad knob value too:
where `/search` answers `200` with an `error` ([When a knob is wrong](#when-a-knob-is-wrong)),
this route answers `400`. A wrong type is the usual `422`. A reranker that fails
gives `200` with `error` set to `Reranking failed: …`, and any other failure
gives `200` with `Evidence pack failed (…)`. Both come with empty `evidence` and
`retrieval`.

### Laya reranker (experimental)

!!! warning "Experimental — off by default"
    `rerank: laya` is built and covered by tests, but it has never been scored
    on the eval sets, so whether it ranks better than the default is
    unmeasured. It ships behind `retrieval.laya.enabled: false`.

One more rerank method. A Laya model, fine-tuned off-machine on the user's own
notes, gives every candidate passage a probability that it answers the question,
and that probability (`rerank_score`, in [0, 1]) orders the pool. Pick it per
call (`"rerank": "laya"`) or as the default (`retrieval.rerank_mode: laya`). The
same scorer can also back the [relevance gate](#relevance-gate).

- **Refused while off.** A call that asks for it comes back with an error naming
  `retrieval.laya.enabled` (`Reranking failed: …`, see
  [When a knob is wrong](#when-a-knob-is-wrong)), never answered by another
  reranker. The same refusal covers a missing `laya` package or checkpoint.
  `GET /schema` lists `laya` under `rerank_modes`, and `GET /compare/options`
  offers a branch for it, whether or not it is enabled.
- **It ignores `rerank_instruction`**, and the echo says so. `reranker_model` and
  `reranker_max_length` are `null` for it: a checkpoint reads its own length from
  its config.
- **It needs** the optional `laya` package, deliberately not in
  `requirements.txt`, and a checkpoint under `retrieval.laya.model_dir`;
  `docs/laya-finetune.md` is the manual for producing one. The checkpoint loads
  on the first call that uses it (a `cold` call), and one loaded scorer serves
  both the reranker and the gate.
- **Switch it on** from the console's Settings > Experimental features, or with
  `retrieval.laya.enabled` in config, then restart `:8051`. `GET :8052/api/settings`
  reports its state (`experimental`, id `laya_rerank`).
- **Graduation:** [Evaluation](evaluation.md#the-relevance-gate-and-the-laya-reranker)
  says what it has to beat before it stops being experimental.

### Readiness and failure

`GET /health` distinguishes three states, because a service that is still
loading and one that failed to build call for opposite responses:

| `state` | Meaning | What to do |
|---|---|---|
| `ready` | Serving. | — |
| `loading` | Indexes and models are still loading. | Wait. |
| `failed` | The pipeline could **not** be built from the current config. | Act — waiting cannot fix it. `error` names the cause. |

A `failed` service still answers: every retrieval endpoint returns **503** with
the same reason attached, rather than refusing the connection. The usual causes
are a model id that cannot be downloaded, an embedding model with a typo, or an
index path that moved — normally the setting that was changed most recently.
A configured `generated_mode: route` with generated rows and no route sidecar
fails as `RouteUnavailableError`, and its `hint` names the build
([Route mode](#route-mode)).
`GET /api/service/log` on `:8052` returns the startup log.

!!! warning "Reranker failures are retrieval failures"
    A response beginning `Reranking failed:` never reached generation. Preserve
    the underlying model error and check `GET /config` for the active reranker.
    Do not relabel it as a generation-provider outage.

---

## Management console API — `:8052`

Endpoints covering the whole corpus lifecycle, plus the eval review queue.
`GET /api/schema` returns them with a **permission tier** on every operation; read
the live list rather than counting on this page.

### Permission tiers

These tiers are a contract the **caller** honours; the optional token decides
who may call, not which operation. Any agent toolkit driving this API should
gate on them. The [MCP server](mcp.md) does: it maps each tier to its tools'
annotations and prompts.

| Tier | Meaning |
|---|---|
| `read` | Safe. No confirmation needed. |
| `mutating` | Changes local state, or invokes a potentially billed external action. Confirm with the operator first. |
| `destructive` | Removes content. Always confirm, echo exactly what will be deleted, never run unprompted. |

### Inspect the corpus — `read`

| Endpoint | Purpose |
|---|---|
| `GET /api/schema` | This capability map, with tiers. |
| `GET /api/overview` | Corpus summary: chunk/doc counts, per-domain and per-JSONL breakdown with each JSONL's last-modified time (`jsonl_mtime`, epoch seconds), vector count, disk use, query-API health, recent jobs. |
| `GET /api/documents` | Search / filter indexed documents. |
| `GET /api/facets` | Available domains, group labels, tags, file types. |
| `GET /api/documents/preview` | Preview a document's indexed chunks. |
| `GET /api/vault/tree` | Browse the vault tree, with in-index status per file. |
| `GET /api/vault/search` | Find files by name in the vault. |
| `GET /api/browse` | Filesystem folder picker (for choosing paths the container can see). |
| `GET /api/settings` | The editable config surface, provider registry status, reranker suggestions, device capabilities, taxonomy, and `embedders`: each known local model and `embedding.providers` entry with the finished chunk collections in the active Chroma dir that it built. |
| `GET /api/vaults` | Registered vaults and their per-vault settings (index paths, lane chunk files, figure/backup/staging folders, inbox and tree root, the embedder). |
| `GET /api/ocr/status` | What OCR this install can actually do right now, per engine. |
| `GET /api/service/log` | Tail the query API's startup log. |
| `GET /api/releases` | The version on disk (`VERSION`), each part of the install (this console, the query API, the installed agent skills, a packaged bundle) as `current`, `behind`, `missing`, `not running` or `invalid` with what to do, and the release notes from `CHANGELOG.md`. The console's header version chip opens it. |
| `GET /api/jobs`, `GET /api/jobs/{id}`, `GET /api/jobs/{id}/log` | Job queue state and output. A job record carries `vault` (the vault active when it was queued; blank for older records), `elapsed_s`, `queue_position`, `progress` (the current phase's `{done, total, line}`, read from the log) and an ETA (`typical` over past successful runs of that kind, and `eta_s`; both null without history). The log takes `tail` (last N lines) and `wait` (a long poll of up to 60 s that returns as soon as there is new output or a status change). |
| `GET /api/inbox`, `GET /api/import/converted`, `GET /api/import/file` | Staged-file listings and raw fetch. |
| `GET /api/import/ocr_scan` | Which pages of a staged PDF carry no extractable text — run this before OCRing anything. |
| `POST /api/rerank/check` | Preflight a cross-encoder: reachable? already cached (and where)? within its context limit? Downloads nothing. |
| `GET /api/engines` | Every engine per component, with availability and the cost of a switch. See [Engines map](#engines-map). |
| `GET /api/eval/progress`, `GET /api/eval/questions`, `GET /api/eval/questions/{qid}` | The eval review queue: per-suite progress, the filtered question list, and one question with its gold texts and validator findings. See [Eval review queue](#eval-review-queue). |

### Change things — `mutating`

| Endpoint | Purpose |
|---|---|
| `POST /api/settings` | Persist whitelisted config values into `config.yaml` in place, comments preserved. Nothing hot-applies; the response says which service to restart. |
| `POST /api/engines/set` | Switch one component's engine persistently, after a preflight. See [Engines map](#engines-map). |
| `POST /api/providers/key` | Set or clear one provider key in `.env`. Only variable names the registry already declares are accepted. Never returns the value; rejects a declared credential-prefix mismatch. |
| `POST /api/service/restart` | Restart the warm query API. Returns the relaunch generation counter; fails loudly if the supervisor does not actually relaunch. |
| `POST /api/jobs`, `.../retry`, `.../cancel` | Queue, retry and cancel jobs. |
| `POST /api/upload`, `POST /api/ingest_inbox`, `POST /api/ingest_custom` | Stage files and run ingest passes. |
| `POST /api/import/fetch`, `/convert`, `/promote` | Pull a URL as markdown or printed PDF, convert any upload to markdown, promote a staged result into ingest. |
| `POST /api/documents/retag` | Metadata-only update (domain / group label / tags). **No re-embedding.** |
| `POST /api/ocr/warm` | Load a vision-OCR model now rather than stalling the first ingest page. May bill an external endpoint. |
| `POST /api/vaults/switch`, `POST /api/vaults/forget` | Swap the whole per-vault path set atomically; drop a registration. |
| `POST /api/eval/questions/{qid}` | Verify, reject or edit one eval question and rewrite its sets file. See [Eval review queue](#eval-review-queue). |

### Remove things — `destructive`

| Endpoint | Purpose |
|---|---|
| `POST /api/documents/delete` | Remove documents from the index: their vectors and their rows in the chunk files, and with `rebuild: true` a queued sparse rebuild. `keep_jsonls` names chunk files whose rows stay (the file a refused `index --append` will append again); a name that is no chunk file is a `400`. Vault files are never touched. Restart `:8051` afterwards. |
| `POST /api/inbox/delete` | **Permanently deletes** staged files (`names`, plain filenames) from the inbox or its `_converted` folder on disk: no `.trash` copy, and the index is untouched. `folder` (`inbox` or `converted`) says which folder to delete from. Without it, a name that both folders hold is refused with a `400` before anything is deleted. A name it cannot find comes back under `missing`. The console's two × buttons each send their own `folder`. |
| `POST /api/ocrstudio/remove` | Take one document out of the OCR queue by **moving** it into the vault's `.trash`; nothing is deleted. See [OCR Studio](#ocr-studio). |

### Engines map

`GET /api/engines` lists, for each component (generation, reranker, embeddings,
sparse, ocr, fetch, chunking), the active engine, the setting a switch writes,
and every option with:

| Field | Meaning |
|---|---|
| `available` | A fast probe says it can be used now. The probes load no model and call no internet; the local OCR sidecars get short health checks. |
| `swappable` | `POST /api/engines/set` accepts it: a probe can check it, the switch is checked before it is written, and it fails loudly instead of falling back. An option that is not is still listed. |
| `reason` | Why it is unavailable or not swappable; `null` when it is both. |
| `switch` | The cheapest way to use it: `per_call`, `per_job`, `restart`, `build` (a one-off job first) or `reembed`. |

`POST /api/engines/set` takes `{component, engine, model?}`. It refuses an
unknown or unswappable engine with its reason, then runs that component's
preflight: for the reranker, the local cache and then the model's hub entry
(with a timeout) and the context limit; for embeddings, only an already-built
collection whose fingerprint matches the configured embedder; for the sparse
lane, the tantivy index must exist; for OCR, the engine must answer now. It
writes through the same validator as `POST /api/settings`, and a refusal
writes nothing. It never restarts anything: the response is
`{written, takes_effect, next_op, warnings}`. `next_op` is the restart to run,
`POST /api/service/restart`, which restarts `:8051` only; it is `null` when
nothing needs a restart and also when only `:8052` does, so read
`takes_effect`. When the console must restart, `warnings` says so: restart it
by hand. `warnings` also carries the Settings writer's notes, such as a model
carried along with a provider. `model` applies to `generation` (the model id)
and to the reranker's `cross_encoder`.

### Canvases (the Graph tab)

The Graph tab is a client of these routes. Of them, only `POST /api/canvas/file`
writes the canvas itself, and only on an explicit save.

| Endpoint | Tier | Behaviour |
|---|---|---|
| `GET /api/canvas/folders` | `read` | Which vault trees hold `.canvas` files, with counts: the scope picker of the canvas ingest lane. One cached vault walk; `refresh=1` rescans. |
| `POST /api/canvas/preview` | `read` | Runs the canvas loader for real into a scratch file, then deletes it, and reports the chunks, nodes and edges, the share of chunks that carry edges, the text that `context_depth` duplicates, and one composed sample. Body: `include_path`, `min_chunk_size`, `max_chunk_size`, `chunking`, `context_depth`. Indexes nothing. |
| `GET /api/canvas/files` | `read` | Every `.canvas` in the vault, vault-relative, from the same cached walk. |
| `GET /api/canvas/file` | `read` | One canvas (`path`), parsed, with the `sha256` of its bytes that a save must send back and its indexed chunks. |
| `POST /api/canvas/file` | `mutating` | Save edits: `{path, base_sha256, data}`. Refused with `409` when the file changed on disk since it was loaded. The edit is merged onto the file as it is on disk, so only what the editor edits changes (node text, positions and sizes, edge labels, added and deleted text nodes and edges) and everything else stays as Obsidian wrote it. The old bytes are backed up to `canvas.backup_dir`, outside the vault, and the new ones replace the file atomically; a save with no edits writes nothing. |
| `POST /api/canvas/reindex` | `mutating` | Re-index one canvas (`path`): removes its chunks from the index and the chunk files, then queues a scoped `ingest_canvas` into its own `data/canvas_edit_*_chunks.jsonl` and an `index_append`, and `graph_embed` third when `graph.embedding_provider` is set. A broken embedder setting is refused before anything is removed. Restart `:8051` when the last job finishes. |
| `POST /api/canvas/suggest_label` | `mutating` | A suggested label for one edge, from a language model ([below](#suggest-label)). May bill; writes nothing. |
| `GET /api/canvas/suggest_options` | `read` | The suggest-label picker: the default backend and model and which setting they come from, every `providers:` entry with its key status and `vision` flag, the relation vocabulary, the long-node settings, and whether label `#tags` are indexed. |
| `GET /api/canvas/embedder` | `read` | Graph mode's seed embedder: the setting, the seed collection's fingerprint and freshness, a verdict with its reason, and the embedders it can be built with. See [Graph mode](#graph-mode). Reads only. |

To take a canvas out of the index, the Graph tab calls the ordinary
`POST /api/documents/delete` with its `source_files` and `rebuild: true`, then
queues `graph_embed` when a graph embedder is set, then offers the `:8051`
restart. The `.canvas` file stays in the vault.

**Tags in labels.** A label can carry Obsidian `#tags` (`#` and then a letter).
With `canvas.label_tags: true` (default `false`), the next ingest of that canvas
copies them into the `tags` of the chunks of both nodes the edge joins.
Metadata only: chunk ids and text do not change.

#### Suggest label

`POST /api/canvas/suggest_label` asks a language model for one edge's label.
Its body is `path` (an existing canvas), `edge_id`, `data` (the `{nodes, edges}`
the editor holds, unsaved edits included), and optionally `provider`, `model`
and `guidance` (free text appended to the prompt, up to 2000 characters).

What the model is shown is built from the posted canvas, not the file:

- **Both endpoint nodes.** File nodes are read from the vault, read-only: a note
  without its frontmatter, plain text and code as they are, a sub-canvas as its
  node titles, a group as its label and members, a link as its URL (never
  fetched). A PDF or other binary, a missing file, or a path outside the vault
  becomes a placeholder, never an error. An endpoint longer than
  `canvas.augment.node_chars` is summarised by the same model first (one more
  call, reading at most `canvas.augment.summarize_input_chars`), or cut, with
  `canvas.augment.long_nodes: truncate`.
- **The arrow** as drawn (`→`, `←`, `↔` or `—`), the canvas title, each
  endpoint's group and the canvas's groups.
- **The labels around it:** the other edges at each endpoint, and the canvas's
  labels as examples. A label with no letter once its `#tags` are removed (a
  step number, a bare `#p`) names no relation, so it is never an example and its
  edge shows only the neighbour's title.
- **Images**, only for a `providers:` entry marked `vision: true` and up to
  `canvas.augment.max_image_mb`. For any other entry the prompt names the image
  instead; the LLM client refuses to send images to an entry without the flag.

Who answers: the request's `provider` and `model`, else `canvas.augment.provider`
and `canvas.augment.model`, else the generation role's backend and model. A
backend that cannot be built is refused, never replaced by another. The output
cap is `canvas.augment.max_tokens`, sized for a model that reasons before it
answers.

The reply is read robustly: reasoning blocks are dropped and the last JSON
object with a non-empty `label` wins. The response carries `label`, `relation`
(a term of `canvas.augment.vocabulary`, read along the arrow, or `null`, with a
warning, when the model names another), `tags`, `alternatives`, `confidence`
(`high`, `medium`, `low` or `null`), `why` and `warnings`; the `backend`, `protocol`
and routed `model` that answered; `usage` over every call; `prompt_version`;
`label_tags_indexed`; and `context`, exactly what was sent, including how each
endpoint was fitted and which images went or why not. Nothing is written: the
label reaches the file only through a save.

Errors are typed by `stage`, and each `502` names the `backend` and `model`:

| Status | `stage` | When |
|---|---|---|
| `400` | — | The path is not a `.canvas` in the vault; `data` is not `{nodes, edges}`; the edge, or one of its endpoints, is not in `data`; `canvas.augment.long_nodes` is neither `summarize` nor `truncate`. A canvas that does not exist is a `404`. |
| `400` | `client` | An unknown provider (the error names the field that set it), a missing or wrong-type key, or a `vision` value that is not a boolean. |
| `502` | `summary:from`, `summary:to` | A long endpoint's summary call failed or came back empty. |
| `502` | `label` | The label call failed: transport, HTTP or timeout. |
| `502` | `parse` | The reply holds no JSON object with a non-empty `label`; `reply` carries the start of the raw reply. |

### OCR Studio

The OCR tab's routes. Studio work (regions, clean-up, masks, page status) is
kept in `ocr_studio/` beside the chunk files, never in the vault.

| Endpoint | Tier | Behaviour |
|---|---|---|
| `GET /api/ocrstudio/files` | `read` | The OCR queue: imports (`ocr_studio.upload_dir`), the inbox's top-level PDFs and images, other documents with studio work, and with `q` (two characters or more) a filename search of the whole vault. Every row carries its studio state and working page set. |
| `POST /api/ocrstudio/upload` | `mutating` | Import one PDF or image (multipart `file`) into `ocr_studio.upload_dir`, whole and byte for byte. `pages`, a typed range, becomes its working page set, and every page keeps its real number. Never overwrites a name, never indexes; `413` above `ocr_studio.upload_max_mb`. |
| `POST /api/ocrstudio/remove` | `destructive` | Take one document out of the queue ([below](#taking-a-document-out-of-the-queue)). |
| `GET /api/ocrstudio/doc`, `/page`, `/page.png`, `/mask.png` | `read` | One document's pages with the scanner's verdict per page, its progress and its indexed chunks; one page's regions and assembled markdown; the page rendered with its clean-up; its healing mask. |
| `POST /api/ocrstudio/page`, `/mask`, `/pageset` | `mutating` | Save one page's regions, clean-up and status; store or clear its mask; set or clear the working pages (`409` when saved work would fall outside them). |
| `POST /api/ocrstudio/propose` | `read` | Region proposals for one page, from the page layout or the PaddleOCR sidecar's lines. Saves nothing. |
| `POST /api/ocrstudio/ocr` | `mutating` | OCR one region with its own engine and language, and store the result. A failing engine is that region's error, never another engine's output. May call the sidecar or a vision model. |
| `GET /api/ocrstudio/engines`, `/quality` | `read` | Each engine's availability and languages; the character error rate per engine over the pages marked done. |
| `POST /api/ocrstudio/ingest` | `mutating` | Chunk the done (or every saved) page into the document's own `data/ocrstudio_*_chunks.jsonl` and queue its ingest and append. `append` is refused (`409`) for a document that already has chunks; `replace` first removes them from the index and every chunk file; `dry_run` only reports. |
| `POST /api/ocrstudio/export` | `read` | A zip of the corrected markdown per document, real page numbers as `<!-- page N -->` markers, with its figure crops. Writes nothing. |

#### Taking a document out of the queue

`POST /api/ocrstudio/remove {path}` accepts exactly what the queue lists as an
import or an inbox document: a top-level PDF or image of `ocr_studio.upload_dir`
or of the inbox (`webui.inbox_dir`). Anything else is a `400`, a missing file a
`404`.

- It **moves** the file into the vault's `.trash` folder, under its own
  vault-relative sub-path, with a ` (n)` suffix when that name is taken. Nothing
  is deleted, and the reply names `moved_to`.
- The studio work and figure crops stay. Moving the file back to its old path
  brings the document back with its work attached.
- It is refused with `409` while the document has chunks in the index. Remove
  those first, with `POST /api/documents/delete` or the Documents tab.

The Ingest tab's inbox delete is a different operation. `POST /api/inbox/delete`
deletes a staged file from disk for good, with no trash copy; the queue's
remove only ever moves.

### Eval review queue

The labelled questions behind the [eval bench](evaluation.md) start life as
drafts that a person has to check. These endpoints are that review queue:
they read and rewrite the question-set files in the directory named by
`eval.sets_dir` (default `eval/sets`; local data, because it quotes the vault).

| Endpoint | Tier | Behaviour |
|---|---|---|
| `GET /api/eval/progress` | `read` | Per suite: its `quota`, its `total`, and how many questions sit in each review status (`draft`, `verified`, `edited`, `rejected`), plus a `totals` row over the suites. The quotas are `SUITES` in `eval/bench/questions.py`; read them here instead of copying them. |
| `GET /api/eval/questions` | `read` | The queue, one row per question: `id`, `suite`, `tier`, `split`, `status`, `author`, `question`, and `n_errors` / `n_warnings` (the validator's cached findings). Filters: `suite`, `status`, `split`; an empty filter matches everything. Rejected questions are listed. |
| `GET /api/eval/questions/{qid}` | `read` | `{record, gold_texts, findings}`: the question as stored, the text of the chunks its gold locators resolve to, and the validator findings. The last two come from the review cache and are empty lists when it does not cover the question. `404` for an unknown id. |
| `POST /api/eval/questions/{qid}` | `mutating` | Body: `action` (`verify`, `reject` or `edit`) and, for `edit`, the whole replacement `record`. Sets `provenance.status` (`verified`, `rejected` or `edited`) and `reviewed_at`, rewrites that question's whole sets file, and returns `{ok: true, record}`. |

Behaviour worth relying on:

- **The cache is a snapshot.** `findings` and `gold_texts` come from
  `<sets_dir>/.review_cache.json`, written by
  `python main.py bench validate --write-cache`, and are as old as that run.
  Without a cache they are empty, which means "not validated", not "no findings".
- **An edit is validated like a load.** `action: "edit"` sends the whole
  replacement `record`, checked by the same schema the bench loads questions with.
  A record that breaks it is a `400` carrying the schema error, and nothing is
  written. `record` is required with `edit` and refused with anything else.
- **Identity is locked.** An edit cannot change `id`, `suite` or `split` (`400`).
  The split is assigned once by `bench split`, and moving a question between dev
  and test would leak the sealed test set. The server sets only
  `provenance.status` and `reviewed_at`; the record keeps whatever
  `provenance.author` it carries, so an edited draft is still a draft unless the
  editor says otherwise.
- **Writes are atomic and serialised.** The sets file is replaced as a whole
  (a temp file, then a rename) under a lock, so a reader never sees half a file and
  two reviewers cannot overwrite each other with stale copies.
- **Real status codes.** Unlike the query API's retrieval knobs, failures here are
  `400` / `404`, with `{"ok": false, "error": …}` on the write path. A sets file
  the loader rejects (a hand edit with a typo) is a `500` whose error names the
  file, the question and the problem.

Verifying is the human check that makes a drafted question trustworthy, so a
client should confirm each verify, reject or edit with the operator rather than
looping over the queue.

### Which service do I call?

- **Ask, search, compare, inspect evidence, or change warm retrieval defaults**
  → `:8051`.
- **Manage the corpus, persistent settings, jobs, OCR, provider secrets, or the
  eval review queue** → `:8052`.

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
