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
| `lanes` | Restrict the call to a subset of the lanes (names under [Lane weights](#lane-weights)), e.g. `["sparse"]`: a lane outside the list is not run at all. It only restricts — a conditional lane (code, scope, `omnisearch`, `hype`) still needs its own trigger. An empty list or an unknown name is an error ([below](#when-a-knob-is-wrong)); the echo's `lanes_requested` and `lanes_run` show what ran. |
| `metadata_boost` | `true` / `false` forces the course/domain/tag metadata boost on or off for this call; unset follows `retrieval.metadata_boost`. |
| `gate` / `gate_threshold` / `gate_scorer` | The post-rerank **relevance gate**: drop chunks scoring below a cutoff before generation, and abstain when nothing is left. Off by default; a `gate_threshold` alone turns it on for the call. See [Relevance gate](#relevance-gate). |
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
configured one without its own `gate_threshold`.

A wrong **type** is different: `top_k: "ten"`, `lanes: "sparse"` (a string, not
a list), a non-numeric `gate_threshold` — or a number outside a field's
documented range — is rejected by request validation before retrieval runs, as
FastAPI's usual `422` with a `detail` list.

Two habits follow. Read `error` / `confidence` before reading `results` or
`sources`: an empty list that comes with an `error` is a rejected request, not
a gap in the corpus. And change the knob rather than resending the same body.

The rest keep their real status codes. [`/graph/expand` and `/answer`](#graph-mode)
answer `400` for a bad seed or document list, and `/compare` answers `400` for
duplicate branch ids or too many query-mode branches. A generation failure on
`/query` is also reported in the body (`confidence: "ERROR"`), but keeps the
`sources` and `retrieval` that were already retrieved.

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
| `cold` | `true` when this call loaded the Chroma collection, the BM25 payload, the cross-encoder or the Laya checkpoint. `/health` reports `ready` before any of them loads, so the first search after a start pays for them: its `timings` are a one-off spike, not steady state. |
| `hyde_cache` | Why the HyDE text is what it is: `off` (HyDE not used on this call), `bypass` (a code-intent signal skipped it), `hit` (draft replayed from the cache, no LLM call), `miss` (written fresh and stored), `nocache` (written fresh, no cache attached), `error` (the LLM failed, so the raw query was used). |
| `gate` | What the relevance gate did: `{"enabled": false}` when it did not run, otherwise `{enabled, scorer, threshold, kept, dropped, abstained}`. See [Relevance gate](#relevance-gate). |

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

Fusion is Reciprocal Rank Fusion over **eight** lanes, not two:

| Lane | When it opens |
|---|---|
| `dense` / `sparse` | Always. Vector and BM25. |
| `dense_code` / `sparse_code` | On code-intent queries, filtered to the configured code file types. |
| `dense_scope` / `sparse_scope` | When the query names a domain or a content type, filtered to that scope. |
| `omnisearch` | When the live-vault lane is enabled and reachable. |
| `hype` | When the hypothetical-question collection exists and the lane is on. |

A document's fused score is the sum over lanes of
`weight(lane) / (rrf_k + rank_in_that_lane)`. Every weight defaults to `1.0`,
which is plain unweighted RRF — an untouched config ranks exactly as it did
before weights existed.

Set defaults in `retrieval.lane_weights`, per preset inside
`retrieval.presets.*`, or per call. All three **merge lane by lane**, so setting
one lane leaves the other seven where they were.

A weight of `0` is not an off switch: the lane still contributes candidates for
the reranker to consider, it just stops influencing the fused order. To drop a
lane, turn the lane off.

An unknown lane name or a negative weight is an error, never a setting that
silently does nothing: it comes back as HTTP 200 with an `error` field (see
[When a knob is wrong](#when-a-knob-is-wrong)), and a non-number is a `422`. The
retrieval echo reports every lane's effective weight, so the ones you did not
override are visible too.

!!! tip "Tune these against an eval, not by feel"
    `main.py eval --retrieval-only` runs the golden set without a generation
    backend, and `main.py bench run` scores labelled questions per named
    configuration; a configuration in `eval/configs.yaml` can set `lane_weights`
    like any other search override. See [Evaluation](evaluation.md).

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
    tests, but its retrieval benefit has never been scored against the golden
    set, so it is not part of the default workflow. While the flag is off,
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

The response carries `nodes` (each with its hop depth and graph metadata),
`tree` (parent → child with each edge's label and direction), `cross_edges`
(edges back onto nodes already reached), and `stats`.

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

**`POST /answer`** — generation over a document set the caller chose. No
retrieval runs.

| Field | Effect |
|---|---|
| `q` | The question these documents should answer. |
| `docs` | 1–100 chunk ids. An unknown id is a `400`, never a quietly smaller context: an answer grounded on fewer documents than you chose is a different answer. |
| `provider` / `model` / `max_tokens` / `include_text` / `max_sources` | As on `/query`. |

Evidence ids that are not indexed records — `live:` excerpts and `parent:`
sections — cannot ground an answer and are rejected by name.

This shape is what makes the human-in-the-loop fork work without server
session state: *answer from the graph alone* and *merge with the previous
result* are the same call with a different id list, so "what counts as the
previous stack" stays with the caller.

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

Endpoints covering the whole corpus lifecycle, plus the eval review queue.
`GET /api/schema` returns them with a **permission tier** on every operation; read
the live list rather than counting on this page.

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
| `GET /api/eval/progress`, `GET /api/eval/questions`, `GET /api/eval/questions/{qid}` | The eval review queue: per-suite progress, the filtered question list, and one question with its gold texts and validator findings. See [Eval review queue](#eval-review-queue). |

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
| `POST /api/eval/questions/{qid}` | Verify, reject or edit one eval question and rewrite its sets file. See [Eval review queue](#eval-review-queue). |

### Remove things — `destructive`

| Endpoint | Purpose |
|---|---|
| `POST /api/documents/delete` | Remove documents from the index. |
| `POST /api/inbox/delete` | Remove staged files. |

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
