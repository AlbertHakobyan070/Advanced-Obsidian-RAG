# Agent integration

This system was built to be driven by a program as much as by a person. Both
services are plain JSON over HTTP, both publish a live capability schema, and
the management surface labels every operation with a permission tier so a
toolkit can gate it.

If you are wiring an LLM agent, a script, a bot, or a CI job into this, start
here. The endpoint-by-endpoint listing is in the [API reference](api.md).

## Discover, don't hard-code

Presets, branch limits, provider names, job kinds and taxonomy all come from the
operator's config. Fetch them instead of baking a copy into a prompt that will
quietly go stale:

```bash
curl -s http://127.0.0.1:8051/schema           # query capabilities + live presets
curl -s http://127.0.0.1:8051/compare/options  # ready-to-post comparison branches
curl -s http://127.0.0.1:8051/providers        # configured generation backends
curl -s http://127.0.0.1:8052/api/schema       # management ops + permission tiers
```

Pair the two maps. **Ask, search, compare, inspect evidence, or change warm
retrieval defaults** → `:8051`. **Manage the corpus, persistent settings, or
provider secrets** → `:8052`.

## `/search` is the primary call, not `/query`

An agent that is already a strong reasoner usually wants the *raw material*, not
a paraphrase of it.

| | `POST /search` | `POST /query` |
|---|---|---|
| Returns | reranked chunks + labels + text | a written, cited answer |
| Needs a generation backend? | **No** — fully local | Yes |
| Who reasons over it? | **The caller** | The configured generator |
| Token cost | one HTTP call; you set `include_text` | the same retrieval, plus a generation you then re-read |
| Failure mode | "no chunks" | backend outage → `confidence: "ERROR"` |

Paying for a generation pass to summarise chunks you are going to read anyway is
double spend. **Default to `/search`; escalate to `/query`** only when you
deliberately want the system's own grounded synthesis — a cited write-up for a
human, for instance.

```bash
curl -s -X POST http://127.0.0.1:8051/search \
  -H "Content-Type: application/json" \
  -d '{"q":"how is retry backoff implemented","preset":"code","include_text":700,"max_sources":5}'
```

## Five efficiency principles

1. **One warm endpoint, one request.** The server loads the indexes and models
   once and stays warm. Never shell out to the CLI per question (it reloads
   everything, every time), and never drive the browser console for data —
   each screenshot is a vision round-trip for something a single request returns.
2. **Retrieve narrow.** `include_text` of 400–800 characters is usually enough to
   judge relevance, and `max_sources` of 4–6 is enough to triage. Fetch the full
   chunk with `GET /chunks/{id}` only for the one or two you decide to build on.
3. **Retrieve late.** Search at the moment you need a specific thing, using that
   thing's real vocabulary. A speculative search is pure waste.
4. **Cache within the task.** The corpus does not change mid-task. Reuse the
   chunks you already pulled instead of re-querying for them.
5. **Spend generation deliberately.** Explore on `/search` — it returns scores,
   so you can see whether a knob helped. Reserve `/query` with a right-sized
   `max_tokens` for an actual deliverable.

## The grounding loop

```text
for each sub-problem:
  1. FRAME    turn it into a query in the corpus's own vocabulary
  2. SEARCH   POST /search (preset by shape, small include_text, 4-6 sources)
  3. TRIAGE   read labels + scores — is the real source material here?
  4. GROUND   yes -> pull the top 1-2 chunks in full, build on them, cite labels
              no  -> climb the ladder ONCE (scope, preset, expansion, wider pool)
                     still nothing -> answer from general knowledge and SAY SO
  5. VERIFY   for each non-trivial claim, point at the chunk backing it;
              drop or re-search anything unsupported
```

The discipline that makes this trustworthy rather than noisy: **ground only in
what actually came back.** If retrieval returns nothing on topic, say the corpus
does not appear to cover it and keep the two kinds of answer visibly separate.
A fabricated citation costs more trust than an admitted gap.

### Choosing a preset by question shape

| Shape | Preset | Why |
|---|---|---|
| "How was X implemented here?" | `code` | Wide pools, HyDE off, code-boosted — surfaces real scripts and notebooks over prose that merely mentions X. |
| "What is the technique for X?" | `concept` | Tight top-k plus HyDE — grounds an approach in the corpus's explanation of it. |
| "Connect X and Y" / "the whole approach to Z" | `synthesis` | Wide, HyDE on, parent/neighbour context — spans several documents. |
| Unsure | *omit* | Code-intent queries self-apply the `code` preset. |

Naming a language or library in the query (`sql`, `pytorch`, `ggplot`) is itself
a routing signal: it opens the code lane and turns HyDE off.

## When one search is not enough

Do not relay a weak result — branch:

- **Decompose** a multi-part request into its parts and search each.
- **Re-frame** with the vocabulary the corpus actually uses, not the user's.
- **Widen** once: raise `dense_top_k` / `sparse_top_k`, or turn on
  `parent_context` for more surrounding text.
- **Compare** if you genuinely do not know which setting is right —
  `POST /compare` runs the branches in one call and reports overlap and rank
  shifts instead of leaving you to diff two responses by eye. Call
  `GET /compare/options` first: it returns branch objects you can post as-is,
  the real branch caps, and the reason any backend is unavailable, so you never
  have to join `/schema` and `/providers` by hand.
- **State the criterion** rather than re-querying, when the results are on-topic
  but the wrong *kind* of thing. `rerank_instruction` ("prefer worked procedures
  over definitions") reorders the pool you already have; it costs no extra
  retrieval and cannot drop anything. Check `rerank_instruction_applied` in the
  echo — `lexical` and `none` ignore it by design.
- Check `GET /history` before re-tuning: it lists what was already tried, with
  the retrieval echo of what each run actually did.

## Permission tiers on the management API

The local API has no authentication. The tiers in `GET /api/schema` are a
contract the **caller** enforces:

| Tier | Rule |
|---|---|
| `read` | Safe. Call freely. |
| `mutating` | Changes local state or invokes a potentially billed external action. Confirm with the operator first. |
| `destructive` | Removes content. Always confirm, echo exactly what will be deleted, never run unprompted. |

Two concrete rules worth stating outright: never write a secret anywhere except
through `POST /api/providers/key` (which accepts only variable *names* the
provider registry already declares, and never returns a value), and never delete
indexed documents without echoing the exact list first.

## Failure handling

Distinguish these — they have different fixes, and conflating them sends the
operator hunting in the wrong place:

| Symptom | Meaning |
|---|---|
| `GET /health` → `state: "loading"` | Indexes/models still loading. Wait. |
| `GET /health` → `state: "failed"` | The pipeline could not be built from the current config. Waiting will not help. `error` names the cause; `GET /api/service/log` on `:8052` has the traceback. |
| Answer begins `Reranking failed:` | A **retrieval** failure. It never reached generation. Check `GET /config` for the active reranker. Do not report it as a generation outage. |
| `confidence: "ERROR"` | The generation backend failed or is unreachable. Retrieval still works — fall back to `/search`. |
| 503 from a retrieval endpoint | The service is up but not ready; the body carries the same reason as `/health`. |

## After changing the index

Anything that changes what is indexed requires the warm query service to reload:

```bash
curl -s -X POST http://127.0.0.1:8052/api/service/restart
curl -s http://127.0.0.1:8051/health        # poll until ready
```

The restart returns a relaunch counter and fails loudly if the service did not
actually come back, rather than reporting a restart that never happened. Setting
`webui.auto_restart_rag: true` does this automatically after every successful
index-changing job.

## Anti-patterns

- **Driving the browser console to read data.** Every capability it has is an
  HTTP call; screenshots cost orders of magnitude more.
- **Calling the CLI per question.** It reloads the whole index each time.
- **Citing the corpus for something retrieval did not return.** The citation
  contract is the product; breaking it silently is worse than saying "not found".
- **Copying preset or provider lists into a prompt.** They are per-install. Fetch
  the schema.
- **Treating a reranker error as a provider outage.** Different subsystem,
  different fix.
