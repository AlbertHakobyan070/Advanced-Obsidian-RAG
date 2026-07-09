# Agentic RAG for Code-Writing — a token-efficient implementation guide

**Audience:** a coding agent (any tool-using LLM with shell + file tools) that
has a personal RAG available and is writing or modifying code. **Goal:**
use the RAG as a *grounding tool inside the build loop* — pulling the user's
own prior implementations, the textbook code, and the relevant technique
notes into the code you write — while spending the fewest possible tokens
to do it.

This guide is about *how an agent wires retrieval into a coding workflow*.
For the endpoint reference, see the `personal-rag` skill (`SKILL.md`); for
corpus management, the `rag-ops` skill (`SKILL_RAGOPS.md`). Read those for
syntax; read *this* for the loop.

---

## 0. The one-paragraph version

When you're about to write code for a sub-problem the user has plausibly
touched before, **`POST /search` the corpus first, reason over the returned
chunks yourself, and ground your implementation in what came back** — citing
the source labels in a comment or your summary. Use `/search` (not `/query`)
so retrieval stays local, free, and returns raw material you reason over
directly; reserve the one generation pass (`/query`) for when you want the
RAG itself to synthesize. Retrieve narrow, retrieve late, cache what you
got, and never screen-scrape.

## 1. Why `/search`, not `/query`, is the agent's primary tool

The RAG exposes two modes. For an agent *that is already a strong reasoner*,
`/search` is almost always the right one:

| | `POST /search` (retrieval only) | `POST /query` (retrieval + generation) |
|---|---|---|
| What it returns | reranked chunks + labels + text | a written, cited answer |
| Needs a generation endpoint? | **No** — fully local | Yes (any OpenAI-compatible endpoint) |
| Who reasons? | **You do**, over the chunks | a second, weaker model does |
| Token cost | one HTTP call, you control `include_text` | same retrieval + a generation you then re-read |
| Failure mode | none but "no chunks" | endpoint-down `confidence:"ERROR"` |

For code-writing you want the *raw material* (the user's actual function, the
textbook's actual algorithm), not a paraphrase of it. `/search` hands you
that. You're a better synthesizer than the free-tier generator, so paying
for `/query` to have it summarize chunks you'll re-read anyway is double
spend. **Default to `/search`; escalate to `/query` only when you
deliberately want the RAG's own grounded synthesis** (e.g. to show the user
a cited write-up).

```bash
# the agent's bread-and-butter call — local, no generation backend needed
curl.exe -X POST http://127.0.0.1:8051/search -H "Content-Type: application/json" \
  -d "{\"q\":\"how did I implement Thompson sampling\",\"preset\":\"code\",\"include_text\":700,\"max_sources\":5}"
```

## 2. The token-efficiency principles (internalize these five)

1. **One warm endpoint, one curl.** The server loads the index once and
   stays warm. Never `python main.py query` (reloads the index per call,
   10-30 s) and never drive the Streamlit/console *browser* (each
   click/snapshot is a vision round-trip). One curl = one grounded result.
2. **Retrieve narrow.** Set `include_text` to what you'll actually read
   (400-800 chars usually shows enough of a function to judge relevance) and
   `max_sources` to 4-6. The defaults return more text than a triage pass
   needs. Fetch the *full* chunk only for the 1-2 sources you decide to
   build on.
3. **Retrieve late.** Don't pre-search speculatively. Search at the moment
   you're about to implement a specific sub-problem, with the sub-problem's
   real vocabulary in the query. A search you didn't need is pure token
   waste.
4. **Cache within the task.** The corpus doesn't change mid-build. Keep the
   chunks you retrieved in working memory and reuse them across related
   sub-problems instead of re-querying the same thing.
5. **Let retrieval be local; spend generation sparingly.** Every `/search`
   is free of the generation budget. Do all your *exploration* on `/search`
   (it returns `score` so you can see if a config helped), and spend a
   `/query` generation pass — with a right-sized `max_tokens` — only for a
   deliverable write-up. Small `max_tokens` for a terse fact; large only
   for real synthesis.

## 3. The core loop — retrieval-augmented code writing

A single sub-problem, expressed as a loop you run per unit of code:

```
for each sub-problem you're about to implement:
  1. FRAME    turn the sub-problem into a retrieval query in the user's vocabulary
              (name the technique + a course/library so scope routing fires)
  2. SEARCH   POST /search  (preset by shape; small include_text; max_sources 4-6)
  3. TRIAGE   read labels + scores; is the user's own prior work / textbook code here?
  4. GROUND   if yes -> read the top 1-2 chunks' full text; write code modeled on
              them; cite the source label in a comment / your explanation
              if no  -> climb the hyperparameter ladder ONCE (scope, preset,
              hype, widen). Still nothing -> write from general knowledge and
              SAY it's not grounded in the corpus (keep the two visibly separate)
  5. VERIFY   self-RAG: for each non-trivial claim/choice, point at the chunk
              that backs it; drop or re-search anything unsupported
```

The discipline that makes this *powerful* rather than noisy: **you only
ground in what actually came back.** If the search returns nothing
on-topic, you don't invent a citation to the corpus — you say "the corpus
doesn't seem to have this; here's a standard implementation" and move on.
Trust is the whole point.

### Preset selection by sub-problem shape

- **"How did *I* do X" / "my implementation of X"** -> `preset:"code"` (wide
  nets, HyDE off, code-boosted). Pulls the user's notebooks/scripts over
  lecture prose. This is the primary preset for code-writing.
- **"What's the technique/algorithm for X"** -> `preset:"concept"` (tight
  top-5 + HyDE). For grounding an approach in the textbook's *explanation*
  before you code it.
- **"Connect X and Y" / "the whole approach to Z"** -> `preset:"synthesis"`
  (wide + HyDE + parent/neighbor context). For designing something spanning
  several chapters or notes.
- Omit the preset for auto: code-intent queries self-apply `code`.

### Framing queries so retrieval finds the right material

- Put the **library/language** in the query (`ggplot`, `sklearn`,
  `pytorch`, `sql`, `dplyr`) — these are scope + HyDE-skip signals; they
  route to the code lane and turn HyDE off automatically.
- Use **"my" / a course name** for personal-work recall (e.g. "my thesis",
  "my NLP homework") — the retriever boosts course/domain matches when
  you've populated `parser.course_taxonomy` for your corpus.
- For a raw-script language now covered by `ingest-code` (JS/TS/SQL/Go/…),
  name the language ("my sql schema", "the typescript handler") —
  `code_scripts` scope routing points at those chunks.

## 4. When one search isn't enough — multi-query & decomposition

If the sub-problem is multi-part, or a first `/search` is thin, **don't
relay a weak result — branch**, exactly like the self-RAG loop in
`SKILL.md`:

- **Decompose.** "Build a cached retry wrapper with backoff" -> search
  ("my retry/backoff code"), ("caching decorator I've written"),
  ("exception handling pattern") separately; merge the distinct chunks;
  compose from the union. Each `/search` is local and cheap.
- **Query-variation.** Same intent, 2-3 phrasings/synonyms ("regularization"
  / "weight decay" / "L2 penalty"), then dedupe by source label. Catches
  material written in different vocabulary than your first guess.
- **Ladder a single leg.** For one stubborn leg, escalate cheapest-first:
  add scope words -> switch preset -> raise `top_k` -> flip `hyde` ->
  `hype:true` -> `parent_context:true`. Stop as soon as the top chunks
  are on-topic.

Budget rule of thumb: 2-4 `/search` calls to fully ground a non-trivial
feature is normal and cheap; if you're past ~5 without the material
surfacing, it's a corpus-coverage gap — write from general knowledge
(labeled) or, if it's content the user clearly *should* have, offer to
ingest it (next section).

## 5. Closing the loop with `rag-ops` (when the gap is missing content)

Sometimes the right move isn't more tuning — it's that the material
genuinely isn't in the corpus yet. That's the `rag-ops` skill's job
(management API on `:8052`, permission-gated):

- **Confirm it's absent** with `GET :8052/api/documents?q=...` or
  `GET :8052/api/vault/search?q=...` (is the file even in the vault but
  unindexed?).
- If it's an unindexed vault file -> offer to ingest it (a `mutating` op:
  ask first), then re-search after the append job + a `:8051` restart.
- If it's brand-new code you just wrote and the user wants it retrievable
  later -> `ingest-code` on that path.

Don't reflexively ingest mid-build — usually you just retrieve and move
on. But knowing the ops path exists means "the corpus doesn't cover
this" can become "...so I added it," when that's what the user wants.

## 6. A worked example

**Task:** "Add a Thompson-sampling bandit to pick between two email
subject lines." The user has a Bayesian A/B testing chapter / lecture in
their corpus.

```
FRAME    "how did I implement Thompson sampling with a Bernoulli reward"
SEARCH   POST /search {preset:"code", include_text:700, max_sources:5}
TRIAGE   top hit (score ~3.9): "Bayesian-Ab-Testing > Thompson Sampling with
         Bernoulli Reward, p.178" - the user's own note. Second: a
         Beta-Binomial conjugate-prior chunk. That's the exact prior art.
GROUND   read those two chunks in full; implement the Beta(alpha,beta)
         posterior update + argmax-of-samples selection modeled on the
         formulation; comment the code with "// modeled on the corpus's
         Bayesian A/B Testing notes, p.178"
VERIFY   the conjugate update in the code matches the chunk's
         p(theta|X) propto p(X|theta)p(theta); the sampling step is
         standard (no chunk needed) - note that separately
DELIVER  optionally, ONE /query {q:"summarize my Thompson sampling approach",
         max_tokens:400} to hand the user a short cited write-up of what
         you based it on - not to write the code (you already did,
         grounded).
```

Total generation spend: zero for the code itself (all `/search`), one
small `max_tokens` `/query` only for the human-facing summary. That's the
pattern.

## 7. Anti-patterns (each one burns tokens or trust)

- **Screen-scraping the Streamlit page or the console UI.** Every
  snapshot is a vision round-trip. There is always a JSON endpoint - use
  it.
- **`python main.py query` in a loop.** Reloads the index each call. Use
  the warm `:8051` endpoint.
- **`/query` when you only need chunks.** Pays a generation pass to
  summarize material you'll re-read anyway. Use `/search`.
- **Over-fetching `include_text`.** 6000-char sources on a 6-source
  triage pass is ~36 KB of text you skim and discard. Triage at
  400-800; expand only the keepers.
- **HyDE on for a code/exact-term query.** The hypothetical answer biases
  toward lecture prose and buries the actual code. The `code` preset
  turns it off; if you hand-roll, set `hyde:false` for "show me my code"
  queries.
- **Fabricating grounding.** Never cite the corpus for something the
  search didn't return. "Not in your corpus, here's a standard version"
  is the correct, trust-preserving answer.
- **Speculative pre-fetching.** Searching before you know the
  sub-problem's real vocabulary wastes calls and returns off-target
  chunks.

## 8. Quick reference (the calls that matter)

```bash
# health (free, instant) - confirm the warm endpoint is up
curl.exe http://127.0.0.1:8051/health

# ground a code sub-problem (primary agent call)
curl.exe -X POST http://127.0.0.1:8051/search -H "Content-Type: application/json" \
  -d "{\"q\":\"<technique> in my <course/library>\",\"preset\":\"code\",\"include_text\":700,\"max_sources\":5}"

# concept grounding before you code an approach
curl.exe -X POST http://127.0.0.1:8051/search -H "Content-Type: application/json" \
  -d "{\"q\":\"<technique> explanation\",\"preset\":\"concept\",\"include_text\":600}"

# human-facing cited write-up (spend ONE generation pass, right-sized)
curl.exe -X POST http://127.0.0.1:8051/query -H "Content-Type: application/json" \
  -d "{\"q\":\"summarize my approach to <X>\",\"max_tokens\":400}"

# discover the knobs if unsure
curl.exe http://127.0.0.1:8051/schema
```

PowerShell: swap `curl.exe ... -d "{...}"` for
`Invoke-RestMethod -Uri ... -Method Post -ContentType "application/json" -Body '{"..":".."}'`.

**The whole guide in one line:** search local and narrow, ground in what
actually came back, spend generation only on deliverables, and never
pretend the corpus said something it didn't.
