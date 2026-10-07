# Evaluation

Two harnesses live in this repo, built for different questions:

| | Measures | Run with |
|---|---|---|
| **The bench** (`eval/bench/`) | Retrieval, per pipeline *configuration*, against labelled questions whose gold is a source locator. Built to say which *component* helps, and by how much. | `python main.py bench …` |
| **The golden suite** (`eval/golden_queries.yaml`) | Retrieval, answer and calibration tiers over a fixed query list. Cheap regression checks. | `python main.py eval …` |

The bench supersedes the golden suite's **retrieval** tier: its questions carry
gold sources, it varies the pipeline per call, and every run records which
pipeline it measured. It does not yet cover what the golden suite's answer and
calibration tiers cover (generated answers, citations, confidence), so those stay
on the golden suite until the bench grows an answer tier.

Neither harness certifies that an answer is correct. Both report what they
measured and over how many questions.

## The bench

### What it is for

The pipeline is a stack: lanes, fusion, a reranker, optional context expansion,
generation. Each layer's ceiling is set by the one below it, so the bench
measures bottom-up and **per configuration**. A configuration is a set of
per-call overrides to `RAGPipeline.search()`: which lanes run, whether the
metadata boost, the reranker and HyDE are on, and so on. One pipeline is built
in-process and varied per call, so the index and models load once and no HTTP
latency sits in the numbers. (It also means the bench holds a second pipeline in
memory: on a small machine, do not run it beside the warm query API.)

That is what lets it answer what a fixed query list cannot: does a component help
at all, how much of the result is owed to each component, and what does each
cost in latency.

### Suites and tiers

Questions belong to a **suite**, each aimed at one capability, and to a **tier**,
each a rung of difficulty. The suite list and every suite's target size are
`SUITES` in `eval/bench/questions.py`; `bench validate` prints have / target per
suite, so this page does not copy them.

| Suite | Stresses |
|---|---|
| `lexical` | BM25: identifiers, formula names, error strings, acronyms |
| `paraphrase` | The dense lane and HyDE: wording unlike the source |
| `code` | The code lanes and the code preset |
| `scoped` | Scope routing and the metadata boost: the question names a course or area |
| `books` | The PDF lane: page-level precision, OCR'd scans |
| `multihop` | Fusion breadth and synthesis: the answer needs more than one source |
| `canvas` | Canvas chunks and the graph lane |
| `personal` | Daily notes, strategies, the user's own projects |
| `unanswerable` | Abstention: topics adjacent to, but absent from, the corpus |

| Tier | Meaning |
|---|---|
| `T1` | One chunk, close to the source's wording. |
| `T2` | Paraphrased, or the right source must be picked from similar ones. |
| `T3` | Two or more sources or sections. |
| `T4` | Needs iteration: a follow-up search to resolve a reference. |

Every suite is re-run whenever a component changes. A suite aimed at a component
shows where it should help; the other suites catch what it breaks.

### A question record

One YAML record per question, in `eval/sets/<suite>.yaml` (the directory is
`eval.sets_dir`). The schema, its validation and the loader are in
`eval/bench/questions.py`:

```yaml
- id: lex-0007            # lowercase prefix + four digits; never reused
  question: "…"
  suite: lexical
  tier: T1
  split: dev              # dev | test; null until `bench split` assigns it
  answerable: true        # false: the right behaviour is to abstain, and gold is empty
  gold:                   # where the answer lives, as LOCATORS
    - file: "<vault-relative path>"
      pages: [212, 213]             # optional: inclusive PDF page range
      heading: "Ch 4 > Bisection"   # optional: heading-path prefix
      required: true      # multihop: every required source must be found
  nuggets: ["…"]          # atomic facts a complete answer contains
  expect_course: null
  provenance:
    author: draft         # draft | owner (written by the vault's owner)
    status: draft         # draft | verified | edited | rejected
    reviewed_at: null     # drafts also carry drafted_by and seed_chunks
  notes: ""
```

These decisions are worth knowing:

- **Gold is a locator, never a chunk id.** Chunk ids change with every re-chunk or
  embedding swap, which are exactly the components the bench exists to ablate. A
  chunk is relevant to a gold entry when its file matches (separators, case and
  Unicode form are normalised) and, where given, its page range overlaps `pages`
  or its heading path starts with `heading`. A chunk with no pages (markdown,
  notebooks, code, canvases) never satisfies a paged gold entry: no page is
  guessed. The rule is `eval/bench/relevance.py`, and a document-level twin (file
  match only) is reported beside every chunk-level number.
- **Some suites' gold is effectively file-level.** A notebook's or a canvas's chunks
  share one heading path (the file's own), so a heading locator there matches every
  chunk of the file: chunk-level scores for `code` and `canvas` (and canvas-based
  `personal` questions) measure finding the right file, not the right cell or node.
  Read them beside the document-level numbers; finer locators (cell ranges, canvas
  node ids) are future work.
- **The split is stored, not computed.** It is assigned once and written into the
  YAML, so adding questions never reshuffles existing ones.
- **One writer at a time.** The drafter, the console's review save and `bench split`
  each re-read and rewrite a sets file under a shared lock file (`<sets>/.sets.lock`),
  so reviewing while a draft run goes on loses nothing, and two drafters never reuse
  an id.
- **Validation is strict and loud.** A bad record raises an error naming the file,
  the question id and the problem; it never scores as a quiet zero. The schema
  check does not know the corpus; whether a gold file exists is `bench validate`'s
  job.

### From corpus to a verified question set

1. **Sample.** `python main.py bench sample [--suites …] [--seed N] [--factor F]`
   draws deterministic, stratified **seed packs** from the chunk files into
   `eval/seeds/<suite>.jsonl`. A pack is spread over domains, courses and files
   instead of clumping in the biggest one, is over-drawn by `--factor` so a
   drafter can skip unusable seeds, and never hands out a seed that an existing
   question already used. `multihop` seeds are related chunks from different files;
   `unanswerable` gets topic neighbourhoods to avoid rather than seeds to draft from.
2. **Draft.** `python main.py bench draft --suite S [--n N] [--provider P] [--model M]`
   drafts from that suite's seed pack with a model from the provider registry
   (`freellmapi` unless `--provider`). The model writes the question, its nuggets,
   a tier and a note. It never writes gold: the locator is read off the seed chunk
   (a PDF's pages, a note's heading path), and twins (the same passage in another
   file) come from the warm query API, so a model cannot invent a file or a page.
   Where a suite fixes the tier by construction the rubric overrides the model
   (`multihop` is T3 or T4, `paraphrase` at least T2). Every record must pass the
   schema and the validator's checks, with one retry that carries the finding back
   to the model; a seed that still fails is skipped, and asked again on the next
   run. Records land in `eval/sets/<suite>.yaml` with `split: null`,
   `author: draft` and `status: draft`, and the run is resumable: it stops when the
   suite holds its quota (or `--n`), and a re-run continues the ids. For
   `unanswerable` the model proposes questions next to a course's topics, and one is
   kept only when `/search`'s best score for it stays below `--max-score`.
   Questions a person writes use the same schema with `author: owner`.
3. **Validate.** `python main.py bench validate [--suites …] [--write-cache]` checks
   every question against the corpus. *Errors* (exit code 1): `gold-file-missing`
   and `locator-empty`, either of which scores zero in *every* configuration and
   would masquerade as a retrieval failure. *Warnings*, which a human decides:
   `nugget-unsupported`, `copied-phrasing` (not for `lexical`, whose point is the
   exact term) and `near-duplicate`. It also prints the quota table.
   `--write-cache` writes `<sets>/.review_cache.json` (gold chunk text plus
   findings) for the review queue; it covers the whole sets directory, so it
   cannot be combined with `--suites`.
4. **Review.** A person verifies, edits or rejects each draft through the
   [review API](api.md#eval-review-queue), and may replace any question with their
   own. Each run row records the question's `author`, so drafted and hand-written
   questions can be told apart.
5. **Split.** `python main.py bench split` gives every live question that has no
   split a `dev` or a `test` one, and stores it. A deterministic balancer works
   within each suite × tier stratum, breaking ties by a hash of the id, so the
   outcome depends only on the ids. Existing splits are never touched, a rejected
   question takes none, and re-running it after adding questions only fills gaps.

Question sets, seed packs, the review cache and run records quote the user's notes
and name their files. They are local data and stay out of the public release
(`eval/runs/`, `eval/seeds/` and the review cache are gitignored).

### Running configurations

```bash
python main.py bench run [--sets DIR] [--suites …] [--split dev|test|all] \
    [--unseal-test] [--configs SPEC] [--limit N] [--out-root DIR] [--verbose]
```

`--configs` takes one of the **generated** sets, or comma-separated **named**
configurations:

| Spec | Runs |
|---|---|
| `ladder` (default) | One rung per added component, cumulative: BM25 alone (`R0-bm25`) and dense alone (`R1-dense`), then hybrid (`R2-hybrid`), then + scope routing and the metadata boost (`R3-routing`), + the code lanes and code preset (`R4-code`), + the cross-encoder (`R5-rerank`), + HyDE (`R6-hyde`, the shipped default). |
| `loo` | **Leave-one-out.** The full pipeline (`full`), then the full pipeline minus each factor in turn (`-sparse`, `-dense`, `-routing`, `-code`, `-rerank`, `-hyde`). |
| `factorial` | **Every subset** of the factors, named `F:<factors joined by +>` (`F:none` is the empty set). A subset with neither dense nor sparse has no lane to search and scores 0 without searching. |
| `name[,name…]` | Hand-written configurations from `eval/configs.yaml`. |

The generated sets are built from the same binary factors (`FACTORS` in
`eval/bench/configs.py`: `sparse`, `dense`, `routing`, `code`, `rerank`, `hyde`) and the
ladder is `LADDER` in the same file, so they cannot drift apart and the YAML needs no
entry for them.

Every configuration is scored on the same footing (`EVAL_DEFAULTS` in the same
file): a fixed final-list depth, with `omnisearch`, `hype`, `parent_context` and
`neighbor_context` off. Omnisearch queries the live vault, hype covers only part
of the corpus, and parent/neighbor context rewrite the very chunks the metrics
judge; leave any on and two configurations stop being comparable. These are the
eval protocol, not pipeline tunables, so they live with the harness and not in
`config.yaml`.

Named configurations are `search()` overrides merged over those defaults. The
file is the authoritative list; add a hand-written variant by adding an entry. The
ones that exist for this page's purposes:

| Name | What it is |
|---|---|
| `default-k10` | The shipped pipeline as configured, scored at the eval depth. |
| `wide-pools` | Bigger candidate pools: does that help the reranker? |
| `gate-rerank-score` | The [relevance gate](api.md#relevance-gate) on the reranker's own score: the cheap baseline a learned gate must beat. |
| `gate-laya` | **Experimental.** The gate on Laya's P(relevant). Needs `retrieval.laya.enabled`, the `laya` package and a checkpoint. |
| `laya-rerank` | **Experimental.** `rerank: laya`. Same prerequisites. |

How a run behaves:

- The HyDE disk cache is always attached (`retrieval.hyde_cache`), so two runs
  retrieve with identical expansion text; a question's first HyDE draft calls the
  generation backend, and later runs replay it. The row records `hyde_cache`.
- One discarded warm-up search per distinct rerank mode runs before anything is
  timed, because the first embed and the first rerank in a process pay one-off costs.
- `--split dev` is the default. `test` and `all` are sealed, see below.
- `--limit N` keeps the first N questions of the split, for smoke runs;
  `eval/smoke/` holds a few hand-checked questions for exercising the runner
  (`--sets eval/smoke`).
- A question that fails is a **row**, not an abort: its `error` is recorded, the
  summary counts it, and the command exits 1. A mistake in the command (an unknown
  suite, a missing sets directory, a sealed split, nothing to run) prints
  `ERROR: …` and exits 2. A component that is switched off is never quietly
  swapped: a run that asks for `laya-rerank` while Laya is disabled fails with the
  refusal text instead of scoring another reranker.

### Metrics and statistics

Per question and configuration, over the final list: `hit@1`, `hit@5`, `hit@10`,
`recall@10`, `complete@10`, `mrr@10`, `ndcg@10`, `p@5`, and the document-level
`doc_hit@5` and `doc_recall@10` (`eval/bench/metrics.py`). Gain is counted per
**required gold source**, not per chunk: any number of chunks of one book earn one gain, so
a retriever cannot inflate nDCG by filling the list with near-duplicates.
`required: false` marks optional context that does not count toward recall,
completeness or nDCG. Only answerable questions are scored.

- Every mean carries a 95% percentile-bootstrap interval with a **fixed seed**
  (recorded in `run.json`), so re-running the same rows reproduces the interval.
- A-versus-B comparisons are **paired**: the same questions, and the interval is on
  the per-question difference, which has far less variance than two independent
  means. Binary metrics also get an exact McNemar test. A ladder step is called
  real only when its paired 95% interval excludes zero, and the report says so per
  step.
- Each suite reports its `n`. A suite scored on fewer questions than `DIAG_BELOW`
  (`eval/bench/report.py`) is marked *diag*: a diagnostic, not a result.
- Latency is per stage (p50 / p95 / p99), over **warm** rows only. A row that had to
  load the index or a model is `cold`, and is counted beside the table instead of
  averaged into it.

### Attribution: ladder, leave-one-out, factorial

- The **ladder** is one ordering of the components: each rung adds one thing, so the
  paired step between neighbours is that component's gain *at that point*.
- **Leave-one-out** is the converse: what does the full pipeline lose when one
  component is removed?
- The **factorial** scores every subset. Given a value function over subsets (the
  design's is mean dev nDCG@10), `shapley()` in `eval/bench/stats.py` computes each
  factor's **Shapley value**: its marginal contribution averaged over every order
  the factors could be switched on, an order-independent share where the ladder
  gives only one order.

What exists today: the runner scores each of these sets and records every
configuration's numbers in `summary.json`, and the report renders the ladder's
paired steps. **`bench report` does not yet render leave-one-out deltas or Shapley
values.** `shapley()` is implemented and tested, and a factorial run's `summary.json`
holds the per-subset scores it needs.

### The sealed test split

Tuning happens on **dev**. The **test** split is sealed: `--split test` and
`--split all` (which contains it) refuse to run without `--unseal-test`, and a
refused run writes nothing. Every opening is appended to `eval/test_ledger.jsonl`
(time, run id, split, configurations, git state) **before the first search**, so
the ledger records that a run started even if it never finished. A run record
stores how many entries the ledger held (`test_ledger_count`), and the report
header prints "test split opened N time(s)", so a reader can see how many looks
the test numbers have had.

The point is procedural. Choose configurations and thresholds on dev, then look at
test once; never add `--unseal-test` to see whether something "holds" before it is
decided what should hold, and never edit the ledger.

### Run records and reports

Every run writes a new directory, `eval/runs/<run_id>/`, where the id is a
timestamp plus the short git sha. It is gitignored because it quotes the vault. A
directory is never reused and a record is never overwritten.

| File | Holds |
|---|---|
| `run.json` | What was measured: git sha and a dirty flag, a digest of the effective config, the **index fingerprint** (collection and BM25 counts, build time, models, a digest of the chunk files), a digest of the question set, the split, suites, each configuration with its overrides, the bootstrap seed and timestamps. Written first with status `running`, rewritten at the end as `done`, or as `failed` with the reason; a crash never leaves a stale `running`. |
| `per_query.jsonl` | One row per (question, configuration), flushed as the run goes: the retrieved ids and files, which gold sources each rank matched, the metrics, `timings`, `lanes_run`, `cold`, `hyde_cache`, or an `error`. |
| `summary.json` | Per configuration: `overall`, `by_suite` and `by_tier` metrics (mean, interval, n), warm latency percentiles per stage, cold-row and error counts, and a `paired` comparison against the previous ladder rung where both ran. |
| `report.md` | The scorecard, rendered from `summary.json` and `run.json`. `python main.py bench report <run_dir>` re-renders it exactly, so a layout change never needs a re-run. |

### The relevance gate and the Laya reranker

Both are optional, **off by default** and unmeasured: no result for either is
claimed anywhere in this repo. They are the first components the bench is meant to
judge, under the same rules as any other.

- **Relevance gate.** Whether a cutoff after the rerank helps depends on its
  threshold, and a threshold means something only for one scorer. The named
  configurations `gate-rerank-score` and `gate-laya` carry *starting* thresholds,
  not tuned ones. The intended protocol is a sweep on the **dev** split, with the
  chosen threshold then reported once on test. A sweep today is several named
  configurations that differ only in `gate_threshold` (add entries to
  `eval/configs.yaml`), run together so each is scored on the same questions. What
  the bench does not have yet are the metrics a sweep is really about: the share of
  retrieved gold chunks the gate removed, and abstention accuracy on `unanswerable`.
  Until they exist, a gate's cost shows up as lost recall and nDCG on answerable
  questions, its benefit (abstaining correctly) is not measured, and the bench
  scores retrieval only, so the gate's effect on a generated answer is outside it.
  The per-row record does not keep the `gate` echo either.
- **Laya reranker.** Compare it with the default on dev:
  `python main.py bench run --configs default-k10,laya-rerank --split dev`. The
  graduation rule is the design spec's §11 and applies to any experimental
  component (graph mode, a new embedding model, Laya): it graduates only if it beats
  the current default on dev with a paired interval that excludes zero, and then
  holds on test. Until then it stays experimental and off by default. The manual
  for producing a checkpoint is `docs/laya-finetune.md`; this page covers how to
  judge one.

The same section of the spec sets the rule for components that are already on: one
stays on by default only if its leave-one-out delta on test is not negative (its
paired interval not entirely below zero) and no adequately sized suite regresses
beyond the tolerance the spec sets.

### What exists, and what does not yet

Built: `sample`, `validate` (with the review cache), `split`, `run` (generated and
named configurations) and `report`, plus the review API.

Specified in the design but **not built**: answer-side metrics and a pinned judge
(nugget recall, faithfulness, abstention accuracy, calibration); isolation probes
(feeding a component oracle input from the layer below); per-question failure
attribution; `bench compare`; rendering of leave-one-out and Shapley results; a
stage cache that would make the factorial cheap (today each configuration runs the
whole question set); and console dashboards with a job kind for runs.

The design and the plans behind this live in `docs/superpowers/` (local-only, not
part of the public release).

## The golden suite (v1)

Quality is also measured with the labelled suite in `eval/golden_queries.yaml`. Read that
file for the current cases and coverage; do not rely on a copied query count. Entries
can label expected domains, groups, source files, scope routing, keywords, and gold
answers independently, so missing labels drop out of the corresponding metric instead
of becoming false zeroes.

```bash
python main.py eval --retrieval-only # retrieval tier only; no generation backend
python main.py eval                  # retrieval, answer, and calibration tiers
python main.py eval --judge          # add the advisory LLM-judge fields
```

### What each tier measures

| Tier | Measures |
|---|---|
| Retrieval | hit-rate@k, MRR, labelled-source recall@k, scope precision/recall, and group routing |
| Answer | keyword recall, citation validity, deterministic groundedness floor, optional citation-auditor support, and answered rate |
| Calibration | whether stated confidence tracks keyword recall, groundedness, and optional judge correctness |

`--judge` adds advisory correctness and groundedness scores. A gold answer enables
reference-based correctness; without one, in-process judge correctness is not treated
as a meaningful score. Judge failures remain visible in the report.

### Read these honestly

The automatic tiers are **proxy metrics**, and they are framed that way on purpose:

- **Keyword recall** checks that expected terms *appear* — not that the explanation is
  correct.
- **Retrieval hit** checks that the expected *domain* landed in the top-k — not that the
  exact passage did. (The bench's gold locators do check the passage.)

They exist to **catch regressions** cheaply and repeatedly. They do **not** certify
answer faithfulness. The LLM judge is advisory too. Three other mechanisms help, and
none replaces reading the cited source:

1. the **second-pass citation auditor**, which checks each citation supports its sentence;
2. the **per-answer confidence line**; and
3. the citations themselves, which point straight back to the source.

### What misses tell you

A miss can be informative rather than a bug. Cross-domain questions may retrieve the
domain where the material actually lives, and a small code artifact can lose to prose
that repeats the same terms. Inspect the per-query rows before tuning a global default.

### Reproducibility

- `--retrieval-only` removes the LLM from the loop, so retrieval numbers are stable and
  fast to regenerate.
- For generation-side numbers, pin `generation.model` so runs are comparable.
- Each run writes structured and Markdown reports; compare them explicitly rather than
  hardcoding a copied baseline in documentation.
- `--judge-export` / `--judge-import` allow an external model to grade a complete
  bundle without giving that model access to the vault or local service.
