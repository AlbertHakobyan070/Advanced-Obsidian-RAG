# Evaluation

The repo ships a **small illustrative golden suite** to exercise the runner
out of the box. Replace it with your own set for serious regression work —
the runner and metrics are the same.

## What ships in the box

`eval/golden_queries.yaml` contains six queries across three shapes:

| Shape | Count | What it tests |
|---|---|---|
| `concept` | 3 | Clean concept questions; expected terms live in textbook prose. |
| `exam-like` | 2 | Chapter-review / final-style questions (no exam leakage; the same shape would appear in any textbook). |
| `code` | 1 | Code-shape query using the `code` retrieval preset. |

For each query, you can supply:

- `expect_keywords` — terms that should appear in the top-k retrieved chunks.
- `domain` — the `domain` metadata value you expect in top-k (lets the runner
  score `retrieval_hit`).
- `expect_course` — (optional) the `course_name` metadata value the #1 source
  should carry. Useful only after you've populated `parser.course_taxonomy`
  in `config.yaml` for your corpus.
- `preset` — `"code" | "concept" | "synthesis"`, defaulting to the pipeline
  default.

## Running it

```bash
python main.py eval                  # full run (retrieval + generation)
python main.py eval --retrieval-only # offline retrieval regression, no LLM, runs in minutes
```

## What the metrics mean

| Metric | What it measures |
|---|---|
| **Keyword recall** | Fraction of `expect_keywords` that appear in the top-k chunks (or the answer, in the full run). |
| **Retrieval hit** | Whether the expected `domain` metadata value landed in the top-k. |
| **Course hit** *(optional)* | Whether the #1 source's `course_name` metadata matches `expect_course`. |
| **Citation support** *(full run)* | Fraction of cited sources the second-pass auditor marked supported. |
| **Confidence / Answered** *(full run)* | Distribution of HIGH/MEDIUM/LOW and the fraction of non-punting answers. |

## Read these honestly

These are **automatic proxy metrics**, and they're framed that way on purpose:

- **Keyword recall** checks that expected terms *appear* — not that the
  explanation is correct.
- **Retrieval hit** checks that the expected *domain* landed in the top-k —
  not that the exact passage did.

They exist to **catch regressions** cheaply and repeatedly. They do **not**
certify answer faithfulness. Three other mechanisms do that job, and none
replaces reading the cited source:

1. the **second-pass citation auditor**, which checks each citation supports
   its sentence;
2. the **per-answer confidence line**; and
3. the citations themselves, which point straight back to the source.

## What the remaining misses tell you

A handful of "misses" can be *informative*, not bugs. For example,
cross-domain questions (a technique taught in one area but asked about in
another's language) legitimately retrieve the domain where the material
actually lives, and single-chunk artefacts can lose to longer prose that
repeats the same keyword. Document these as expectations rather than
silently tuning them away, so the score stays a faithful signal.

## Reproducibility

- `--retrieval-only` removes the LLM from the loop, so retrieval numbers are
  stable and fast to regenerate.
- For generation-side numbers, pin `generation.model` so runs are comparable.
- Baseline result files (`eval/results.json`) are written next to the suite,
  so any regression shows up as a diff in version control.

## Writing your own golden set

For a real regression suite, write 30-100 queries drawn from *your* corpus
— questions you actually ask, with `expect_keywords` and `domain` taken
straight from the source material. The runner doesn't care how many
queries you have; the harness is the same.
