# Evaluation

Retrieval quality is measured with a **94-question, exam-grounded golden suite**
(`eval/golden_queries.yaml`) scored automatically. The suite mixes conceptual questions
across all nine domains, questions mined from real exam material, and own-code /
problem-situation queries.

```bash
python main.py eval                  # full run (retrieval + generation)
python main.py eval --retrieval-only # offline retrieval regression, no LLM, runs in minutes
```

## Current baseline

| Metric | Result |
|---|---|
| Retrieval hit-rate (expected domain in top-k) | **~96%** |
| Keyword recall (expected terms in the answer) | **~96%** |
| Course-routing accuracy | **~78%** |
| Questions answered | **100%** |

## Read these honestly

These are **automatic proxy metrics**, and they're framed that way on purpose:

- **Keyword recall** checks that expected terms *appear* — not that the explanation is
  correct.
- **Retrieval hit** checks that the expected *domain* landed in the top-k — not that the
  exact passage did.

They exist to **catch regressions** cheaply and repeatedly. They do **not** certify
answer faithfulness. Three other mechanisms do that job, and none replaces reading the
cited source:

1. the **second-pass citation auditor**, which checks each citation supports its sentence;
2. the **per-answer confidence line**; and
3. the citations themselves, which point straight back to the source.

## What the remaining misses tell you

The suite is kept honest — a handful of "misses" are *informative*, not bugs. For
example, cross-domain questions (a technique taught in one course but asked about in
another's language) legitimately retrieve the domain where the material actually lives,
and single-chunk artefacts can lose to longer prose that repeats the same keyword. These
are documented as expectations rather than silently tuned away, so the score stays a
faithful signal.

## Reproducibility

- `--retrieval-only` removes the LLM from the loop, so retrieval numbers are stable and
  fast to regenerate.
- For generation-side numbers, pin `generation.model` so runs are comparable.
- Baseline result files are versioned alongside the suite, so any regression shows up as
  a diff.
