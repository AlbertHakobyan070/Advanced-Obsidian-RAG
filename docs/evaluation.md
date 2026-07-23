# Evaluation

Quality is measured with the labelled suite in `eval/golden_queries.yaml`. Read that
file for the current cases and coverage; do not rely on a copied query count. Entries
can label expected domains, courses, source files, scope routing, keywords, and gold
answers independently, so missing labels drop out of the corresponding metric instead
of becoming false zeroes.

```bash
python main.py eval --retrieval-only # retrieval tier only; no generation backend
python main.py eval                  # retrieval, answer, and calibration tiers
python main.py eval --judge          # add the advisory LLM-judge fields
```

## What each tier measures

| Tier | Measures |
|---|---|
| Retrieval | hit-rate@k, MRR, labelled-source recall@k, scope precision/recall, and course routing |
| Answer | keyword recall, citation validity, deterministic groundedness floor, optional citation-auditor support, and answered rate |
| Calibration | whether stated confidence tracks keyword recall, groundedness, and optional judge correctness |

`--judge` adds advisory correctness and groundedness scores. A gold answer enables
reference-based correctness; without one, in-process judge correctness is not treated
as a meaningful score. Judge failures remain visible in the report.

## Read these honestly

The automatic tiers are **proxy metrics**, and they are framed that way on purpose:

- **Keyword recall** checks that expected terms *appear* — not that the explanation is
  correct.
- **Retrieval hit** checks that the expected *domain* landed in the top-k — not that the
  exact passage did.

They exist to **catch regressions** cheaply and repeatedly. They do **not** certify
answer faithfulness. The LLM judge is advisory too. Three other mechanisms help, and
none replaces reading the cited source:

1. the **second-pass citation auditor**, which checks each citation supports its sentence;
2. the **per-answer confidence line**; and
3. the citations themselves, which point straight back to the source.

## What misses tell you

A miss can be informative rather than a bug. Cross-domain questions may retrieve the
domain where the material actually lives, and a small code artifact can lose to prose
that repeats the same terms. Inspect the per-query rows before tuning a global default.

## Reproducibility

- `--retrieval-only` removes the LLM from the loop, so retrieval numbers are stable and
  fast to regenerate.
- For generation-side numbers, pin `generation.model` so runs are comparable.
- Each run writes structured and Markdown reports; compare them explicitly rather than
  hardcoding a copied baseline in documentation.
- `--judge-export` / `--judge-import` allow an external model to grade a complete
  bundle without giving that model access to the vault or local service.
