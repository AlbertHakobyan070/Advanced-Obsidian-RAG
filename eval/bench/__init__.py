"""
bench — the v2 evaluation harness
(spec: docs/superpowers/specs/2026-10-04-eval-system-design.md).

Labelled questions are scored per NAMED pipeline configuration, in-process, so
models and indexes load once. The modules are small and pure where they can be
(they take plain data and touch no index, LLM or network), which is what makes
each rule testable in milliseconds without a corpus.

The legacy harness (eval/eval_runner.py, `main.py eval`, golden_queries.yaml)
is not part of this package and stays runnable: v2 is a new set, not a
migration.
"""
