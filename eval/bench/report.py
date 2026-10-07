"""
report.py — summary.json + run.json -> the scorecard (report.md).

render() is a pure function of the two dicts, so `bench report <run>` can
re-render an old run exactly (and a changed layout never needs a re-run). It
reports; it does not judge, with one exception that is the rule of the whole
eval: a ladder step is called real only when its PAIRED 95% interval excludes
zero (spec §6), and the table says so per step instead of leaving it to the
reader's eye.

Two honesty devices. A suite with fewer scored questions than DIAG_BELOW is
marked *diag*: a mean over a dozen questions is a diagnostic, not a result.
And latency is warm-only — the cold rows (a search that had to load the index
or the cross-encoder) are counted beside the table, never averaged into it.
"""
from __future__ import annotations

from eval.bench.questions import AUTHORS, SUITES

DIAG_BELOW = 30        # a suite scored on fewer questions than this is diagnostic only
DASH = "–"             # an empty cell: the config has no number for it


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    line = lambda cells: "| " + " | ".join(cells) + " |"
    return [line(header), line(["---"] * len(header))] + [line(r) for r in rows]


def _ci(m: dict) -> str:
    return f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def _signed(m: dict) -> str:
    return f"{m['mean']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}]"


def _header(summary: dict, meta: dict) -> list[str]:
    git = meta["git"]
    lines = [f"- **git** `{git['sha'][:7]}`{' (dirty)' if git['dirty'] else ''} · "
             f"**config digest** `{meta['config_digest']}`"]
    index = meta.get("index") or {}
    parts = []
    if "dense_count" in index:
        parts.append(f"dense {index['dense_count']}")
    if "bm25" in index:
        bm25 = index["bm25"]
        parts.append(f"bm25 {bm25['count']} (built {bm25['built_at']})" if bm25 else "bm25 n/a")
    if parts:
        lines.append("- **index** " + " · ".join(parts))
    lines.append(f"- **split** {meta['split']} · **questions** {meta['n_questions']} · "
                 f"**configs** {len(summary)} · **suites** {', '.join(meta['suites'])}")
    lines.append(f"- **test split opened** {meta['test_ledger_count']} time(s)")
    return lines


def _metric_table(summary: dict) -> list[str]:
    scored = {name: c["overall"] for name, c in summary.items() if c["overall"]}
    if not scored:
        return ["No answerable question was scored."]
    metrics = list(next(iter(scored.values())))
    rows = []
    for name, c in summary.items():
        o = c["overall"]
        rows.append([name, str(next(iter(o.values()))["n"]) if o else DASH]
                    + [_ci(o[m]) if o else DASH for m in metrics])
    return ["Mean [95% bootstrap CI] over answerable questions; n is how many were scored.", ""] \
        + _table(["config", "n"] + metrics, rows)


def _ladder_table(summary: dict) -> list[str]:
    rows = []
    for name, c in summary.items():
        p = c.get("paired")
        if not p:
            continue
        d = p["ndcg@10"]
        real = "yes" if d["lo"] > 0 or d["hi"] < 0 else "no"
        rows.append([f"{name} vs {p['vs']}", str(p["n"]), _signed(d),
                     f"{p['hit@5_mcnemar_p']:.3f}", real])
    if not rows:
        return []
    return ["## Ladder steps (paired)", "",
            "Δ nDCG@10 is B minus A over the questions both configurations scored. A step "
            "is real only when its paired 95% CI excludes zero.", ""] \
        + _table(["step (B vs A)", "n", "Δ nDCG@10 [95% CI]", "McNemar p (hit@5)",
                  "CI excludes 0"], rows) + [""]


def _group_table(summary: dict, key: str, groups, title: str, intro: str = "") -> list[str]:
    """nDCG@10 per configuration, one column per group of questions; `key` is the
    summary's grouping (by_suite, by_author) and `groups` its columns in order."""
    present = [g for g in groups if any(g in c[key] for c in summary.values())]
    if not present:
        return []
    heads = []
    for g in present:
        n = max(c[key][g]["ndcg@10"]["n"] for c in summary.values() if g in c[key])
        heads.append(f"{g} (n={n}, *diag*)" if n < DIAG_BELOW else f"{g} (n={n})")
    rows = [[name] + [f"{c[key][g]['ndcg@10']['mean']:.3f}" if g in c[key] else DASH
                      for g in present]
            for name, c in summary.items()]
    return [f"## nDCG@10 by {title}", "",
            f"{intro}*diag* = fewer than {DIAG_BELOW} scored questions: a diagnostic, not a result.", ""] \
        + _table(["config"] + heads, rows) + [""]


def _suite_table(summary: dict) -> list[str]:
    return _group_table(summary, "by_suite", SUITES, "suite")


def _author_table(summary: dict) -> list[str]:
    if not any("by_author" in c for c in summary.values()):   # a run recorded before by_author existed
        return []
    return _group_table(summary, "by_author", AUTHORS, "author",
                        "`draft` = drafted by a model, `owner` = hand-written. ")


def _latency_table(summary: dict) -> list[str]:
    stages: list[str] = []
    for c in summary.values():
        stages += [s for s in c["latency"] if s not in stages]
    out = ["## Latency (ms, p50 / p95, warm rows only)", ""]
    if not stages:
        return out + ["No warm search was timed."]
    if "total" in stages:                       # the sum reads best at the right-hand end
        stages.remove("total")
        stages.append("total")
    rows = [[name] + [f"{c['latency'][s]['p50']:.1f} / {c['latency'][s]['p95']:.1f}"
                      if s in c["latency"] else DASH for s in stages]
            for name, c in summary.items()]
    out += _table(["config"] + stages, rows)
    cold = [f"{name}: {c['cold_rows']}" for name, c in summary.items() if c["cold_rows"]]
    if cold:
        out += ["", "Cold rows (a search that loaded the index or the cross-encoder) are left "
                    "out of the percentiles: " + ", ".join(cold) + "."]
    return out


def _errors(summary: dict) -> list[str]:
    failed = [f"{name}: {c['errors']}" for name, c in summary.items() if c["errors"]]
    if not failed:
        return ["Failed rows: none."]
    return ["Failed rows — " + ", ".join(failed) + ". Each carries an `error` in per_query.jsonl."]


def render(summary: dict, meta: dict) -> str:
    out = [f"# Bench run {meta['run_id']}", "", *_header(summary, meta), "",
           "## Retrieval by configuration", "", *_metric_table(summary), "",
           *_ladder_table(summary), *_suite_table(summary), *_author_table(summary),
           *_latency_table(summary), "",
           "## Errors", "", *_errors(summary), ""]
    return "\n".join(out)
