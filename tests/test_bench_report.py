import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.report import render


def _m(mean, lo, hi, n):
    return {"mean": mean, "lo": lo, "hi": hi, "n": n}


def _lat(p50, p95, n):
    return {"p50": p50, "p95": p95, "p99": p95 + 1, "n": n}


SUMMARY = {
    "R2-hybrid": {
        "overall": {"hit@5": _m(0.6, 0.5, 0.7, 52), "ndcg@10": _m(0.4, 0.31, 0.49, 52)},
        "by_suite": {"lexical": {"ndcg@10": _m(0.41, 0.3, 0.5, 40)},
                     "code": {"ndcg@10": _m(0.25, 0.1, 0.4, 12)}},
        "by_tier": {"T1": {"ndcg@10": _m(0.4, 0.3, 0.5, 52)}},
        "latency": {"embed": _lat(11.0, 20.0, 51), "total": _lat(120.0, 300.0, 51)},
        "cold_rows": 1, "errors": 0,
    },
    "R3-routing": {
        "overall": {"hit@5": _m(0.65, 0.55, 0.75, 50), "ndcg@10": _m(0.45, 0.36, 0.54, 50)},
        "by_suite": {"lexical": {"ndcg@10": _m(0.46, 0.3, 0.6, 38)},
                     "code": {"ndcg@10": _m(0.3, 0.1, 0.5, 12)}},
        "by_tier": {"T1": {"ndcg@10": _m(0.45, 0.3, 0.6, 50)}},
        # `rerank` comes after `total` here, as it would when a later config adds a stage
        "latency": {"embed": _lat(12.0, 21.0, 52), "total": _lat(130.0, 310.0, 52),
                    "rerank": _lat(80.0, 150.0, 52)},
        "cold_rows": 0, "errors": 2,
        "paired": {"vs": "R2-hybrid", "n": 50,
                   "ndcg@10": {"mean": 0.05, "lo": 0.01, "hi": 0.09}, "hit@5_mcnemar_p": 0.04},
    },
    "R4-code": {
        "overall": {"hit@5": _m(0.66, 0.56, 0.76, 52), "ndcg@10": _m(0.46, 0.37, 0.55, 52)},
        "by_suite": {"lexical": {"ndcg@10": _m(0.47, 0.3, 0.6, 40)}},
        "by_tier": {}, "latency": {}, "cold_rows": 0, "errors": 0,
        "paired": {"vs": "R3-routing", "n": 50,
                   "ndcg@10": {"mean": 0.01, "lo": -0.02, "hi": 0.04}, "hit@5_mcnemar_p": 0.7},
    },
}

META = {
    "run_id": "20261006-101112-abc1234", "status": "done",
    "git": {"sha": "abc1234def5678", "dirty": True}, "config_digest": "d1g3st",
    "index": {"dense_count": 161576,
              "bm25": {"count": 161570, "built_at": "2026-10-05T01:02:03"}},
    "split": "dev", "suites": ["lexical", "code"], "n_questions": 52, "test_ledger_count": 2,
}


def test_header_carries_what_was_measured():
    text = render(SUMMARY, META)
    assert "20261006-101112-abc1234" in text
    assert "`abc1234`" in text and "(dirty)" in text and "`d1g3st`" in text
    assert "dense 161576" in text and "bm25 161570" in text and "2026-10-05T01:02:03" in text
    assert "**split** dev" in text and "**questions** 52" in text
    assert "**test split opened** 2 time(s)" in text


def test_a_clean_checkout_is_not_marked_dirty():
    assert "(dirty)" not in render(SUMMARY, {**META, "git": {"sha": "abc1234def5678", "dirty": False}})


def test_metric_table_shows_mean_and_interval_per_config():
    text = render(SUMMARY, META)
    assert "0.600 [0.500, 0.700]" in text and "0.450 [0.360, 0.540]" in text
    assert all(f"| {name} |" in text for name in SUMMARY)


def test_small_suites_are_marked_diagnostic_and_big_ones_are_not():
    text = render(SUMMARY, META)
    assert "code (n=12, *diag*)" in text
    assert "lexical (n=40)" in text                    # the widest n any config scored


def _with_authors(summary):
    out = copy.deepcopy(summary)
    for c in out.values():
        c["by_author"] = {"draft": {"ndcg@10": _m(0.40, 0.3, 0.5, 40)},
                          "owner": {"ndcg@10": _m(0.62, 0.4, 0.8, 12)}}
    return out


def test_drafted_and_hand_written_questions_get_their_own_table():
    text = render(_with_authors(SUMMARY), META)
    section = text.split("## nDCG@10 by author")[1].split("\n## ")[0]
    assert "draft (n=40)" in section and "owner (n=12, *diag*)" in section
    assert "0.400" in section and "0.620" in section
    assert "hand-written" in section                      # the legend says which author is which
    assert all(f"| {name} |" in section for name in SUMMARY)
    assert "## nDCG@10 by suite" in text                  # the suite table is still there


def test_a_summary_written_before_by_author_existed_still_renders():
    # bench report re-renders old runs; theirs have no by_author, and no empty table appears
    assert "by author" not in render(SUMMARY, META)


def test_ladder_steps_say_whether_the_paired_interval_excludes_zero():
    lines = render(SUMMARY, META).splitlines()
    real = next(l for l in lines if l.startswith("| R3-routing vs R2-hybrid"))
    flat = next(l for l in lines if l.startswith("| R4-code vs R3-routing"))
    assert "+0.050 [+0.010, +0.090]" in real and real.rstrip(" |").endswith("yes")
    assert "+0.010 [-0.020, +0.040]" in flat and flat.rstrip(" |").endswith("no")
    assert "0.040" in real and "0.700" in flat           # the McNemar p-values


def test_no_ladder_pair_means_no_ladder_section():
    solo = {"custom": {k: v for k, v in SUMMARY["R2-hybrid"].items()}}
    assert "Ladder" not in render(solo, META)
    assert "Ladder" in render(SUMMARY, META)


def test_latency_table_has_p50_over_p95_and_flags_cold_rows():
    text = render(SUMMARY, META)
    assert "11.0 / 20.0" in text and "120.0 / 300.0" in text
    assert "R2-hybrid: 1" in text                        # the cold rows left out of the percentiles
    quiet = copy.deepcopy(SUMMARY)
    quiet["R2-hybrid"]["cold_rows"] = 0
    assert "cold" not in render(quiet, META).lower().split("## latency")[1]


def test_total_is_the_last_latency_column():
    header = next(l for l in render(SUMMARY, META).splitlines() if l.startswith("| config | embed"))
    assert header.index("embed") < header.index("rerank") < header.index("total")


def test_errors_line_names_the_configs_that_failed():
    assert "R3-routing: 2" in render(SUMMARY, META).split("## Errors")[1]
    clean = {k: {**v, "errors": 0} for k, v in SUMMARY.items()}
    assert "none" in render(clean, META).split("## Errors")[1]


def test_a_run_with_nothing_scored_still_renders():
    empty = {"x": {"overall": {}, "by_suite": {}, "by_tier": {}, "latency": {},
                   "cold_rows": 0, "errors": 0}}
    assert "No answerable question was scored" in render(empty, META)


def test_missing_index_details_do_not_break_the_header():
    text = render(SUMMARY, {**META, "index": {}})
    assert "dense" not in text.split("## ")[0]
    text = render(SUMMARY, {**META, "index": {"dense_count": 3, "bm25": None}})
    assert "dense 3" in text and "bm25 n/a" in text
