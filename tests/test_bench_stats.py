import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.stats import bootstrap_ci, mcnemar_exact, paired_diff_ci, percentiles, shapley


def test_bootstrap_degenerate_and_deterministic():
    assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0, 1.0)
    assert all(math.isnan(x) for x in bootstrap_ci([]))
    a = bootstrap_ci([0, 1] * 20, seed=7)
    assert a == bootstrap_ci([0, 1] * 20, seed=7)
    assert a[0] == 0.5 and a[1] < 0.5 < a[2]


def test_paired_diff_sign_and_zero():
    assert paired_diff_ci([0, 0, 0], [1, 1, 1])[0] == 1.0
    assert paired_diff_ci([1, 0, 1, 0], [1, 0, 1, 0]) == (0.0, 0.0, 0.0)


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(0, 6), 2 * 0.5 ** 6)
    assert mcnemar_exact(3, 3) == 1.0


def test_shapley_additive_and_interaction_games():
    w = {"a": 1.0, "b": 2.0, "c": 0.5}
    phi = shapley(lambda s: sum(w[x] for x in s), ["a", "b", "c"])
    assert all(math.isclose(phi[k], w[k]) for k in w)
    phi = shapley(lambda s: 1.0 if {"a", "b"} <= s else 0.0, ["a", "b", "c"])
    assert math.isclose(phi["a"], 0.5) and math.isclose(phi["b"], 0.5) and phi["c"] == 0.0
    assert math.isclose(sum(phi.values()), 1.0)          # efficiency: sums to v(N) - v(empty)


def test_percentiles():
    p = percentiles(list(range(1, 101)))
    assert p["p50"] == 50.5 and p["p95"] > 94 and p["p99"] > 98
