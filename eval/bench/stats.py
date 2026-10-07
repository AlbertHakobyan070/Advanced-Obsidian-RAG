"""
stats.py — the statistics the bench reports, and nothing more.

Every mean carries a percentile-bootstrap 95% CI with a FIXED seed, so
re-running the same rows reproduces the same interval; an interval that moves
on a re-run reads as a different result. A-vs-B comparisons are PAIRED (same
questions, same order): the per-question difference has far less variance than
two independent means, so paired_diff_ci bootstraps that difference, and
mcnemar_exact is the exact two-sided test for binary metrics, from the
discordant counts alone. shapley() splits a value function's total across
factors by exact enumeration (cheap for the six pipeline factors): the ladder
shows one ordering of the components, Shapley averages over all of them.
"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n_resamples: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean: (mean, lo, hi). Fixed seed, so a
    re-run of the same rows reports the same interval."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (math.nan, math.nan, math.nan)
    mean = float(arr.mean())
    if arr.size == 1 or np.all(arr == arr[0]):
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (mean, float(lo), float(hi))


def paired_diff_ci(a: Sequence[float], b: Sequence[float], **kw) -> tuple[float, float, float]:
    """CI of mean(b - a) over PAIRED rows (same questions, same order)."""
    if len(a) != len(b):
        raise ValueError(f"paired_diff_ci: {len(a)} vs {len(b)} rows — not paired")
    return bootstrap_ci([y - x for x, y in zip(a, b)], **kw)


def mcnemar_exact(n01: int, n10: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts
    (n01: A wrong & B right, n10: A right & B wrong)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def shapley(value: Callable[[frozenset], float], factors: Sequence[str]) -> dict[str, float]:
    """Exact Shapley values: each factor's marginal contribution averaged over
    every order in which the factors could be switched on."""
    n = len(factors)
    out = {}
    for f in factors:
        others = [x for x in factors if x != f]
        total = 0.0
        for r in range(n):
            w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for subset in combinations(others, r):
                s = frozenset(subset)
                total += w * (value(s | {f}) - value(s))
        out[f] = total
    return out


def percentiles(values: Sequence[float], qs=(50, 95, 99)) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {f"p{q}": math.nan for q in qs}
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}
