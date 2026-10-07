"""
configs.py — the named pipeline configurations the bench runs.

A configuration is a dict of RAGPipeline.search() keyword overrides. The
ladder, leave-one-out and factorial sets are GENERATED from six binary
factors, so they cannot drift apart and eval/configs.yaml needs no entry for
them; one-off variants are written by hand in that file.

Every configuration is scored on the same footing (EVAL_DEFAULTS): a final
list of ten, and the optional lanes that would make a run non-deterministic or
incomplete switched off. omnisearch queries the LIVE vault, hype covers only
part of the corpus, and parent/neighbor context rewrite the very chunks the
metrics judge — leave any of them on and two configurations stop being
comparable. These are the eval protocol, not pipeline tunables, so they live
here and not in config.yaml.

`lanes` only RESTRICTS what retrieve() may run; a conditional lane still needs
its own trigger (a detected scope, code intent). A factor is therefore
"allowed and able": `routing` lists the scope lanes and switches the metadata
boost on, `code` lists the code lanes and lets auto_preset fire the code preset.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import yaml

FACTORS = ("sparse", "dense", "routing", "code", "rerank", "hyde")

EVAL_DEFAULTS = {"top_k": 10, "omnisearch": False, "hype": False,
                 "parent_context": False, "neighbor_context": False}


def factor_config(active: frozenset[str]) -> dict:
    """search() overrides for the pipeline with exactly the `active` factors on.

    Composition rule: `lanes` = the active families among {dense, sparse}, plus
    their `*_scope` lanes when `routing` is on, plus their `*_code` lanes when
    `code` is on (sorted); `metadata_boost` = routing; `auto_preset` = code;
    `rerank` = "cross_encoder" if rerank else "none"; `hyde` = hyde; everything
    else from EVAL_DEFAULTS. A set with neither dense nor sparse yields
    `lanes = []`, which the runner scores 0 without searching.
    """
    families = [f for f in ("dense", "sparse") if f in active]
    lanes = list(families)
    if "routing" in active:
        lanes += [f"{f}_scope" for f in families]
    if "code" in active:
        lanes += [f"{f}_code" for f in families]
    return {
        **EVAL_DEFAULTS,
        "lanes": sorted(lanes),
        "metadata_boost": "routing" in active,
        "auto_preset": "code" in active,
        "rerank": "cross_encoder" if "rerank" in active else "none",
        "hyde": "hyde" in active,
    }


# Cumulative from R2 on: each rung adds one thing to the one before. R0 and R1
# are the two single-lane baselines that R2 (hybrid) is measured against.
LADDER: list[tuple[str, frozenset[str]]] = [
    ("R0-bm25", frozenset({"sparse"})),
    ("R1-dense", frozenset({"dense"})),
    ("R2-hybrid", frozenset({"dense", "sparse"})),
    ("R3-routing", frozenset({"dense", "sparse", "routing"})),
    ("R4-code", frozenset({"dense", "sparse", "routing", "code"})),
    ("R5-rerank", frozenset({"dense", "sparse", "routing", "code", "rerank"})),
    ("R6-hyde", frozenset(FACTORS)),          # the shipped default
]


def ladder() -> list[tuple[str, dict]]:
    return [(name, factor_config(s)) for name, s in LADDER]


def leave_one_out() -> list[tuple[str, dict]]:
    """The shipped default, then the default minus each factor in turn."""
    full = frozenset(FACTORS)
    return [("full", factor_config(full))] + [(f"-{f}", factor_config(full - {f})) for f in FACTORS]


def factorial() -> list[tuple[str, dict, frozenset[str]]]:
    """Every subset of FACTORS (2^6 = 64) as (name, overrides, subset); the
    subset is what Shapley attribution keys its value function on."""
    out = []
    for r in range(len(FACTORS) + 1):
        for combo in combinations(FACTORS, r):
            s = frozenset(combo)
            out.append(("F:" + ("+".join(sorted(s)) or "none"), factor_config(s), s))
    return out


def load_named(path) -> dict[str, dict]:
    """The hand-written configurations in `path`, each merged over EVAL_DEFAULTS."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping with a top-level 'configs:' key")
    named = raw.get("configs") or {}
    if not isinstance(named, dict):
        raise ValueError(f"{path}: 'configs' must map each name to its search() overrides")
    out = {}
    for name, overrides in named.items():
        overrides = {} if overrides is None else overrides
        if not isinstance(overrides, dict):
            raise ValueError(f"{path}: config {name!r} must be a mapping of search() overrides")
        out[str(name)] = {**EVAL_DEFAULTS, **overrides}
    return out


def resolve(spec: str, path) -> list[tuple[str, dict]]:
    """A `--configs` value to [(name, overrides)]: `ladder`, `loo` or `factorial`
    (generated; `path` is not read), or comma-separated names from `path`."""
    if spec == "ladder":
        return ladder()
    if spec == "loo":
        return leave_one_out()
    if spec == "factorial":
        return [(name, overrides) for name, overrides, _ in factorial()]
    named = load_named(path)
    out = []
    for name in (n.strip() for n in spec.split(",")):
        if name not in named:
            raise KeyError(f"unknown config {name!r}; known: {', '.join(sorted(named)) or '(none)'} "
                           f"(or one of ladder, loo, factorial)")
        out.append((name, named[name]))
    return out
