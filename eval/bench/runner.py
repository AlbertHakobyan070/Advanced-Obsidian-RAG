"""
runner.py — score labelled questions per named pipeline configuration, in-process.

One RAGPipeline is built (or handed in) and a configuration is just a dict of
search() keyword overrides, so the embedding model, the Chroma collection and
the BM25 payload load once for the whole run. No HTTP: that would put network
noise in the latency numbers and cannot reach config-only components.

Decisions that live here, each for a reason:

  The test split is sealed. `test` and `all` (which contains test) refuse to run
  without unseal_test, and every opening is appended to the ledger BEFORE the
  first search, so the log says a run started even if it never finishes. A
  refused run writes nothing at all.

  Records are written as the run goes: run.json first (status "running", with
  everything that was measured: code, config, index, question set), then one
  flushed per_query.jsonl row per (question, config), then summary.json and
  report.md, then run.json again with the final status. A crash leaves the rows
  already written and a run.json saying "failed" and why — never a stale
  "running". A run directory is never reused; a clash raises.

  A question that fails is a ROW, not an abort: the exception text is recorded
  as `error`, the row has no metrics, the summary counts it, the CLI exits
  non-zero. Anything that goes wrong outside search() (a broken echo contract,
  a bug here) is not the question's fault and propagates.

  `lanes == []` scores 0 without searching: the factorial's empty-lane
  combinations mean "no retrieval at all". search() would rightly reject an
  empty lane set as a typo, so the runner never asks it.

  HyDE is cached for the run. Its drafts come from an LLM at temperature 0.3, so
  without the disk cache two runs of one configuration retrieve different text
  and any comparison between configurations is noise.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from eval.bench.configs import LADDER
from eval.bench.fingerprint import config_digest, git_state, index_fingerprint
from eval.bench.metrics import retrieval_metrics
from eval.bench.questions import AUTHORS, SUITES, TIERS, question_to_dict
from eval.bench.relevance import Judged, judge
from eval.bench.report import render
from eval.bench.stats import bootstrap_ci, mcnemar_exact, paired_diff_ci, percentiles
from src.retrieval.hyde_cache import HydeCache

# Anchored at the checkout, not the working directory: the ledger is the audit
# trail of the sealed split, and a run started from another directory must not
# quietly start a second one.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = REPO_ROOT / "eval" / "runs"
DEFAULT_LEDGER = REPO_ROOT / "eval" / "test_ledger.jsonl"

SEED = 0                          # bootstrap seed; recorded in run.json
SPLITS = ("dev", "test", "all")
SEALED = ("test", "all")


class SealedSplitError(PermissionError):
    """A run would open the sealed test split without unseal_test."""


class RunSetupError(ValueError):
    """The run as specified cannot start: nothing to run, or a bad argument."""


@contextmanager
def _quiet_src_logs(quiet: bool):
    """Raise the `src` loggers to WARNING for the duration: 64 configurations of
    hundreds of questions of per-search INFO lines would bury the progress lines
    and bloat logs/rag.log. It only ever RAISES the threshold (a config that
    already logs less stays that quiet) and puts back exactly what was set."""
    lg = logging.getLogger("src")
    before = lg.level
    if quiet:
        lg.setLevel(max(lg.getEffectiveLevel(), logging.WARNING))
    try:
        yield
    finally:
        lg.setLevel(before)


def _select(questions, split: str) -> list:
    live = [q for q in questions if q.provenance.get("status") != "rejected"]
    return live if split == "all" else [q for q in live if q.split == split]


def _questions_digest(questions) -> str:
    # default=str: a hand-typed `reviewed_at: 2026-10-05` loads as a date, which json cannot dump.
    blob = json.dumps([question_to_dict(q) for q in sorted(questions, key=lambda q: q.id)],
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _ledger_count(ledger: Path) -> int:
    if not ledger.exists():
        return 0
    return sum(1 for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip())


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_json(path: Path, obj) -> None:
    # Temp + replace: run.json is rewritten at the end, and a crash in the middle
    # of that write must not cost the record of the run.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _run_one(rag, q, name: str, overrides: dict) -> dict:
    row = {"qid": q.id, "config": name, "suite": q.suite, "tier": q.tier, "split": q.split,
           "author": q.provenance.get("author"), "answerable": q.answerable,
           "ids": [], "files": [], "match": [], "doc_match": [],
           "timings": {}, "lanes_run": {}, "cold": False, "hyde_cache": None}
    lanes = overrides.get("lanes")
    if lanes is not None and len(lanes) == 0:
        row["skipped"] = "no lanes"
        if q.answerable:
            row["metrics"] = retrieval_metrics(Judged([], []), q.gold)     # all 0.0
        return row
    try:
        docs, info = rag.search(q.question, **overrides)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    metas = [d.metadata for d in docs]
    j = judge(metas, q.gold)
    row.update(ids=[d.id for d in docs], files=[m.get("source_file") for m in metas],
               match=[sorted(s) for s in j.matches], doc_match=[sorted(s) for s in j.doc_matches],
               timings=info["timings"], lanes_run=info["lanes_run"], cold=info["cold"],
               hyde_cache=info["hyde_cache"])
    if q.answerable:
        row["metrics"] = retrieval_metrics(j, q.gold)
    return row


# --- summary ------------------------------------------------------------------------

def _agg(rows: list[dict]) -> dict:
    """{metric: {mean, lo, hi, n}} over rows that carry metrics; {} when none do."""
    if not rows:
        return {}
    out = {}
    for metric in rows[0]["metrics"]:
        mean, lo, hi = bootstrap_ci([r["metrics"][metric] for r in rows], seed=SEED)
        out[metric] = {"mean": mean, "lo": lo, "hi": hi, "n": len(rows)}
    return out


def _by(rows: list[dict], key: str, order) -> dict:
    out = {}
    for group in order:
        agg = _agg([r for r in rows if r[key] == group])
        if agg:
            out[group] = agg
    return out


def _latency(rows: list[dict]) -> dict:
    """Per stage over WARM rows. A cold row paid for loading the index or the
    cross-encoder, a one-off spike that belongs beside the percentiles, not in
    them. Rows that never searched (skipped, failed) have no timings."""
    stages: dict[str, list[float]] = {}
    for r in rows:
        if r["cold"] or not r["timings"]:
            continue
        for stage, ms in r["timings"].items():
            stages.setdefault(stage, []).append(ms)
    return {stage: {**percentiles(v), "n": len(v)} for stage, v in stages.items()}


def _paired(rows_a: list[dict], rows_b: list[dict]) -> dict | None:
    """B minus A over the questions BOTH scored: the per-question difference has
    far less variance than two independent means."""
    a = {r["qid"]: r["metrics"] for r in rows_a if "metrics" in r}
    b = {r["qid"]: r["metrics"] for r in rows_b if "metrics" in r}
    qids = [q for q in a if q in b]
    if not qids:
        return None
    mean, lo, hi = paired_diff_ci([a[q]["ndcg@10"] for q in qids],
                                  [b[q]["ndcg@10"] for q in qids], seed=SEED)
    only_b = sum(1 for q in qids if a[q]["hit@5"] == 0.0 and b[q]["hit@5"] == 1.0)
    only_a = sum(1 for q in qids if a[q]["hit@5"] == 1.0 and b[q]["hit@5"] == 0.0)
    return {"n": len(qids), "ndcg@10": {"mean": mean, "lo": lo, "hi": hi},
            "hit@5_mcnemar_p": mcnemar_exact(only_b, only_a)}


def summarise(rows_by_config: dict[str, list[dict]]) -> dict:
    """Per configuration: overall / by_suite / by_author / by_tier metric CIs over
    answerable rows without an error, warm latency percentiles, cold_rows, errors;
    and, for each pair of neighbouring ladder rungs that both ran, `paired` on the
    later one. by_author keeps drafted questions (`draft`) apart from hand-written
    ones (`owner`), so a gap between the two shows instead of averaging away."""
    summary = {}
    for name, rows in rows_by_config.items():
        scored = [r for r in rows if "metrics" in r]
        summary[name] = {
            "overall": _agg(scored), "by_suite": _by(scored, "suite", SUITES),
            "by_author": _by(scored, "author", AUTHORS),
            "by_tier": _by(scored, "tier", TIERS), "latency": _latency(rows),
            "cold_rows": sum(1 for r in rows if r["cold"]),
            "errors": sum(1 for r in rows if "error" in r),
        }
    for (a, _), (b, _) in zip(LADDER, LADDER[1:]):
        if a in rows_by_config and b in rows_by_config:
            paired = _paired(rows_by_config[a], rows_by_config[b])
            if paired:
                summary[b]["paired"] = {"vs": a, **paired}
    return summary


# --- the run ------------------------------------------------------------------------

def _warm_up(rag, configs) -> None:
    """One discarded search per distinct rerank mode, before anything is timed. The
    first embed and the first rerank in a process pay one-off costs (kernel and
    tokenizer initialisation as well as the lazy loads) — measured at ~5 s on the
    first "warm" row of a live run — and the cold flag does not see all of them."""
    modes = {o.get("rerank") for _, o in configs if o.get("lanes") != []}
    for mode in sorted(modes, key=str):
        kw = {"hyde": False, "top_k": 1}
        if mode is not None:
            kw["rerank"] = mode
        rag.search("warm-up", **kw)


def run_bench(cfg, questions, configs, *, split: str = "dev", unseal_test: bool = False,
              out_root: Path = DEFAULT_RUNS, rag=None, ledger: Path = DEFAULT_LEDGER,
              progress=print, limit: int | None = None, quiet_logs: bool = True,
              warmup: bool = False) -> Path:
    """Run `configs` ([(name, search-overrides)]) over the selected `questions`
    and return the run directory. `split` is dev | test | all; test and all need
    `unseal_test`. `limit` keeps the first N questions AFTER the split filter.
    `progress` is called once per finished configuration with one line of text.
    """
    if split not in SPLITS:
        raise RunSetupError(f"split must be one of {', '.join(SPLITS)}, got {split!r}")
    if split in SEALED and not unseal_test:
        raise SealedSplitError(
            f"split {split!r} opens the sealed test split: pass --unseal-test to run it "
            f"(every opening is appended to {ledger})")
    if not configs:
        raise RunSetupError("no configurations to run")
    names = [n for n, _ in configs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise RunSetupError(f"duplicate config name(s) {', '.join(dupes)}: rows are keyed by name")
    if limit is not None and limit < 1:
        raise RunSetupError(f"limit must be at least 1, got {limit}")
    selected = _select(questions, split)
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        # a rejected question never gets a split, so it is not what `bench split` would fix
        unsplit = sum(1 for q in _select(questions, "all") if q.split is None)
        raise RunSetupError(
            f"no questions to run: split {split!r} selects none of the {len(questions)} given"
            + (f" ({unsplit} have no split yet: run `main.py bench split`)" if unsplit else ""))

    git = git_state(REPO_ROOT)
    started = datetime.now()
    run_id = f"{started:%Y%m%d-%H%M%S}-{git['sha'][:7]}"
    if rag is None:
        from src.pipeline import RAGPipeline          # heavy: torch, chroma, the models
        rag = RAGPipeline.from_config(cfg)
    if rag.hyde.cache is None:
        rag.hyde.cache = HydeCache(cfg.path("retrieval.hyde_cache.path", "data/cache/hyde.sqlite"))
    meta = {
        "run_id": run_id, "status": "running", "started_at": started.isoformat(timespec="seconds"),
        "git": git, "config_digest": config_digest(cfg), "index": index_fingerprint(cfg, rag),
        "questions_digest": _questions_digest(selected), "split": split,
        "suites": [s for s in SUITES if any(q.suite == s for q in selected)],
        "n_questions": len(selected), "limit": limit, "seed": SEED,
        "configs": {name: dict(overrides) for name, overrides in configs},
    }

    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True)                     # FileExistsError on a clash: records are never overwritten
    if split in SEALED:
        entry = {"at": _now(), "run_id": run_id, "split": split, "configs": names, "git": git}
        Path(ledger).parent.mkdir(parents=True, exist_ok=True)
        with open(ledger, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    meta["test_ledger_count"] = _ledger_count(Path(ledger))
    _write_json(run_dir / "run.json", meta)

    t_run = time.perf_counter()
    rows_by_config: dict[str, list[dict]] = {}
    try:
        with _quiet_src_logs(quiet_logs), \
                open(run_dir / "per_query.jsonl", "w", encoding="utf-8", newline="\n") as out:
            if warmup:
                _warm_up(rag, configs)
            for i, (name, overrides) in enumerate(configs, 1):
                t_cfg = time.perf_counter()
                rows = rows_by_config[name] = []
                for q in selected:
                    row = _run_one(rag, q, name, overrides)
                    rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                errors = sum(1 for r in rows if "error" in r)
                ndcg = [r["metrics"]["ndcg@10"] for r in rows if "metrics" in r]
                mean = f"{sum(ndcg) / len(ndcg):.3f}" if ndcg else "n/a"
                progress(f"[{i}/{len(configs)}] {name}: {len(rows)} questions, {errors} error(s), "
                         f"nDCG@10 {mean}, {time.perf_counter() - t_cfg:.1f}s")
        summary = summarise(rows_by_config)
        _write_json(run_dir / "summary.json", summary)
        meta.update(status="done", finished_at=_now(),
                    elapsed_s=round(time.perf_counter() - t_run, 1))
        (run_dir / "report.md").write_text(render(summary, meta), encoding="utf-8", newline="\n")
        _write_json(run_dir / "run.json", meta)
    except BaseException as e:
        meta.update(status="failed", error=f"{type(e).__name__}: {e}", finished_at=_now())
        _write_json(run_dir / "run.json", meta)
        raise
    return run_dir
