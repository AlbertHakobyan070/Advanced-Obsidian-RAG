"""
cli.py — `main.py bench …`, the command line of the v2 eval.

    bench run       score questions per named configuration     (runner.py)
    bench report    re-render a finished run's report.md         (report.py)
    bench split     give each live question a dev/test split once, and store it
    bench validate  check a question set against the corpus      (validate.py)
    bench sample    draw seed packs from the chunk files         (sample.py)
    bench draft     draft questions from the seed packs with an LLM (draft.py)

main.py imports this module just to register the subcommand, so everything
heavy (numpy, the pipeline, the chunk files, the validate and sample modules) is
imported inside the handler that needs it: every other main.py command starts as
fast as it did before.

A mistake of the user's (an unknown suite, a missing sets directory, a sealed
split, nothing to run) ends as `ERROR: …` on stderr and exit 2, the code argparse
uses for a bad command line. A run in which any question failed exits 1.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import NoReturn

import yaml

from eval.bench.configs import resolve
from eval.bench.questions import SUITES, SchemaError, assign_splits, load_sets, save_suite, sets_lock
from eval.bench.report import render
from src.utils.config_loader import load_config

# Anchored at the checkout, not the working directory: runner.py does the same for
# the ledger and the run folders, so `main.py bench …` writes into the same eval/
# wherever it is started from. Named configurations are part of the harness, not
# something a --flag points elsewhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_FILE = REPO_ROOT / "eval" / "configs.yaml"
DEFAULT_SEEDS = REPO_ROOT / "eval" / "seeds"      # where `bench sample` writes and `bench draft` reads


def _fail(msg: str) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def _check_suites(names: list[str] | None) -> list[str] | None:
    """load_sets silently returns nothing for a suite it has never heard of, so a
    typo would run (or validate) zero questions and look like success."""
    unknown = sorted(set(names or []) - set(SUITES))
    if unknown:
        _fail(f"unknown suite(s) {', '.join(unknown)}; valid suites: {', '.join(SUITES)}")
    return names


def _sets_dir(args, must_exist: bool = True) -> Path:
    """--sets as typed, else config's eval.sets_dir, resolved exactly as the
    console's review queue resolves it, so the CLI and the review queue can never
    look at two different directories. A typed --sets never reads the config."""
    p = Path(args.sets) if args.sets else load_config(args.config).path("eval.sets_dir", "eval/sets")
    if must_exist and not p.is_dir():
        _fail(f"sets directory {p} does not exist")      # load_sets would read it as "no questions"
    return p


def _load(sets: Path, **kw) -> list:
    try:
        return load_sets(sets, **kw)
    except SchemaError as e:
        _fail(str(e))


def _data_dir(args) -> Path:
    d = Path(args.data_dir) if args.data_dir else load_config(args.config).path("paths.chunks_file").parent
    if not d.is_dir():
        _fail(f"data directory {d} does not exist")
    return d


def _seed_count(factor: float, quota: int) -> int:
    """ceil(factor x quota); the rounding strips float noise (1.1 x 50 is
    55.00000000000001, which a bare ceil would turn into 56)."""
    return math.ceil(round(factor * quota, 6))


def _cmd_run(args) -> None:
    suites = _check_suites(args.suites)
    sets = _sets_dir(args)
    questions = _load(sets, suites=suites)
    if not questions:
        _fail(f"no questions in {sets}" + (f" for suites {', '.join(suites)}" if suites else ""))
    try:
        configs = resolve(args.configs, CONFIGS_FILE)
    except KeyError as e:
        _fail(e.args[0])
    cfg = load_config(args.config)
    from eval.bench import runner

    print(f"bench run: split {args.split}, {len(configs)} configuration(s), questions from {sets}")
    try:
        run_dir = runner.run_bench(
            cfg, questions, configs, split=args.split, unseal_test=args.unseal_test,
            out_root=Path(args.out_root) if args.out_root else runner.DEFAULT_RUNS,
            ledger=runner.DEFAULT_LEDGER, limit=args.limit,
            quiet_logs=not args.verbose, warmup=True)
    except (runner.SealedSplitError, runner.RunSetupError) as e:
        _fail(str(e))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    errors = sum(c["errors"] for c in summary.values())
    print(f"\nRun record: {run_dir}\nReport:     {run_dir / 'report.md'}")
    if errors:
        print(f"ERROR: {errors} question row(s) failed; see \"error\" in "
              f"{run_dir / 'per_query.jsonl'}", file=sys.stderr)
        sys.exit(1)


def _cmd_report(args) -> None:
    run_dir = Path(args.run_dir)
    files = [run_dir / "summary.json", run_dir / "run.json"]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        _fail(f"not a finished bench run: {', '.join(missing)} not found")
    summary, meta = (json.loads(p.read_text(encoding="utf-8")) for p in files)
    (run_dir / "report.md").write_text(render(summary, meta), encoding="utf-8", newline="\n")
    print(f"Wrote {run_dir / 'report.md'}")


def _cmd_split(args) -> None:
    sets = _sets_dir(args)
    # One read-modify-write of every sets file: a console review or a draft run writing
    # in between would be overwritten, so the whole of it holds the sets lock.
    with sets_lock(sets):
        _split(sets)


def _split(sets: Path) -> None:
    everything = _load(sets, include_rejected=True)      # loaded so the rewrite keeps them
    # A rejected question takes no split: one un-rejected later is assigned then,
    # and a dead question must not tip the balance of its stratum.
    live = [q for q in everything if q.provenance.get("status") != "rejected"]
    before = {q.id: q.split for q in everything}
    assign_splits(live)
    by_id = {q.id: q for q in everything}
    rewritten = []
    for path in sorted(sets.glob("*.yaml")):
        # File membership comes from the file itself: a file may hold several suites
        # (eval/smoke/smoke.yaml does), so regrouping by suite would move questions.
        members = [by_id[d["id"]] for d in yaml.safe_load(path.read_text(encoding="utf-8"))]
        if any(q.split != before[q.id] for q in members):      # untouched files stay byte-identical
            save_suite(path, members)
            rewritten.append(path.name)
    counts = Counter(q.split for q in live)
    print(f"Assigned {sum(1 for q in everything if q.split != before[q.id])} split(s); "
          f"rewrote {len(rewritten)} file(s){': ' + ', '.join(rewritten) if rewritten else ''}")
    print(f"Live questions: dev {counts['dev']}, test {counts['test']}")


def _cmd_validate(args) -> None:
    suites = _check_suites(args.suites)
    if args.write_cache and suites:
        _fail("--write-cache writes the review cache for the WHOLE sets directory; drop --suites "
              "(a partial cache would hide the other suites' gold texts from the review queue)")
    sets = _sets_dir(args)
    from eval.bench.validate import load_gold_chunks, quota_table, validate, write_review_cache

    questions = _load(sets, suites=suites)
    gold_chunks = load_gold_chunks(_data_dir(args), {s.file for q in questions for g in q.gold for s in g.sources()})
    findings = validate(questions, gold_chunks)

    by_code: dict[str, list] = {}
    for f in findings:
        by_code.setdefault(f.code, []).append(f)
    for code, items in by_code.items():
        print(f"\n{code} ({len(items)})")
        for f in items:
            print(f"  [{f.level}] {f.qid}: {f.message}")
    print("\nQuota (suite: have / target)")
    for suite, have, target in quota_table(questions):
        print(f"  {suite:<14}{have:>4} / {target}")
    errors = sum(1 for f in findings if f.level == "error")
    print(f"\n{errors} error(s), {len(findings) - errors} warning(s) in {len(questions)} question(s)")
    if args.write_cache:
        cache = sets / ".review_cache.json"
        write_review_cache(cache, questions, gold_chunks, findings)
        print(f"Review cache: {cache}")
    if errors:
        sys.exit(1)


def _cmd_sample(args) -> None:
    suites = _check_suites(args.suites) or list(SUITES)
    if args.factor <= 0:
        _fail(f"--factor must be positive, got {args.factor}")
    # Unlike every other command, a sets directory that does not exist is fine here:
    # before the first draft there is nothing to exclude, and sampling comes first.
    sets = _sets_dir(args, must_exist=False)
    if not sets.is_dir():
        print(f"note: {sets} does not exist yet, so no seed chunk is excluded")
    from eval.bench.sample import (
        STRATA, multihop_pairs, sample_suite, topic_neighbourhoods, used_seed_ids, write_pack)

    data_dir = _data_dir(args)
    exclude = used_seed_ids(sets)
    out = Path(args.out) if args.out else DEFAULT_SEEDS
    for suite in suites:
        n = _seed_count(args.factor, SUITES[suite])
        if suite == "unanswerable":
            records = topic_neighbourhoods(data_dir)       # topics to avoid, not seeds to draft from
        elif suite == "multihop":
            records = multihop_pairs(data_dir, n, seed=args.seed, exclude_ids=exclude)
        else:
            records = sample_suite(suite, data_dir, n, seed=args.seed, exclude_ids=exclude)
        write_pack(out / f"{suite}.jsonl", records)
        short = f" (asked for {n}: no more qualify)" if suite != "unanswerable" and len(records) < n else ""
        print(f"{suite}: {len(records)} record(s) -> {out / (suite + '.jsonl')}{short}")
        if suite in STRATA and records:          # multihop pairs and unanswerable topics have none
            per_stratum = Counter(STRATA[suite](rec) for rec in records)
            print(f"  per stratum ({len(per_stratum)}):")
            for stratum, n_seeds in sorted(per_stratum.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    {n_seeds:>4}  {stratum or '(blank)'}")


def _cmd_draft(args) -> None:
    for flag, value in (("--n", args.n), ("--max-calls", args.max_calls)):
        if value is not None and value < 1:
            _fail(f"{flag} must be at least 1, got {value}")
    sets = _sets_dir(args, must_exist=False)           # drafting may be what creates it
    pack = (Path(args.seeds) if args.seeds else DEFAULT_SEEDS) / f"{args.suite}.jsonl"
    if not pack.is_file():
        _fail(f"no seed pack at {pack}; draw one with `bench sample --suites {args.suite}`")
    data_dir = _data_dir(args)
    from eval.bench import draft
    from src.llm.llm_client import LLMClient

    cfg = load_config(args.config)
    # The query API's address is the console's own setting (webui.rag_api, same default).
    search_url = args.search_url or cfg.get("webui.rag_api", "http://127.0.0.1:8051")
    try:
        llm = LLMClient.from_provider_override(cfg, args.provider, model=args.model)
    except (ValueError, RuntimeError) as e:           # an unknown provider; a missing or mistyped key
        _fail(str(e))
    try:
        drafter = draft.draft_suite(
            args.suite, llm, draft.http_fetch(search_url), seeds_path=pack, sets_dir=sets,
            data_dir=data_dir, n=args.n or SUITES[args.suite], max_score=args.max_score,
            max_calls=args.max_calls)
    except (SchemaError, draft.DraftError) as e:
        _fail(str(e))
    print("\n" + drafter.summary())
    if drafter.skipped["llm-error"]:                  # as in `bench run`: lost work exits 1
        sys.exit(1)


def bench_cmd(args) -> None:
    {"run": _cmd_run, "report": _cmd_report, "split": _cmd_split,
     "validate": _cmd_validate, "sample": _cmd_sample,
     "draft": _cmd_draft}[args.bench_command](args)


def add_bench_parser(subparsers) -> None:
    b = subparsers.add_parser(
        "bench", help="The v2 eval: labelled questions scored per named pipeline configuration")
    sub = b.add_subparsers(dest="bench_command", required=True)
    suite_help = f"suite names, any of: {', '.join(SUITES)}"

    sets_help = "directory of question-set YAML files (default: eval.sets_dir from config.yaml)"

    r = sub.add_parser("run", help="Score questions per configuration; write an immutable run record")
    r.add_argument("--sets", default=None, metavar="DIR", help=sets_help)
    r.add_argument("--suites", nargs="+", default=None, metavar="SUITE",
                   help=f"only these suites (default: all); {suite_help}")
    r.add_argument("--split", choices=("dev", "test", "all"), default="dev",
                   help="which split to run; test and all (which contains test) are sealed "
                        "and need --unseal-test")
    r.add_argument("--unseal-test", action="store_true", dest="unseal_test",
                   help="allow a sealed split; every opening is appended to eval/test_ledger.jsonl")
    r.add_argument("--configs", default="ladder", metavar="SPEC",
                   help="ladder | loo | factorial, or comma-separated names from eval/configs.yaml "
                        "(default: ladder)")
    r.add_argument("--limit", type=int, default=None, metavar="N",
                   help="run only the first N questions of the split (smoke runs)")
    r.add_argument("--out-root", default=None, dest="out_root", metavar="DIR",
                   help="where the run directory is created (default: eval/runs in the "
                        "checkout, wherever this is run from)")
    r.add_argument("--verbose", action="store_true",
                   help="keep the pipeline's INFO logging during the run (silenced by default: "
                        "it is a block of lines per search)")

    rp = sub.add_parser("report", help="Re-render report.md of a finished run")
    rp.add_argument("run_dir", metavar="RUN_DIR", help="a directory under eval/runs")

    sp = sub.add_parser("split", help="Assign the dev/test split to questions that have none")
    sp.add_argument("--sets", default=None, metavar="DIR", help=sets_help)

    v = sub.add_parser("validate", help="Check question sets against the corpus")
    v.add_argument("--sets", default=None, metavar="DIR", help=sets_help)
    v.add_argument("--suites", nargs="+", default=None, metavar="SUITE",
                   help=f"only these suites (default: all); {suite_help}")
    v.add_argument("--data-dir", default=None, metavar="DIR", dest="data_dir",
                   help="folder holding the *chunks.jsonl files (default: the folder of "
                        "paths.chunks_file)")
    v.add_argument("--write-cache", action="store_true", dest="write_cache",
                   help="also write <sets>/.review_cache.json for the review queue "
                        "(whole sets directory only; not with --suites)")

    s = sub.add_parser("sample", help="Draw seed packs from the corpus for drafting questions")
    s.add_argument("--suites", nargs="+", default=None, metavar="SUITE",
                   help=f"only these suites (default: all); {suite_help}")
    s.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    s.add_argument("--factor", type=float, default=1.6,
                   help="seeds per suite = ceil(factor x its quota), so drafters can skip "
                        "unusable ones (default: 1.6)")
    s.add_argument("--data-dir", default=None, metavar="DIR", dest="data_dir",
                   help="folder holding the *chunks.jsonl files (default: the folder of "
                        "paths.chunks_file)")
    s.add_argument("--sets", default=None, metavar="DIR",
                   help="existing question sets; their seed chunks are not handed out again "
                        "(default: eval.sets_dir from config.yaml)")
    s.add_argument("--out", default=None, metavar="DIR",
                   help="where <suite>.jsonl seed packs are written (default: eval/seeds in "
                        "the checkout, wherever this is run from)")

    d = sub.add_parser(
        "draft", help="Draft questions from seed packs with the configured LLM (resumable; the author "
                      "verifies every draft in the console)")
    d.add_argument("--suite", required=True, choices=list(SUITES), metavar="SUITE",
                   help=f"the suite to draft, one of: {', '.join(SUITES)}")
    d.add_argument("--n", type=int, default=None, metavar="N",
                   help="stop when the suite holds N live questions (default: the suite's quota)")
    d.add_argument("--provider", default="freellmapi", metavar="NAME",
                   help="a `providers:` name from config.yaml (default: freellmapi, the free local proxy)")
    d.add_argument("--model", default=None, metavar="ID",
                   help="override the provider's default model")
    d.add_argument("--seeds", default=None, metavar="DIR",
                   help="folder of <suite>.jsonl seed packs from `bench sample` (default: "
                        "eval/seeds in the checkout, wherever this is run from)")
    d.add_argument("--sets", default=None, metavar="DIR", help=sets_help)
    d.add_argument("--data-dir", default=None, metavar="DIR", dest="data_dir",
                   help="folder holding the *chunks.jsonl files (default: the folder of "
                        "paths.chunks_file)")
    d.add_argument("--search-url", default=None, metavar="URL", dest="search_url",
                   help="the warm query API (`rag serve`), used for twin detection and the "
                        "unanswerable check (default: webui.rag_api from config.yaml)")
    d.add_argument("--max-calls", type=int, default=None, metavar="K", dest="max_calls",
                   help="stop after K LLM calls, failed tries included (default: no cap)")
    d.add_argument("--max-score", type=float, default=-2.0, metavar="X", dest="max_score",
                   help="unanswerable: keep a question only if /search's best score for it is below X, "
                        "in the reranker's own units (default: -2.0, cross-encoder logits)")

    b.set_defaults(func=bench_cmd)
