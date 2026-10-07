import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench import cli
from eval.bench.questions import load_sets, question_from_dict, save_suite


def _parse(*argv):
    import main                  # imported here, as test_logging_reconfigure does: it configures logging
    return main.build_parser().parse_args(list(argv))


def _run(*argv):
    args = _parse(*argv)
    args.func(args)


def _q(i, suite="lexical", tier="T1", split="dev", status="draft"):
    return question_from_dict({
        "id": f"{suite[:3]}-{i:04d}", "question": f"q{i}?", "suite": suite, "tier": tier,
        "split": split, "answerable": True, "gold": [{"file": "a.md"}], "nuggets": ["n"],
        "expect_course": None, "provenance": {"author": "draft", "status": status}}, "t")


def _sets(tmp_path, *questions):
    d = tmp_path / "sets"
    d.mkdir(exist_ok=True)
    save_suite(d / "lexical.yaml", list(questions))
    return d


@pytest.fixture
def fake_bench(tmp_path, monkeypatch):
    """Replace the runner's run_bench (and its ledger path) so `bench run` can be
    driven without a pipeline; records what it was called with."""
    state = SimpleNamespace(errors=0, kw=None, questions=None, configs=None)

    def fake(cfg, questions, configs, **kw):
        state.kw, state.questions = kw, [q.id for q in questions]
        state.configs = [n for n, _ in configs]
        run = kw["out_root"] / "fake-run"
        run.mkdir(parents=True, exist_ok=True)
        (run / "summary.json").write_text(
            json.dumps({"R0-bm25": {"errors": state.errors}}), encoding="utf-8")
        return run

    monkeypatch.setattr("eval.bench.runner.run_bench", fake)
    monkeypatch.setattr("eval.bench.runner.DEFAULT_LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(cli, "load_config", lambda path=None: object())
    return state


def test_bench_is_wired_into_the_main_parser():
    a = _parse("bench", "run", "--sets", "s")
    assert a.func is cli.bench_cmd and a.bench_command == "run"
    assert (a.split, a.configs, a.unseal_test, a.verbose, a.limit) == ("dev", "ladder", False, False, None)
    assert _parse("bench", "run").sets is None               # None = the config's eval.sets_dir
    for sub in ("split", "validate", "sample"):
        assert _parse("bench", sub).bench_command == sub
    assert _parse("bench", "report", "some/dir").run_dir == "some/dir"
    with pytest.raises(SystemExit):
        _parse("bench", "run", "--split", "train")


def test_main_docstring_lists_the_bench_command():
    import main
    assert "bench" in main.__doc__


def test_unknown_suites_are_an_error_listing_the_valid_ones(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        _run("bench", "run", "--sets", str(_sets(tmp_path, _q(1))), "--suites", "lexical", "nope")
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "nope" in err and "paraphrase" in err and "unanswerable" in err


def test_a_sets_directory_that_does_not_exist_is_an_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        _run("bench", "run", "--sets", str(tmp_path / "absent"))
    assert e.value.code == 2 and "absent" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        _run("bench", "split", "--sets", str(tmp_path / "absent"))
    assert e.value.code == 2


def test_sets_defaults_to_the_configs_eval_sets_dir(tmp_path, monkeypatch, fake_bench):
    # The review queue reads the same key: the CLI must look where the console looks.
    sets = _sets(tmp_path, _q(1), _q(2, split=None))
    asked = []

    def path(key, default=None):
        asked.append((key, default))
        return sets

    monkeypatch.setattr(cli, "load_config", lambda p=None: SimpleNamespace(path=path))
    _run("bench", "run", "--out-root", str(tmp_path / "r"))
    assert fake_bench.questions == ["lex-0001", "lex-0002"] and ("eval.sets_dir", "eval/sets") in asked

    _run("bench", "split")                                   # the same default for the other commands
    assert all(q.split in ("dev", "test") for q in load_sets(sets))


def test_a_typed_sets_never_reads_the_config(tmp_path, monkeypatch):
    def boom(path=None):
        raise AssertionError("the config must not be read when --sets is given")

    monkeypatch.setattr(cli, "load_config", boom)
    _run("bench", "split", "--sets", str(_sets(tmp_path, _q(1, split=None))))


def test_an_empty_sets_directory_is_an_error_not_a_silent_run(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit) as e:
        _run("bench", "run", "--sets", str(tmp_path / "empty"))
    assert e.value.code == 2 and "no questions" in capsys.readouterr().err


def test_an_unknown_config_name_is_an_error_naming_it_and_the_known_ones(tmp_path, capsys, fake_bench):
    with pytest.raises(SystemExit) as e:
        _run("bench", "run", "--sets", str(_sets(tmp_path, _q(1))), "--configs", "nope")
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "nope" in err and "wide-pools" in err and "ladder" in err


def test_run_passes_its_flags_through_and_defaults_to_the_ladder(tmp_path, fake_bench):
    sets = _sets(tmp_path, _q(1), _q(2, suite="paraphrase"))
    _run("bench", "run", "--sets", str(sets), "--out-root", str(tmp_path / "r"))
    kw = fake_bench.kw
    assert (kw["split"], kw["unseal_test"], kw["limit"], kw["quiet_logs"]) == ("dev", False, None, True)
    assert kw["out_root"] == tmp_path / "r" and kw["ledger"] == tmp_path / "ledger.jsonl"
    assert fake_bench.configs[0] == "R0-bm25" and fake_bench.configs[-1] == "R6-hyde"
    assert fake_bench.questions == ["lex-0001", "par-0002"]

    _run("bench", "run", "--sets", str(sets), "--suites", "paraphrase", "--split", "all",
         "--unseal-test", "--configs", "wide-pools", "--limit", "3", "--verbose",
         "--out-root", str(tmp_path / "r"))
    kw = fake_bench.kw
    assert (kw["split"], kw["unseal_test"], kw["limit"], kw["quiet_logs"]) == ("all", True, 3, False)
    assert fake_bench.questions == ["par-0002"] and fake_bench.configs == ["wide-pools"]


def test_run_exits_one_when_any_row_failed_and_zero_otherwise(tmp_path, fake_bench, capsys):
    base = ["bench", "run", "--sets", str(_sets(tmp_path, _q(1))), "--out-root", str(tmp_path / "r")]
    _run(*base)                                              # clean: returns normally
    fake_bench.errors = 2
    with pytest.raises(SystemExit) as e:
        _run(*base)
    assert e.value.code == 1 and "2" in capsys.readouterr().err


def test_a_sealed_split_without_unsealing_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda path=None: object())
    monkeypatch.setattr("eval.bench.runner.DEFAULT_LEDGER", tmp_path / "ledger.jsonl")
    sets = _sets(tmp_path, _q(1, split="test"))
    with pytest.raises(SystemExit) as e:
        _run("bench", "run", "--sets", str(sets), "--split", "test", "--out-root", str(tmp_path / "r"))
    assert e.value.code == 2 and "--unseal-test" in capsys.readouterr().err
    assert not (tmp_path / "ledger.jsonl").exists() and not (tmp_path / "r").exists()


def test_a_split_with_no_questions_points_at_bench_split(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda path=None: object())
    sets = _sets(tmp_path, _q(1, split=None))               # nothing has been split yet
    with pytest.raises(SystemExit) as e:
        _run("bench", "run", "--sets", str(sets), "--out-root", str(tmp_path / "r"))
    assert e.value.code == 2 and "bench split" in capsys.readouterr().err


def test_split_assigns_missing_splits_and_rewrites_only_what_changed(tmp_path):
    sets = tmp_path / "sets"
    sets.mkdir()
    # a.yaml mixes two suites (as eval/smoke/smoke.yaml does) and holds a rejected question
    save_suite(sets / "a.yaml", [_q(1, split=None), _q(2, split=None),
                                 _q(3, suite="paraphrase", split=None),
                                 _q(4, split=None, status="rejected")])
    # b.yaml is fully split already: it must come out byte-identical, comments and all
    b = sets / "b.yaml"
    save_suite(b, [_q(5, suite="code", tier="T2", split="test")])
    b.write_text("# hand-written note\n" + b.read_text(encoding="utf-8"), encoding="utf-8")
    b_before = b.read_bytes()

    _run("bench", "split", "--sets", str(sets))

    got = {q.id: q for q in load_sets(sets, include_rejected=True)}
    assert set(got) == {"lex-0001", "lex-0002", "par-0003", "lex-0004", "cod-0005"}   # nothing lost
    assert all(got[i].split in ("dev", "test") for i in ("lex-0001", "lex-0002", "par-0003"))
    assert got["lex-0004"].split is None                    # a rejected question is not given one
    assert {got["lex-0001"].split, got["lex-0002"].split} == {"dev", "test"}   # balanced in its stratum
    ids = [d["id"] for d in yaml.safe_load((sets / "a.yaml").read_text(encoding="utf-8"))]
    assert ids == ["lex-0001", "lex-0002", "par-0003", "lex-0004"]    # file membership and order kept
    assert b.read_bytes() == b_before

    after = (sets / "a.yaml").read_bytes()
    _run("bench", "split", "--sets", str(sets))             # idempotent: nothing left to assign
    assert (sets / "a.yaml").read_bytes() == after


def test_report_rerenders_from_the_run_files_and_leaves_the_records_alone(tmp_path):
    from eval.bench.report import render
    run = tmp_path / "r"
    run.mkdir()
    summary = {"x": {"overall": {}, "by_suite": {}, "by_tier": {}, "latency": {},
                     "cold_rows": 0, "errors": 0}}
    meta = {"run_id": "r1", "git": {"sha": "abc1234", "dirty": False}, "config_digest": "d",
            "index": {}, "split": "dev", "n_questions": 1, "suites": ["lexical"],
            "test_ledger_count": 0}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    (run / "report.md").write_text("stale", encoding="utf-8")

    _run("bench", "report", str(run))

    assert (run / "report.md").read_text(encoding="utf-8") == render(summary, meta)
    assert json.loads((run / "summary.json").read_text(encoding="utf-8")) == summary
    assert json.loads((run / "run.json").read_text(encoding="utf-8")) == meta


def test_report_on_something_that_is_not_a_run_is_a_clean_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        _run("bench", "report", str(tmp_path))
    assert e.value.code == 2 and "summary.json" in capsys.readouterr().err


def test_seed_count_is_ceil_of_factor_times_quota_without_float_noise():
    assert cli._seed_count(1.6, 40) == 64
    assert cli._seed_count(1.1, 50) == 55                    # 1.1 * 50 is 55.00000000000001
    assert cli._seed_count(1.5, 45) == 68                    # 67.5 rounds UP


# ---- sample and validate, and where the commands write by default --------------
# Temp directories only: never the real eval/sets, eval/seeds or eval/runs.

PROSE = ("The bisection method halves the bracketing interval at every step until the "
         "tolerance is met. ") * 4                           # past the sampler's 300-char floor


def _chunk(doc_id, source_file, course="Numerical Methods", text=PROSE):
    return {"doc_id": doc_id, "text": text,
            "metadata": {"source_file": source_file, "file_type": "note",
                         "course_name": course, "domain": "math"}}


def _corpus(tmp_path, *records):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    (data / "x_chunks.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return data


def _scoped_corpus(tmp_path):
    return _corpus(tmp_path,
                   *[_chunk(f"c{i}", f"calc\\f{i}.md", course="Calculus") for i in range(4)],
                   *[_chunk(f"s{i}", f"stat\\f{i}.md", course="Statistics") for i in range(4)],
                   _chunk("a0", "alg\\f0.md", course="Algebra"))


def test_the_default_output_folders_are_anchored_at_the_checkout():
    from eval.bench import runner
    root = Path(cli.__file__).resolve().parents[2]
    assert (root / "main.py").is_file()                       # that is the checkout
    assert cli.DEFAULT_SEEDS == root / "eval" / "seeds"
    assert Path(runner.DEFAULT_RUNS) == root / "eval" / "runs"


def test_run_writes_to_the_checkouts_runs_folder_not_the_working_directory(
        tmp_path, monkeypatch, fake_bench):
    """The runner anchors the ledger at the checkout for exactly this reason: a run
    started from another directory must not quietly start a second eval/runs."""
    runs = tmp_path / "checkout" / "eval" / "runs"
    monkeypatch.setattr("eval.bench.runner.DEFAULT_RUNS", runs)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _run("bench", "run", "--sets", str(_sets(tmp_path, _q(1))))
    assert fake_bench.kw["out_root"] == runs
    assert list(elsewhere.iterdir()) == []                    # nothing made under the working directory


def test_sample_writes_to_the_checkouts_seeds_folder_not_the_working_directory(
        tmp_path, monkeypatch):
    seeds = tmp_path / "checkout" / "eval" / "seeds"
    monkeypatch.setattr(cli, "DEFAULT_SEEDS", seeds)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _run("bench", "sample", "--suites", "scoped", "--factor", "0.1",
         "--data-dir", str(_scoped_corpus(tmp_path)), "--sets", str(tmp_path / "no-sets-yet"))
    assert (seeds / "scoped.jsonl").is_file()
    assert list(elsewhere.iterdir()) == []


def test_sample_prints_how_many_seeds_came_from_each_stratum(tmp_path, capsys):
    out = tmp_path / "seeds"
    _run("bench", "sample", "--suites", "scoped", "--factor", "0.2", "--out", str(out),
         "--data-dir", str(_scoped_corpus(tmp_path)), "--sets", str(tmp_path / "no-sets-yet"))
    said = capsys.readouterr().out
    pack = [json.loads(line) for line in (out / "scoped.jsonl").read_text(encoding="utf-8").splitlines()]
    per_course = Counter(rec["course_name"] for rec in pack)
    assert len(pack) == 8 and set(per_course) == {"Calculus", "Statistics", "Algebra"}
    assert "per stratum (3)" in said
    for course, n in per_course.items():                      # the scoped suite's stratum is the course
        assert re.search(rf"^\s+{n}\s+{course}$", said, re.MULTILINE), (course, n, said)


def test_a_suite_with_no_strata_prints_only_its_total(tmp_path, capsys):
    # unanswerable draws topic neighbourhoods, not seeds: there is no stratum to count
    _run("bench", "sample", "--suites", "unanswerable", "--out", str(tmp_path / "seeds"),
         "--data-dir", str(_scoped_corpus(tmp_path)), "--sets", str(tmp_path / "no-sets-yet"))
    said = capsys.readouterr().out
    assert "unanswerable: 3 record(s)" in said and "per stratum" not in said


@pytest.mark.parametrize("factor", ["0", "-1.5"])
def test_sample_refuses_a_factor_that_is_not_positive_before_reading_anything(
        tmp_path, capsys, factor):
    out = tmp_path / "seeds"
    with pytest.raises(SystemExit) as e:
        _run("bench", "sample", "--factor", factor, "--out", str(out),
             "--data-dir", str(tmp_path / "no-data"), "--sets", str(tmp_path / "no-sets"))
    assert e.value.code == 2
    assert "--factor must be positive" in capsys.readouterr().err   # not "data directory ... does not exist"
    assert not out.exists()


def test_validate_exits_one_when_a_question_cannot_be_scored(tmp_path, capsys):
    sets = _sets(tmp_path, _q(1))                              # gold is a.md ...
    data = _corpus(tmp_path, _chunk("o1", "other.md"))         # ... which the corpus does not hold
    with pytest.raises(SystemExit) as e:
        _run("bench", "validate", "--sets", str(sets), "--data-dir", str(data))
    assert e.value.code == 1
    said = capsys.readouterr().out
    assert "gold-file-missing" in said and "[error] lex-0001" in said and "1 error(s)" in said


def test_validate_exits_zero_when_the_set_agrees_with_the_corpus(tmp_path, capsys):
    sets = _sets(tmp_path, _q(1))
    data = _corpus(tmp_path, _chunk("g1", "a.md"))
    _run("bench", "validate", "--sets", str(sets), "--data-dir", str(data))    # returns, no SystemExit
    assert "0 error(s)" in capsys.readouterr().out


def test_validate_writes_the_review_cache_where_the_console_reads_it(
        tmp_path, monkeypatch, management_module):
    sets = _sets(tmp_path, _q(1))
    data = _corpus(tmp_path, _chunk("g1", "a.md"))
    _run("bench", "validate", "--sets", str(sets), "--data-dir", str(data), "--write-cache")
    cache = sets / ".review_cache.json"
    written = json.loads(cache.read_text(encoding="utf-8"))
    assert list(written) == ["lex-0001"] and written["lex-0001"]["gold_texts"][0]["file"] == "a.md"
    # the console's review queue reads the very same file
    monkeypatch.setattr(management_module, "EVAL_SETS_DIR", sets)
    assert management_module._eval_cache() == written


def test_validate_without_write_cache_leaves_no_cache(tmp_path):
    sets = _sets(tmp_path, _q(1))
    _run("bench", "validate", "--sets", str(sets), "--data-dir", str(_corpus(tmp_path, _chunk("g1", "a.md"))))
    assert not (sets / ".review_cache.json").exists()


def test_validate_refuses_write_cache_together_with_suites(tmp_path, capsys):
    sets = _sets(tmp_path, _q(1))
    with pytest.raises(SystemExit) as e:
        _run("bench", "validate", "--sets", str(sets), "--suites", "lexical", "--write-cache",
             "--data-dir", str(tmp_path / "no-data"))
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "--write-cache" in err and "WHOLE" in err           # the refusal, not a missing data directory
    assert not (sets / ".review_cache.json").exists()
