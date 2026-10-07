import json
import logging
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.questions import question_from_dict
from eval.bench.runner import RunSetupError, SealedSplitError, run_bench
from src.retrieval.hyde_cache import HydeCache


class FakeDoc:
    def __init__(self, i, f):
        self.id, self.metadata = i, {"source_file": f}


class FakeHyde:
    cache = object()            # pretend a cache is attached


class FakeRetriever:
    def _get_collection(self):
        return type("C", (), {"count": staticmethod(lambda: 3)})()


class FakeRag:
    hyde, retriever = FakeHyde(), FakeRetriever()

    def __init__(self):
        self.calls = []

    def search(self, q, **kw):
        self.calls.append(kw)
        if "boom" in q:
            raise RuntimeError("kaboom")
        docs = [FakeDoc("1", "x.md"), FakeDoc("2", "a.md")]
        return docs, {"timings": {"total": 5.0}, "lanes_run": {"dense": 2}, "cold": False,
                      "hyde_cache": "off"}


def _q(i, split, text="q?", **over):
    d = {
        "id": f"lex-{i:04d}", "question": text, "suite": "lexical", "tier": "T1",
        "split": split, "answerable": True, "gold": [{"file": "a.md"}], "nuggets": ["n"],
        "expect_course": None, "provenance": {"author": "draft", "status": "draft"}}
    d.update(over)
    return question_from_dict(d, "t")


def test_dev_run_writes_records(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.bench.runner.index_fingerprint", lambda cfg, rag: {"dense_count": 3})
    monkeypatch.setattr("eval.bench.runner.git_state", lambda repo: {"sha": "abc", "dirty": False})
    monkeypatch.setattr("eval.bench.runner.config_digest", lambda cfg: "d")
    rag = FakeRag()
    qs = [_q(1, "dev"), _q(2, "test"), _q(3, "dev", "boom")]
    run = run_bench(None, qs, [("R2-hybrid", {"lanes": ["dense", "sparse"], "top_k": 10}),
                               ("F:none", {"lanes": [], "top_k": 10})],
                    split="dev", out_root=tmp_path, rag=rag, progress=lambda *_: None)
    rows = [json.loads(l) for l in (run / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["qid"] for r in rows} == {"lex-0001", "lex-0003"}          # test split excluded
    ok = next(r for r in rows if r["qid"] == "lex-0001" and r["config"] == "R2-hybrid")
    assert ok["metrics"]["hit@5"] == 1.0 and ok["metrics"]["mrr@10"] == 0.5
    assert any(r.get("error", "").startswith("RuntimeError") for r in rows)
    assert all(r.get("skipped") == "no lanes" for r in rows if r["config"] == "F:none")
    assert all("lanes" in kw for kw in rag.calls) and len(rag.calls) == 2   # F:none never searched
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["R2-hybrid"]["errors"] == 1
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert meta["status"] == "done" and (run / "report.md").exists()


@pytest.mark.parametrize("split", ["test", "all"])
def test_sealed_splits_need_unsealing_and_are_ledgered(tmp_path, monkeypatch, split):
    monkeypatch.setattr("eval.bench.runner.index_fingerprint", lambda cfg, rag: {})
    monkeypatch.setattr("eval.bench.runner.git_state", lambda repo: {"sha": "abc", "dirty": False})
    monkeypatch.setattr("eval.bench.runner.config_digest", lambda cfg: "d")
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(SealedSplitError):
        run_bench(None, [_q(1, "test")], [("x", {"lanes": ["dense"]})], split=split,
                  out_root=tmp_path, rag=FakeRag(), ledger=ledger, progress=lambda *_: None)
    assert not ledger.exists()
    run_bench(None, [_q(1, "test")], [("x", {"lanes": ["dense"]})], split=split,
              unseal_test=True, out_root=tmp_path, rag=FakeRag(), ledger=ledger,
              progress=lambda *_: None)
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1


# --- everything below pins behaviour the plan's two tests do not reach --------------

ONE = [("x", {"lanes": ["dense"]})]


@pytest.fixture
def stamped(monkeypatch):
    """Stand-ins for the three fingerprints: no git, no index, no real config."""
    monkeypatch.setattr("eval.bench.runner.index_fingerprint", lambda cfg, rag: {"dense_count": 3})
    monkeypatch.setattr("eval.bench.runner.git_state",
                        lambda repo: {"sha": "abc1234def", "dirty": False})
    monkeypatch.setattr("eval.bench.runner.config_digest", lambda cfg: "d")


def _go(tmp_path, qs, configs=ONE, rag=None, **kw):
    """run_bench with every output path inside tmp_path (never the repo's real
    ledger or eval/runs)."""
    kw.setdefault("split", "dev")
    kw.setdefault("out_root", tmp_path / "runs")
    kw.setdefault("ledger", tmp_path / "ledger.jsonl")
    kw.setdefault("progress", lambda *_: None)
    return run_bench(kw.pop("cfg", None), qs, configs, rag=rag or FakeRag(), **kw)


def _rows(run):
    return [json.loads(l) for l in (run / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]


def _json(run, name):
    return json.loads((run / name).read_text(encoding="utf-8"))


def test_all_split_runs_dev_and_test_and_ledger_entry_has_the_agreed_keys(tmp_path, stamped):
    ledger = tmp_path / "ledger.jsonl"
    run = _go(tmp_path, [_q(1, "dev"), _q(2, "test")], split="all", unseal_test=True)
    assert {r["qid"] for r in _rows(run)} == {"lex-0001", "lex-0002"}
    (entry,) = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines()]
    assert set(entry) == {"at", "run_id", "split", "configs", "git"}
    assert entry["split"] == "all" and entry["configs"] == ["x"] and entry["run_id"] == run.name
    assert entry["git"] == {"sha": "abc1234def", "dirty": False}
    assert _json(run, "run.json")["test_ledger_count"] == 1
    # a dev run neither writes the ledger nor forgets that the test split was opened
    dev = _go(tmp_path, [_q(1, "dev")], out_root=tmp_path / "again")
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    assert _json(dev, "run.json")["test_ledger_count"] == 1


def test_a_sealed_run_is_ledgered_before_its_first_search(tmp_path, stamped):
    ledger = tmp_path / "ledger.jsonl"
    seen = []

    class Spy(FakeRag):
        def search(self, q, **kw):
            seen.append(ledger.exists() and len(ledger.read_text(encoding="utf-8").splitlines()))
            return super().search(q, **kw)

    _go(tmp_path, [_q(1, "test")], rag=Spy(), split="test", unseal_test=True)
    assert seen == [1]


def test_rejected_questions_are_never_run(tmp_path, stamped):
    rejected = _q(2, "dev", provenance={"author": "draft", "status": "rejected"})
    assert [r["qid"] for r in _rows(_go(tmp_path, [_q(1, "dev"), rejected]))] == ["lex-0001"]


def test_a_selection_that_leaves_nothing_is_refused_before_anything_is_written(tmp_path, stamped):
    with pytest.raises(RunSetupError, match="split"):
        _go(tmp_path, [_q(1, "test")])                       # dev asked, only test exists
    with pytest.raises(RunSetupError, match="bench split"):  # unassigned splits get a hint
        _go(tmp_path, [_q(1, None)])
    dead = _q(1, None, provenance={"author": "draft", "status": "rejected"})
    with pytest.raises(RunSetupError) as e:                  # ...but a rejected one is not fixable by it
        _go(tmp_path, [dead])
    assert "bench split" not in str(e.value)
    with pytest.raises(RunSetupError):
        _go(tmp_path, [_q(1, "dev")], limit=0)
    with pytest.raises(RunSetupError):
        _go(tmp_path, [_q(1, "dev")], configs=[])
    with pytest.raises(RunSetupError, match="dup"):
        _go(tmp_path, [_q(1, "dev")], configs=[("x", {"lanes": ["dense"]}), ("x", {"lanes": ["sparse"]})])
    with pytest.raises(RunSetupError, match="train"):
        _go(tmp_path, [_q(1, "dev")], split="train")
    assert not (tmp_path / "runs").exists() and not (tmp_path / "ledger.jsonl").exists()


def test_limit_keeps_the_first_n_questions_after_the_split_filter(tmp_path, stamped):
    qs = [_q(1, "test"), _q(2, "dev"), _q(3, "dev"), _q(4, "dev")]
    run = _go(tmp_path, qs, limit=2)
    assert [r["qid"] for r in _rows(run)] == ["lex-0002", "lex-0003"]
    assert _json(run, "run.json")["n_questions"] == 2


def test_run_json_records_what_was_measured(tmp_path, stamped):
    cfgs = [("R2-hybrid", {"lanes": ["dense", "sparse"], "top_k": 10})]
    run = _go(tmp_path, [_q(1, "dev"), _q(2, "dev")], cfgs)
    meta = _json(run, "run.json")
    assert meta["status"] == "done" and meta["run_id"] == run.name
    assert meta["git"] == {"sha": "abc1234def", "dirty": False} and meta["config_digest"] == "d"
    assert meta["index"] == {"dense_count": 3} and meta["split"] == "dev"
    assert meta["suites"] == ["lexical"] and meta["n_questions"] == 2 and meta["seed"] == 0
    assert meta["configs"] == {"R2-hybrid": {"lanes": ["dense", "sparse"], "top_k": 10}}
    assert len(meta["questions_digest"]) == 16 and meta["test_ledger_count"] == 0
    assert meta["started_at"] <= meta["finished_at"] and meta["elapsed_s"] >= 0


def test_question_digest_survives_a_date_and_ignores_order(tmp_path, stamped):
    # A hand-typed `reviewed_at: 2026-10-05` loads as a date, which json cannot dump.
    prov = {"author": "draft", "status": "draft", "reviewed_at": date(2026, 10, 5)}
    one, two = _q(1, "dev", "one?", provenance=prov), _q(2, "dev", "two?", provenance=prov)
    other = _q(2, "dev", "changed?", provenance=prov)
    digests = [_json(_go(tmp_path, qs, out_root=tmp_path / tag), "run.json")["questions_digest"]
               for tag, qs in (("a", [one, two]), ("b", [two, one]), ("c", [one, other]))]
    assert digests[0] == digests[1] != digests[2]


def test_a_crash_outside_a_question_marks_the_run_failed_and_propagates(tmp_path, stamped):
    class Broken(FakeRag):
        def search(self, q, **kw):
            return [], {}                  # a search() that broke its own echo contract

    with pytest.raises(KeyError):
        _go(tmp_path, [_q(1, "dev")], rag=Broken())
    (run,) = (tmp_path / "runs").iterdir()
    meta = _json(run, "run.json")
    assert meta["status"] == "failed" and "KeyError" in meta["error"]


def test_src_logs_are_quiet_during_the_run_and_restored_after(tmp_path, stamped):
    lg = logging.getLogger("src")
    before = lg.level
    seen = []

    class Spy(FakeRag):
        def search(self, q, **kw):
            seen.append(lg.getEffectiveLevel())
            return super().search(q, **kw)

    try:
        lg.setLevel(logging.INFO)
        _go(tmp_path, [_q(1, "dev")], rag=Spy(), out_root=tmp_path / "quiet")
        assert seen == [logging.WARNING] and lg.level == logging.INFO
        seen.clear()
        _go(tmp_path, [_q(1, "dev")], rag=Spy(), out_root=tmp_path / "loud", quiet_logs=False)
        assert seen == [logging.INFO]                       # --verbose leaves them alone
        seen.clear()
        lg.setLevel(logging.ERROR)                          # already quieter: never LOWER it
        _go(tmp_path, [_q(1, "dev")], rag=Spy(), out_root=tmp_path / "quieter")
        assert seen == [logging.ERROR] and lg.level == logging.ERROR
    finally:
        lg.setLevel(before)


def test_logs_are_restored_even_when_the_run_crashes(tmp_path, stamped):
    lg = logging.getLogger("src")
    before = lg.level

    class Broken(FakeRag):
        def search(self, q, **kw):
            return [], {}

    try:
        lg.setLevel(logging.INFO)
        with pytest.raises(KeyError):
            _go(tmp_path, [_q(1, "dev")], rag=Broken())
        assert lg.level == logging.INFO
    finally:
        lg.setLevel(before)


def test_a_search_that_finds_nothing_scores_zero_without_an_error(tmp_path, stamped):
    class Empty(FakeRag):
        def search(self, q, **kw):
            return [], {"timings": {"total": 1.0}, "lanes_run": {}, "cold": False, "hyde_cache": "off"}

    (row,) = _rows(_go(tmp_path, [_q(1, "dev")], rag=Empty()))
    assert "error" not in row and row["ids"] == [] and set(row["metrics"].values()) == {0.0}


def test_unanswerable_questions_are_searched_but_carry_no_retrieval_metrics(tmp_path, stamped):
    una = _q(2, "dev", suite="unanswerable", answerable=False, gold=[], nuggets=[])
    run = _go(tmp_path, [_q(1, "dev"), una])
    row = next(r for r in _rows(run) if r["qid"] == "lex-0002")
    assert row["answerable"] is False and "metrics" not in row and row["ids"] == ["1", "2"]
    s = _json(run, "summary.json")["x"]
    assert s["overall"]["ndcg@10"]["n"] == 1 and set(s["by_suite"]) == {"lexical"}
    assert s["latency"]["total"]["n"] == 2          # ...yet it still counts toward latency


class Scripted(FakeRag):
    """Misses on 'miss' questions, reports the first call as cold, and times call k at 10k ms."""
    n = 0

    def search(self, q, **kw):
        self.n += 1
        docs, info = super().search(q, **kw)
        if "miss" in q:
            docs = [FakeDoc("9", "z.md")]
        return docs, {**info, "cold": self.n == 1,
                      "timings": {"total": 10.0 * self.n, "rerank": 1.0}}


def test_summary_has_cis_groups_and_warm_only_latency(tmp_path, stamped):
    qs = [_q(1, "dev"), _q(2, "dev", "miss?"), _q(3, "dev"), _q(4, "dev", tier="T2")]
    s = _json(_go(tmp_path, qs, rag=Scripted()), "summary.json")["x"]
    hit = s["overall"]["hit@5"]
    assert set(hit) == {"mean", "lo", "hi", "n"} and hit["n"] == 4
    assert hit["mean"] == 0.75 and hit["lo"] < 0.75 < hit["hi"]
    assert set(s["by_suite"]) == {"lexical"} and set(s["by_tier"]) == {"T1", "T2"}
    assert s["by_tier"]["T2"]["hit@5"]["n"] == 1
    assert s["cold_rows"] == 1 and s["errors"] == 0
    total = s["latency"]["total"]                    # calls 2..4 -> 20, 30, 40; the cold call is out
    assert total["n"] == 3 and total["p50"] == 30.0 and {"p95", "p99"} <= set(total)
    assert s["latency"]["rerank"]["n"] == 3
    assert "paired" not in s


def test_summary_reports_drafted_and_hand_written_questions_separately(tmp_path, stamped):
    hand = {"author": "owner", "status": "verified"}
    qs = [_q(1, "dev"), _q(2, "dev", "miss?"), _q(3, "dev", provenance=hand)]
    run = _go(tmp_path, qs, rag=Scripted())
    by_author = _json(run, "summary.json")["x"]["by_author"]
    assert list(by_author) == ["draft", "owner"]                # the schema's order
    assert by_author["draft"]["hit@5"]["n"] == 2 and by_author["owner"]["hit@5"]["n"] == 1
    assert by_author["draft"]["hit@5"]["mean"] == 0.5            # the drafted miss stays in its own row
    assert by_author["owner"]["hit@5"]["mean"] == 1.0
    assert "by author" in (run / "report.md").read_text(encoding="utf-8")
    only_drafts = _go(tmp_path, [_q(1, "dev")], out_root=tmp_path / "drafts")
    assert list(_json(only_drafts, "summary.json")["x"]["by_author"]) == ["draft"]


class Staged(FakeRag):
    """The gold document ranks first only when the scope lane is allowed."""

    def search(self, q, **kw):
        docs, info = super().search(q, **kw)
        return ([FakeDoc("2", "a.md")] if "dense_scope" in kw["lanes"] else [FakeDoc("9", "z.md")]), info


def test_consecutive_ladder_rungs_get_paired_comparisons(tmp_path, stamped):
    base = ["dense", "sparse"]
    cfgs = [("R2-hybrid", {"lanes": base}), ("R3-routing", {"lanes": base + ["dense_scope"]}),
            ("R4-code", {"lanes": base + ["dense_scope", "dense_code"]})]
    s = _json(_go(tmp_path, [_q(i, "dev") for i in range(1, 11)], cfgs, rag=Staged()), "summary.json")
    assert "paired" not in s["R2-hybrid"]
    p = s["R3-routing"]["paired"]
    assert p["vs"] == "R2-hybrid" and p["n"] == 10
    assert p["ndcg@10"]["mean"] == 1.0 and p["ndcg@10"]["lo"] > 0       # B minus A
    assert p["hit@5_mcnemar_p"] == pytest.approx(2 * 0.5 ** 10)          # 10 only-B-right, 0 only-A-right
    q = s["R4-code"]["paired"]
    assert q["vs"] == "R3-routing" and q["ndcg@10"]["mean"] == 0.0 and q["hit@5_mcnemar_p"] == 1.0


def test_rungs_that_are_not_neighbours_are_not_compared(tmp_path, stamped):
    cfgs = [("R2-hybrid", {"lanes": ["dense"]}), ("R4-code", {"lanes": ["dense", "dense_scope"]}),
            ("custom", {"lanes": ["dense"]})]
    s = _json(_go(tmp_path, [_q(1, "dev")], cfgs, rag=Staged()), "summary.json")
    assert not any("paired" in v for v in s.values())


def test_a_pair_only_uses_questions_both_configs_scored(tmp_path, stamped):
    class Flaky(Staged):
        def search(self, q, **kw):
            if "boom" in q and "dense_scope" in kw["lanes"]:
                raise RuntimeError("only R3 fails here")
            return super().search(q.replace("boom", ""), **kw)

    cfgs = [("R2-hybrid", {"lanes": ["dense"]}), ("R3-routing", {"lanes": ["dense", "dense_scope"]})]
    s = _json(_go(tmp_path, [_q(1, "dev"), _q(2, "dev", "boom")], cfgs, rag=Flaky()), "summary.json")
    assert s["R3-routing"]["errors"] == 1 and s["R3-routing"]["paired"]["n"] == 1


def test_a_hyde_cache_is_attached_when_the_pipeline_has_none(tmp_path, stamped):
    rag = FakeRag()
    rag.hyde = SimpleNamespace(cache=None)
    asked = []

    def path(key, default=None):
        asked.append((key, default))
        return tmp_path / "cache" / "h.sqlite"

    _go(tmp_path, [_q(1, "dev")], rag=rag, cfg=SimpleNamespace(path=path))
    assert isinstance(rag.hyde.cache, HydeCache) and (tmp_path / "cache" / "h.sqlite").exists()
    assert asked == [("retrieval.hyde_cache.path", "data/cache/hyde.sqlite")]


def test_an_existing_hyde_cache_is_left_alone(tmp_path, stamped):
    mine = object()
    rag = FakeRag()
    rag.hyde = SimpleNamespace(cache=mine)
    _go(tmp_path, [_q(1, "dev")], rag=rag)          # cfg=None: touching cfg.path would raise
    assert rag.hyde.cache is mine


def test_the_search_overrides_reach_search_untouched(tmp_path, stamped):
    rag = FakeRag()
    ov = {"lanes": ["dense"], "top_k": 10, "rerank": "none", "hyde": False, "auto_preset": False}
    _go(tmp_path, [_q(1, "dev")], [("x", ov)], rag=rag)
    assert rag.calls == [ov]


def test_progress_prints_one_line_per_config(tmp_path, stamped):
    lines = []
    _go(tmp_path, [_q(1, "dev")], [("a", {"lanes": ["dense"]}), ("b", {"lanes": []})],
        progress=lines.append)
    assert len(lines) == 2 and "a" in lines[0] and "b" in lines[1] and "[2/2]" in lines[1]


def test_report_md_is_written_and_names_the_run(tmp_path, stamped):
    run = _go(tmp_path, [_q(1, "dev")])
    text = (run / "report.md").read_text(encoding="utf-8")
    assert run.name in text and "x" in text and "Errors" in text


# ---- warm-up (added 2026-10-06: the first embed/rerank paid ~5 s on a "warm" row) ----

def test_warm_up_is_one_discarded_search_per_rerank_mode(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.bench.runner.index_fingerprint", lambda cfg, rag: {"dense_count": 3})
    monkeypatch.setattr("eval.bench.runner.git_state", lambda repo: {"sha": "abc", "dirty": False})
    monkeypatch.setattr("eval.bench.runner.config_digest", lambda cfg: "d")
    rag = FakeRag()
    run = run_bench(None, [_q(1, "dev")],
                    [("a", {"lanes": ["dense"], "rerank": "none"}),
                     ("b", {"lanes": ["dense"], "rerank": "cross_encoder"}),
                     ("c", {"lanes": ["dense", "sparse"], "rerank": "cross_encoder"}),
                     ("F:none", {"lanes": []})],
                    split="dev", out_root=tmp_path, rag=rag, progress=lambda *_: None,
                    ledger=tmp_path / "ledger.jsonl", warmup=True)
    warm = [kw for kw in rag.calls if "lanes" not in kw]
    assert sorted(kw["rerank"] for kw in warm) == ["cross_encoder", "none"]
    assert all(kw["hyde"] is False and kw["top_k"] == 1 for kw in warm)
    rows = (run / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 4          # 1 question x 4 configs: warm-ups are never recorded


def test_no_warm_up_unless_asked(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.bench.runner.index_fingerprint", lambda cfg, rag: {})
    monkeypatch.setattr("eval.bench.runner.git_state", lambda repo: {"sha": "abc", "dirty": False})
    monkeypatch.setattr("eval.bench.runner.config_digest", lambda cfg: "d")
    rag = FakeRag()
    run_bench(None, [_q(1, "dev")], [("a", {"lanes": ["dense"]})], split="dev",
              out_root=tmp_path, rag=rag, progress=lambda *_: None,
              ledger=tmp_path / "ledger.jsonl")
    assert all("lanes" in kw for kw in rag.calls)
