"""External-judge import tests (session 16).

Run:  python -m pytest tests/ -q

apply_judge_scores() is a pure post-process over two files — no pipeline, no
index, no LLM — so the whole merge path is testable offline. What matters here:
a partial scores file must be REFUSED rather than silently averaged over a
subset, because a judge run that quietly covered 60 of 100 questions would
report a confident number for a suite it never finished grading.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from eval.eval_runner import apply_judge_scores
from src.utils.config_loader import Config


def make_results(n=3):
    per_query = []
    for i in range(n):
        per_query.append({
            "question": f"q{i}", "domain": "ml",
            "retrieval_hit": True, "reciprocal_rank": 1.0, "first_hit_rank": 1,
            "recall_at_k": None,
            "scope": {"fired": True, "fired_domains": ["ml"],
                      "correct": True, "recalled": True},
            "course_hit": None,
            "keyword_recall": 1.0, "citation_validity": 1.0,
            "dangling_citations": [], "groundedness": 0.5,
            "citation_support": None,
            "confidence": ["HIGH", "MEDIUM", "LOW"][i % 3],
            "answered": True, "n_citations": 2, "answer_preview": "...",
        })
    return {
        "meta": {"generated_at": "2026-07-21T00:00:00", "golden_file": "g.yaml",
                 "mode": "full", "judge": False, "ngram_n": 2,
                 "n_queries": n, "elapsed_seconds": 1.0, "seconds_per_query": 0.3},
        "retrieval": {"course_hit_rate": {"value": None, "n_scored": 0}},
        "answer": {}, "calibration": {}, "summary": {},
        "per_query": per_query,
    }


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "eval").mkdir()
    res = tmp_path / "eval" / "results.json"
    res.write_text(json.dumps(make_results()), encoding="utf-8")
    return Config({}, tmp_path), tmp_path


def write_scores(tmp_path, rows):
    p = tmp_path / "eval" / "scores.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return "eval/scores.jsonl"


def test_merge_rebuilds_the_judge_tier(workspace):
    cfg, tmp = workspace
    scores = write_scores(tmp, [
        {"i": 0, "correct": 1.0, "grounded": 1.0, "unsupported_claims": []},
        {"i": 1, "correct": 0.5, "grounded": 0.5, "unsupported_claims": ["x"]},
        {"i": 2, "correct": 0.0, "grounded": 0.0, "unsupported_claims": ["y", "z"]},
    ])
    out = apply_judge_scores(cfg, "eval/results.json", scores)
    assert out["answer"]["judge_correctness"] == {"value": 0.5, "n_scored": 3}
    assert out["answer"]["judge_groundedness"] == {"value": 0.5, "n_scored": 3}
    assert out["meta"]["judge"] == "external"
    assert out["per_query"][2]["judge_unsupported"] == ["y", "z"]


def test_calibration_is_rebuilt_from_the_merged_scores(workspace):
    cfg, tmp = workspace
    scores = write_scores(tmp, [
        {"i": 0, "correct": 1.0, "grounded": 1.0},     # HIGH
        {"i": 1, "correct": 0.6, "grounded": 0.6},     # MEDIUM
        {"i": 2, "correct": 0.2, "grounded": 0.2},     # LOW
    ])
    out = apply_judge_scores(cfg, "eval/results.json", scores)
    buckets = out["calibration"]["by_judge_correctness"]
    assert buckets["buckets"]["HIGH"]["mean"] == 1.0
    assert buckets["buckets"]["LOW"]["mean"] == 0.2
    assert buckets["gap"] == 0.8          # calibrated: HIGH scores above LOW


def test_partial_scores_are_refused(workspace):
    """The important one — silence about missing questions would be a lie."""
    cfg, tmp = workspace
    scores = write_scores(tmp, [{"i": 0, "correct": 1.0, "grounded": 1.0}])
    with pytest.raises(ValueError, match="1/3"):
        apply_judge_scores(cfg, "eval/results.json", scores)


def test_out_of_range_scores_are_clamped(workspace):
    cfg, tmp = workspace
    scores = write_scores(tmp, [
        {"i": 0, "correct": 7, "grounded": -3},
        {"i": 1, "correct": 1.0, "grounded": 1.0},
        {"i": 2, "correct": 1.0, "grounded": 1.0},
    ])
    out = apply_judge_scores(cfg, "eval/results.json", scores)
    assert out["per_query"][0]["judge_correct"] == 1.0
    assert out["per_query"][0]["judge_grounded"] == 0.0


def test_failed_generations_are_excluded_not_scored_zero(tmp_path):
    """An endpoint 502 must not read as a bad answer.

    Questions whose generation raised are never exported to the judge, so the
    scores file legitimately lacks them — and they must stay unscored rather
    than counting as 0.0 and dragging the judge average down.
    """
    (tmp_path / "eval").mkdir()
    res = make_results(3)
    res["per_query"][1]["generation_error"] = "InternalServerError: 502"
    (tmp_path / "eval" / "results.json").write_text(json.dumps(res), encoding="utf-8")
    cfg = Config({}, tmp_path)

    scores = write_scores(tmp_path, [                 # note: no i=1
        {"i": 0, "correct": 1.0, "grounded": 1.0},
        {"i": 2, "correct": 1.0, "grounded": 1.0},
    ])
    out = apply_judge_scores(cfg, "eval/results.json", scores)
    assert out["per_query"][1]["judge_correct"] is None
    # averaged over the 2 gradable questions, NOT 2/3
    assert out["answer"]["judge_correctness"] == {"value": 1.0, "n_scored": 2}


def test_missing_a_gradable_question_still_raises(tmp_path):
    (tmp_path / "eval").mkdir()
    res = make_results(3)
    res["per_query"][1]["generation_error"] = "boom"
    (tmp_path / "eval" / "results.json").write_text(json.dumps(res), encoding="utf-8")
    cfg = Config({}, tmp_path)
    scores = write_scores(tmp_path, [{"i": 0, "correct": 1.0, "grounded": 1.0}])
    with pytest.raises(ValueError, match="1/2 gradable"):
        apply_judge_scores(cfg, "eval/results.json", scores)


def test_bom_prefixed_scores_file_is_accepted(workspace):
    """PowerShell's `Out-File -Encoding utf8` writes a BOM.

    The scores file arrives from outside this program by design, so a BOM must
    not be a parse error — it made the very first CLI round-trip fail.
    """
    cfg, tmp = workspace
    p = tmp / "eval" / "scores.jsonl"
    rows = [{"i": i, "correct": 1.0, "grounded": 1.0} for i in range(3)]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8-sig")
    assert p.read_bytes().startswith(b"\xef\xbb\xbf")
    out = apply_judge_scores(cfg, "eval/results.json", "eval/scores.jsonl")
    assert out["answer"]["judge_correctness"] == {"value": 1.0, "n_scored": 3}


def test_unparseable_score_line_names_the_line(workspace):
    cfg, tmp = workspace
    p = tmp / "eval" / "scores.jsonl"
    p.write_text('{"i":0,"correct":1}\nNOT JSON\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2"):
        apply_judge_scores(cfg, "eval/results.json", "eval/scores.jsonl")


def test_non_numeric_score_becomes_unscored_not_zero(workspace):
    cfg, tmp = workspace
    scores = write_scores(tmp, [
        {"i": 0, "correct": None, "grounded": 1.0},
        {"i": 1, "correct": "n/a", "grounded": 1.0},
        {"i": 2, "correct": 1.0, "grounded": 1.0},
    ])
    out = apply_judge_scores(cfg, "eval/results.json", scores)
    assert out["per_query"][0]["judge_correct"] is None
    assert out["per_query"][1]["judge_correct"] is None
    # only the one real score counts — the others must not be averaged as 0
    assert out["answer"]["judge_correctness"] == {"value": 1.0, "n_scored": 1}
