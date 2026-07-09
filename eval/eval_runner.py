"""
eval_runner.py — Run the golden-query suite and score the pipeline.

Metrics (no human labels needed, all automatic):
  retrieval_hit       — did the expected domain appear in the top-k sources?
  course_hit          — (if expect_course set) did the #1 source match it?
  keyword_recall      — fraction of expect_keywords present in the answer text
  citation_support    — fraction of cited sources the auditor marked supported
  confidence_dist     — distribution of HIGH/MEDIUM/LOW
  answered_rate       — fraction where the model didn't punt ("notes don't cover")

This gives you a fast, repeatable signal of pipeline quality against your
golden set — useful for catching regressions and tracking changes. The
shipped `eval/golden_queries.yaml` is a small illustrative example; replace
it with your own golden set for serious regression work.

    from eval.eval_runner import run_eval
    run_eval(cfg)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.pipeline import RAGPipeline
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

_PUNT_MARKERS = ("don't cover", "dont cover", "not contain", "no relevant", "don't contain")


def _keyword_recall(answer_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    low = answer_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in low)
    return hits / len(keywords)


def _domain_in_sources(domain: str, sources) -> bool:
    for d in sources:
        if str(d.metadata.get("domain", "")).lower() == domain.lower():
            return True
    return False


def _citation_support_rate(answer) -> float | None:
    audited = [c for c in answer.citations if c.supported is not None]
    if not audited:
        return None
    supported = sum(1 for c in audited if c.supported)
    return supported / len(audited)


def run_eval(
    cfg: Config,
    golden_path: str = "eval/golden_queries.yaml",
    out_path: str = "eval/results.json",
    retrieval_only: bool = False,
) -> dict[str, Any]:
    """
    retrieval_only=True skips generation entirely (no LLM needed): keyword
    recall is measured over the top-k CHUNK TEXTS instead of the answer, and
    the citation/confidence/answered metrics are omitted. This makes the
    suite runnable in minutes, offline - ideal for CI smoke tests.
    """
    golden_file = cfg.project_root / golden_path
    with open(golden_file, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)
    queries = suite["queries"]

    rag = RAGPipeline.from_config(cfg)

    per_query: list[dict[str, Any]] = []
    t0 = time.time()

    for i, item in enumerate(queries, 1):
        q = item["question"]
        log.info("[eval %d/%d] %s", i, len(queries), q[:60])

        if retrieval_only:
            top, _info = rag.search(q)
            pool_text = "\n".join(d.text for d in top)
            recall = _keyword_recall(pool_text, item.get("expect_keywords", []))
            ret_hit = _domain_in_sources(item.get("domain", ""), top)
            course_hit = None
            if item.get("expect_course") and top:
                m = top[0].metadata
                src_course = str(m.get("course_name") or m.get("course")
                                 or m.get("course_code") or "")
                course_hit = src_course.lower() == str(item["expect_course"]).lower()
            per_query.append({
                "question": q,
                "domain": item.get("domain"),
                "keyword_recall": round(recall, 3),
                "retrieval_hit": ret_hit,
                "course_hit": course_hit,
                "citation_support": None,
                "confidence": "N/A",
                "answered": True,
                "n_citations": 0,
                "answer_preview": (top[0].text[:200] if top else ""),
            })
            continue

        answer = rag.query(q)

        recall = _keyword_recall(answer.text, item.get("expect_keywords", []))
        ret_hit = _domain_in_sources(item.get("domain", ""), answer.sources)
        course_hit = None
        if item.get("expect_course") and answer.sources:
            m = answer.sources[0].metadata
            src_course = str(m.get("course_name") or m.get("course")
                             or m.get("course_code") or "")
            course_hit = src_course.lower() == str(item["expect_course"]).lower()
        support = _citation_support_rate(answer)
        answered = not any(m in answer.text.lower() for m in _PUNT_MARKERS)

        per_query.append({
            "question": q,
            "domain": item.get("domain"),
            "keyword_recall": round(recall, 3),
            "retrieval_hit": ret_hit,
            "course_hit": course_hit,
            "citation_support": None if support is None else round(support, 3),
            "confidence": answer.confidence,
            "answered": answered,
            "n_citations": len(answer.citations),
            "answer_preview": answer.text[:200],
        })

    elapsed = time.time() - t0

    # Aggregate
    n = len(per_query)
    avg_recall = sum(r["keyword_recall"] for r in per_query) / n
    ret_hit_rate = sum(1 for r in per_query if r["retrieval_hit"]) / n
    answered_rate = sum(1 for r in per_query if r["answered"]) / n
    supports = [r["citation_support"] for r in per_query if r["citation_support"] is not None]
    avg_support = sum(supports) / len(supports) if supports else None
    course_checks = [r["course_hit"] for r in per_query if r["course_hit"] is not None]
    course_rate = sum(1 for c in course_checks if c) / len(course_checks) if course_checks else None

    conf_dist: dict[str, int] = {}
    for r in per_query:
        conf_dist[r["confidence"]] = conf_dist.get(r["confidence"], 0) + 1

    summary = {
        "n_queries": n,
        "avg_keyword_recall": round(avg_recall, 3),
        "retrieval_hit_rate": round(ret_hit_rate, 3),
        "course_hit_rate": None if course_rate is None else round(course_rate, 3),
        "avg_citation_support": None if avg_support is None else round(avg_support, 3),
        "answered_rate": round(answered_rate, 3),
        "confidence_dist": conf_dist,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_query": round(elapsed / n, 2),
    }

    out = {"summary": summary, "per_query": per_query}
    out_file = cfg.project_root / out_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    _print_summary(summary, out_file)
    return out


def _print_summary(s: dict, out_file: Path) -> None:
    print("\n" + "=" * 60)
    print("  EVAL SUMMARY")
    print("=" * 60)
    print(f"  Queries:              {s['n_queries']}")
    print(f"  Avg keyword recall:   {s['avg_keyword_recall']:.1%}")
    print(f"  Retrieval hit rate:   {s['retrieval_hit_rate']:.1%}")
    if s["course_hit_rate"] is not None:
        print(f"  Course hit rate:      {s['course_hit_rate']:.1%}")
    if s["avg_citation_support"] is not None:
        print(f"  Citation support:     {s['avg_citation_support']:.1%}")
    print(f"  Answered rate:        {s['answered_rate']:.1%}")
    print(f"  Confidence dist:      {s['confidence_dist']}")
    print(f"  Speed:                {s['seconds_per_query']}s/query ({s['elapsed_seconds']}s total)")
    print("=" * 60)
    print(f"  Full results: {out_file}")
    print("=" * 60 + "\n")
