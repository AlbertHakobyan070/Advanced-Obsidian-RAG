"""
eval_runner.py — Run the golden-query suite and score the pipeline, in tiers.

Three tiers, each independently runnable and each honest about what it can and
cannot tell you:

  1. RETRIEVAL (free, offline — no LLM at all)
       hit_rate@k        expected domain present in the top-k
       mrr               mean reciprocal rank of the first correct-domain hit
       recall@k          expected source files retrieved  (needs labels)
       scope precision   when the auto scope-router fires, is it right?
       scope recall      when there was a domain to find, did it fire?
     This is the regression harness. It runs the whole suite in minutes with
     the generation endpoint down.

  2. ANSWER (needs generation)
       keyword_recall    expected terms present            (proxy, kept)
       citation_validity [n] markers that resolve to a retrieved doc
       groundedness      content-word n-gram overlap between each cited
                         sentence and the chunk it cites — DETERMINISTIC,
                         no LLM, the floor under the judge
       citation_support  the existing second-pass auditor, when enabled
       judge_*           optional LLM-as-judge pass (--judge): answer
                         correctness against a gold answer, plus a
                         model-scored groundedness and unsupported-claim list

  3. CALIBRATION
       Does the stated confidence track answer quality? Bucket every answer by
       its HIGH/MEDIUM/LOW line and score each bucket. A calibrated system is
       right more often when it says HIGH; `gap` (HIGH minus LOW) makes that
       one number.

Framing that must not get lost: tiers 1-2 are AUTOMATIC PROXIES. They catch
regressions. The judge tier is ADVISORY — it is a language model grading a
language model, useful for ranking two runs against each other, not ground
truth. Nothing here certifies faithfulness; that is still the job of the
citation auditor, the confidence line, and reading the cited source.

Every metric reports the number of questions it was actually scored over. A
metric with no ground truth in the golden set reports null, never zero.

    from eval.eval_runner import run_eval
    run_eval(cfg)                      # full
    run_eval(cfg, retrieval_only=True) # tier 1 only, offline
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eval import metrics as M
from src.llm.llm_client import LLMClient
from src.pipeline import RAGPipeline
from src.prompts.loader import load_prompt
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

_PUNT_MARKERS = ("don't cover", "dont cover", "not contain", "no relevant", "don't contain")


def _keyword_recall(text: str, keywords: list[str]) -> float | None:
    """Fraction of expected terms present. None when none are declared."""
    if not keywords:
        return None
    low = str(text or "").lower()
    return sum(1 for kw in keywords if str(kw).lower() in low) / len(keywords)


def _citation_support_rate(answer) -> float | None:
    audited = [c for c in answer.citations if c.supported is not None]
    if not audited:
        return None
    return sum(1 for c in audited if c.supported) / len(audited)


def _course_of(meta: dict) -> str:
    """Course label across the loaders' key drift (course_name/course/code)."""
    meta = meta or {}
    val = str(meta.get("course_name") or meta.get("course")
              or meta.get("course_code") or "").strip()
    return "" if val.lower() in ("", "unknown") else val


def _scope_truth(item: dict) -> list[str]:
    """RECALL truth only: domains the router SHOULD have fired on.

    ONLY from an explicit `expect_scope` label. This used to fall back to the
    entry's `domain`, which was wrong and made the recall number meaningless:
    `domain` says where the ANSWER should come from, not whether the QUESTION
    names a domain. "Explain the bias-variance tradeoff" has domain=ml but
    names no domain, so a silent router is CORRECT — yet it was scored as a
    recall miss. On the 100-question suite that mislabelled 65 of 68 "misses"
    and reported 32% for a router whose only genuine miss was `ggplot2` failing
    to match the alias `ggplot`.

    Whether a question names a domain is a human judgement, so it needs a
    human label. Unlabelled entries are NOT scored for recall (precision and
    fire-rate need no labels and stay meaningful).
    """
    declared = item.get("expect_scope")
    return [str(d) for d in declared] if declared else []


# ------------------------------------------------------------- judge tier

def _run_judge(llm, prompt: dict, question: str, gold: str | None,
               answer_text: str, docs, max_tokens: int) -> dict[str, Any]:
    """One LLM-as-judge call. Returns the parsed rubric or an error marker.

    `correct` is DISCARDED when the golden entry has no gold_answer — without
    a reference, "correct" would just be the model's own opinion of its own
    output, which measures nothing.

    A malformed or failed judge response is recorded in the row and counted in
    the summary rather than aborting a 94-question run. It is surfaced, not
    swallowed: judge_errors appears in the scorecard.
    """
    from src.generation.generator import Generator

    context = Generator._format_context(list(docs))
    try:
        resp = llm.complete(
            system=prompt["system"],
            user=prompt["user"].format(
                question=question,
                gold_answer=(gold or "(none provided)"),
                answer=answer_text,
                context=context,
            ),
            temperature=0.0,
            max_tokens=max_tokens,
        )
        data = Generator._parse_json(resp.text)
        if not data:
            return {"judge_error": "unparseable judge response"}
    except Exception as e:                       # noqa: BLE001 — recorded below
        log.warning("judge call failed: %s", e)
        return {"judge_error": str(e)}

    def _num(key):
        v = data.get(key)
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return None

    return {
        "judge_correct": _num("correct") if gold else None,
        "judge_grounded": _num("grounded"),
        "judge_unsupported": [str(c) for c in (data.get("unsupported_claims") or [])],
        "judge_error": None,
    }


# ------------------------------------------------------------------ runner

def run_eval(
    cfg: Config,
    golden_path: str = "eval/golden_queries.yaml",
    out_path: str = "eval/results.json",
    retrieval_only: bool = False,
    judge: bool = False,
    limit: int | None = None,
    judge_export: str | None = None,
) -> dict[str, Any]:
    """Score the golden suite.

    retrieval_only  tier 1 only — no generation backend needed. Keyword recall
                    is measured over the retrieved CHUNK TEXTS instead of an
                    answer, which is what the historical *.retrieval.json
                    baselines did.
    judge           add the in-process LLM-as-judge pass (ignored under
                    retrieval_only — there is no answer to judge).
    judge_export    also write a JSONL bundle for an EXTERNAL judge to grade
                    (see _write_judge_bundle / apply_judge_scores).
    limit           score only the first N questions (smoke runs).
    """
    golden_file = cfg.project_root / golden_path
    with open(golden_file, "r", encoding="utf-8") as f:
        suite = yaml.safe_load(f)
    queries = suite["queries"]
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        raise ValueError(f"No queries to run from {golden_file} (limit={limit})")

    ngram_n = int(cfg.get("eval.ngram_n", 2))
    judge = bool(judge and not retrieval_only)
    if judge_export and retrieval_only:
        raise ValueError("--judge-export needs generated answers; drop "
                         "--retrieval-only (there is nothing to judge)")

    judge_max_tokens = int(cfg.get("eval.judge.max_tokens", 700))

    rag = RAGPipeline.from_config(cfg)
    judge_prompt = judge_llm = None
    if judge:
        # Fail here, before the first query, if judge.yaml is missing.
        judge_prompt = load_prompt("judge")
        # A judge that IS the model under test grades its own homework. When
        # eval.judge.provider names a backend, build a separate client for it;
        # otherwise fall back to the generation client and say so.
        if cfg.get("eval.judge.provider"):
            judge_llm = LLMClient.from_config(cfg, role="eval.judge")
        else:
            judge_llm = rag.generator.llm
            log.warning("eval.judge.provider is unset — judging with the SAME "
                        "client that generated the answers (self-grading)")

    per_query: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []      # judge-export payloads, not scored
    any_course_in_corpus = False
    t0 = time.time()

    for i, item in enumerate(queries, 1):
        q = item["question"]
        domain = str(item.get("domain") or "")
        log.info("[eval %d/%d] %s", i, len(queries), q[:60])

        # The scope router is a pure regex pass over the RAW question — cheap
        # to re-run here, and gives the structured Scope instead of the
        # display labels that end up in the retrieval echo.
        detected = rag.scope_router.detect(q).domains

        # search() and generate() are called separately rather than through
        # query() so that a generation failure cannot destroy the retrieval
        # measurement for this question — or the 99 other questions. Upstream
        # 5xx/410s are routine when the endpoint is a proxy fanning out to many
        # providers, and a batch scorer that dies on question 32 is useless.
        gen_error = None
        top, info = rag.search(q)
        answer = None
        if retrieval_only:
            scored_text = "\n".join(d.text for d in top)
        else:
            try:
                answer = rag.generator.generate(q, top)
                answer.retrieval = info
                scored_text = answer.text
            except Exception as e:                   # noqa: BLE001 — recorded
                # NOT swallowed: recorded per question, counted in the report,
                # and the row scores null (not zero) on every answer metric, so
                # a failed question can never look like a bad answer. If the
                # endpoint is misconfigured, generation_errors == n_queries and
                # the scorecard says so in the headline.
                gen_error = f"{type(e).__name__}: {e}"
                log.error("[eval %d/%d] generation FAILED: %s", i, len(queries), e)
                scored_text = ""

        for d in top:
            if _course_of(d.metadata):
                any_course_in_corpus = True
                break

        # Tier 1 scores the top-k as configured. `top` can be LONGER than k —
        # neighbor_context appends adjacent chunks after the rerank — and
        # counting those would quietly inflate every retrieval metric. The
        # answer tier keeps the full `top`, because citation [n] indexes into
        # exactly the list the generator was handed.
        k = info.get("rerank_top_k") or len(top)
        pool = top[:k] if k else top

        row: dict[str, Any] = {
            "question": q,
            "domain": domain or None,
            # --- tier 1
            "retrieval_hit": M.hit_at_k(domain, pool),
            "reciprocal_rank": M.reciprocal_rank(domain, pool),
            "first_hit_rank": M.first_domain_rank(domain, pool),
            "recall_at_k": M.recall_at_k(item.get("expected_source_files"), pool),
            # precision truth = the expected answer-domain; recall truth = an
            # explicit "this question names a domain" label. See scope_verdict.
            "scope": M.scope_verdict(detected, [domain] if domain else [],
                                     _scope_truth(item)),
            "course_hit": None,
            # --- tier 2
            # A question whose generation failed has no answer to score: every
            # tier-2 field stays None so it drops OUT of the averages instead
            # of counting as a zero. Its tier-1 numbers above are still real —
            # retrieval ran fine.
            "generation_error": gen_error,
            "keyword_recall": (None if gen_error else
                               _keyword_recall(scored_text, item.get("expect_keywords", []))),
            "citation_validity": None,
            "dangling_citations": [],
            "groundedness": None,
            "citation_support": None,
            "confidence": ("ERROR" if gen_error else "N/A"),
            "answered": (None if gen_error else True),
            "n_citations": (None if gen_error else 0),
            "answer_preview": (top[0].text[:200] if (retrieval_only and top)
                               else (scored_text[:200] if scored_text else "")),
        }

        if item.get("expect_course") and pool:
            row["course_hit"] = (_course_of(pool[0].metadata).lower()
                                 == str(item["expect_course"]).lower())

        if answer is not None:
            validity, dangling = M.citation_validity(answer.text, len(top))
            ground = M.groundedness_floor(answer.text, top, n=ngram_n)
            row.update({
                "citation_validity": validity,
                "dangling_citations": dangling,
                "groundedness": ground["score"],
                "groundedness_detail": {
                    "sentences_scored": ground["n_sentences_scored"],
                    "sentences_uncited": ground["n_sentences_uncited"],
                    "worst": ground["worst"],
                },
                "citation_support": _citation_support_rate(answer),
                "confidence": answer.confidence,
                "answered": not any(m in answer.text.lower() for m in _PUNT_MARKERS),
                "n_citations": len(answer.citations),
            })

            if judge:
                row.update(_run_judge(
                    judge_llm, judge_prompt, q, item.get("gold_answer"),
                    answer.text, top, judge_max_tokens,
                ))

        bundles.append({
            "answer": answer.text if answer is not None else "",
            "n_retrieved": len(top),
            "cited_chunks": [] if answer is None else [
                {"n": c.number, "label": c.source_label, "chunk_id": c.chunk_id,
                 "source_file": M.doc_source_file(top[c.number - 1]),
                 "text": top[c.number - 1].text}
                for c in answer.citations if 1 <= c.number <= len(top)
            ],
        })
        per_query.append(row)

    elapsed = time.time() - t0

    out = _build_report(
        per_query,
        retrieval_only=retrieval_only,
        judge=judge,
        any_course_in_corpus=any_course_in_corpus,
        golden_file=golden_file,
        ngram_n=ngram_n,
        elapsed=elapsed,
    )

    if judge_export:
        _write_judge_bundle(cfg, judge_export, queries, per_query, bundles)
        print(f"  Judge bundle: {cfg.project_root / judge_export}")

    _emit(cfg, out, out_path)
    return out


def _build_report(per_query: list[dict], *, retrieval_only: bool, judge: bool,
                  any_course_in_corpus: bool, golden_file, ngram_n: int,
                  elapsed: float) -> dict[str, Any]:
    """Aggregate rows into the tiered report.

    Split out of run_eval so --judge-import can rebuild every tier from an
    existing results file after merging externally-produced scores, instead of
    re-running 100 queries to recompute averages.
    """
    n = len(per_query)

    def agg_mean(key):
        return M.mean_scored(r.get(key) for r in per_query)

    def agg_rate(key):
        return M.rate_scored(r.get(key) for r in per_query)

    def scored(value_and_count, digits=3):
        """(value, n_scored) -> the JSON shape every metric reports."""
        value, count = value_and_count
        return {"value": None if value is None else round(value, digits),
                "n_scored": count}

    # --- tier 1
    retrieval_tier = {
        "hit_rate_at_k": scored(agg_rate("retrieval_hit")),
        "mrr": scored(agg_mean("reciprocal_rank")),
        "recall_at_k": scored(agg_mean("recall_at_k")),
        "scope_precision": scored(M.rate_scored(r["scope"]["correct"] for r in per_query)),
        "scope_recall": scored(M.rate_scored(r["scope"]["recalled"] for r in per_query)),
        "scope_fire_rate": scored(M.rate_scored(r["scope"]["fired"] for r in per_query)),
    }

    # Course routing auto-skips on a corpus that has no courses at all (a
    # non-academic vault), instead of scoring a confident 0%.
    course_rate = scored(agg_rate("course_hit"))
    if not any_course_in_corpus:
        course_rate = {"value": None, "n_scored": 0,
                       "skipped": "no course metadata anywhere in the retrieved corpus"}
    retrieval_tier["course_hit_rate"] = course_rate

    # --- tier 2
    answer_tier: dict[str, Any] = {
        "keyword_recall": scored(agg_mean("keyword_recall")),
    }
    if not retrieval_only:
        answer_tier.update({
            "citation_validity": scored(agg_mean("citation_validity")),
            "groundedness_floor": scored(agg_mean("groundedness")),
            "citation_support": scored(agg_mean("citation_support")),
            "answered_rate": scored(agg_rate("answered")),
            "mean_citations": scored(agg_mean("n_citations"), digits=2),
            "n_dangling_citations": sum(len(r["dangling_citations"]) for r in per_query),
            "generation_errors": sum(1 for r in per_query if r.get("generation_error")),
        })
        if judge:
            answer_tier.update({
                "judge_correctness": scored(agg_mean("judge_correct")),
                "judge_groundedness": scored(agg_mean("judge_grounded")),
                "judge_errors": sum(1 for r in per_query if r.get("judge_error")),
            })

    # --- tier 3
    conf_dist: dict[str, int] = {}
    for r in per_query:
        conf_dist[r["confidence"]] = conf_dist.get(r["confidence"], 0) + 1
    calibration_tier: dict[str, Any] = {"confidence_dist": conf_dist}
    if not retrieval_only:
        calibration_tier["by_keyword_recall"] = M.bucket_by(
            per_query, "confidence", "keyword_recall")
        calibration_tier["by_groundedness"] = M.bucket_by(
            per_query, "confidence", "groundedness")
        if judge:
            calibration_tier["by_judge_correctness"] = M.bucket_by(
                per_query, "confidence", "judge_correct")

    # Flat headline block, legacy key names preserved, so the pre-v2 baseline
    # files under eval/ stay comparable at a glance without a converter.
    summary = {
        "n_queries": n,
        "avg_keyword_recall": answer_tier["keyword_recall"]["value"],
        "retrieval_hit_rate": retrieval_tier["hit_rate_at_k"]["value"],
        "course_hit_rate": retrieval_tier["course_hit_rate"]["value"],
        "avg_citation_support": (answer_tier.get("citation_support") or {}).get("value"),
        "answered_rate": (answer_tier.get("answered_rate") or {}).get("value"),
        "confidence_dist": conf_dist,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_query": round(elapsed / n, 2),
    }

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "golden_file": str(golden_file),
            "mode": "retrieval_only" if retrieval_only else "full",
            "judge": judge,
            "ngram_n": ngram_n,
            "n_queries": n,
            "elapsed_seconds": round(elapsed, 1),
            "seconds_per_query": round(elapsed / n, 2) if n else 0,
        },
        "retrieval": retrieval_tier,
        "answer": answer_tier,
        "calibration": calibration_tier,
        "summary": summary,
        "per_query": per_query,
    }


def _emit(cfg: Config, out: dict, out_path: str) -> dict[str, Any]:
    """Write the JSON + markdown scorecard pair and print the console summary."""
    out_file = cfg.project_root / out_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    card_file = out_file.with_suffix(".md")
    with open(card_file, "w", encoding="utf-8") as f:
        f.write(_scorecard(out))

    _print_summary(out, out_file, card_file)
    return out


# ------------------------------------------- external judge (export/import)

def _write_judge_bundle(cfg: Config, path: str, queries: list[dict],
                        per_query: list[dict], bundles: list[dict]) -> None:
    """Write one JSONL record per question for an EXTERNAL judge to grade.

    This is the escape hatch from "the RAG can only be judged by a model the
    RAG can call". Everything a grader needs travels in the file — question,
    optional gold answer, the generated answer, and the full text of the chunks
    the answer actually cited — so a judge with no access to this machine (a
    human, or a stronger model in another session) can score it and hand back a
    scores JSONL for --judge-import.

    Only CITED chunks are included: the judge scores whether the answer is
    supported by what it pointed at, and shipping all k chunks would bloat the
    bundle for no gain.
    """
    out_file = cfg.project_root / path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for i, (item, row, bundle) in enumerate(zip(queries, per_query, bundles)):
            # A question whose generation failed has no answer to grade.
            # Including it would invite a 0, turning an endpoint outage into a
            # fake quality signal. apply_judge_scores expects it to be absent.
            if row.get("generation_error"):
                skipped += 1
                continue
            f.write(json.dumps({
                "i": i,
                "question": row["question"],
                "domain": row["domain"],
                "gold_answer": item.get("gold_answer"),
                "expect_keywords": item.get("expect_keywords", []),
                "answer": bundle["answer"],
                "confidence": row["confidence"],
                "cited_chunks": bundle["cited_chunks"],
                "n_retrieved": bundle["n_retrieved"],
            }, ensure_ascii=False) + "\n")
    if skipped:
        log.warning("judge bundle omits %d question(s) whose generation failed",
                    skipped)


def apply_judge_scores(cfg: Config, results_path: str, scores_path: str,
                       out_path: str | None = None) -> dict[str, Any]:
    """Merge an external judge's scores into an existing results file.

    scores JSONL, one object per line:
        {"i": 0, "correct": 0.9, "grounded": 1.0,
         "unsupported_claims": [], "notes": "optional"}

    `i` must match the bundle's `i`. Correctness is DISCARDED for questions
    with no gold_answer only when the judge itself is the generator; an
    external judge grading on subject-matter merit is a different thing, so
    what it returns is kept and labelled as external in the report.
    """
    res_file = cfg.project_root / results_path
    with open(res_file, "r", encoding="utf-8") as f:
        out = json.load(f)

    # utf-8-sig, not utf-8: these files come from OUTSIDE this program — a
    # PowerShell redirect, an editor on another machine — and Windows tooling
    # writes a BOM by default. utf-8-sig strips one if present and is
    # byte-identical to utf-8 when it isn't. Malformed JSON still raises.
    scores: dict[int, dict] = {}
    with open(cfg.project_root / scores_path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{scores_path}:{ln} is not valid JSON: {e}") from e
            if "i" not in rec:
                raise ValueError(f"{scores_path}:{ln} has no 'i' field")
            scores[int(rec["i"])] = rec

    per_query = out["per_query"]
    # Questions whose generation failed were never exported, so they are not
    # expected in the scores file — but every question that DID produce an
    # answer must be graded, or the judge tier would report a confident average
    # over whatever subset the judge happened to finish.
    gradable = [i for i, r in enumerate(per_query) if not r.get("generation_error")]
    missing = [i for i in gradable if i not in scores]
    if missing:
        raise ValueError(
            f"scores file covers {len(scores)}/{len(gradable)} gradable questions; "
            f"missing indices: {missing[:12]}{'...' if len(missing) > 12 else ''}"
        )

    def clamp(v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return None

    for i, row in enumerate(per_query):
        rec = scores.get(i)
        if rec is None:                    # generation failed: nothing to grade
            row["judge_correct"] = row["judge_grounded"] = None
            continue
        row["judge_correct"] = clamp(rec.get("correct"))
        row["judge_grounded"] = clamp(rec.get("grounded"))
        row["judge_unsupported"] = [str(c) for c in (rec.get("unsupported_claims") or [])]
        row["judge_notes"] = str(rec.get("notes", ""))
        row["judge_error"] = None

    meta = out["meta"]
    rebuilt = _build_report(
        per_query,
        retrieval_only=(meta["mode"] == "retrieval_only"),
        judge=True,
        # course_hit_rate was already decided on the original run; a null there
        # means the corpus had no courses, and merging scores cannot change it.
        any_course_in_corpus=out["retrieval"]["course_hit_rate"].get("skipped") is None,
        golden_file=meta["golden_file"],
        ngram_n=meta.get("ngram_n", 2),
        elapsed=meta.get("elapsed_seconds", 0),
    )
    rebuilt["meta"]["judge"] = "external"
    rebuilt["meta"]["judged_by"] = scores.get(0, {}).get("judge", "external")
    rebuilt["meta"]["generated_at"] = meta["generated_at"]
    rebuilt["meta"]["judged_at"] = datetime.now().isoformat(timespec="seconds")

    return _emit(cfg, rebuilt, out_path or results_path)


# ---------------------------------------------------------------- reporting

def _fmt(entry: dict | None, pct: bool = True) -> str:
    """A metric cell: value + the coverage it was scored over, or why not."""
    if not entry:
        return "—"
    if entry.get("value") is None:
        why = entry.get("skipped") or "no ground truth in the golden set"
        return f"not scored ({why})"
    v = entry["value"]
    shown = f"{v:.1%}" if pct else f"{v:.3f}"
    return f"**{shown}** _(n={entry['n_scored']})_"


def _scorecard(out: dict) -> str:
    meta, ret, ans, cal = out["meta"], out["retrieval"], out["answer"], out["calibration"]
    L = [
        "# Eval scorecard",
        "",
        f"- **Run:** {meta['generated_at']} · mode `{meta['mode']}`"
        f"{' · judge on' if meta['judge'] else ''}",
        f"- **Questions:** {meta['n_queries']} · {meta['seconds_per_query']}s/query "
        f"({meta['elapsed_seconds']}s total)",
        f"- **Golden set:** `{Path(meta['golden_file']).name}`",
        "",
        "## Tier 1 — Retrieval (offline, no LLM)",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Hit-rate@k (expected domain in top-k) | {_fmt(ret['hit_rate_at_k'])} |",
        f"| MRR (first correct-domain hit) | {_fmt(ret['mrr'], pct=False)} |",
        f"| Recall@k (expected source files) | {_fmt(ret['recall_at_k'])} |",
        f"| Scope precision (router fired → was right) | {_fmt(ret['scope_precision'])} |",
        f"| Scope recall (needs `expect_scope` labels) | {_fmt(ret['scope_recall'])} |",
        f"| Scope fire-rate (diagnostic) | {_fmt(ret['scope_fire_rate'])} |",
        f"| Course-routing accuracy | {_fmt(ret['course_hit_rate'])} |",
        "",
        "## Tier 2 — Answer",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Keyword recall (proxy) | {_fmt(ans['keyword_recall'])} |",
    ]
    if "citation_validity" in ans:
        L += [
            f"| Citation validity ([n] resolves to a retrieved doc) | {_fmt(ans['citation_validity'])} |",
            f"| Groundedness floor (n-gram overlap, deterministic) | {_fmt(ans['groundedness_floor'])} |",
            f"| Citation support (second-pass auditor) | {_fmt(ans['citation_support'])} |",
            f"| Answered rate | {_fmt(ans['answered_rate'])} |",
            f"| Mean citations per answer | {_fmt(ans['mean_citations'], pct=False)} |",
            f"| Dangling citation markers | {ans['n_dangling_citations']} |",
        ]
        errs = ans.get("generation_errors", 0)
        if errs:
            L += [f"| **Generation failures** (endpoint errors, unscored) | "
                  f"**{errs} / {meta['n_queries']}** |"]
    if "judge_correctness" in ans:
        L += [
            f"| Judge — answer correctness _(advisory)_ | {_fmt(ans['judge_correctness'])} |",
            f"| Judge — groundedness _(advisory)_ | {_fmt(ans['judge_groundedness'])} |",
            f"| Judge call failures | {ans['judge_errors']} |",
        ]

    L += ["", "## Tier 3 — Calibration", "",
          f"Confidence distribution: `{cal['confidence_dist']}`", ""]
    for label, block in cal.items():
        if label == "confidence_dist" or not isinstance(block, dict):
            continue
        L += [f"**{label.replace('_', ' ')}** — gap (HIGH − LOW): "
              f"`{block.get('gap')}`", "",
              "| Confidence | n | scored | mean |", "|---|---|---|---|"]
        for conf, b in block.get("buckets", {}).items():
            mean = "—" if b["mean"] is None else f"{b['mean']:.3f}"
            L += [f"| {conf} | {b['n']} | {b['n_scored']} | {mean} |"]
        L += [""]

    L += [
        "---",
        "",
        "Tiers 1–2 are **automatic proxy metrics**: they catch regressions, they do",
        "not certify that an answer is correct. Keyword recall checks that expected",
        "terms appear, not that the explanation is right; hit-rate checks the domain,",
        "not the passage. The groundedness floor is deterministic and cheap — low",
        "overlap means a cited sentence barely resembles the source it points at, but",
        "high overlap is not proof of correctness. The judge tier is **advisory**: a",
        "language model grading a language model, useful for comparing two runs, not",
        "as ground truth.",
        "",
    ]
    return "\n".join(L)


def _print_summary(out: dict, out_file: Path, card_file: Path) -> None:
    meta, ret, ans = out["meta"], out["retrieval"], out["answer"]

    def line(label: str, entry: dict | None, pct: bool = True) -> str:
        if not entry:
            return ""
        if entry.get("value") is None:
            return f"  {label:<34}—  ({entry.get('skipped') or 'no labels'})"
        v = entry["value"]
        shown = f"{v:.1%}" if pct else f"{v:.3f}"
        return f"  {label:<34}{shown:>8}   (n={entry['n_scored']})"

    print("\n" + "=" * 68)
    print(f"  EVAL SUMMARY — {meta['n_queries']} questions, mode={meta['mode']}"
          f"{', judge on' if meta['judge'] else ''}")
    print("=" * 68)
    print("  -- tier 1: retrieval (offline) " + "-" * 35)
    for label, key, pct in (("Hit-rate@k", "hit_rate_at_k", True),
                            ("MRR", "mrr", False),
                            ("Recall@k (source files)", "recall_at_k", True),
                            ("Scope precision", "scope_precision", True),
                            ("Scope recall", "scope_recall", True),
                            ("Course routing", "course_hit_rate", True)):
        print(line(label, ret.get(key), pct))
    print("  -- tier 2: answer " + "-" * 48)
    for label, key, pct in (("Keyword recall", "keyword_recall", True),
                            ("Citation validity", "citation_validity", True),
                            ("Groundedness floor", "groundedness_floor", True),
                            ("Citation support", "citation_support", True),
                            ("Answered rate", "answered_rate", True),
                            ("Judge correctness (advisory)", "judge_correctness", True),
                            ("Judge groundedness (advisory)", "judge_groundedness", True)):
        text = line(label, ans.get(key), pct)
        if text:
            print(text)
    if ans.get("generation_errors"):
        print(f"  {'Generation FAILURES':<34}{ans['generation_errors']:>8}   "
              f"of {meta['n_queries']} (unscored, see per_query.generation_error)")
    if meta["mode"] != "retrieval_only":
        gap = out["calibration"].get("by_groundedness", {}).get("gap")
        print("  -- tier 3: calibration " + "-" * 43)
        print(f"  {'Confidence dist':<34}{out['calibration']['confidence_dist']}")
        print(f"  {'Groundedness gap (HIGH-LOW)':<34}{gap}")
    print("=" * 68)
    print(f"  Speed:  {meta['seconds_per_query']}s/query ({meta['elapsed_seconds']}s total)")
    print(f"  JSON:   {out_file}")
    print(f"  Card:   {card_file}")
    print("=" * 68 + "\n")
