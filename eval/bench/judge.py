"""
judge.py — grade the answers of a `bench run --answers` (eval spec phase 4).

One judge call per answered row, temperature 0, with the question, its nuggets,
the answer and the SAME numbered sources the generator answered from. The judge
returns per-nugget and per-citation verdicts, which repeat far better than a
holistic score:

    abstained  the answer declined instead of answering
    stated[i]  the answer states nugget i
    supported[i]  nugget i is in the sources
    citations  for each cited [n]: does source n support what cites it

Metrics (per configuration, bootstrap CIs, paired across BASELINES):
    nugget_recall           mean share of nuggets stated (answerable rows)
    grounded_recall         ... stated AND in the sources
    citation_precision      cited sources that support their sentence
    uncited_rate            answers to answerable questions that cite nothing
    abstention_accuracy     share of unanswerable questions abstained on
    false_abstention        share of answerable questions abstained on
    calibration             nugget_recall per HIGH / MEDIUM / LOW
    attribution             first layer that lost each answerable question:
                            retrieval (no gold chunk among the answer's
                            sources) -> generation (a nugget missing) ->
                            citation (an unsupported or absent citation) -> pass

The judge runs only through a replay store (--llm), so the grading model is
fixed and recorded. A verdict that does not parse is a row ERROR, counted and
reported, never dropped. Judged numbers are ADVISORY until the judge-agreement
study (the owner labels a sample; Cohen's kappa) has been done.
"""
from __future__ import annotations

import glob
import json
import re
from datetime import datetime
from pathlib import Path

from eval.bench.configs import BASELINES
from eval.bench.stats import bootstrap_ci, paired_diff_ci
from src.prompts.loader import load_prompt
from src.utils.citations import CITATION_RE

SEED = 0
CONFIDENCES = ("HIGH", "MEDIUM", "LOW", "UNKNOWN")


def load_chunk_texts(data_dir: Path, ids: set[str]) -> dict[str, str]:
    """doc_id -> text for the wanted ids, streamed from every *chunks*.jsonl."""
    out: dict[str, str] = {}
    for f in sorted(glob.glob(str(Path(data_dir) / "*chunks*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("doc_id") in ids:
                    out[r["doc_id"]] = r["text"]
    return out


def build_prompt(prompt: dict, question: str, nuggets: list[str], answer: str,
                 sources: list[str]) -> tuple[str, str]:
    """(system, user). Sources are numbered from 1 in the order the generator saw them."""
    user = prompt["user"].format(
        question=question,
        nuggets="\n".join(f"{i}. {n}" for i, n in enumerate(nuggets, 1)) or "(none)",
        answer=answer,
        sources="\n\n".join(f"[{i}] {t.strip()}" for i, t in enumerate(sources, 1)))
    return prompt["system"], user


def parse_verdict(text: str, n_nuggets: int, cited: list[int]) -> dict:
    """The judge's JSON, checked against what it was asked. Raises ValueError."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"judge: no JSON object in {text[:200]!r}")
    try:
        v = json.loads(t[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"judge: unparseable verdict {text[:200]!r} ({e})") from None
    if not isinstance(v.get("abstained"), bool):
        raise ValueError("judge: 'abstained' must be true or false")
    for key in ("stated", "supported"):
        vals = v.get(key)
        if not isinstance(vals, list) or len(vals) != n_nuggets or not all(isinstance(x, bool) for x in vals):
            raise ValueError(f"judge: '{key}' must be {n_nuggets} booleans, got {vals!r}")
    cites = v.get("citations")
    if not isinstance(cites, list) or not all(
            isinstance(c, dict) and isinstance(c.get("n"), int) and isinstance(c.get("supports"), bool)
            for c in cites):
        raise ValueError(f"judge: 'citations' must be a list of {{n, supports}}, got {cites!r}")
    if sorted(c["n"] for c in cites) != sorted(cited):
        raise ValueError(f"judge: verdicts for citations {sorted(c['n'] for c in cites)}, "
                         f"the answer cites {sorted(cited)}")
    return {"abstained": v["abstained"], "stated": v["stated"], "supported": v["supported"],
            "citations": [{"n": c["n"], "supports": c["supports"]} for c in cites]}


def _layer(row: dict, v: dict, k: int) -> str:
    if not any(row["match"][:k]):
        return "retrieval"
    if v["abstained"] or not all(v["stated"]):
        return "generation"
    if not v["citations"] or not all(c["supports"] for c in v["citations"]):
        return "citation"
    return "pass"


def _ci(values: list[float]) -> dict | None:
    if not values:
        return None
    mean, lo, hi = bootstrap_ci(values, seed=SEED)
    return {"mean": mean, "lo": lo, "hi": hi, "n": len(values)}


def summarise(judged: list[dict]) -> dict:
    """judged: rows of {qid, config, answerable, confidence, k, match, verdict | error}."""
    by_cfg: dict[str, list[dict]] = {}
    for r in judged:
        by_cfg.setdefault(r["config"], []).append(r)
    out = {}
    for cfg, rows in by_cfg.items():
        ok = [r for r in rows if "verdict" in r]
        ans = [r for r in ok if r["answerable"]]
        una = [r for r in ok if not r["answerable"]]
        recall = {r["qid"]: (sum(r["verdict"]["stated"]) / len(r["verdict"]["stated"])
                             if r["verdict"]["stated"] else 0.0) for r in ans}
        grounded = [sum(s and p for s, p in zip(r["verdict"]["stated"], r["verdict"]["supported"]))
                    / max(1, len(r["verdict"]["stated"])) for r in ans]
        cited = [r for r in ans if not r["verdict"]["abstained"]]
        prec = [sum(c["supports"] for c in r["verdict"]["citations"]) / len(r["verdict"]["citations"])
                for r in cited if r["verdict"]["citations"]]
        layers: dict[str, int] = {}
        for r in ans:
            layer = _layer(r, r["verdict"], r["k"])
            layers[layer] = layers.get(layer, 0) + 1
        out[cfg] = {
            "nugget_recall": _ci(list(recall.values())),
            "grounded_recall": _ci(grounded),
            "citation_precision": _ci(prec),
            "uncited_rate": _ci([0.0 if r["verdict"]["citations"] else 1.0 for r in cited]),
            "abstention_accuracy": _ci([1.0 if r["verdict"]["abstained"] else 0.0 for r in una]),
            "false_abstention": _ci([1.0 if r["verdict"]["abstained"] else 0.0 for r in ans]),
            "calibration": {c: _ci([recall[r["qid"]] for r in ans if r["confidence"] == c])
                            for c in CONFIDENCES if any(r["confidence"] == c for r in ans)},
            "attribution": layers,
            "errors": sum(1 for r in rows if "error" in r),
            "_recall_by_qid": recall,
        }
    for b, a in BASELINES.items():
        if a in out and b in out:
            ra, rb = out[a]["_recall_by_qid"], out[b]["_recall_by_qid"]
            qids = [q for q in ra if q in rb]
            if qids:
                mean, lo, hi = paired_diff_ci([ra[q] for q in qids], [rb[q] for q in qids], seed=SEED)
                out[b]["paired_nugget_recall"] = {"vs": a, "n": len(qids), "mean": mean, "lo": lo, "hi": hi}
    for s in out.values():
        s.pop("_recall_by_qid")
    return out


def render(summary: dict, meta: dict) -> str:
    def f(x):
        return "—" if not x else f"{x['mean']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] (n={x['n']})"
    lines = [f"# Judged answers — {meta['run_id']}", "",
             f"Judge: `{meta['judge_model']}` (store `{meta['store']}`), {meta['judged_at']}. "
             "ADVISORY until the judge-agreement study is done.", ""]
    for cfg, s in summary.items():
        lines += [f"## {cfg}", "", "| metric | mean [95% CI] |", "|---|---|"]
        for m in ("nugget_recall", "grounded_recall", "citation_precision", "uncited_rate",
                  "abstention_accuracy", "false_abstention"):
            lines.append(f"| {m} | {f(s[m])} |")
        if "paired_nugget_recall" in s:
            p = s["paired_nugget_recall"]
            lines.append(f"| nugget_recall vs {p['vs']} | {p['mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] "
                         f"(n={p['n']}) |")
        lines += ["", "Calibration (nugget recall by stated confidence): "
                  + ", ".join(f"{c} {f(x)}" for c, x in s["calibration"].items()),
                  "", "Attribution (first layer that lost the question): "
                  + ", ".join(f"{k} {v}" for k, v in sorted(s["attribution"].items())),
                  "", f"Errors: {s['errors']}", ""]
    return "\n".join(lines)


def judge_run(run_dir: Path, questions: dict, llm, data_dir: Path, progress=print) -> dict:
    """Grade every answered row of `run_dir` with `llm`; write judge.jsonl,
    judge_summary.json and judge.md beside the run. Returns the summary."""
    run_dir = Path(run_dir)
    rows = [json.loads(l) for l in (run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows if "answer" in r]
    if not rows:
        raise ValueError(f"{run_dir}: no answered rows (run it with --answers)")
    texts = load_chunk_texts(data_dir, {i for r in rows for i in r["ids"][:r["answer"]["k"]]})
    prompt = load_prompt("judge_nuggets")
    judged = []
    for r in rows:
        q = questions[r["qid"]]
        k = r["answer"]["k"]
        missing = [i for i in r["ids"][:k] if i not in texts]
        base = {"qid": r["qid"], "config": r["config"], "answerable": q.answerable,
                "confidence": r["answer"]["confidence"], "k": k, "match": r["match"]}
        if missing:
            judged.append({**base, "error": f"source chunk(s) not in the JSONLs: {missing}"})
            continue
        system, user = build_prompt(prompt, q.question, list(q.nuggets), r["answer"]["text"],
                                    [texts[i] for i in r["ids"][:k]])
        # Every marker in the text, not the generator's resolved list: that one
        # drops a [9] against 7 sources, which the judge still reads. A marker
        # past the sources points at nothing, so it never supports anything.
        markers = sorted({int(n) for n in CITATION_RE.findall(r["answer"]["text"])})
        try:
            resp = llm.complete(system=system, user=user, temperature=0.0, max_tokens=600)
            verdict = parse_verdict(resp.text, len(q.nuggets), markers)
            for c in verdict["citations"]:
                if not 1 <= c["n"] <= k:
                    c["supports"] = False
            judged.append({**base, "verdict": verdict})
        except Exception as e:
            judged.append({**base, "error": f"{type(e).__name__}: {e}"})
    with open(run_dir / "judge.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for j in judged:
            f.write(json.dumps({k: v for k, v in j.items() if k != "match"}, ensure_ascii=False) + "\n")
    summary = summarise(judged)
    meta = {"run_id": run_dir.name, "judge_model": llm.model, "store": str(getattr(llm, "store", "")),
            "judged_at": datetime.now().isoformat(timespec="seconds")}
    (run_dir / "judge_summary.json").write_text(
        json.dumps({"meta": meta, "configs": summary}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n")
    (run_dir / "judge.md").write_text(render(summary, meta), encoding="utf-8", newline="\n")
    progress(f"judged {sum(1 for j in judged if 'verdict' in j)} of {len(judged)} answered rows")
    return summary
