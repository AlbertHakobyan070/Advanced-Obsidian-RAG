"""
metrics.py — Pure scoring functions for the golden-query eval suite.

Nothing in here touches the index, the LLM or the filesystem: every function
takes plain data and returns a number. That keeps the metric layer testable in
milliseconds with no corpus at all (tests/test_eval_metrics.py) and leaves
eval_runner.py doing nothing but orchestration.

One rule runs through the whole module:

    None means "not scored". It never means "scored zero".

A golden entry that doesn't declare `expected_source_files` must not drag
recall@k down to 0 — it drops out of the average entirely, and the runner
reports how many entries the metric actually covered. Every aggregate here
returns (value, n_scored) so that coverage is impossible to lose.

Tiers:
  retrieval    reciprocal_rank, hit_at_k, recall_at_k, scope_verdict
  answer       citation_validity, groundedness_floor  (both LLM-free)
  calibration  bucket_by
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

# Citation markers the generator emits: [1], [2], ...
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Sentence-ish split: after .!? or on a newline. Bullets and numbered lists in
# the generated answers are newline-separated, so both matter.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# A fragment that is nothing but citation markers belongs to the sentence
# BEFORE it ("...gradient clipping. [1][2]") — merged back during splitting.
_ONLY_CITATIONS_RE = re.compile(r"^(?:\s*\[\d+\]\s*)+[.,;]?$")

_WORD_RE = re.compile(r"[a-z0-9]+(?:[-_'][a-z0-9]+)*")

# U+2010..U+2015 (hyphen, non-breaking hyphen, figure/en/em dash, horizontal
# bar) + U+2212 minus. Written as escapes on purpose: the literal characters
# are indistinguishable from `-` in most editors, which is how they get into
# paths unnoticed in the first place.
_DASH_RE = re.compile("[\u2010-\u2015\u2212]")

# Deliberately small: enough to stop "the/of/and" inflating overlap, not so
# aggressive that it strips technical prose. Not a tunable — widening this
# changes what the groundedness floor means, so it lives in code, versioned.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here
is are was were be been being am do does did doing done
have has had having will would shall should can could may might must
of in on at to for from by with without within into onto over under
as it its it's we you they he she i our your their his her them us me
not no nor so such both each few more most other some any own same
about above below after before between during through against
what which who whom whose when where why how
""".split())


# ---------------------------------------------------------------- text utils

def _fold_dashes(s: str) -> str:
    """en/em/figure dashes and the minus sign -> plain hyphen.

    the author's vault paths are full of U+2013 (`Lecture – 3.pdf`). A golden entry
    typed with a plain hyphen must still match, or `expected_source_files`
    silently scores 0 for reasons nobody can see. Same class of bug as the
    curl/PowerShell en-dash gotcha in the handoff.
    """
    return _DASH_RE.sub("-", s)


def normalize_path(p: Any) -> str:
    """Case/separator/dash/whitespace-insensitive form for path comparison."""
    s = unicodedata.normalize("NFKC", str(p or ""))
    s = _fold_dashes(s).replace("\\", "/").lower()
    return re.sub(r"\s+", " ", s).strip()


def content_tokens(text: str) -> list[str]:
    """Lowercased alphanumeric words with stopwords removed."""
    s = _fold_dashes(unicodedata.normalize("NFKC", str(text or ""))).lower()
    return [w for w in _WORD_RE.findall(s) if w not in _STOPWORDS]


def ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    """Content-word n-grams.

    Short spans degrade to the largest n they can support (a 1-token sentence
    is compared as a unigram) rather than scoring 0 for being short. This is a
    defined part of the metric, not a silent fallback — see the module test.
    """
    if not tokens:
        return set()
    n = max(1, min(n, len(tokens)))
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def split_sentences(text: str) -> list[str]:
    """Split into sentences, re-attaching trailing citation-only fragments."""
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(str(text or ""))]
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        if out and _ONLY_CITATIONS_RE.match(part):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


# ------------------------------------------------------- tier 1: retrieval

def doc_domain(doc: Any) -> str:
    return str((getattr(doc, "metadata", None) or {}).get("domain", "")).lower()


def doc_source_file(doc: Any) -> str:
    meta = getattr(doc, "metadata", None) or {}
    return str(meta.get("source_file") or meta.get("filename") or "")


def first_domain_rank(domain: str, docs: Sequence[Any]) -> int | None:
    """1-indexed rank of the first doc in `domain`, or None if absent."""
    if not domain:
        return None
    want = str(domain).lower()
    for i, d in enumerate(docs, start=1):
        if doc_domain(d) == want:
            return i
    return None


def hit_at_k(domain: str, docs: Sequence[Any], k: int | None = None) -> bool | None:
    """Did the expected domain appear in the top-k? None = no domain declared."""
    if not domain:
        return None
    pool = docs if k is None else docs[:k]
    return first_domain_rank(domain, pool) is not None


def reciprocal_rank(domain: str, docs: Sequence[Any]) -> float | None:
    """1/rank of the first correct-domain hit; 0.0 when it never appears.

    0.0 is a real score here (the domain was declared and simply missed), which
    is why this returns None only when there is nothing to score against.
    """
    if not domain:
        return None
    rank = first_domain_rank(domain, docs)
    return 0.0 if rank is None else 1.0 / rank


def recall_at_k(expected_source_files: Iterable[str] | None,
                docs: Sequence[Any]) -> float | None:
    """Fraction of expected source files present among the retrieved docs.

    An expected entry matches when its normalized form is a substring of a
    retrieved doc's normalized `source_file` (so a golden set can name either
    a full path or just a trailing fragment like `NLP/Lecture 3.md`).
    """
    expected = [normalize_path(e) for e in (expected_source_files or [])]
    expected = [e for e in expected if e]
    if not expected:
        return None
    got = [normalize_path(doc_source_file(d)) for d in docs]
    found = sum(1 for e in expected if any(e in g for g in got if g))
    return found / len(expected)


def scope_verdict(detected_domains: Iterable[str],
                  truth_domains: Iterable[str] | None,
                  recall_truth: Iterable[str] | None = None) -> dict[str, Any]:
    """Score the auto scope-router on one query.

    PRECISION and RECALL need DIFFERENT ground truths, and conflating them was
    a real bug:

      `truth_domains`  the entry's `domain` — where the answer should come
                       from. Valid for PRECISION, which is conditioned on the
                       router having already fired: given that it routed
                       somewhere, did it route at the right domain?

      `recall_truth`   an explicit `expect_scope` label meaning "this question
                       NAMES a domain, so the router SHOULD fire". Recall needs
                       this and cannot use `domain`: most questions ("Explain
                       the bias-variance tradeoff") name no domain at all, so
                       silence is correct and scoring it as a miss defames the
                       router. Absent label -> recall is not scored.

    Reported per query:

      fired      the router routed at least one domain
      correct    a fired domain matched truth_domains  (None if it never fired)
      recalled   a fired domain matched recall_truth   (None if unlabelled)
    """
    fired_set = {str(d).lower() for d in (detected_domains or []) if str(d).strip()}
    truth_set = {str(d).lower() for d in (truth_domains or []) if str(d).strip()}
    recall_set = {str(d).lower() for d in (recall_truth or []) if str(d).strip()}
    return {
        "fired": bool(fired_set),
        "fired_domains": sorted(fired_set),
        # Needs both sides: a fire with no expected domain has nothing to be
        # right or wrong about.
        "correct": (bool(fired_set & truth_set)
                    if (fired_set and truth_set) else None),
        "recalled": (bool(fired_set & recall_set) if recall_set else None),
    }


# ---------------------------------------------- tier 2: groundedness (no LLM)

def citation_validity(answer_text: str, n_docs: int) -> tuple[float | None, list[int]]:
    """Do the answer's [n] markers resolve to docs that were actually retrieved?

    Returns (rate, dangling_numbers). None when the answer cites nothing —
    that's "not scored"; whether an answer *should* have cited is the separate
    n_citations / answered_rate signal.

    This is the cheapest possible hallucination check: the generator drops
    out-of-range citations when it builds Citation objects, so a marker like
    [9] against a top-7 context is invisible downstream. Here it is not.
    """
    used = [int(n) for n in _CITATION_RE.findall(str(answer_text or ""))]
    if not used:
        return None, []
    dangling = sorted({n for n in used if not (1 <= n <= n_docs)})
    valid = sum(1 for n in used if 1 <= n <= n_docs)
    return valid / len(used), dangling


def groundedness_floor(answer_text: str, docs: Sequence[Any],
                       n: int = 2) -> dict[str, Any]:
    """Deterministic groundedness proxy — the LLM-free floor.

    For every sentence carrying a [n] marker, measure the fraction of the
    sentence's content-word n-grams that also occur in the chunk(s) it cites
    (best cited chunk wins). Average over cited sentences.

    What it is: a floor. High overlap can't prove the claim is right, but LOW
    overlap means the sentence's wording has almost nothing to do with the
    source it points at, which is exactly the shape of a fabricated citation.
    Sentences with no citation marker are not scored (a topic sentence isn't
    ungrounded), and neither are sentences with no content words left after
    stopword removal.

    Returns {score, n_sentences_scored, n_sentences_uncited, worst}.
    `score` is None when no sentence was scorable.
    """
    docs = list(docs or [])
    doc_ngrams: list[set[tuple[str, ...]]] = [
        ngrams(content_tokens(getattr(d, "text", "")), n) for d in docs
    ]

    scored: list[float] = []
    worst: dict[str, Any] | None = None
    uncited = 0

    for sentence in split_sentences(answer_text):
        cited = [int(m) for m in _CITATION_RE.findall(sentence)]
        cited = [c for c in cited if 1 <= c <= len(docs)]
        if not cited:
            uncited += 1
            continue
        # The marker itself must not count as content.
        bare = _CITATION_RE.sub(" ", sentence)
        sent_ngrams = ngrams(content_tokens(bare), n)
        if not sent_ngrams:
            continue
        overlap = max(
            len(sent_ngrams & doc_ngrams[c - 1]) / len(sent_ngrams) for c in cited
        )
        scored.append(overlap)
        if worst is None or overlap < worst["overlap"]:
            worst = {"overlap": round(overlap, 3),
                     "sentence": sentence[:200],
                     "cited": cited}

    return {
        "score": (sum(scored) / len(scored)) if scored else None,
        "n_sentences_scored": len(scored),
        "n_sentences_uncited": uncited,
        "worst": worst,
    }


# ------------------------------------------------------- tier 3: calibration

def bucket_by(rows: Iterable[dict], bucket_key: str, score_key: str,
              order: Sequence[str] = ("HIGH", "MEDIUM", "LOW", "UNKNOWN"),
              ) -> dict[str, Any]:
    """Group rows by a label and average a score inside each group.

    Used for the calibration question: does the stated confidence track how
    good the answer actually was? A calibrated system scores higher in HIGH
    than in LOW. `gap` is HIGH minus LOW — positive means correctly ordered,
    None when either bucket is empty or unscored.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get(bucket_key) or "UNKNOWN").upper()
        b = buckets.setdefault(label, {"n": 0, "_scores": []})
        b["n"] += 1
        v = row.get(score_key)
        if v is not None:
            b["_scores"].append(float(v))

    out: dict[str, Any] = {}
    for label in list(order) + [k for k in buckets if k not in order]:
        if label not in buckets:
            continue
        scores = buckets[label].pop("_scores")
        out[label] = {
            "n": buckets[label]["n"],
            "n_scored": len(scores),
            "mean": round(sum(scores) / len(scores), 3) if scores else None,
        }

    hi, lo = out.get("HIGH", {}).get("mean"), out.get("LOW", {}).get("mean")
    gap = round(hi - lo, 3) if (hi is not None and lo is not None) else None
    return {"buckets": out, "gap": gap}


# ------------------------------------------------------------- aggregation

def mean_scored(values: Iterable[Any]) -> tuple[float | None, int]:
    """Average the non-None values. Returns (mean_or_None, n_scored).

    The n_scored half is the point: every aggregate the runner prints carries
    the number of entries it was actually computed over, so a metric that only
    covered 3 of 94 questions can never be read as a suite-wide result.
    """
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def rate_scored(values: Iterable[Any]) -> tuple[float | None, int]:
    """Same as mean_scored but for booleans (True/False, skipping None)."""
    vals = [bool(v) for v in values if v is not None]
    if not vals:
        return None, 0
    return sum(1 for v in vals if v) / len(vals), len(vals)
