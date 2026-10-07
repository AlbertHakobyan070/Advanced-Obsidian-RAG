"""
relevance_gate.py — A post-rerank relevance cutoff, with an abstain path.

Reranking orders the candidate pool but never says "none of these answers the
question": the top-k always goes to the generator, however weak, and the model
then writes an answer from passages that do not contain it. The gate is the
decision step in between. After the rerank it scores every candidate against
the ORIGINAL question (never the HyDE text: that is a hypothetical answer, not
what the user asked), drops those below a threshold, and when nothing passes
the pipeline hands the generator an empty list. The generator already answers
that with a fixed LOW "nothing relevant" reply and no LLM call — that is the
abstain path, and nothing here duplicates it.

    rerank -> GATE -> small-to-big expansion -> generation

Two scorers (GATE_SCORERS), which are NOT interchangeable:
  rerank_score  The active reranker's own score, read off the docs it already
                scored: free, and the BASELINE any learned gate has to beat.
                Its unit is that reranker's scale (the MiniLM cross-encoder
                emits logits, lexical a coverage score), so a threshold means
                something for ONE reranker only. It needs a scoring rerank
                mode: `none` scores nothing, and a gate has no business
                passing docs nobody scored.
  laya          P(relevant) in [0, 1] from the experimental Laya scorer, one
                call per query. Only usable when that scorer was built; it is
                never quietly replaced by rerank_score.

Off by default. A per-call `enabled` / `threshold` / `scorer` beats the config,
and a per-call threshold or scorer alone is enough to switch the gate on for
that call — a knob that did nothing would be silently ignored. A per-call
scorer other than the configured one must bring its own threshold: the
configured threshold is in the configured scorer's units.

Misconfiguration is an error, never a pass-through. A bad config block (an
unknown scorer, a threshold that is not a finite number, an enabled gate with
no threshold, an enabled laya gate with no Laya scorer) raises when the
pipeline is built, so the service does not come up half-configured. Per call,
a gate asked for with no threshold anywhere, a rerank_score gate over unscored
docs, or a laya gate with no scorer raises when the call is made.

Usage:
    gate = RelevanceGate.from_config(cfg, reranker)
    kept, info = gate.apply(question, reranked, enabled=None, threshold=None)
"""
from __future__ import annotations

import math

from src.retrieval.reranker import RERANK_MODES
from src.utils.config_loader import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


GATE_SCORERS = ("rerank_score", "laya")

# The rerank modes that leave a score on the docs. Derived, so a mode added to
# RERANK_MODES shows up in the error below instead of being left out of it.
_SCORING_MODES = tuple(m for m in RERANK_MODES if m != "none")

_THRESHOLD_KEY = "retrieval.relevance_gate.threshold"

_LAYA_MISSING = (
    "the relevance gate's 'laya' scorer needs the experimental Laya scorer, "
    "which is switched off: set retrieval.laya.enabled: true (it also needs the "
    "laya package and a checkpoint, see docs/laya-finetune.md) and restart "
    ":8051. The gate does not fall back to rerank_score: choose that scorer "
    "explicitly, or leave the gate off")


def _finite(value, where: str) -> float | None:
    """A threshold as a float, None passing through. Anything that is not a
    finite number is an ERROR: NaN or +inf would drop every doc, so the gate
    would abstain on everything and still look like it works."""
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f"{where} must be a finite number, got {value!r}")
    return float(value)


class RelevanceGate:
    """Drops reranked docs that score below a threshold.

    `laya_scorer` is anything with `score(query, texts) -> list[float]` (one
    P(relevant) per text, in order); None when the Laya package is not built.
    Validates on construction, so a half-configured gate never exists."""

    def __init__(self, enabled: bool = False, scorer: str = "rerank_score",
                 threshold: float | None = None, laya_scorer=None):
        if scorer not in GATE_SCORERS:
            raise ValueError(
                f"retrieval.relevance_gate.scorer must be one of {GATE_SCORERS}, "
                f"got {scorer!r}")
        threshold = _finite(threshold, _THRESHOLD_KEY)
        if enabled and threshold is None:
            raise ValueError(
                f"retrieval.relevance_gate.enabled is true but {_THRESHOLD_KEY} "
                "is not set: choose a cutoff (in the scorer's own units) on the "
                "dev split, or turn the gate off")
        if enabled and scorer == "laya" and laya_scorer is None:
            raise ValueError(_LAYA_MISSING)
        self.enabled = bool(enabled)
        self.scorer = scorer
        self.threshold = threshold
        self.laya_scorer = laya_scorer

    @classmethod
    def from_config(cls, cfg: Config, reranker, laya_scorer=None) -> "RelevanceGate":
        """Build from `retrieval.relevance_gate`. The laya scorer defaults to
        the reranker's own (`reranker.laya_scorer`, None while
        retrieval.laya.enabled is off), so the checkpoint loads once for both
        roles; rerank_score reads the scores the reranker put on the docs."""
        if laya_scorer is None:
            laya_scorer = getattr(reranker, "laya_scorer", None)
        block = cfg.get("retrieval.relevance_gate", {}) or {}
        if not isinstance(block, dict):
            raise ValueError(
                "retrieval.relevance_gate must be a mapping with enabled / "
                f"scorer / threshold, got {block!r}")
        return cls(
            enabled=bool(block.get("enabled", False)),
            scorer=block.get("scorer", "rerank_score"),
            threshold=block.get("threshold"),
            laya_scorer=laya_scorer,
        )

    def is_on(self, enabled: bool | None = None,
              threshold: float | None = None, scorer: str | None = None) -> bool:
        """Does a call with these per-call settings gate? An explicit `enabled`
        wins; failing that, any per-call threshold or scorer turns the gate on;
        failing that, the configured flag decides."""
        if enabled is not None:
            return bool(enabled)
        if threshold is not None or scorer is not None:
            return True
        return self.enabled

    def _scores(self, query: str, docs: list, scorer: str) -> list[float]:
        if scorer == "laya":
            if self.laya_scorer is None:
                raise ValueError(_LAYA_MISSING)
            if not docs:
                return []
            try:
                raw = self.laya_scorer.score(query, [d.text for d in docs])
            except Exception as e:
                # Name the stage: the same scorer may also be the reranker.
                raise RuntimeError(
                    f"relevance gate: the laya scorer failed — "
                    f"{type(e).__name__}: {e}") from e
            scores = [float(s) for s in raw]
            if len(scores) != len(docs):
                # A misaligned list would pin scores on the wrong docs.
                raise ValueError(
                    f"the laya scorer returned {len(scores)} score(s) for "
                    f"{len(docs)} doc(s)")
            return scores
        unscored = sum(1 for d in docs if d.rerank_score is None)
        if unscored:
            raise ValueError(
                f"the relevance gate's 'rerank_score' scorer needs scored "
                f"candidates, but {unscored} of {len(docs)} carry no "
                "rerank_score: rerank mode 'none' scores nothing. Use a scoring "
                f"rerank mode ({', '.join(_SCORING_MODES)}) or the 'laya' "
                "scorer, or turn the gate off for this call")
        return [d.rerank_score for d in docs]

    def apply(self, query: str, docs: list, *, enabled: bool | None = None,
              threshold: float | None = None,
              scorer: str | None = None) -> tuple[list, dict]:
        """Keep the docs scoring at or above the threshold, in their order.

        Returns (kept, info). Off, the docs come back untouched with
        {"enabled": False}. On, info is {enabled, scorer, threshold, kept,
        dropped, abstained}; `abstained` is kept == 0 — also when retrieval
        itself returned nothing. Every scored doc, dropped ones included, gets
        its score in debug["gate_score"] so a run can be re-thresholded offline.
        Raises ValueError — never a pass-through — for a gate that is on but
        cannot work (see module docstring) or whose scorer returned the wrong
        number of scores or a non-finite one."""
        if scorer is not None and scorer not in GATE_SCORERS:
            # Checked even when the gate is off: a typo must not hide behind
            # gate=false and surface only once someone turns the gate on.
            raise ValueError(
                f"gate_scorer must be one of {GATE_SCORERS}, got {scorer!r}")
        if not self.is_on(enabled, threshold, scorer):
            return docs, {"enabled": False}
        use = scorer if scorer is not None else self.scorer
        cutoff = _finite(threshold, "gate_threshold")
        if cutoff is None:
            if use != self.scorer:
                raise ValueError(
                    f"gate_scorer {use!r} is not the configured scorer "
                    f"({self.scorer!r}), and the configured threshold is in that "
                    "scorer's units: send gate_threshold with it")
            cutoff = self.threshold
        if cutoff is None:
            raise ValueError(
                "the relevance gate is on but has no threshold: send "
                f"gate_threshold, or set {_THRESHOLD_KEY}")
        scores = self._scores(query, docs, use)
        if not all(math.isfinite(s) for s in scores):
            # NaN compares False against every cutoff, so it would quietly drop
            # its doc: a numeric failure in the scorer must not read as "not
            # relevant".
            raise ValueError(
                f"the {use!r} scorer returned a non-finite score; "
                "refusing to gate on it")
        kept = []
        for doc, score in zip(docs, scores):
            doc.debug["gate_score"] = score
            if score >= cutoff:
                kept.append(doc)
        log.info("relevance gate (%s >= %s): kept %d of %d",
                 use, cutoff, len(kept), len(docs))
        return kept, {
            "enabled": True,
            "scorer": use,
            "threshold": cutoff,
            "kept": len(kept),
            "dropped": len(docs) - len(kept),
            "abstained": len(kept) == 0,
        }
