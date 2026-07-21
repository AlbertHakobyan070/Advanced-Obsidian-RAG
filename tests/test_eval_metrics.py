"""Eval-suite v2 metric tests (session 16).

Run:  python -m pytest tests/ -q          (project venv)

eval/metrics.py is deliberately pure — no index, no LLM, no filesystem — so
the scoring layer can be pinned down here in milliseconds. The behaviour these
tests guard is the part that is easy to get quietly wrong:

  * None means "not scored", never "scored zero" — a golden entry without
    labels must drop OUT of an average, not drag it down.
  * en-dash vault paths still match hyphens typed into the golden set.
  * the groundedness floor scores only CITED sentences, and a citation
    pointing at an unrelated chunk actually scores low.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics as M


class Doc:
    """Minimal stand-in for RetrievedDoc (id/text/metadata is all we touch)."""

    def __init__(self, text="", domain="", source_file="", doc_id="x"):
        self.id = doc_id
        self.text = text
        self.metadata = {"domain": domain, "source_file": source_file}


# ------------------------------------------------------------- text utils

def test_dash_folding_matches_vault_paths():
    # The real vault path uses U+2013; the golden set types a plain hyphen.
    assert M.normalize_path("A:/Vault/00 – AUA_DS/Lecture – 3.pdf") == \
           M.normalize_path("A:\\Vault\\00 - AUA_DS\\Lecture - 3.pdf")


def test_content_tokens_drop_stopwords_keep_hyphenates():
    toks = M.content_tokens("The cross-entropy of a model is not the same")
    assert "cross-entropy" in toks
    assert "the" not in toks and "is" not in toks


def test_ngrams_degrade_for_short_spans():
    # A 1-token sentence is compared as a unigram rather than scoring 0.
    assert M.ngrams(["adam"], 2) == {("adam",)}
    assert M.ngrams([], 2) == set()
    assert M.ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}


def test_split_sentences_reattaches_trailing_citations():
    out = M.split_sentences("Gradients vanish over long sequences. [1][2]\nUse clipping. [3]")
    assert len(out) == 2
    assert out[0].endswith("[1][2]")
    assert out[1].endswith("[3]")


# --------------------------------------------------------- tier 1: retrieval

def test_reciprocal_rank_and_hit():
    docs = [Doc(domain="ml"), Doc(domain="nlp"), Doc(domain="nlp")]
    assert M.reciprocal_rank("nlp", docs) == 0.5      # first nlp hit at rank 2
    assert M.reciprocal_rank("ml", docs) == 1.0
    assert M.hit_at_k("nlp", docs) is True
    # declared-but-missing is a real 0.0, not "unscored"
    assert M.reciprocal_rank("stats", docs) == 0.0
    assert M.hit_at_k("stats", docs) is False


def test_hit_at_k_respects_the_cutoff():
    docs = [Doc(domain="ml"), Doc(domain="nlp")]
    assert M.hit_at_k("nlp", docs, k=1) is False
    assert M.hit_at_k("nlp", docs, k=2) is True


def test_no_declared_domain_is_unscored_not_zero():
    docs = [Doc(domain="ml")]
    assert M.reciprocal_rank("", docs) is None
    assert M.hit_at_k("", docs) is None


def test_recall_at_k_matches_trailing_fragments():
    docs = [Doc(source_file="A:/Vault/NLP/Lecture – 3.md"),
            Doc(source_file="A:/Vault/ML/Notes.md")]
    assert M.recall_at_k(["NLP/Lecture - 3.md"], docs) == 1.0
    assert M.recall_at_k(["NLP/Lecture - 3.md", "Stats/Missing.md"], docs) == 0.5
    # unlabelled entries are skipped entirely
    assert M.recall_at_k(None, docs) is None
    assert M.recall_at_k([], docs) is None


def test_scope_truth_comes_only_from_an_explicit_label():
    """Regression: truth must NOT fall back to the entry's `domain`.

    It used to, and that made scope_recall meaningless — `domain` says where
    the answer should come from, not whether the question names a domain.
    "Explain the bias-variance tradeoff" (domain=ml) names no domain, so a
    silent router is correct; scoring it as a miss mislabelled 65 of 68
    "misses" on the real 100-question suite.
    """
    from eval.eval_runner import _scope_truth
    assert _scope_truth({"question": "Explain the bias-variance tradeoff.",
                         "domain": "ml"}) == []
    assert _scope_truth({"domain": "ml", "expect_scope": ["ml"]}) == ["ml"]


def test_scope_verdict_separates_precision_from_recall():
    """Precision and recall take DIFFERENT truths, on purpose.

    precision truth = the expected answer-domain (valid because precision is
    conditioned on the router having fired); recall truth = an explicit
    "this question names a domain" label.
    """
    # fired and right, and the query was labelled as one that should fire
    v = M.scope_verdict(["nlp"], ["nlp"], ["nlp"])
    assert (v["fired"], v["correct"], v["recalled"]) == (True, True, True)
    # fired at the wrong domain
    v = M.scope_verdict(["ml"], ["nlp"], ["nlp"])
    assert (v["fired"], v["correct"], v["recalled"]) == (True, False, False)
    # never fired: precision has nothing to say; labelled -> a real recall miss
    v = M.scope_verdict([], ["nlp"], ["nlp"])
    assert (v["fired"], v["correct"], v["recalled"]) == (False, None, False)
    # THE IMPORTANT CASE: unlabelled query (names no domain). Precision still
    # scores because it fired; recall must stay unscored, not count as a miss.
    v = M.scope_verdict(["nlp"], ["nlp"], [])
    assert (v["fired"], v["correct"], v["recalled"]) == (True, True, None)
    # silent router on an unlabelled query: nothing to score at all
    v = M.scope_verdict([], ["ml"], [])
    assert (v["fired"], v["correct"], v["recalled"]) == (False, None, None)


# ------------------------------------------------------ tier 2: groundedness

def test_citation_validity_catches_dangling_markers():
    rate, dangling = M.citation_validity("Backprop is used [1], and also [9].", 3)
    assert dangling == [9]
    assert rate == 0.5
    # nothing cited -> not scored (the n_citations signal covers that instead)
    assert M.citation_validity("No citations here.", 3) == (None, [])


def test_groundedness_floor_rewards_real_overlap():
    docs = [Doc(text="Gradient clipping rescales the gradient when its norm "
                     "exceeds a threshold, preventing exploding gradients.")]
    good = M.groundedness_floor(
        "Gradient clipping rescales the gradient when its norm exceeds a "
        "threshold [1].", docs, n=2)
    assert good["score"] > 0.8
    assert good["n_sentences_scored"] == 1


def test_groundedness_floor_punishes_an_unrelated_citation():
    docs = [Doc(text="Gradient clipping rescales the gradient when its norm "
                     "exceeds a threshold.")]
    bad = M.groundedness_floor(
        "The Baroque period produced harpsichord concertos in Venice [1].",
        docs, n=2)
    assert bad["score"] == 0.0
    assert bad["worst"]["cited"] == [1]


def test_groundedness_floor_ignores_uncited_sentences():
    docs = [Doc(text="Adam combines momentum with RMSProp scaling.")]
    out = M.groundedness_floor(
        "Here is an overview.\nAdam combines momentum with RMSProp scaling [1].",
        docs, n=2)
    assert out["n_sentences_scored"] == 1
    assert out["n_sentences_uncited"] == 1
    # an answer with no citations at all is unscored, not zero
    assert M.groundedness_floor("No citations at all.", docs)["score"] is None


def test_groundedness_floor_ignores_out_of_range_citations():
    docs = [Doc(text="Adam combines momentum with RMSProp scaling.")]
    out = M.groundedness_floor("Adam is an optimizer [7].", docs, n=2)
    assert out["score"] is None            # [7] resolves to nothing to compare
    assert out["n_sentences_uncited"] == 1


# ------------------------------------------------------- tier 3 + aggregates

def test_bucket_by_orders_confidence_and_reports_gap():
    rows = [
        {"confidence": "HIGH", "score": 1.0},
        {"confidence": "HIGH", "score": 0.8},
        {"confidence": "LOW", "score": 0.2},
        {"confidence": "MEDIUM", "score": None},
    ]
    out = M.bucket_by(rows, "confidence", "score")
    assert list(out["buckets"]) == ["HIGH", "MEDIUM", "LOW"]
    assert out["buckets"]["HIGH"]["mean"] == 0.9
    assert out["buckets"]["MEDIUM"] == {"n": 1, "n_scored": 0, "mean": None}
    assert out["gap"] == 0.7               # calibrated: HIGH scores above LOW


def test_bucket_gap_is_none_when_a_bucket_is_empty():
    out = M.bucket_by([{"confidence": "HIGH", "score": 1.0}], "confidence", "score")
    assert out["gap"] is None


def test_aggregates_report_coverage_and_skip_nones():
    assert M.mean_scored([1.0, None, 0.0]) == (0.5, 2)
    assert M.mean_scored([None, None]) == (None, 0)
    assert M.rate_scored([True, False, None]) == (0.5, 2)
    assert M.rate_scored([]) == (None, 0)
