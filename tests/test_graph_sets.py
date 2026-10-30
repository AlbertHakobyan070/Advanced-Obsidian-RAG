"""eval/graph_sets, the slice E pilot graph question set (M1). The rules are the ones the bench applies
(docs/evaluation.md, "Graph question sets"): bench validate checks the gold against the corpus,
this file checks what needs no corpus, so a bad hand edit fails here and not in a bench run."""
from pathlib import Path

import pytest

from eval.bench.questions import load_sets
from eval.bench.relevance import norm_file

SETS = Path(__file__).resolve().parents[1] / "eval" / "graph_sets"
# The sets quote the vault, so the public release and the Docker context leave them out (like eval/sets).
pytestmark = pytest.mark.skipif(not SETS.is_dir(), reason="eval/graph_sets is vault-derived: not in this checkout")
GENERATED = ("(generated)", "- chapter notes/", "- module notes/")   # the generated paths of the pilot


def _live():
    return load_sets(SETS)


def _count(suite, split=None):
    return sum(1 for q in _live() if q.suite == suite and (split is None or q.split == split))


def test_the_pilot_set_has_its_quotas():
    """30 multi-hop per split = report.DIAG_BELOW: below it a suite is a diagnostic, not a result."""
    assert _count("multihop", "dev") >= 30 and _count("multihop", "test") >= 30, (_count("multihop", "dev"), _count("multihop", "test"))
    assert _count("books") >= 10, _count("books")
    assert _count("unanswerable") >= 5, _count("unanswerable")


def test_every_live_question_has_its_split():
    assert all(q.split in ("dev", "test") for q in _live())


def test_every_multihop_item_chains_two_required_hops():
    for q in _live():
        if q.suite == "multihop":
            assert q.answerable and q.strategy not in (None, "fact"), q.id
            assert {1, 2} <= {g.hop for g in q.gold if g.required}, q.id


def test_gold_never_names_a_generated_source():
    for q in _live():
        for g in q.gold:
            for s in g.sources():
                assert not any(m in norm_file(s.file) for m in GENERATED), (q.id, s.file)


def test_controls_are_single_hop_facts_and_unanswerables_have_no_gold():
    for q in _live():
        if q.suite == "books":
            assert q.strategy == "fact" and sum(g.required for g in q.gold) == 1, q.id
        if q.suite == "unanswerable":
            assert not q.answerable and not q.gold, q.id


def _overlaps(a, b):
    return norm_file(a.file) == norm_file(b.file) and (a.pages is None or b.pages is None
                                 or not (a.pages[1] < b.pages[0] or b.pages[1] < a.pages[0]))


def test_no_test_gold_overlaps_a_dev_gold():
    """The sealed test shares no evidence with dev: no test locator overlaps a dev one."""
    side = {"dev": [], "test": []}
    for q in _live():
        side.setdefault(q.split, []).extend(s for g in q.gold for s in g.sources())
    bad = [(d.file, d.pages, t.pages) for d in side["dev"] for t in side["test"] if _overlaps(d, t)]
    assert not bad, bad
