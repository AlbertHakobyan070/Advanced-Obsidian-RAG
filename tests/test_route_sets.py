"""eval/route_sets, the blind route acceptance set (route-then-cite Task 16; spec §5.8). bench validate
checks the gold against the corpus; this file checks the set's own rules, so a bad hand edit fails here
and not in the gate. eval/route_sets/README.md records how the set was drawn."""
from pathlib import Path

import pytest

from eval.bench.questions import load_sets
from eval.bench.route_strata import (STRATA_PREFIXES, deck_files, in_scope, locators_overlap,
                                     named_source, with_twins)
from src.utils.config_loader import load_config
from src.utils.provenance import provenance_of, provenance_rule

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "eval" / "route_sets"
# The sets quote the vault, so the public release and the Docker context leave them out (like
# eval/graph_sets); the folder holds only its README until the questions are drafted. The rules
# these guards apply (NAMED, locators_overlap) are pinned in test_route_strata.py, which always runs.
pytestmark = pytest.mark.skipif(not any(SETS.glob("*.yaml")),
                                reason="eval/route_sets holds no question set in this checkout")


def _config():
    path = ROOT / "config.yaml"
    if not path.exists():                # a fresh clone and the archives carry no config.yaml
        pytest.skip("no config.yaml in this checkout")
    return load_config(path)


def test_ids_suites_and_status():
    qs = load_sets(SETS)
    assert {q.id.split("-")[0] for q in qs} == set(STRATA_PREFIXES)
    assert {q.suite for q in qs} <= {"books", "multihop"} and all(q.provenance["status"] == "draft" for q in qs)
    assert all(q.provenance["author"] == "draft" for q in qs)


def test_at_least_60_answerable_dev_questions():
    assert sum(q.split == "dev" and q.answerable for q in load_sets(SETS)) >= 60


def test_gold_is_in_scope_primary_and_seeded():
    rule = provenance_rule(_config())
    for q in load_sets(SETS):
        assert q.provenance.get("seed_chunks"), q.id
        for g in q.gold:
            for s in g.sources():
                assert in_scope(s.file), (q.id, s.file)
                assert provenance_of({"source_file": s.file}, rule) != "generated", (q.id, s.file)


def test_questions_name_no_scoped_title():
    bad = [(q.id, hit) for q in load_sets(SETS) if (hit := named_source(q.question))]
    assert not bad, bad


def test_each_stratum_has_both_splits():
    qs = load_sets(SETS)
    for prefix in STRATA_PREFIXES:
        assert {q.split for q in qs if q.id.startswith(prefix + "-")} == {"dev", "test"}, prefix


def test_every_multihop_item_chains_two_required_hops():
    for q in load_sets(SETS):
        if q.suite == "multihop":
            assert q.answerable and q.strategy not in (None, "fact"), q.id
            assert {1, 2} <= {g.hop for g in q.gold if g.required}, q.id


def test_no_test_gold_overlaps_a_dev_gold():
    """The sealed test shares no evidence with dev: no chunk could satisfy a dev locator and a test
    one (route_strata.locators_overlap, the relevance rule: file, pages, heading prefix)."""
    side = {"dev": [], "test": []}
    for q in load_sets(SETS):
        side.setdefault(q.split, []).extend(s for g in q.gold for s in g.sources())
    bad = [(d.file, d.pages, d.heading, t.pages, t.heading)
           for d in side["dev"] for t in side["test"] if locators_overlap(d, t)]
    assert not bad, bad


def test_every_required_deck_gold_lists_its_twin():
    """R11: a deck gold whose same-stem twin (.pdf <-> .md) is in the corpus lists it, so retrieving
    the twin is not scored a miss for one arm only. Reads the corpus the config names."""
    data = _config().path("paths.chunks_file").parent
    if not any(data.glob("*chunks.jsonl")):
        pytest.skip(f"no chunk JSONLs under {data}: the twin guard reads the corpus")
    decks = deck_files(data)
    missing = [(q.id, g.file) for q in load_sets(SETS) for g in q.gold
               if g.required and with_twins(g, decks) != g]
    assert not missing, missing
