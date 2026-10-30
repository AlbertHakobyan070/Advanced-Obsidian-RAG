import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.bench.metrics import (
    complete_at, hit_at, mrr_at, ndcg_at, precision_at, recall_at, required,
)
from eval.bench.questions import GoldSource

F = frozenset
G2 = [GoldSource(file="a"), GoldSource(file="b")]


def test_required_defaults_to_all_when_none_flagged():
    assert required([GoldSource(file="a", required=False)]) == [0]
    assert required([GoldSource(file="a"), GoldSource(file="b", required=False)]) == [0]


def test_hit_mrr_precision():
    m = [F(), F(), F({0}), F({0})]
    assert hit_at(m, 2) == 0.0 and hit_at(m, 3) == 1.0
    assert mrr_at(m, 10) == 1 / 3
    assert precision_at(m, 4) == 0.5


def test_recall_and_complete_count_sources_not_chunks():
    m = [F({0}), F({0}), F({1})]
    assert recall_at(m, G2, 2) == 0.5 and complete_at(m, G2, 2) == 0.0
    assert recall_at(m, G2, 3) == 1.0 and complete_at(m, G2, 3) == 1.0


def test_ndcg_rewards_new_sources_only():
    ideal = 1 + 1 / math.log2(3)
    assert ndcg_at([F({0}), F({1})], G2, 10) == 1.0
    assert math.isclose(ndcg_at([F({0}), F({0}), F({1})], G2, 10), (1 + 1 / 2) / ideal)
    assert ndcg_at([F(), F()], G2, 10) == 0.0


def test_hop_recall_and_gold_only_via():
    from eval.bench.metrics import gold_only_via, hop_recall
    gold = [GoldSource(file="a", hop=1), GoldSource(file="b", hop=2), GoldSource(file="c", hop=2),
            GoldSource(file="d"), GoldSource(file="e", hop=1, required=False)]
    m = [F({0}), F(), F({1}), F({3}), F()]
    assert hop_recall(m, gold, 10) == {"1": 1.0, "2": 0.5}     # e is optional context: never counted
    assert hop_recall(m, gold, 2) == {"1": 1.0, "2": 0.0}
    assert hop_recall(m, [GoldSource(file="a")], 10) == {}      # no entry names a hop
    flagged = [False, False, True, True, True]                  # ranks 2-4 came from the walk
    assert gold_only_via(m, flagged, gold, 10, searched=[]) == 2             # b and d, found nowhere else
    assert gold_only_via(m, flagged, gold, 3, searched=[]) == 1              # rank 3 (d) is past k
    assert gold_only_via([F({0}), F({0})], [False, True], gold, 10, searched=[]) == 0   # the search found it too
    assert gold_only_via([F({4})], [True], gold, 10, searched=[]) == 0       # optional gold is not counted


def test_gold_the_search_had_before_the_union_rerank_is_not_the_walks():
    """The union rerank can push the search's own chunk of a gold file out of the top k
    while a walked chunk of the same file takes its place: that gold is not the walk's."""
    from eval.bench.metrics import gold_only_via
    gold = [GoldSource(file="a"), GoldSource(file="b")]
    final, flagged = [F(), F({0}), F({1})], [False, True, True]
    assert gold_only_via(final, flagged, gold, 10, searched=[F(), F({0})]) == 1      # only b is the walk's
    assert gold_only_via(final, flagged, gold, 10, searched=[F()] * 10 + [F({0})]) == 2   # past the search's k
