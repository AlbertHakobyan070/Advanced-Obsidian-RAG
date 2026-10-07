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
