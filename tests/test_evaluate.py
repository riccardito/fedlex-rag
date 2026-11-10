import pytest

from fedlex_rag.evaluate import Question, evaluate, recall_at, reciprocal_rank
from fedlex_rag.loader import Article


def test_recall_and_reciprocal_rank():
    ranked = ["a", "b", "c", "d"]
    assert recall_at(ranked, ["c"], 2) == 0 and recall_at(ranked, ["c"], 3) == 1
    assert recall_at(ranked, ["a", "z"], 5) == 0.5
    assert reciprocal_rank(ranked, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, ["z"]) == 0


class Fixed:
    name = "fixed"

    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, k=5):
        return [(Article(i, "OR", "1", "", "", "", ""), 1.0) for i in self.ranking[query][:k]]


def test_evaluate_skips_unanswerable_and_lists_misses():
    qs = [Question("q1", "x", ["A"]), Question("q2", "y", ["B"]), Question("q3", "z", [])]
    r = evaluate(Fixed({"x": ["A", "C"], "y": ["C", "D", "E", "F", "G", "B"]}), qs, k=5)
    assert r.n == 2 and r.recall == 0.5 and r.hit1 == 0.5
    assert r.mrr == pytest.approx((1 + 1 / 6) / 2)
    assert r.misses == ["q2"]
