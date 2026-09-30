import pytest

from market_research_agent.eval.retrieval_metrics import ndcg_at_k


def test_ndcg_perfect_and_imperfect():
    rel = {1: 2.0, 2: 1.0}
    assert ndcg_at_k([1, 2, 9], rel) == pytest.approx(1.0)
    assert 0 < ndcg_at_k([2, 1, 9], rel) < 1
    assert ndcg_at_k([8, 9], rel) == 0.0
    assert ndcg_at_k([1], {}) == 0.0
