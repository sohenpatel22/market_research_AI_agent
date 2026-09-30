"""Ranking metrics for retrieval quality."""

import math


def dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg_at_k(retrieved_ids: list, relevance: dict, k: int = 6) -> float:
    """NDCG@k. `relevance` maps doc id -> graded relevance (0 if absent)."""
    gains = [relevance.get(i, 0.0) for i in retrieved_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(gains) / best if best > 0 else 0.0
