"""Per-query ranking metrics with binary relevance.

All metrics are defined only for answerable questions (non-empty `relevant`): an empty set has
no meaningful recall, and unanswerable questions are scored separately as abstention.

`ranking` is a best-first sequence of unique chunk ids; duplicates would inflate every metric.
"""

import math
from collections.abc import Sequence, Set


def recall_at_k(ranking: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant chunks found in the top `k`."""
    _validate(ranking, relevant, k)
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def reciprocal_rank(ranking: Sequence[str], relevant: Set[str]) -> float:
    """1 / rank of the first relevant chunk, or 0.0 if none is retrieved."""
    _validate(ranking, relevant, 1)
    for rank, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: Set[str], k: int) -> float:
    """DCG of the top `k` divided by the DCG of an ideal top `k`; gain 1 per relevant chunk."""
    _validate(ranking, relevant, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranking[:k], start=1)
        if chunk_id in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def _validate(ranking: Sequence[str], relevant: Set[str], k: int) -> None:
    if not relevant:
        raise ValueError("metric is undefined for a question without relevant chunks")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains duplicate chunk ids")
