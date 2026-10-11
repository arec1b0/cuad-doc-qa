"""Ranking metrics against hand-computed values.

Reference: ranking [x, r1, y, r2] with relevant {r1, r2, r3}.
  recall@2 = 1/3, recall@4 = 2/3, RR = 1/2
  nDCG@2 = (1/log2 3) / (1 + 1/log2 3)                        = 0.386853
  nDCG@4 = (1/log2 3 + 1/log2 5) / (1 + 1/log2 3 + 1/log2 4)  = 0.498189
"""

import pytest

from cuad_doc_qa.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

RANKING = ["x", "r1", "y", "r2"]
RELEVANT = {"r1", "r2", "r3"}


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1 / 3), (4, 2 / 3), (10, 2 / 3)])
def test_recall_at_k(k: int, expected: float) -> None:
    assert recall_at_k(RANKING, RELEVANT, k) == pytest.approx(expected)


def test_reciprocal_rank_uses_the_first_relevant_hit() -> None:
    assert reciprocal_rank(RANKING, RELEVANT) == pytest.approx(0.5)
    assert reciprocal_rank(["r2", "x"], RELEVANT) == pytest.approx(1.0)


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.386853), (4, 0.498189)])
def test_ndcg_at_k(k: int, expected: float) -> None:
    assert ndcg_at_k(RANKING, RELEVANT, k) == pytest.approx(expected, abs=1e-6)


def test_ideal_dcg_is_capped_by_the_number_of_relevant_chunks() -> None:
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}, 10) == pytest.approx(0.919721, abs=1e-6)
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 10) == pytest.approx(1.0)


@pytest.mark.parametrize("ranking", [[], ["x", "y"]], ids=["empty-ranking", "no-relevant-hit"])
def test_missing_every_relevant_chunk_scores_zero(ranking: list[str]) -> None:
    assert recall_at_k(ranking, RELEVANT, 10) == 0.0
    assert reciprocal_rank(ranking, RELEVANT) == 0.0
    assert ndcg_at_k(ranking, RELEVANT, 10) == 0.0


@pytest.mark.parametrize(
    ("ranking", "relevant", "k", "message"),
    [
        (["x"], set(), 10, "undefined for a question without relevant chunks"),
        (["r1", "r1"], RELEVANT, 10, "duplicate chunk ids"),
        (["r1"], RELEVANT, 0, "k must be positive"),
    ],
    ids=["unanswerable", "duplicates", "zero-k"],
)
def test_rejects_invalid_input(
    ranking: list[str], relevant: set[str], k: int, message: str
) -> None:
    for metric in (recall_at_k, ndcg_at_k):
        with pytest.raises(ValueError, match=message):
            metric(ranking, relevant, k)
    if k > 0:
        with pytest.raises(ValueError, match=message):
            reciprocal_rank(ranking, relevant)
