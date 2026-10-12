"""reciprocal_rank_fusion and FusedSearch.

Reference, rrf_k = 60: R1 = [a, b, c], R2 = [b, d]
  a = 1/61, b = 1/62 + 1/61, c = 1/63, d = 1/62  ->  b, a, d, c
"""

from itertools import permutations

import pytest

from cuad_doc_qa.domain import SearchHit
from cuad_doc_qa.retrieval.fusion import FusedSearch, reciprocal_rank_fusion


def _ranking(*chunk_ids: str) -> list[SearchHit]:
    return [SearchHit(chunk_id, 100.0 - i) for i, chunk_id in enumerate(chunk_ids)]


R1 = _ranking("a", "b", "c")
R2 = _ranking("b", "d")


def test_fuses_by_reciprocal_rank() -> None:
    fused = reciprocal_rank_fusion([R1, R2], k=10)

    assert [hit.chunk_id for hit in fused] == ["b", "a", "d", "c"]
    assert [hit.score for hit in fused] == pytest.approx([1 / 62 + 1 / 61, 1 / 61, 1 / 62, 1 / 63])


def test_ignores_retriever_scores() -> None:
    rescaled = [SearchHit(hit.chunk_id, hit.score * 1e6) for hit in R1]

    assert reciprocal_rank_fusion([rescaled, R2], k=10) == reciprocal_rank_fusion([R1, R2], k=10)


def test_result_does_not_depend_on_ranking_order() -> None:
    rankings = [_ranking("a", "b", "c"), _ranking("c", "a", "b"), _ranking("b", "c", "a")]

    results = {tuple(reciprocal_rank_fusion(list(p), k=3)) for p in permutations(rankings)}

    assert len(results) == 1
    [result] = results
    assert [hit.chunk_id for hit in result] == ["a", "b", "c"]
    assert len({hit.score for hit in result}) == 1


def test_single_ranking_keeps_its_order() -> None:
    assert [hit.chunk_id for hit in reciprocal_rank_fusion([R1], k=10)] == ["a", "b", "c"]


def test_k_caps_the_result_and_empty_input_gives_nothing() -> None:
    assert [hit.chunk_id for hit in reciprocal_rank_fusion([R1, R2], k=2)] == ["b", "a"]
    assert reciprocal_rank_fusion([], k=5) == []
    assert reciprocal_rank_fusion([[], []], k=5) == []


@pytest.mark.parametrize(
    ("rankings", "k", "rrf_k", "message"),
    [
        ([R1], 0, 60, "k must be positive"),
        ([R1], 5, -1, "rrf_k must be non-negative"),
        ([_ranking("a", "a")], 5, 60, "listed twice"),
    ],
    ids=["zero-k", "negative-rrf-k", "duplicate-in-ranking"],
)
def test_rejects_invalid_input(
    rankings: list[list[SearchHit]], k: int, rrf_k: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion(rankings, k, rrf_k)


class RecordingSearch:
    def __init__(self, ranking: list[SearchHit]) -> None:
        self.ranking = ranking
        self.calls: list[tuple[str, int, str | None]] = []

    def __call__(self, query: str, k: int, doc_id: str | None) -> list[SearchHit]:
        self.calls.append((query, k, doc_id))
        return self.ranking[:k]


def test_fused_search_queries_every_retriever_at_depth_and_fuses() -> None:
    first, second = RecordingSearch(R1), RecordingSearch(R2)
    search = FusedSearch([first, second], depth=50)

    hits = search("governing law", 3, "D1")

    assert first.calls == second.calls == [("governing law", 50, "D1")]
    assert [hit.chunk_id for hit in hits] == ["b", "a", "d"]


def test_fused_search_never_asks_for_fewer_candidates_than_k() -> None:
    first = RecordingSearch(R1)

    FusedSearch([first], depth=2)("q", 10, None)

    assert first.calls == [("q", 10, None)]


@pytest.mark.parametrize(
    ("searches", "depth", "message"),
    [([], 10, "at least one search"), ([RecordingSearch(R1)], 0, "depth must be positive")],
    ids=["no-searches", "zero-depth"],
)
def test_fused_search_rejects_invalid_configuration(
    searches: list[RecordingSearch], depth: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FusedSearch(searches, depth=depth)
