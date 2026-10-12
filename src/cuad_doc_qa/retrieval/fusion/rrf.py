"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009).

score(chunk) = sum over rankings of 1 / (rrf_k + rank), rank starting at 1; a chunk absent
from a ranking contributes nothing. Only positions are used, so BM25 scores and cosine
similarities never need to be put on a common scale. rrf_k = 60 is the paper's value, not tuned.

Contributions are summed with math.fsum (correctly rounded), so a fused score does not depend
on the order in which rankings are passed, and equal rank profiles tie exactly; ties are
broken by chunk id.
"""

import math
from collections import defaultdict
from collections.abc import Sequence

from cuad_doc_qa.domain import Search, SearchHit

RRF_K = 60
DEFAULT_DEPTH = 100


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]], k: int, rrf_k: int = RRF_K
) -> list[SearchHit]:
    """Top-`k` chunks by fused score, best first; ties broken by chunk id.

    Raises ValueError if k < 1, rrf_k < 0, or a ranking lists the same chunk twice.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be non-negative, got {rrf_k}")

    contributions: dict[str, list[float]] = defaultdict(list)
    for ranking in rankings:
        seen: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            if hit.chunk_id in seen:
                raise ValueError(f"{hit.chunk_id}: listed twice in one ranking")
            seen.add(hit.chunk_id)
            contributions[hit.chunk_id].append(1.0 / (rrf_k + rank))

    fused = sorted(
        ((chunk_id, math.fsum(parts)) for chunk_id, parts in contributions.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return [SearchHit(chunk_id=chunk_id, score=score) for chunk_id, score in fused[:k]]


class FusedSearch:
    """A Search that asks every retriever for `depth` candidates and fuses them with RRF."""

    def __init__(
        self, searches: Sequence[Search], depth: int = DEFAULT_DEPTH, rrf_k: int = RRF_K
    ) -> None:
        if not searches:
            raise ValueError("at least one search is required")
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        self._searches = tuple(searches)
        self._depth = depth
        self._rrf_k = rrf_k

    def __call__(self, query: str, k: int = 10, doc_id: str | None = None) -> list[SearchHit]:
        candidates = max(self._depth, k)
        rankings = [search(query, candidates, doc_id) for search in self._searches]
        return reciprocal_rank_fusion(rankings, k, self._rrf_k)
