"""Two-stage retrieval: a first-stage Search proposes candidates, a cross-encoder reorders them.

The reranker can only reorder what the first stage returns, so recall@k after reranking is
bounded by the first stage's recall at `depth`. Cost grows linearly with `depth`: one
cross-encoder pass per (query, candidate) pair. The default depth was chosen on the dev split;
the commit that set it records the measurement.
"""

from collections.abc import Iterable

from cuad_doc_qa.domain import Chunk, Search, SearchHit
from cuad_doc_qa.retrieval.rerank.scorer import PairScorer

DEFAULT_DEPTH = 50


class RerankedSearch:
    """A Search that rescores the first stage's top `depth` candidates with `scorer`."""

    def __init__(
        self,
        first_stage: Search,
        scorer: PairScorer,
        chunks: Iterable[Chunk],
        depth: int = DEFAULT_DEPTH,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        self._first_stage = first_stage
        self._scorer = scorer
        self._texts = {chunk.chunk_id: chunk.text for chunk in chunks}
        self._depth = depth

    def __call__(self, query: str, k: int = 10, doc_id: str | None = None) -> list[SearchHit]:
        """Top-`k` candidates by cross-encoder score, best first; ties broken by chunk id.

        Raises ValueError if the first stage returns a chunk this reranker has no text for,
        or if the scorer returns a different number of scores than it was given texts.
        """
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        candidates = self._first_stage(query, max(self._depth, k), doc_id)
        if not candidates:
            return []

        unknown = [hit.chunk_id for hit in candidates if hit.chunk_id not in self._texts]
        if unknown:
            raise ValueError(f"no text for first-stage chunk {unknown[0]}")
        scores = self._scorer.score(query, [self._texts[hit.chunk_id] for hit in candidates])
        if len(scores) != len(candidates):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(candidates)} texts")

        ranked = sorted(
            zip((hit.chunk_id for hit in candidates), scores, strict=True),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [SearchHit(chunk_id=chunk_id, score=score) for chunk_id, score in ranked[:k]]
