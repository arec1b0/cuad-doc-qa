"""RerankedSearch with a deterministic keyword scorer standing in for the cross-encoder."""

from collections.abc import Sequence

import pytest

from cuad_doc_qa.domain import Chunk, SearchHit
from cuad_doc_qa.retrieval.rerank import RerankedSearch


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=chunk_id.split("::")[0], start=0, end=len(text), text=text
    )


CHUNKS = [
    _chunk("A::0000", "Payment is due within thirty days."),
    _chunk("A::0001", "This Agreement is governed by Delaware law."),
    _chunk("A::0002", "Either party may terminate on notice."),
    _chunk("A::0003", "Delaware courts have exclusive jurisdiction under Delaware law."),
]


class KeywordScorer:
    """Score = occurrences of query words in the text; records every call."""

    def __init__(self, extra_scores: int = 0) -> None:
        self.extra_scores = extra_scores
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, list(texts)))
        words = query.lower().split()
        scores = [float(sum(text.lower().count(w) for w in words)) for text in texts]
        return scores + [0.0] * self.extra_scores


class FixedFirstStage:
    """Returns its ranking in order, capped at k; records every call."""

    def __init__(self, *chunk_ids: str) -> None:
        self.ranking = [SearchHit(chunk_id, 1.0) for chunk_id in chunk_ids]
        self.calls: list[tuple[str, int, str | None]] = []

    def __call__(self, query: str, k: int, doc_id: str | None) -> list[SearchHit]:
        self.calls.append((query, k, doc_id))
        return self.ranking[:k]


def test_reorders_first_stage_candidates_by_scorer() -> None:
    first = FixedFirstStage("A::0000", "A::0001", "A::0002", "A::0003")
    scorer = KeywordScorer()
    search = RerankedSearch(first, scorer, CHUNKS, depth=4)

    hits = search("delaware", 2, "A")

    assert first.calls == [("delaware", 4, "A")]
    assert scorer.calls == [("delaware", [c.text for c in CHUNKS])]
    assert [(hit.chunk_id, hit.score) for hit in hits] == [("A::0003", 2.0), ("A::0001", 1.0)]


def test_only_candidates_within_depth_are_scored() -> None:
    first = FixedFirstStage("A::0000", "A::0001", "A::0002", "A::0003")
    scorer = KeywordScorer()

    hits = RerankedSearch(first, scorer, CHUNKS, depth=2)("delaware", 1, None)

    assert first.calls == [("delaware", 2, None)]
    assert len(scorer.calls[0][1]) == 2
    assert [hit.chunk_id for hit in hits] == ["A::0001"]


def test_never_asks_the_first_stage_for_fewer_than_k() -> None:
    first = FixedFirstStage("A::0000")

    RerankedSearch(first, KeywordScorer(), CHUNKS, depth=1)("q", 5, None)

    assert first.calls == [("q", 5, None)]


def test_ties_are_broken_by_chunk_id() -> None:
    first = FixedFirstStage("A::0002", "A::0000")

    hits = RerankedSearch(first, KeywordScorer(), CHUNKS)("unmatched", 2, None)

    assert [hit.chunk_id for hit in hits] == ["A::0000", "A::0002"]


def test_empty_first_stage_skips_the_scorer() -> None:
    scorer = KeywordScorer()

    assert RerankedSearch(FixedFirstStage(), scorer, CHUNKS)("q", 3, None) == []
    assert scorer.calls == []


@pytest.mark.parametrize(
    ("first", "scorer", "message"),
    [
        (FixedFirstStage("Z::0000"), KeywordScorer(), "no text for first-stage chunk Z::0000"),
        (FixedFirstStage("A::0000"), KeywordScorer(extra_scores=1), "returned 2 scores for 1"),
    ],
    ids=["unknown-chunk", "score-count-mismatch"],
)
def test_rejects_inconsistent_stages(
    first: FixedFirstStage, scorer: KeywordScorer, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RerankedSearch(first, scorer, CHUNKS)("q", 1, None)


def test_rejects_invalid_depth_and_k() -> None:
    with pytest.raises(ValueError, match="depth must be positive"):
        RerankedSearch(FixedFirstStage(), KeywordScorer(), CHUNKS, depth=0)
    with pytest.raises(ValueError, match="k must be positive"):
        RerankedSearch(FixedFirstStage(), KeywordScorer(), CHUNKS)("q", 0, None)
