"""LeadChunkSearch over a fixed first stage."""

import pytest

from cuad_doc_qa.domain import Chunk, SearchHit
from cuad_doc_qa.retrieval.lead import LeadChunkSearch

QUESTION = "Who are the parties?"
OTHER = "Which law governs?"


def _chunk(doc_id: str, index: int) -> Chunk:
    text = f"chunk {index} of {doc_id}"
    return Chunk(
        chunk_id=f"{doc_id}::{index:04d}",
        doc_id=doc_id,
        start=index * 100,
        end=index * 100 + len(text),
        text=text,
    )


CHUNKS = [_chunk("A", i) for i in (3, 0, 4, 1, 2)] + [_chunk("B", 0)]


class FixedFirstStage:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits

    def __call__(self, query: str, k: int, doc_id: str | None) -> list[SearchHit]:
        return self.hits[:k]


BM25 = [SearchHit("A::0004", 9.0), SearchHit("A::0000", 5.0), SearchHit("A::0003", 2.0)]


def _search(hits: list[SearchHit], lead: int = 2) -> LeadChunkSearch:
    return LeadChunkSearch(FixedFirstStage(hits), CHUNKS, [QUESTION], lead=lead)


def test_lead_chunks_come_first_in_document_order_with_the_top_score() -> None:
    assert _search(BM25)(QUESTION, 10, "A") == [
        SearchHit("A::0000", 9.0),
        SearchHit("A::0001", 9.0),
        SearchHit("A::0004", 9.0),
        SearchHit("A::0003", 2.0),
    ]


def test_result_is_cut_to_k() -> None:
    assert _search(BM25)(QUESTION, 3, "A") == [
        SearchHit("A::0000", 9.0),
        SearchHit("A::0001", 9.0),
        SearchHit("A::0004", 9.0),
    ]
    assert _search(BM25, lead=4)(QUESTION, 2, "A") == [
        SearchHit("A::0000", 9.0),
        SearchHit("A::0001", 9.0),
    ]


@pytest.mark.parametrize(
    ("query", "doc_id"), [(OTHER, "A"), (QUESTION, None)], ids=["other-query", "no-contract"]
)
def test_other_queries_and_corpus_wide_search_are_unchanged(query: str, doc_id: str | None) -> None:
    assert _search(BM25)(query, 10, doc_id) == BM25


def test_nothing_found_stays_nothing() -> None:
    assert _search([])(QUESTION, 10, "A") == []


def test_a_contract_shorter_than_the_lead_contributes_every_chunk() -> None:
    assert _search([SearchHit("B::0000", 1.5)], lead=3)(QUESTION, 10, "B") == [
        SearchHit("B::0000", 1.5)
    ]


def test_rejects_a_non_positive_lead() -> None:
    with pytest.raises(ValueError, match="lead must be positive"):
        LeadChunkSearch(FixedFirstStage(BM25), CHUNKS, [QUESTION], lead=0)


def test_rejects_a_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        _search(BM25)(QUESTION, 0, "A")
