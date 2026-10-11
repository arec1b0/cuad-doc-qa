"""Bm25Index: ranking, stemming, contract filter, query sanitizing and validation."""

from collections.abc import Iterator

import pytest

from cuad_doc_qa.domain import Chunk
from cuad_doc_qa.retrieval.lexical import Bm25Index


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=chunk_id.split("::")[0], start=0, end=len(text), text=text
    )


CHUNKS = [
    _chunk("A::0000", "This Agreement shall be governed by the laws of the State of Delaware."),
    _chunk("A::0001", "Either party may terminate this Agreement upon thirty days written notice."),
    _chunk("A::0002", "Payment is due within thirty days of the invoice date."),
    _chunk("B::0000", "This Agreement shall be governed by the laws of the State of New York."),
    _chunk("B::0001", "The term of this Agreement is five years from the Effective Date."),
]


@pytest.fixture
def index() -> Iterator[Bm25Index]:
    with Bm25Index.build(CHUNKS) as built:
        yield built


def test_ranks_by_bm25_best_first(index: Bm25Index) -> None:
    hits = index.search("governing law Delaware", k=3)

    assert [hit.chunk_id for hit in hits][:2] == ["A::0000", "B::0000"]
    assert all(hit.score > 0 for hit in hits)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_matches_across_inflections(index: Bm25Index) -> None:
    assert index.search("termination", k=1)[0].chunk_id == "A::0001"


def test_doc_id_restricts_results_to_one_contract(index: Bm25Index) -> None:
    hits = index.search("Agreement governed laws State", k=10, doc_id="B")

    assert {hit.chunk_id for hit in hits} == {"B::0000", "B::0001"}


@pytest.mark.parametrize(
    "query",
    ['governing-law: "Delaware" AND NOT*', "NEAR(laws Delaware)", "laws OR) (Delaware"],
)
def test_fts5_syntax_in_queries_is_treated_as_words(index: Bm25Index, query: str) -> None:
    assert index.search(query, k=1)[0].chunk_id == "A::0000"


def test_query_without_words_returns_nothing(index: Bm25Index) -> None:
    assert index.search(' "-*: () ', k=5) == []


def test_k_caps_the_number_of_hits(index: Bm25Index) -> None:
    assert len(index.search("Agreement", k=2)) == 2
    with pytest.raises(ValueError, match="k must be positive"):
        index.search("Agreement", k=0)


def test_ties_are_broken_by_chunk_id() -> None:
    same = "Confidential information shall not be disclosed."
    with Bm25Index.build([_chunk("C::0001", same), _chunk("C::0000", same)]) as tied:
        assert [hit.chunk_id for hit in tied.search("disclosed", k=2)] == ["C::0000", "C::0001"]


def test_rejects_duplicate_chunk_ids() -> None:
    with pytest.raises(ValueError, match="duplicate chunk id"):
        Bm25Index.build([CHUNKS[0], CHUNKS[0]])


def test_len_counts_indexed_chunks(index: Bm25Index) -> None:
    assert len(index) == len(CHUNKS)
