"""DenseIndex with a deterministic word-count embedder standing in for the ONNX model."""

from collections.abc import Sequence

import numpy as np
import pytest

from cuad_doc_qa.domain import Chunk
from cuad_doc_qa.retrieval.dense import DenseIndex

VOCABULARY = ["governed", "laws", "delaware", "york", "terminate", "notice", "payment", "invoice"]


class WordCountEmbedder:
    """Vector = counts of vocabulary words; `document_scale` stretches document vectors."""

    def __init__(self, document_scale: float = 1.0, rows_override: int | None = None) -> None:
        self.document_scale = document_scale
        self.rows_override = rows_override

    def _vector(self, text: str) -> np.ndarray:
        words = text.lower().replace(".", " ").split()
        return np.array([words.count(term) for term in VOCABULARY], dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self._vector(text) * self.document_scale for text in texts]
        if self.rows_override is not None:
            rows = rows[: self.rows_override]
        return np.array(rows, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=chunk_id.split("::")[0], start=0, end=len(text), text=text
    )


CHUNKS = [
    _chunk("A::0000", "Governed by the laws of Delaware."),
    _chunk("A::0001", "Either party may terminate on notice."),
    _chunk("A::0002", "Payment is due on invoice."),
    _chunk("B::0000", "Governed by the laws of New York."),
]


@pytest.fixture
def index() -> DenseIndex:
    return DenseIndex.build(CHUNKS, WordCountEmbedder())


def test_ranks_by_cosine_similarity_best_first(index: DenseIndex) -> None:
    hits = index.search("laws of delaware", k=4)

    assert [hit.chunk_id for hit in hits][:2] == ["A::0000", "B::0000"]
    assert hits[0].score == pytest.approx(2 / np.sqrt(3) / np.sqrt(2))
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_vector_length_does_not_change_the_ranking() -> None:
    scaled = DenseIndex.build(CHUNKS, WordCountEmbedder(document_scale=7.5))

    hits = scaled.search("laws of delaware", k=2)

    assert [hit.chunk_id for hit in hits] == ["A::0000", "B::0000"]
    assert hits[0].score == pytest.approx(2 / np.sqrt(3) / np.sqrt(2))


def test_doc_id_restricts_results_to_one_contract(index: DenseIndex) -> None:
    assert [hit.chunk_id for hit in index.search("laws", k=10, doc_id="B")] == ["B::0000"]
    assert index.search("laws", k=10, doc_id="missing") == []


def test_ties_are_broken_by_chunk_id() -> None:
    same = "Governed by the laws of Delaware."
    tied = DenseIndex.build([_chunk("C::0001", same), _chunk("C::0000", same)], WordCountEmbedder())

    assert [hit.chunk_id for hit in tied.search("delaware", k=2)] == ["C::0000", "C::0001"]


def test_k_caps_hits_and_must_be_positive(index: DenseIndex) -> None:
    assert len(index.search("laws", k=2)) == 2
    with pytest.raises(ValueError, match="k must be positive"):
        index.search("laws", k=0)


def test_blank_query_returns_nothing(index: DenseIndex) -> None:
    assert index.search("  \n", k=5) == []


def test_rejects_duplicate_chunk_ids() -> None:
    with pytest.raises(ValueError, match="duplicate chunk id"):
        DenseIndex.build([CHUNKS[0], CHUNKS[0]], WordCountEmbedder())


def test_rejects_an_embedding_matrix_of_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="embedder returned shape"):
        DenseIndex.build(CHUNKS, WordCountEmbedder(rows_override=2))


def test_len_counts_indexed_chunks(index: DenseIndex) -> None:
    assert len(index) == len(CHUNKS)
