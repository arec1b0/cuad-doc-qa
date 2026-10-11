"""Shared test assertions."""

from collections.abc import Callable

import pytest

from cuad_doc_qa.domain import Chunk, Document

PartitionCheck = Callable[[Document, list[Chunk], int], None]


@pytest.fixture
def assert_partition() -> PartitionCheck:
    """Chunks are exact, ordered, disjoint slices within budget that cover all non-whitespace."""

    def check(document: Document, chunks: list[Chunk], max_chars: int) -> None:
        previous_end = 0
        for chunk in chunks:
            assert chunk.doc_id == document.doc_id
            assert chunk.text == document.text[chunk.start : chunk.end]
            assert chunk.start >= previous_end
            assert chunk.end - chunk.start <= max_chars
            assert not chunk.text[0].isspace() and not chunk.text[-1].isspace()
            previous_end = chunk.end
        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
        assert "".join("".join(c.text.split()) for c in chunks) == "".join(document.text.split())

    return check
