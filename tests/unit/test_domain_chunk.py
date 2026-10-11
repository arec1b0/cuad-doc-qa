"""Chunk: span invariants are enforced at construction."""

import pytest

from cuad_doc_qa.domain import Chunk


@pytest.mark.parametrize(
    ("start", "end", "text"),
    [(5, 5, ""), (-1, 3, "abcd"), (0, 4, "abc")],
    ids=["empty-span", "negative-start", "length-mismatch"],
)
def test_rejects_inconsistent_span(start: int, end: int, text: str) -> None:
    with pytest.raises(ValueError):
        Chunk(chunk_id="D::0000", doc_id="D", start=start, end=end, text=text)
