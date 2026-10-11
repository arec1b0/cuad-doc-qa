"""chunk_document: boundary preference, budget, and offset invariants."""

from collections.abc import Callable

import pytest

from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.domain import Document

PartitionCheck = Callable[..., None]


def _doc(text: str) -> Document:
    return Document(doc_id="C-EX-10-SUPPLY AGREEMENT", text=text)


def _texts(document: Document, max_chars: int) -> list[str]:
    return [chunk.text for chunk in chunk_document(document, max_chars)]


def test_short_paragraphs_are_packed_up_to_the_budget(assert_partition: PartitionCheck) -> None:
    doc = _doc("  Alpha beta.\n\n   Gamma delta.  \n \nEpsilon zeta.\n")

    assert _texts(doc, 100) == ["Alpha beta.\n\n   Gamma delta.  \n \nEpsilon zeta."]
    assert _texts(doc, 30) == ["Alpha beta.\n\n   Gamma delta.", "Epsilon zeta."]
    assert_partition(doc, chunk_document(doc, 30), 30)


@pytest.mark.parametrize(
    "heading",
    ["7. Termination. Either party may terminate.", "ARTICLE IV", "Section 3", "TERMINATION."],
)
def test_clause_heading_starts_a_new_chunk(heading: str) -> None:
    doc = _doc(f"Preamble text.\n\n{heading}\n\nBody text.")

    assert _texts(doc, 1000)[1].startswith(heading)


def test_numbered_subclause_does_not_force_a_break() -> None:
    doc = _doc("7. Term.\n\n7.1 Initial term of five years.")

    assert _texts(doc, 1000) == ["7. Term.\n\n7.1 Initial term of five years."]


def test_long_flat_paragraph_splits_at_sentence_boundaries(
    assert_partition: PartitionCheck,
) -> None:
    doc = _doc("First sentence here. Second sentence here. 3. Third clause starts. Fourth one.")

    chunks = chunk_document(doc, 45)

    assert [c.text for c in chunks] == [
        "First sentence here. Second sentence here.",
        "3. Third clause starts. Fourth one.",
    ]
    assert_partition(doc, chunks, 45)


def test_overlong_sentence_splits_at_whitespace_then_hard_cuts(
    assert_partition: PartitionCheck,
) -> None:
    doc = _doc("aaaa bbbb cccc " + "x" * 25)

    chunks = chunk_document(doc, 10)

    assert [c.text for c in chunks] == ["aaaa bbbb", "cccc", "xxxxxxxxxx", "xxxxxxxxxx", "xxxxx"]
    assert_partition(doc, chunks, 10)


def test_chunk_ids_are_ordinal_per_document() -> None:
    doc = _doc("One.\n\nARTICLE II\n\nTwo.")

    assert [c.chunk_id for c in chunk_document(doc)] == [
        "C-EX-10-SUPPLY AGREEMENT::0000",
        "C-EX-10-SUPPLY AGREEMENT::0001",
    ]


def test_whitespace_only_document_has_no_chunks() -> None:
    assert chunk_document(_doc(" \n\n\t ")) == []


def test_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_chars must be positive"):
        chunk_document(_doc("text"), 0)
