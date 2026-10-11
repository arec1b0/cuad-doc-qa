"""load_documents: parsing of the CUAD v1 JSON layout."""

from collections.abc import Callable
from pathlib import Path

import pytest

from cuad_doc_qa.domain import Document
from cuad_doc_qa.ingestion import load_documents

ArchiveFactory = Callable[..., Path]


def _entry(title: str, *contexts: str) -> dict:
    return {"title": title, "paragraphs": [{"context": c, "qas": []} for c in contexts]}


def test_preserves_ids_text_and_order_exactly(make_cuad_archive: ArchiveFactory) -> None:
    raw_text = "MASTER AGREEMENT\r\n\r\n1.  Term.\tFive years.  \n"
    accented_id = "Cafe\N{COMBINING ACUTE ACCENT} Supply Agreement"
    archive = make_cuad_archive(
        [_entry("B-EX-10-LICENSE AGREEMENT", raw_text), _entry(accented_id, "Exhibit A")]
    )

    assert load_documents(archive) == [
        Document(doc_id="B-EX-10-LICENSE AGREEMENT", text=raw_text),
        Document(doc_id=accented_id, text="Exhibit A"),
    ]


def test_rejects_entry_with_multiple_paragraphs(make_cuad_archive: ArchiveFactory) -> None:
    archive = make_cuad_archive([_entry("X", "part one", "part two")])

    with pytest.raises(ValueError, match="expected 1 paragraph, found 2"):
        load_documents(archive)


def test_rejects_duplicate_document_ids(make_cuad_archive: ArchiveFactory) -> None:
    archive = make_cuad_archive([_entry("X", "one"), _entry("X", "two")])

    with pytest.raises(ValueError, match="duplicate document id"):
        load_documents(archive)
