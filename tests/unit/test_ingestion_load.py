"""load_documents: parsing of the CUAD v1 JSON layout."""

import json
import zipfile
from pathlib import Path

import pytest

from cuad_doc_qa.domain import Document
from cuad_doc_qa.ingestion import load_documents


def _write_archive(path: Path, entries: list[dict]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("CUADv1.json", json.dumps({"version": "aok_v1.0", "data": entries}))
    return path


def _entry(title: str, *contexts: str) -> dict:
    return {"title": title, "paragraphs": [{"context": c, "qas": []} for c in contexts]}


def test_preserves_ids_text_and_order_exactly(tmp_path: Path) -> None:
    raw_text = "MASTER AGREEMENT\r\n\r\n1.  Term.\tFive years.  \n"
    accented_id = "Café Supply Agreement"
    archive = _write_archive(
        tmp_path / "cuad.zip",
        [_entry("B-EX-10-LICENSE AGREEMENT", raw_text), _entry(accented_id, "Exhibit A")],
    )

    assert load_documents(archive) == [
        Document(doc_id="B-EX-10-LICENSE AGREEMENT", text=raw_text),
        Document(doc_id=accented_id, text="Exhibit A"),
    ]


def test_rejects_entry_with_multiple_paragraphs(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "cuad.zip", [_entry("X", "part one", "part two")])

    with pytest.raises(ValueError, match="expected 1 paragraph, found 2"):
        load_documents(archive)


def test_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "cuad.zip", [_entry("X", "one"), _entry("X", "two")])

    with pytest.raises(ValueError, match="duplicate document id"):
        load_documents(archive)
