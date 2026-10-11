"""Chunking invariants hold on every contract of the real CUAD v1 release.

Needs the CUAD archive: downloaded on first run, cached in data/raw (gitignored).
"""

from collections.abc import Callable
from pathlib import Path

import pytest

from cuad_doc_qa.chunking import DEFAULT_MAX_CHARS, chunk_document
from cuad_doc_qa.ingestion import download, load_documents

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.network
def test_every_contract_is_partitioned_within_budget(
    assert_partition: Callable[..., None],
) -> None:
    for document in load_documents(download(DATA_RAW)):
        assert_partition(document, chunk_document(document), DEFAULT_MAX_CHARS)
