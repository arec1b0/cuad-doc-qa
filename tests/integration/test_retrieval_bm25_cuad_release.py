"""Bm25Index over every chunk of the real CUAD v1 release.

Needs the CUAD archive: downloaded on first run, cached in data/raw (gitignored).
"""

from pathlib import Path

import pytest

from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.ingestion import download, load_documents
from cuad_doc_qa.retrieval.lexical import Bm25Index

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.network
def test_indexes_the_full_corpus_and_scopes_search_to_one_contract() -> None:
    documents = load_documents(download(DATA_RAW))
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    contract = documents[0].doc_id

    with Bm25Index.build(chunks) as index:
        hits = index.search("governing law of the state", k=10, doc_id=contract)

        assert len(index) == len(chunks)
        assert hits
        assert all(hit.chunk_id.startswith(f"{contract}::") for hit in hits)
