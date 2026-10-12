"""FusedSearch over the real BM25 and bge-small retrievers on three real CUAD contracts.

Needs internet access on first run: the CUAD archive is cached in data/raw and the ONNX model
(67 MB) in artifacts/models, both gitignored.
"""

from pathlib import Path

import pytest

from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.ingestion import download, load_documents, load_questions
from cuad_doc_qa.retrieval.dense import DenseIndex, FastEmbedEmbedder
from cuad_doc_qa.retrieval.fusion import FusedSearch
from cuad_doc_qa.retrieval.lexical import Bm25Index

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.network
def test_hybrid_hits_come_from_the_contract_and_from_either_retriever() -> None:
    archive = download(REPO / "data" / "raw")
    documents = load_documents(archive)[:3]
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    question = load_questions(archive)["Governing Law"]
    contract = documents[0].doc_id
    dense = DenseIndex.build(chunks, FastEmbedEmbedder(cache_dir=REPO / "artifacts" / "models"))

    with Bm25Index.build(chunks) as bm25:
        hybrid = FusedSearch([bm25.search, dense.search], depth=20)
        hits = hybrid(question, 10, contract)
        candidates = {
            hit.chunk_id
            for search in (bm25.search, dense.search)
            for hit in search(question, 20, contract)
        }

    assert len(hits) == 10
    assert all(hit.chunk_id.startswith(f"{contract}::") for hit in hits)
    assert {hit.chunk_id for hit in hits} <= candidates
