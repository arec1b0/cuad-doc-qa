"""DenseIndex with the real bge-small-en-v1.5 model on three real CUAD contracts.

Needs internet access on first run: the CUAD archive is cached in data/raw and the ONNX model
(67 MB) in artifacts/models, both gitignored.
"""

from pathlib import Path

import pytest

from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.ingestion import download, load_documents, load_questions
from cuad_doc_qa.retrieval.dense import DenseIndex, FastEmbedEmbedder

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.network
def test_bge_small_ranks_chunks_inside_one_contract() -> None:
    archive = download(REPO / "data" / "raw")
    documents = load_documents(archive)[:3]
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    question = load_questions(archive)["Governing Law"]
    contract = documents[0].doc_id

    index = DenseIndex.build(chunks, FastEmbedEmbedder(cache_dir=REPO / "artifacts" / "models"))
    hits = index.search(question, k=5, doc_id=contract)

    assert len(index) == len(chunks)
    assert len(hits) == 5
    assert all(hit.chunk_id.startswith(f"{contract}::") for hit in hits)
    assert all(-1.0001 <= hit.score <= 1.0001 for hit in hits)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
