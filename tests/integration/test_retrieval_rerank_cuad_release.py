"""RerankedSearch with the real ms-marco-MiniLM-L-6-v2 cross-encoder over BM25 on real CUAD data.

Needs internet access on first run: the CUAD archive is cached in data/raw and the ONNX model
(80 MB) in artifacts/models, both gitignored.
"""

from pathlib import Path

import pytest

from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.ingestion import download, load_documents, load_questions
from cuad_doc_qa.retrieval.lexical import Bm25Index
from cuad_doc_qa.retrieval.rerank import FastEmbedCrossEncoder, RerankedSearch

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.network
def test_cross_encoder_reorders_bm25_candidates_inside_one_contract() -> None:
    archive = download(REPO / "data" / "raw")
    documents = load_documents(archive)[:3]
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    question = load_questions(archive)["Governing Law"]
    contract = documents[0].doc_id
    scorer = FastEmbedCrossEncoder(cache_dir=REPO / "artifacts" / "models")

    with Bm25Index.build(chunks) as bm25:
        candidates = {hit.chunk_id for hit in bm25.search(question, 20, contract)}
        hits = RerankedSearch(bm25.search, scorer, chunks, depth=20)(question, 10, contract)

    assert len(hits) == 10
    assert {hit.chunk_id for hit in hits} <= candidates
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)
