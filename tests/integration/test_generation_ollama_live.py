"""Generator with a real local Ollama model on BM25 results from one real CUAD contract.

Needs a running Ollama server with the model named in CUAD_QA_OLLAMA_MODEL already pulled.
Checks the contract between the model and the parser, not answer quality.
"""

import os
from pathlib import Path

import pytest

from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.generation import Generator, OllamaChat
from cuad_doc_qa.ingestion import download, load_documents, load_questions
from cuad_doc_qa.retrieval.lexical import Bm25Index

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.ollama
def test_local_model_returns_a_parsable_answer_citing_only_given_chunks() -> None:
    model = os.environ.get("CUAD_QA_OLLAMA_MODEL")
    if not model:
        pytest.fail("set CUAD_QA_OLLAMA_MODEL to a pulled Ollama model, e.g. qwen2.5:7b")
    archive = download(REPO / "data" / "raw")
    document = load_documents(archive)[0]
    chunks = chunk_document(document)
    question = load_questions(archive)["Governing Law"]
    chat = OllamaChat(model)

    with Bm25Index.build(chunks) as bm25:
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        context = [by_id[hit.chunk_id] for hit in bm25.search(question, 5)]

    answer = Generator(chat, num_ctx=chat.num_ctx).answer(question, context)

    assert {citation.chunk_id for citation in answer.citations} <= {c.chunk_id for c in context}
    assert all(citation.quote for citation in answer.citations)
