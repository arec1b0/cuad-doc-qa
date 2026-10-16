"""The CUAD session on real data: real BM25, lead chunks, committed gate and grounding; fake model.

The fake model quotes the first 80 characters of the first source, so the test checks the
wiring and the offsets, not answer quality. Needs the CUAD archive (cached in data/raw).
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from cuad_doc_qa.abstention import Reason, load_gate
from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.ingestion import download, load_documents
from cuad_doc_qa.pipeline import DEFAULT_GATE, cuad_session
from cuad_doc_qa.retrieval.lead import DEFAULT_LEAD
from cuad_doc_qa.retrieval.lexical import Bm25Index

REPO = Path(__file__).resolve().parents[2]
DATA_RAW = REPO / "data" / "raw"
GATE = REPO / DEFAULT_GATE


class QuotingChat:
    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        first_source = re.search(r"\[S1\]\n(.*)", messages[1]["content"], re.DOTALL)
        assert first_source is not None
        quote = first_source.group(1)[:80]
        return json.dumps(
            {"found": True, "answer": "Quoted.", "citations": [{"source": "S1", "quote": quote}]}
        )


@pytest.mark.network
def test_free_form_answer_cites_real_document_offsets() -> None:
    texts = {document.doc_id: document.text for document in load_documents(download(DATA_RAW))}

    with cuad_session(QuotingChat(), 8192, DATA_RAW, GATE) as session:
        doc_id = session.contracts[0]
        result = session.pipeline.ask("governing law of the state", doc_id)

    assert result.decision.reason is None
    assert result.check is not None
    [cited] = result.check.grounded
    assert cited.doc_id == doc_id
    assert texts[doc_id][cited.start : cited.end] == cited.text
    assert len(result.retrieved) == 10


@pytest.mark.network
def test_category_questions_are_gated_with_the_committed_thresholds() -> None:
    gate = load_gate(GATE)

    with cuad_session(QuotingChat(), 8192, DATA_RAW, GATE) as session:
        question = session.questions["Governing Law"]
        for doc_id in session.contracts[:5]:
            result = session.pipeline.ask(question, doc_id, "Governing Law")
            top = result.retrieved[0].score if result.retrieved else None
            withheld = result.decision.reason == Reason.LOW_RETRIEVAL_SCORE
            assert withheld is not gate.passes("Governing Law", top)


@pytest.mark.network
def test_only_metadata_questions_see_the_contract_opening_first() -> None:
    documents = load_documents(download(DATA_RAW))
    chunks = [chunk for document in documents for chunk in chunk_document(document)]

    with cuad_session(QuotingChat(), 8192, DATA_RAW, GATE) as session:
        doc_id = session.contracts[0]
        parties = session.pipeline.ask(session.questions["Parties"], doc_id, "Parties")
        law_question = session.questions["Governing Law"]
        law = session.pipeline.ask(law_question, doc_id, "Governing Law")
    with Bm25Index.build(chunks) as bm25:
        plain_parties = bm25.search(session.questions["Parties"], 10, doc_id)
        plain_law = bm25.search(law_question, 10, doc_id)

    assert [hit.chunk_id for hit in parties.retrieved[:DEFAULT_LEAD]] == [
        f"{doc_id}::{index:04d}" for index in range(DEFAULT_LEAD)
    ]
    assert parties.retrieved[0].score == plain_parties[0].score
    assert law.retrieved == tuple(plain_law)
