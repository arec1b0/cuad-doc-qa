"""Prompt construction and the answer schema."""

from cuad_doc_qa.domain import Chunk
from cuad_doc_qa.generation.prompt import (
    SYSTEM_PROMPT,
    answer_schema,
    build_messages,
    source_labels,
)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="D", start=0, end=len(text), text=text)


def test_labels_are_one_based() -> None:
    assert source_labels(3) == ["S1", "S2", "S3"]
    assert source_labels(0) == []


def test_messages_list_sources_in_order_then_the_question() -> None:
    chunks = [
        _chunk("D::0007", "Governed by Delaware law."),
        _chunk("D::0002", "Term: five years."),
    ]

    system, user = build_messages("Which law governs?", chunks)

    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user == {
        "role": "user",
        "content": (
            "Sources:\n\n[S1]\nGoverned by Delaware law.\n\n[S2]\nTerm: five years.\n\n"
            "Question: Which law governs?"
        ),
    }


def test_schema_restricts_sources_to_the_labels_shown() -> None:
    schema = answer_schema(["S1", "S2"])

    citation = schema["properties"]["citations"]["items"]
    assert citation["properties"]["source"]["enum"] == ["S1", "S2"]
    assert citation["required"] == ["source", "quote"]
    assert schema["required"] == ["found", "answer", "citations"]
