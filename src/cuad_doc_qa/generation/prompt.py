"""Prompt and output schema for answering a question from numbered contract excerpts."""

from collections.abc import Sequence
from typing import Any

from cuad_doc_qa.domain import Chunk

SYSTEM_PROMPT = """\
You answer questions about one commercial contract using only the numbered sources below.
Each source is an excerpt of that contract.

Rules:
- If the sources contain what the question asks for, set "found" to true, answer in one or \
two sentences, and add a citation for every source you rely on. A citation quote must be \
copied verbatim from its source: the exact characters, no paraphrase, no ellipses.
- If no source contains it, set "found" to false and leave "answer" and "citations" empty.
- Use no knowledge outside the sources."""


def source_labels(count: int) -> list[str]:
    """Labels shown to the model for `count` sources: S1, S2, ..."""
    return [f"S{number}" for number in range(1, count + 1)]


def build_messages(question: str, chunks: Sequence[Chunk]) -> list[dict[str, str]]:
    """Chat messages with one labelled source per chunk, in the given order."""
    sources = "\n\n".join(
        f"[{label}]\n{chunk.text}"
        for label, chunk in zip(source_labels(len(chunks)), chunks, strict=True)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"},
    ]


def answer_schema(labels: Sequence[str]) -> dict[str, Any]:
    """JSON schema of the model's reply; `source` is restricted to the labels actually shown."""
    return {
        "type": "object",
        "properties": {
            "found": {"type": "boolean"},
            "answer": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": list(labels)},
                        "quote": {"type": "string"},
                    },
                    "required": ["source", "quote"],
                },
            },
        },
        "required": ["found", "answer", "citations"],
    }
