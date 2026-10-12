"""Generator with a fake chat model: label mapping, short-circuits and reply validation."""

import json
from typing import Any

import pytest

from cuad_doc_qa.domain import Answer, Chunk, Citation
from cuad_doc_qa.generation import Generator

CHUNKS = [
    Chunk(chunk_id="D::0007", doc_id="D", start=0, end=25, text="Governed by Delaware law."),
    Chunk(chunk_id="D::0002", doc_id="D", start=30, end=47, text="Term: five years."),
]


class FakeChat:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        self.calls.append((messages, schema))
        return self.reply


def _reply(found: object, answer: object, citations: list[dict[str, str]]) -> str:
    return json.dumps({"found": found, "answer": answer, "citations": citations})


def test_maps_source_labels_back_to_chunk_ids() -> None:
    chat = FakeChat(
        _reply(
            True,
            "Delaware law governs.",
            [{"source": "S1", "quote": "Delaware law"}, {"source": "S2", "quote": "five years"}],
        )
    )

    answer = Generator(chat, num_ctx=8192).answer("Which law governs?", CHUNKS)

    assert answer == Answer(
        found=True,
        text="Delaware law governs.",
        citations=(Citation("D::0007", "Delaware law"), Citation("D::0002", "five years")),
    )
    [(messages, schema)] = chat.calls
    assert messages[1]["content"].endswith("Question: Which law governs?")
    assert schema["properties"]["citations"]["items"]["properties"]["source"]["enum"] == [
        "S1",
        "S2",
    ]


def test_not_found_is_returned_as_the_model_said_it() -> None:
    answer = Generator(FakeChat(_reply(False, "", [])), num_ctx=8192).answer("q", CHUNKS)

    assert answer == Answer(found=False, text="", citations=())


def test_no_chunks_means_no_model_call() -> None:
    chat = FakeChat("unused")

    assert Generator(chat, num_ctx=8192).answer("q", []) == Answer(False, "", ())
    assert chat.calls == []


def test_prompt_that_may_not_fit_is_refused_before_the_call() -> None:
    chat = FakeChat("unused")
    long_chunk = Chunk(chunk_id="D::0000", doc_id="D", start=0, end=4000, text="x" * 4000)

    with pytest.raises(ValueError, match="pass fewer chunks or raise num_ctx"):
        Generator(chat, num_ctx=2048, reply_tokens=1024).answer("q", [long_chunk])
    assert chat.calls == []


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        json.dumps({"found": True, "answer": "x"}),
        _reply(True, "x", [{"source": "S9", "quote": "q"}]),
        _reply(True, "x", [{"source": "S1"}]),
        _reply("yes", "x", []),
        _reply(True, None, []),
    ],
    ids=[
        "invalid-json",
        "missing-key",
        "unknown-label",
        "citation-without-quote",
        "found-str",
        "answer-null",
    ],
)
def test_rejects_replies_that_break_the_schema(reply: str) -> None:
    with pytest.raises(ValueError, match="does not follow the answer schema"):
        Generator(FakeChat(reply), num_ctx=8192).answer("q", CHUNKS)
