"""Pipeline with a fake retriever and a fake chat model behind the real Generator."""

import json
from typing import Any

import pytest

from cuad_doc_qa.abstention import Decision, Reason, ScoreGate
from cuad_doc_qa.domain import Chunk, SearchHit
from cuad_doc_qa.generation import Generator
from cuad_doc_qa.pipeline import Pipeline

LAW = "This Agreement shall be governed by the laws of the State of Delaware."
TERM = "The term of this Agreement is five years."
CHUNKS = [
    Chunk(chunk_id="D::0000", doc_id="D", start=100, end=100 + len(TERM), text=TERM),
    Chunk(chunk_id="D::0001", doc_id="D", start=500, end=500 + len(LAW), text=LAW),
]
GATE = ScoreGate({"Governing Law": 5.0})


class FakeSearch:
    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[tuple[str, int, str | None]] = []

    def __call__(self, query: str, k: int, doc_id: str | None) -> list[SearchHit]:
        self.calls.append((query, k, doc_id))
        return self.hits[:k]


class FakeChat:
    def __init__(self, found: bool, citations: list[dict[str, str]]) -> None:
        self.reply = json.dumps({"found": found, "answer": "Delaware law.", "citations": citations})
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        self.calls.append(messages)
        return self.reply


def _pipeline(search: FakeSearch, chat: FakeChat) -> Pipeline:
    return Pipeline(search, CHUNKS, Generator(chat, num_ctx=8192), GATE, k=2)


HITS = [SearchHit("D::0001", 9.0), SearchHit("D::0000", 2.0)]


def test_answer_with_grounded_citation_is_shown() -> None:
    search = FakeSearch(HITS)
    chat = FakeChat(
        True, [{"source": "S1", "quote": "governed by the laws of the State of Delaware"}]
    )

    result = _pipeline(search, chat).ask("Which law governs?", "D", "Governing Law")

    assert search.calls == [("Which law governs?", 2, "D")]
    assert "[S1]\n" + LAW in chat.calls[0][1]["content"]
    assert result.decision == Decision(None)
    assert result.retrieved == tuple(HITS)
    [cited] = result.check.grounded
    assert (cited.start, cited.end) == (500 + LAW.index("governed"), 500 + len(LAW) - 1)
    assert set(result.seconds) == {"retrieval", "generation", "grounding", "total"}
    assert all(value >= 0 for value in result.seconds.values())


def test_low_top_score_is_withheld_before_the_model_is_called() -> None:
    chat = FakeChat(True, [])

    result = _pipeline(FakeSearch([SearchHit("D::0001", 4.9)]), chat).ask("q", "D", "Governing Law")

    assert result.decision == Decision(Reason.LOW_RETRIEVAL_SCORE)
    assert (result.answer, result.check) == (None, None)
    assert chat.calls == []
    assert set(result.seconds) == {"retrieval", "total"}


@pytest.mark.parametrize(
    ("category", "hits", "reason"),
    [(None, HITS[:1], None), (None, [], Reason.LOW_RETRIEVAL_SCORE)],
    ids=["free-form-not-gated", "no-hits-withheld"],
)
def test_gate_for_free_form_questions(
    category: str | None, hits: list[SearchHit], reason: Reason | None
) -> None:
    chat = FakeChat(True, [{"source": "S1", "quote": "laws of the State of Delaware"}])

    result = _pipeline(FakeSearch(hits), chat).ask("Which law governs?", "D", category)

    assert result.decision == Decision(reason)


@pytest.mark.parametrize(
    ("found", "citations", "reason"),
    [
        (False, [], Reason.NOT_FOUND),
        (True, [], Reason.NO_GROUNDED_CITATION),
        (True, [{"source": "S2", "quote": "governed by English law"}], Reason.REJECTED_CITATION),
    ],
    ids=["not-found", "uncited", "quote-not-in-chunk"],
)
def test_post_generation_abstentions(
    found: bool, citations: list[dict[str, str]], reason: Reason
) -> None:
    result = _pipeline(FakeSearch(HITS), FakeChat(found, citations)).ask("q", "D", "Governing Law")

    assert result.decision == Decision(reason)
    assert result.answer is not None and result.check is not None


def test_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        Pipeline(FakeSearch([]), CHUNKS, Generator(FakeChat(False, []), num_ctx=8192), GATE, k=0)
