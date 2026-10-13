"""cuad-qa ask: argument rules, contract resolution, rendering. The CUAD session is faked."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from cuad_doc_qa.abstention import ScoreGate
from cuad_doc_qa.cli import app, main
from cuad_doc_qa.domain import Chunk, SearchHit
from cuad_doc_qa.generation import Generator
from cuad_doc_qa.pipeline import CuadSession, Pipeline

DOC = "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
LAW = "This Agreement shall be governed by the laws of the State of Illinois."
CHUNK = Chunk(chunk_id=f"{DOC}::0040", doc_id=DOC, start=9000, end=9000 + len(LAW), text=LAW)
QUESTIONS = {"Governing Law": "Which state's law governs the contract?"}


class FakeChat:
    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = json.dumps(reply)

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        return self.reply


@pytest.fixture
def session_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    reply = {
        "found": True,
        "answer": "Illinois law governs.",
        "citations": [{"source": "S1", "quote": "governed by the laws of the State of Illinois"}],
    }

    def search(query: str, k: int, doc_id: str | None) -> list[SearchHit]:
        return [SearchHit(CHUNK.chunk_id, 12.5)] if doc_id == DOC else []

    @contextmanager
    def fake_session(chat: Any, num_ctx: int, *args: Any) -> Iterator[CuadSession]:
        calls.append({"model": chat.model, "num_ctx": num_ctx, "args": args})
        generator = Generator(FakeChat(reply), num_ctx=num_ctx)
        pipeline = Pipeline(search, [CHUNK], generator, ScoreGate({"Governing Law": 20.0}), k=2)
        yield CuadSession(pipeline, QUESTIONS, (DOC, "LIMESTONE-EX-10-SUPPLY", "OTHER"))

    monkeypatch.setattr(app, "cuad_session", fake_session)
    monkeypatch.delenv(app.MODEL_ENV, raising=False)
    return calls


def test_free_form_question_prints_answer_with_located_citation(
    session_calls: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["ask", "--contract", "LIMEENERGY", "--question", "Which law?", "--model", "qwen2.5:7b"]
    )

    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert session_calls == [
        {
            "model": "qwen2.5:7b",
            "num_ctx": 8192,
            "args": (Path("data/raw"), Path("configs/bm25-gate.json"), 10),
        }
    ]
    assert out[:4] == [
        f"contract: {DOC}",
        "question: Which law?",
        "decision: answer",
        "answer: Illinois law governs.",
    ]
    assert out[4] == "[1] chars 9024-9069 (verbatim): governed by the laws of the State of Illinois"
    assert out[5].startswith("time: retrieval ")


def test_category_uses_the_cuad_question_and_the_gate(
    session_calls: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(app.MODEL_ENV, "llama3.1:8b")

    code = main(["ask", "--contract", DOC, "--category", "Governing Law"])

    out = capsys.readouterr().out.splitlines()
    assert code == 0
    assert session_calls[0]["model"] == "llama3.1:8b"
    assert out[1] == "question: Which state's law governs the contract?"
    assert out[2] == "decision: abstain (low_retrieval_score)"


@pytest.mark.parametrize(
    ("contract", "category", "message"),
    [
        ("LIME", None, "'LIME' matches 2 contracts"),
        ("NOPE", None, "'NOPE' matches 0 contracts"),
        (DOC, "Governing law", "unknown category 'Governing law'"),
    ],
    ids=["ambiguous-prefix", "no-match", "unknown-category"],
)
def test_bad_contract_or_category_exits_1(
    session_calls: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
    contract: str,
    category: str | None,
    message: str,
) -> None:
    query = ["--category", category] if category else ["--question", "q"]

    assert main(["ask", "--contract", contract, *query, "--model", "m"]) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["ask", "--contract", DOC, "--question", "q"],
        ["ask", "--contract", DOC, "--model", "m"],
        [
            "ask",
            "--contract",
            DOC,
            "--question",
            "q",
            "--category",
            "Governing Law",
            "--model",
            "m",
        ],
    ],
    ids=["no-model", "no-query", "question-and-category"],
)
def test_invalid_usage_exits_2(session_calls: list[dict[str, Any]], argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert exit_info.value.code == 2
    assert session_calls == []
