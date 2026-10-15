"""answer_eval: question order, the JSONL log, resuming, and the consecutive-error abort."""

import json
from pathlib import Path
from typing import Any

import pytest

from cuad_doc_qa.abstention import ScoreGate
from cuad_doc_qa.domain import Chunk, SearchHit
from cuad_doc_qa.evaluation.answers import MAX_CONSECUTIVE_ERRORS, answer_eval, read_question_log
from cuad_doc_qa.generation import Generator
from cuad_doc_qa.pipeline import CuadSession, Pipeline

LAW = "This Agreement is governed by the laws of the State of Delaware."
CHUNKS = [
    Chunk(chunk_id=f"{doc}::0000", doc_id=doc, start=10, end=10 + len(LAW), text=LAW)
    for doc in ("A", "B")
]
QUESTIONS = {"Governing Law": "Which law governs?", "Term": "What is the term?"}
GOLD = {
    ("A", "Governing Law"): ((10, 40),),
    ("A", "Term"): (),
    ("B", "Governing Law"): ((10, 40),),
    ("B", "Term"): (),
}
CONFIG = {"run": "answers", "model": {"name": "fake"}}
GOOD_REPLY = json.dumps(
    {"found": True, "answer": "Delaware.", "citations": [{"source": "S1", "quote": "governed by"}]}
)


class ScriptedChat:
    """Plays its script in order, then repeats the last entry; exceptions are raised."""

    def __init__(self, *script: str | Exception) -> None:
        self.script = list(script)
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(step, Exception):
            raise step
        return step


def _session(chat: ScriptedChat) -> CuadSession:
    def search(query: str, k: int, doc_id: str | None) -> list[SearchHit]:
        return [SearchHit(f"{doc_id}::0000", 9.0)]

    pipeline = Pipeline(search, CHUNKS, Generator(chat, num_ctx=8192), ScoreGate({}), k=1)
    return CuadSession(pipeline, QUESTIONS, ("A", "B"))


def _log(report: Path) -> list[dict[str, Any]]:
    path = report.with_name(report.stem + ".questions.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_every_category_of_every_contract_is_asked_and_logged(tmp_path: Path) -> None:
    report_path = tmp_path / "answers" / "run.json"
    chat = ScriptedChat(GOOD_REPLY)

    report = answer_eval(_session(chat), GOLD, ["A", "B"], report_path, CONFIG)

    assert chat.calls == 4
    log = _log(report_path)
    assert log[0] == {"config": CONFIG}
    assert [r["question_id"] for r in log[1:]] == [
        "A__Governing Law",
        "A__Term",
        "B__Governing Law",
        "B__Term",
    ]
    assert report["questions"] == {"total": 4, "answerable": 2, "unanswerable": 2}
    assert report["metrics"]["answer_recall"] == 1.0
    assert report["metrics"]["false_answer_rate"] == 1.0
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["provenance"] == CONFIG


def test_resume_skips_finished_questions(tmp_path: Path) -> None:
    report_path = tmp_path / "run.json"
    answer_eval(_session(ScriptedChat(GOOD_REPLY)), GOLD, ["A"], report_path, CONFIG)
    chat = ScriptedChat(GOOD_REPLY)

    report = answer_eval(_session(chat), GOLD, ["A", "B"], report_path, CONFIG)

    assert chat.calls == 2
    assert report["questions"]["total"] == 4


def test_resume_under_another_configuration_is_refused(tmp_path: Path) -> None:
    report_path = tmp_path / "run.json"
    answer_eval(_session(ScriptedChat(GOOD_REPLY)), GOLD, ["A"], report_path, CONFIG)

    with pytest.raises(ValueError, match="different configuration"):
        answer_eval(_session(ScriptedChat(GOOD_REPLY)), GOLD, ["A"], report_path, {"run": "x"})


def test_a_single_bad_reply_is_recorded_and_retried_on_the_next_run(tmp_path: Path) -> None:
    report_path = tmp_path / "run.json"

    first = answer_eval(
        _session(ScriptedChat("not json", GOOD_REPLY)), GOLD, ["A"], report_path, CONFIG
    )
    chat = ScriptedChat(GOOD_REPLY)
    second = answer_eval(_session(chat), GOLD, ["A"], report_path, CONFIG)

    assert first["metrics"]["error_rate"] == 0.5
    assert "ValueError: reply does not follow" in _log(report_path)[1]["error"]
    assert chat.calls == 1
    assert second["metrics"]["error_rate"] == 0.0


def test_consecutive_errors_abort_and_keep_the_log(tmp_path: Path) -> None:
    report_path = tmp_path / "run.json"
    down = ScriptedChat(RuntimeError("Ollama is not reachable"))

    with pytest.raises(RuntimeError, match=f"aborted after {MAX_CONSECUTIVE_ERRORS} consecutive"):
        answer_eval(_session(down), GOLD, ["A", "B"], report_path, CONFIG)

    assert not report_path.exists()
    assert [r["decision"] for r in _log(report_path)[1:]] == ["error"] * MAX_CONSECUTIVE_ERRORS
    recovered = answer_eval(
        _session(ScriptedChat(GOOD_REPLY)), GOLD, ["A", "B"], report_path, CONFIG
    )
    assert recovered["metrics"]["error_rate"] == 0.0
    assert recovered["questions"]["total"] == 4


def test_question_log_keeps_the_last_record_of_a_retried_question(tmp_path: Path) -> None:
    log = tmp_path / "run.questions.jsonl"
    lines = [
        {"config": CONFIG},
        {"question_id": "A__x", "decision": "error"},
        {"question_id": "B__x", "decision": "answer"},
        {"question_id": "A__x", "decision": "not_found"},
    ]
    log.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")

    config, records = read_question_log(log)

    assert config == CONFIG
    assert {qid: record["decision"] for qid, record in records.items()} == {
        "A__x": "not_found",
        "B__x": "answer",
    }


def test_a_file_without_a_config_line_is_not_a_question_log(tmp_path: Path) -> None:
    log = tmp_path / "run.questions.jsonl"
    log.write_text('{"question_id": "A__x"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="not a question log"):
        read_question_log(log)
