"""cuad-qa evaluate-answers: arguments, default report path and summary. The run is faked."""

from pathlib import Path
from typing import Any

import pytest

from cuad_doc_qa.cli import app, main


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    recorded: list[dict[str, Any]] = []

    def fake(chat: Any, split: Path, out: Path, *args: Any) -> dict[str, Any]:
        sample, seed, k, gate, data_dir = args
        recorded.append(
            {
                "model": chat.model,
                "timeout": chat.timeout,
                "split": split,
                "out": out,
                "sample": sample,
                "seed": seed,
                "k": k,
                "gate": gate,
                "data_dir": data_dir,
            }
        )
        return {
            "questions": {"total": 410, "answerable": 120, "unanswerable": 290},
            "metrics": {"answer_recall": 0.5, "false_answer_rate": 0.1},
        }

    monkeypatch.setattr(app, "run_answer_eval", fake)
    monkeypatch.delenv(app.MODEL_ENV, raising=False)
    monkeypatch.delenv(app.TIMEOUT_ENV, raising=False)
    return recorded


def test_defaults_and_a_windows_safe_report_name(
    calls: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["evaluate-answers", "--model", "qwen2.5:7b"]) == 0

    assert calls == [
        {
            "model": "qwen2.5:7b",
            "timeout": 300.0,
            "split": Path("data/eval/splits/dev.txt"),
            "out": Path("reports/answers/dev-sample10-seed0-qwen2.5-7b.json"),
            "sample": 10,
            "seed": 0,
            "k": 10,
            "gate": Path("configs/bm25-gate.json"),
            "data_dir": Path("data/raw"),
        }
    ]
    assert capsys.readouterr().out == (
        "report: reports/answers/dev-sample10-seed0-qwen2.5-7b.json\n"
        "questions=410 answerable=120 answer_recall=0.5 false_answer_rate=0.1\n"
    )


def test_model_from_environment_and_custom_sample(
    calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(app.MODEL_ENV, "llama3.1:8b-instruct-q4_K_M")

    main(["evaluate-answers", "--sample", "2", "--seed", "3", "--k", "5"])

    assert calls[0]["out"] == Path(
        "reports/answers/dev-sample2-seed3-llama3.1-8b-instruct-q4_K_M.json"
    )
    assert (calls[0]["sample"], calls[0]["seed"], calls[0]["k"]) == (2, 3, 5)


def test_missing_model_exits_2(calls: list[dict[str, Any]]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate-answers"])

    assert exit_info.value.code == 2
    assert calls == []


def test_timeout_from_environment_and_flag(
    calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(app.TIMEOUT_ENV, "1800")

    main(["evaluate-answers", "--model", "m"])
    main(["evaluate-answers", "--model", "m", "--timeout", "90.5"])

    assert [call["timeout"] for call in calls] == [1800.0, 90.5]


@pytest.mark.parametrize("value", ["0", "-5", "abc", "inf", "nan"])
def test_invalid_timeout_exits_2(
    calls: list[dict[str, Any]], capsys: pytest.CaptureFixture[str], value: str
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate-answers", "--model", "m", "--timeout", value])

    assert exit_info.value.code == 2
    assert "expected a positive number of seconds" in capsys.readouterr().err
    assert calls == []


def test_invalid_timeout_in_environment_exits_2(
    calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(app.TIMEOUT_ENV, "five minutes")

    with pytest.raises(SystemExit) as exit_info:
        main(["evaluate-answers", "--model", "m"])

    assert exit_info.value.code == 2
    assert calls == []
