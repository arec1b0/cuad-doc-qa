"""cuad-qa: parsing, dispatch, output and exit codes. Library calls are replaced by fakes."""

from importlib.metadata import entry_points
from pathlib import Path

import pytest

from cuad_doc_qa import __version__
from cuad_doc_qa.cli import app, main
from cuad_doc_qa.evaluation.runner import RetrievalEvaluation, RetrievalScores
from cuad_doc_qa.ingestion import ChecksumMismatchError

SCORES = RetrievalScores(
    n=3, recall_at_5=0.5, recall_at_10=0.75, mrr_at_10=0.25, ndcg_at_10=0.123456
)


class FakeBaseline:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, Path]] = []

    def __call__(self, split_path: Path, report_path: Path, data_dir: Path) -> RetrievalEvaluation:
        self.calls.append((split_path, report_path, data_dir))
        return RetrievalEvaluation(overall=SCORES, per_category={}, unanswerable_skipped=1)


def test_console_script_points_at_main() -> None:
    [script] = [ep for ep in entry_points(group="console_scripts") if ep.name == "cuad-qa"]

    assert script.value == "cuad_doc_qa.cli:main"


def test_evaluate_runs_the_split_and_prints_the_scores(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = FakeBaseline()
    monkeypatch.setattr(app, "run_bm25_baseline", baseline)

    assert main(["evaluate", "--split", "test"]) == 0

    assert baseline.calls == [
        (
            Path("data/eval/splits/test.txt"),
            Path("reports/bm25-baseline-test.json"),
            Path("data/raw"),
        )
    ]
    assert capsys.readouterr().out == (
        "report: reports/bm25-baseline-test.json\n"
        "n=3 recall@5=0.5000 recall@10=0.7500 mrr@10=0.2500 ndcg@10=0.1235\n"
    )


def test_evaluate_honours_out_and_data_dir(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = FakeBaseline()
    monkeypatch.setattr(app, "run_bm25_baseline", baseline)

    main(["evaluate", "--split", "dev", "--out", "custom/r.json", "--data-dir", "elsewhere"])

    assert baseline.calls == [
        (Path("data/eval/splits/dev.txt"), Path("custom/r.json"), Path("elsewhere"))
    ]
    assert capsys.readouterr().out.startswith("report: custom/r.json\n")


def test_download_prints_the_archive_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(app, "download", lambda data_dir: data_dir / "cuad_v1_data.zip")

    assert main(["download"]) == 0
    assert capsys.readouterr().out == "archive: data/raw/cuad_v1_data.zip\n"


@pytest.mark.parametrize(
    "error",
    [
        ChecksumMismatchError("bad sha"),
        OSError("offline"),
        RuntimeError("no fts5"),
        ValueError("x"),
    ],
    ids=["checksum", "network", "fts5", "data"],
)
def test_handled_failures_exit_1_with_a_one_line_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    def fail(data_dir: Path) -> Path:
        raise error

    monkeypatch.setattr(app, "download", fail)

    assert main(["download"]) == 1
    assert capsys.readouterr().err == f"error: {error}\n"


@pytest.mark.parametrize(
    "argv",
    [[], ["evaluate"], ["evaluate", "--split", "train"], ["unknown"]],
    ids=["no-command", "missing-split", "unknown-split", "unknown-command"],
)
def test_invalid_usage_exits_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert exit_info.value.code == 2


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"cuad-qa {__version__}\n"
