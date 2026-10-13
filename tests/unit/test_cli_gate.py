"""cuad-qa gate: output, exit codes and tolerance parsing on copies of the committed dev report."""

import json
from pathlib import Path

import pytest

from cuad_doc_qa.cli import main

DEV = Path(__file__).resolve().parents[2] / "reports" / "bm25-baseline-dev.json"


@pytest.fixture
def reports(tmp_path: Path) -> tuple[Path, Path]:
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(DEV.read_bytes())
    report = json.loads(DEV.read_text(encoding="utf-8"))
    report["metrics"]["recall@10"] = round(report["metrics"]["recall@10"] - 0.03, 4)
    report["metrics"]["recall@5"] = round(report["metrics"]["recall@5"] + 0.01, 4)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(report), encoding="utf-8")
    return baseline, candidate


def test_unchanged_report_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["gate", "--baseline", str(DEV), "--candidate", str(DEV)]) == 0
    assert capsys.readouterr().out == (
        "recall@10: 0.6734 -> 0.6734 (drop +0.0000, tolerance 0.0200) ok\n"
    )


def test_regression_exits_3(reports: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    baseline, candidate = reports

    assert main(["gate", "--baseline", str(baseline), "--candidate", str(candidate)]) == 3
    assert capsys.readouterr().out == (
        "recall@10: 0.6734 -> 0.6434 (drop +0.0300, tolerance 0.0200) REGRESSION\n"
    )


def test_tolerances_replace_the_default(
    reports: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    baseline, candidate = reports
    argv = ["gate", "--baseline", str(baseline), "--candidate", str(candidate)]

    code = main([*argv, "--tolerance", "recall@5=0", "--tolerance", "recall@10=0.05"])

    assert code == 0
    assert capsys.readouterr().out.splitlines() == [
        "recall@5: 0.5620 -> 0.5720 (drop -0.0100, tolerance 0.0000) ok",
        "recall@10: 0.6734 -> 0.6434 (drop +0.0300, tolerance 0.0500) ok",
    ]


def test_incomparable_reports_exit_1(capsys: pytest.CaptureFixture[str]) -> None:
    test = DEV.with_name("bm25-baseline-test.json")

    assert main(["gate", "--baseline", str(DEV), "--candidate", str(test)]) == 1
    assert "not comparable: split" in capsys.readouterr().err


@pytest.mark.parametrize("tolerance", ["recall@10", "=0.1", "recall@10=abc"])
def test_malformed_tolerance_exits_2(tolerance: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["gate", "--baseline", str(DEV), "--candidate", str(DEV), "--tolerance", tolerance])

    assert exit_info.value.code == 2
