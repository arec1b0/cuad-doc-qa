"""Committed answer reports are recomputed exactly from their committed question logs.

Model output is not reproducible across machines, so the question log is the evidence and the
report must be a pure function of it: the last record per question, aggregated, under the log's
configuration. A committed run must also be complete, with no question left in error.
"""

import json
from pathlib import Path

import pytest

from cuad_doc_qa.evaluation.answers import aggregate, read_question_log

ANSWERS = Path(__file__).resolve().parents[2] / "reports" / "answers"
REPORTS = sorted(ANSWERS.glob("*.json"))


def test_the_qwen_run_is_committed() -> None:
    assert [path.name for path in REPORTS] == ["dev-sample10-seed0-qwen2.5-7b.json"]


@pytest.mark.parametrize("report_path", REPORTS, ids=lambda path: path.stem)
def test_report_is_recomputed_from_its_log(report_path: Path) -> None:
    config, records = read_question_log(report_path.with_suffix(".questions.jsonl"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report == {"provenance": config, **aggregate(records.values())}
    assert report["questions"]["total"] == len(records)
    assert report["metrics"]["error_rate"] == 0.0
