"""compare: comparability rules and the drop tolerance."""

import copy
from pathlib import Path
from typing import Any

import pytest

from cuad_doc_qa.evaluation.gate import compare, load_report

REPORTS = Path(__file__).resolve().parents[2] / "reports"


def _report(**metrics: float) -> dict[str, Any]:
    return {
        "provenance": {
            "split": "dev.txt",
            "split_sha256": "abc",
            "archive_sha256": "def",
            "k": 10,
            "retriever": "bm25",
        },
        "metrics": {"n": 5458, "recall@5": 0.56, "recall@10": 0.6734, **metrics},
    }


@pytest.mark.parametrize(
    ("candidate_recall", "regressed"),
    [(0.6734, False), (0.7000, False), (0.6534, False), (0.6533, True), (0.5000, True)],
    ids=["equal", "better", "drop-equals-tolerance", "drop-over-tolerance", "large-drop"],
)
def test_recall_at_10_drop_is_judged_at_report_precision(
    candidate_recall: float, regressed: bool
) -> None:
    [check] = compare(_report(), _report(**{"recall@10": candidate_recall}))

    assert check.metric == "recall@10"
    assert check.regressed is regressed
    assert check.drop == round(0.6734 - candidate_recall, 4)


def test_custom_tolerances_check_each_metric_in_order() -> None:
    checks = compare(
        _report(),
        _report(**{"recall@5": 0.50, "recall@10": 0.67}),
        {"recall@5": 0.05, "recall@10": 0.0},
    )

    assert [(c.metric, c.regressed) for c in checks] == [("recall@5", True), ("recall@10", True)]


def test_retriever_may_differ_between_reports() -> None:
    candidate = _report()
    candidate["provenance"]["retriever"] = "bm25 + rerank"

    assert not compare(_report(), candidate)[0].regressed


@pytest.mark.parametrize("field", ["split", "split_sha256", "archive_sha256", "k"])
def test_reports_on_different_questions_are_not_comparable(field: str) -> None:
    candidate = copy.deepcopy(_report())
    candidate["provenance"][field] = "other"

    with pytest.raises(ValueError, match=f"not comparable: {field}"):
        compare(_report(), candidate)


def test_different_answerable_count_is_not_comparable() -> None:
    candidate = _report()
    candidate["metrics"]["n"] = 5457

    with pytest.raises(ValueError, match="not comparable: n is 5458 vs 5457"):
        compare(_report(), candidate)


@pytest.mark.parametrize(
    ("tolerances", "message"),
    [({"mrr@20": 0.01}, "'mrr@20' is not in both reports"), ({"recall@10": -0.1}, "non-negative")],
)
def test_rejects_invalid_tolerances(tolerances: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compare(_report(), _report(), tolerances)


def test_committed_dev_report_passes_against_itself_and_is_not_comparable_to_test() -> None:
    dev = load_report(REPORTS / "bm25-baseline-dev.json")
    test = load_report(REPORTS / "bm25-baseline-test.json")

    assert not compare(dev, dev)[0].regressed
    with pytest.raises(ValueError, match="not comparable: split"):
        compare(dev, test)
