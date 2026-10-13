"""docs/FAILURES.md cites only numbers that its sources still contain.

Regenerating a report or changing a test assertion without updating the write-up fails here.
"""

import json
from functools import reduce
from operator import getitem
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOC = (REPO / "docs" / "FAILURES.md").read_text(encoding="utf-8")
TEST_REPORT = "reports/bm25-baseline-test.json"
DEV_REPORT = "reports/bm25-baseline-dev.json"
GATE = "configs/bm25-gate.json"
CITATIONS_TEST = "tests/integration/test_citations_cuad_release.py"


def _experiment(retriever: str) -> str:
    return f"reports/experiments/dev-sample40-seed0-{retriever}.json"


REPORT_NUMBERS = [
    (TEST_REPORT, ("metrics", "recall@5"), "0.5514"),
    (TEST_REPORT, ("metrics", "recall@10"), "0.6706"),
    (TEST_REPORT, ("metrics", "mrr@10"), "0.4780"),
    (TEST_REPORT, ("metrics", "ndcg@10"), "0.5048"),
    (TEST_REPORT, ("queries", "answerable"), "1,244"),
    (TEST_REPORT, ("queries", "unanswerable_skipped"), "2,938"),
    (DEV_REPORT, ("per_category", "Governing Law", "recall@10"), "0.9831"),
    (DEV_REPORT, ("per_category", "Governing Law", "n"), "354"),
    (DEV_REPORT, ("per_category", "Parties", "recall@10"), "0.2271"),
    (DEV_REPORT, ("per_category", "Parties", "n"), "407"),
    (DEV_REPORT, ("per_category", "Document Name", "recall@10"), "0.2721"),
    (DEV_REPORT, ("per_category", "Document Name", "n"), "408"),
    (DEV_REPORT, ("per_category", "Agreement Date", "recall@10"), "0.3369"),
    (DEV_REPORT, ("per_category", "Agreement Date", "n"), "377"),
    (GATE, ("provenance", "target_answerable_recall"), "0.95"),
    *[
        (_experiment(retriever), ("metrics", metric), value)
        for retriever, values in {
            "bm25": ("0.5640", "0.6743", "0.5081", "0.5279"),
            "dense": ("0.4893", "0.5843", "0.4358", "0.4469"),
            "hybrid": ("0.5741", "0.6770", "0.5079", "0.5257"),
            "rerank": ("0.5583", "0.6443", "0.4888", "0.5030"),
        }.items()
        for metric, value in zip(
            ("recall@5", "recall@10", "mrr@10", "ndcg@10"), values, strict=True
        )
    ],
    (_experiment("bm25"), ("metrics", "n"), "573"),
    *[
        (_experiment(retriever), ("per_category", category, "recall@10"), value)
        for category, values in {
            "Parties": ("0.2000", "0.2125", "0.2000", "0.3500"),
            "Document Name": ("0.2250", "0.3250", "0.2250", "0.2500"),
            "Agreement Date": ("0.3158", "0.3684", "0.3553", "0.7368"),
        }.items()
        for retriever, value in zip(("bm25", "dense", "hybrid", "rerank"), values, strict=True)
    ],
]


def _format(value: object, like: str) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    decimals = len(like.split(".")[1])
    return f"{value:.{decimals}f}"


@pytest.mark.parametrize(("source", "keys", "cited"), REPORT_NUMBERS)
def test_cited_number_matches_its_report(source: str, keys: tuple[str, ...], cited: str) -> None:
    value = reduce(getitem, keys, json.loads((REPO / source).read_text(encoding="utf-8")))

    assert _format(value, cited) == cited
    assert cited in DOC


def test_gate_category_count_matches_the_committed_gate() -> None:
    thresholds = json.loads((REPO / GATE).read_text(encoding="utf-8"))["thresholds"]

    assert f"{len(thresholds)} of 41" in DOC


@pytest.mark.parametrize("number", ["13_505", "13_372", "9_948", "9_921"])
def test_cited_grounding_counts_are_asserted_by_the_integration_test(number: str) -> None:
    assert number in (REPO / CITATIONS_TEST).read_text(encoding="utf-8")
    assert f"{int(number):,}" in DOC
