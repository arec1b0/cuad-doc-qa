"""docs/FAILURES.md cites only numbers that its sources still contain.

Regenerating a report or changing a test assertion without updating the write-up fails here.
"""

import json
import math
from functools import reduce
from operator import getitem
from pathlib import Path

import pytest

from cuad_doc_qa.retrieval.lead import DEFAULT_LEAD, LEAD_CATEGORIES

REPO = Path(__file__).resolve().parents[2]
DOC = (REPO / "docs" / "FAILURES.md").read_text(encoding="utf-8")
TEST_REPORT = "reports/bm25-baseline-test.json"
DEV_REPORT = "reports/bm25-baseline-dev.json"
GATE = "configs/bm25-gate.json"
CITATIONS_TEST = "tests/integration/test_citations_cuad_release.py"
ANSWERS = "reports/answers/dev-sample10-seed0-qwen2.5-7b.json"
LEAD_TEST = "reports/bm25-lead-baseline-test.json"
LEAD_DEV = "reports/bm25-lead-baseline-dev.json"
ANSWERS_LOG = "reports/answers/dev-sample10-seed0-qwen2.5-7b.questions.jsonl"


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
    (DEV_REPORT, ("metrics", "recall@10"), "0.6734"),
    (LEAD_DEV, ("metrics", "recall@10"), "0.8321"),
    *[
        (LEAD_TEST, ("metrics", metric), value)
        for metric, value in zip(
            ("recall@5", "recall@10", "mrr@10", "ndcg@10"),
            ("0.7176", "0.8119", "0.6184", "0.6455"),
            strict=True,
        )
    ],
    *[
        (report, ("per_category", category, key), value)
        for category, n, before, after in (
            ("Agreement Date", "93", "0.3871", "0.9140"),
            ("Document Name", "102", "0.3333", "0.8431"),
            ("Effective Date", "70", "0.6071", "0.8214"),
            ("Parties", "102", "0.3464", "0.9330"),
        )
        for report, key, value in (
            (TEST_REPORT, "n", n),
            (TEST_REPORT, "recall@10", before),
            (LEAD_TEST, "recall@10", after),
        )
    ],
    (ANSWERS, ("metrics", "answer_recall"), "0.4444"),
    (ANSWERS, ("metrics", "answer_precision"), "0.6747"),
    (ANSWERS, ("metrics", "false_answer_rate"), "0.0563"),
    (ANSWERS, ("metrics", "citation_precision"), "0.5470"),
    (ANSWERS, ("metrics", "abstention_precision"), "0.8196"),
    (ANSWERS, ("questions", "total"), "410"),
    (ANSWERS, ("questions", "answerable"), "126"),
    (ANSWERS, ("questions", "unanswerable"), "284"),
    (ANSWERS, ("grounding", "claimed"), "126"),
    (ANSWERS, ("grounding", "grounded"), "119"),
    (ANSWERS, ("grounding", "fuzzy"), "10"),
    (ANSWERS, ("grounding", "rejected"), "7"),
    (ANSWERS, ("outcomes", "answerable", "correct"), "56"),
    (ANSWERS, ("outcomes", "answerable", "low_retrieval_score"), "7"),
    (ANSWERS, ("outcomes", "answerable", "rejected_citation"), "5"),
    (ANSWERS, ("outcomes", "unanswerable", "low_retrieval_score"), "103"),
    (ANSWERS, ("outcomes", "unanswerable", "not_found"), "163"),
    (ANSWERS, ("outcomes", "unanswerable", "rejected_citation"), "2"),
    (ANSWERS, ("outcomes", "unanswerable", "answer"), "16"),
    (ANSWERS, ("latency_seconds", "p50"), "83.17"),
    (ANSWERS, ("latency_seconds", "p95"), "135.16"),
    (ANSWERS, ("per_category", "Parties", "correct"), "0"),
    (ANSWERS, ("per_category", "Parties", "shown"), "5"),
    (ANSWERS, ("per_category", "Document Name", "correct"), "0"),
    (ANSWERS, ("per_category", "Agreement Date", "correct"), "1"),
    *[
        (ANSWERS, ("per_category", category, "answerable"), "10")
        for category in ("Parties", "Document Name", "Agreement Date")
    ],
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


def _answers() -> dict:
    return json.loads((REPO / ANSWERS).read_text(encoding="utf-8"))


def test_answer_recall_interval_is_the_wilson_interval_of_the_run() -> None:
    report = _answers()
    n = report["questions"]["answerable"]
    p = report["outcomes"]["answerable"]["correct"] / n
    z = 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)

    assert f"between {center - half:.2f} and {center + half:.2f} (95% Wilson" in DOC


def test_gate_counts_in_the_write_up_add_up() -> None:
    report = _answers()
    gated = sum(o["low_retrieval_score"] for o in report["outcomes"].values())

    assert f"spared {gated} of {report['questions']['total']} model calls" in DOC
    assert f"cost of {report['outcomes']['answerable']['low_retrieval_score']} of 126" in DOC


def test_the_one_retried_timeout_is_in_the_question_log() -> None:
    errors = [
        record
        for line in (REPO / ANSWERS_LOG).read_text(encoding="utf-8").splitlines()[1:]
        if (record := json.loads(line))["decision"] == "error"
    ]

    assert [error["error"].split(";")[0] for error in errors] == [
        "RuntimeError: Ollama did not reply within 1800s"
    ]
    assert "one question that still timed out was retried on resume" in DOC


def test_lead_chunks_are_described_as_configured() -> None:
    text = DOC.replace("\n", " ")

    assert f"first {DEFAULT_LEAD} chunks" in text
    assert ", ".join(LEAD_CATEGORIES[:-1]) + " and " + LEAD_CATEGORIES[-1] in text


def test_every_other_category_is_identical_to_bm25_on_test() -> None:
    bm25, lead = (
        json.loads((REPO / path).read_text(encoding="utf-8"))["per_category"]
        for path in (TEST_REPORT, LEAD_TEST)
    )

    others = set(bm25) - set(LEAD_CATEGORIES)
    assert set(lead) == set(bm25) and len(others) == len(bm25) - len(LEAD_CATEGORIES)
    assert {category: lead[category] for category in others} == {
        category: bm25[category] for category in others
    }
    assert all(lead[c]["recall@10"] > bm25[c]["recall@10"] for c in LEAD_CATEGORIES)
