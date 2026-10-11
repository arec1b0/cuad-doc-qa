"""evaluate_retrieval and write_report.

Expected scores, by hand:
  A in D1: gold retrieved at rank 1                  -> recall 1.0, RR 1.0, nDCG 1.0
  A in D2: gold not retrieved                        -> recall 0.0, RR 0.0, nDCG 0.0
  B in D3: 1 of 2 gold at rank 2                     -> recall 0.5, RR 0.5, nDCG 0.386853
  overall recall@10 = (1 + 0 + 0.5) / 3 = 0.5
"""

import json
from pathlib import Path

import pytest

from cuad_doc_qa.domain import SearchHit
from cuad_doc_qa.evaluation.qrels import Judgment
from cuad_doc_qa.evaluation.runner import K, evaluate_retrieval, write_report

QUESTIONS = {"A": "question about A", "B": "question about B", "C": "question about C"}
RANKINGS = {
    ("question about A", "D1"): ["D1::0000", "D1::0001"],
    ("question about A", "D2"): ["D2::0000"],
    ("question about B", "D3"): ["D3::0000", "D3::0001"],
}
JUDGMENTS = [
    Judgment("D1__A", "D1", "A", frozenset({"D1::0000"})),
    Judgment("D2__A", "D2", "A", frozenset({"D2::0005"})),
    Judgment("D3__B", "D3", "B", frozenset({"D3::0001", "D3::0002"})),
    Judgment("D3__C", "D3", "C", frozenset()),
]


class RecordingSearch:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None]] = []

    def __call__(self, query: str, k: int, doc_id: str | None) -> list[SearchHit]:
        self.calls.append((query, k, doc_id))
        return [SearchHit(chunk_id, 1.0) for chunk_id in RANKINGS[(query, doc_id)]]


def test_scores_answerable_questions_within_their_contract() -> None:
    search = RecordingSearch()

    evaluation = evaluate_retrieval(search, JUDGMENTS, QUESTIONS)

    assert search.calls == [
        ("question about A", K, "D1"),
        ("question about A", K, "D2"),
        ("question about B", K, "D3"),
    ]
    assert evaluation.unanswerable_skipped == 1
    assert evaluation.overall.as_dict() == {
        "n": 3,
        "recall@5": 0.5,
        "recall@10": 0.5,
        "mrr@10": 0.5,
        "ndcg@10": round((1.0 + 0.0 + 0.386853) / 3, 4),
    }
    assert list(evaluation.per_category) == ["A", "B"]
    assert evaluation.per_category["A"].as_dict()["recall@10"] == 0.5
    assert evaluation.per_category["B"].as_dict()["ndcg@10"] == 0.3869


def test_rejects_a_run_without_answerable_questions() -> None:
    with pytest.raises(ValueError, match="no answerable judgments"):
        evaluate_retrieval(RecordingSearch(), JUDGMENTS[3:], QUESTIONS)


def test_report_is_deterministic_json_with_lf_endings(tmp_path: Path) -> None:
    evaluation = evaluate_retrieval(RecordingSearch(), JUDGMENTS, QUESTIONS)
    first, second = tmp_path / "a" / "report.json", tmp_path / "b" / "report.json"

    write_report(first, evaluation, {"run": "test", "note": "Cafe\N{COMBINING ACUTE ACCENT}"})
    write_report(second, evaluation, {"note": "Cafe\N{COMBINING ACUTE ACCENT}", "run": "test"})

    raw = first.read_bytes()
    assert raw == second.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    assert "Cafe\N{COMBINING ACUTE ACCENT}".encode() in raw
    report = json.loads(raw)
    assert report["queries"] == {"answerable": 3, "unanswerable_skipped": 1}
    assert report["metrics"]["recall@10"] == 0.5
    assert set(report["per_category"]) == {"A", "B"}
