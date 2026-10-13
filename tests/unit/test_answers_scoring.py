"""score_question and aggregate on hand-built pipeline results."""

import pytest

from cuad_doc_qa.abstention import Decision, Reason
from cuad_doc_qa.citations import CitationCheck, GroundedCitation
from cuad_doc_qa.domain import Answer, Citation, SearchHit
from cuad_doc_qa.evaluation.answers import aggregate, error_record, is_correct, score_question
from cuad_doc_qa.pipeline import PipelineResult

GOLD = ((100, 150), (400, 420))
QUOTE = Citation("D::0001", "governed by Delaware law")


def _cited(start: int, end: int, fuzzy: bool = False) -> GroundedCitation:
    return GroundedCitation(
        QUOTE, "D", start, end, "x" * (end - start), fuzzy, 95.0 if fuzzy else 100.0
    )


GATED_SECONDS = {"retrieval": 0.01, "total": 0.01}
MODEL_SECONDS = {"retrieval": 0.01, "generation": 2.0, "grounding": 0.001, "total": 2.011}


def _result(
    reason: Reason | None,
    grounded: tuple[GroundedCitation, ...] = (),
    rejected: int = 0,
    gated: bool = False,
) -> PipelineResult:
    answer = None if gated else Answer(True, "Delaware law.", (QUOTE,) * (len(grounded) + rejected))
    check = None if gated else CitationCheck(grounded, (QUOTE,) * rejected)
    seconds = GATED_SECONDS if gated else MODEL_SECONDS
    return PipelineResult(
        question="Which law governs?",
        doc_id="D",
        category="Governing Law",
        retrieved=(SearchHit("D::0001", 12.3456789),),
        answer=answer,
        check=check,
        decision=Decision(reason),
        seconds=seconds,
    )


def test_shown_answer_records_which_citations_hit_gold() -> None:
    record = score_question(
        "D__Governing Law", _result(None, (_cited(140, 160), _cited(200, 210, True))), GOLD
    )

    assert record["decision"] == "answer"
    assert record["answerable"] and record["gold_spans"] == 2
    assert [c["overlaps_gold"] for c in record["citations"]] == [True, False]
    assert record["gold_spans_hit"] == 1
    assert (record["claimed"], record["grounded"], record["fuzzy"], record["rejected"]) == (
        2,
        2,
        1,
        0,
    )
    assert record["top_score"] == 12.345679
    assert is_correct(record)


def test_withheld_answer_keeps_counts_but_shows_no_citation() -> None:
    record = score_question(
        "q", _result(Reason.REJECTED_CITATION, (_cited(100, 120),), rejected=1), GOLD
    )

    assert record["decision"] == "rejected_citation"
    assert record["citations"] == [] and record["gold_spans_hit"] == 0
    assert (record["claimed"], record["grounded"], record["rejected"]) == (2, 1, 1)
    assert not is_correct(record)


def test_gated_question_never_reached_the_model() -> None:
    record = score_question("q", _result(Reason.LOW_RETRIEVAL_SCORE, gated=True), ())

    assert record["decision"] == "low_retrieval_score"
    assert not record["answerable"]
    assert (record["claimed"], record["answer"]) == (0, None)
    assert "generation" not in record["seconds"]


def test_error_record_is_neither_shown_nor_withheld() -> None:
    record = error_record("q", "D", "Term", GOLD, RuntimeError("Ollama is not reachable"))

    assert record["decision"] == "error"
    assert record["error"] == "RuntimeError: Ollama is not reachable"


def test_aggregate_metrics_by_hand() -> None:
    records = [
        score_question("a1", _result(None, (_cited(100, 110),)), GOLD),  # correct, 1 of 2 gold
        score_question("a2", _result(None, (_cited(10, 20),)), GOLD),  # shown, wrong citation
        score_question("a3", _result(Reason.NOT_FOUND), GOLD),  # missed
        score_question("u1", _result(None, (_cited(10, 20),)), ()),  # false answer
        score_question("u2", _result(Reason.LOW_RETRIEVAL_SCORE, gated=True), ()),
        score_question("u3", _result(Reason.NOT_FOUND), ()),
        error_record("u4", "D", "Governing Law", (), ValueError("bad reply")),
    ]

    report = aggregate(records)

    assert report["questions"] == {"total": 7, "answerable": 3, "unanswerable": 4}
    assert report["metrics"] == {
        "answer_recall": round(1 / 3, 4),
        "answer_precision": round(1 / 3, 4),
        "false_answer_rate": 0.25,
        "abstention_recall": 0.5,
        "abstention_precision": round(2 / 3, 4),
        "citation_precision": round(1 / 3, 4),
        "gold_span_recall": round(1 / 6, 4),
        "error_rate": round(1 / 7, 4),
    }
    assert report["outcomes"]["unanswerable"] == {
        "answer": 1,
        "correct": 0,
        "error": 1,
        "low_retrieval_score": 1,
        "not_found": 1,
    }
    assert report["grounding"] == {"claimed": 3, "grounded": 3, "fuzzy": 0, "rejected": 0}
    assert report["latency_seconds"] == {"p50": 2.01, "p95": 2.01}
    assert report["per_category"]["Governing Law"]["correct"] == 1


@pytest.mark.parametrize(
    ("values", "expected"),
    [([], {"p50": None, "p95": None}), ([3.0], {"p50": 3.0, "p95": 3.0})],
    ids=["no-model-calls", "one-call"],
)
def test_latency_with_too_few_calls(values: list[float], expected: dict[str, float | None]) -> None:
    shown = score_question("q", _result(None, (_cited(100, 110),)), GOLD)
    records = [shown | {"seconds": {"generation": value, "total": value}} for value in values]

    assert aggregate(records)["latency_seconds"] == expected
