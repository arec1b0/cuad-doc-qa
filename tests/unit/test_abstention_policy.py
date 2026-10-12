"""decide: the post-generation abstention rules."""

import pytest

from cuad_doc_qa.abstention import Decision, Reason, decide
from cuad_doc_qa.citations import CitationCheck, GroundedCitation
from cuad_doc_qa.domain import Answer, Citation

KEPT = Citation("D::0001", "governed by the laws of Delaware")
LOST = Citation("D::0001", "governed by English law")
GROUNDED = GroundedCitation(KEPT, "D", 10, 42, "governed by the laws of Delaware", False, 100.0)


@pytest.mark.parametrize(
    ("found", "check", "reason"),
    [
        (False, CitationCheck(grounded=(GROUNDED,), rejected=()), Reason.NOT_FOUND),
        (True, CitationCheck(grounded=(GROUNDED,), rejected=(LOST,)), Reason.REJECTED_CITATION),
        (True, CitationCheck(grounded=(), rejected=(LOST,)), Reason.REJECTED_CITATION),
        (True, CitationCheck(grounded=(), rejected=()), Reason.NO_GROUNDED_CITATION),
        (True, CitationCheck(grounded=(GROUNDED,), rejected=()), None),
    ],
    ids=["not-found-wins", "one-rejected", "all-rejected", "uncited", "answer"],
)
def test_decision(found: bool, check: CitationCheck, reason: Reason | None) -> None:
    decision = decide(Answer(found=found, text="Delaware law.", citations=()), check)

    assert decision == Decision(reason)
    assert decision.abstain is (reason is not None)


def test_reasons_have_stable_report_names() -> None:
    assert [reason.value for reason in Reason] == [
        "low_retrieval_score",
        "not_found",
        "no_grounded_citation",
        "rejected_citation",
    ]
