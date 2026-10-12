"""When to withhold an answer, and why.

Strict by design: an answer is shown only if the model claims the clause is present and every
quote it gave was located in the contract. One unlocatable quote withholds the whole answer,
because the answer text may rest on it.
"""

from dataclasses import dataclass
from enum import StrEnum

from cuad_doc_qa.citations import CitationCheck
from cuad_doc_qa.domain import Answer


class Reason(StrEnum):
    LOW_RETRIEVAL_SCORE = "low_retrieval_score"
    NOT_FOUND = "not_found"
    NO_GROUNDED_CITATION = "no_grounded_citation"
    REJECTED_CITATION = "rejected_citation"


@dataclass(frozen=True, slots=True)
class Decision:
    """`reason` is None when the answer may be shown."""

    reason: Reason | None

    @property
    def abstain(self) -> bool:
        return self.reason is not None


def decide(answer: Answer, check: CitationCheck) -> Decision:
    """Post-generation decision; the retrieval gate runs before generation and is separate."""
    if not answer.found:
        return Decision(Reason.NOT_FOUND)
    if check.rejected:
        return Decision(Reason.REJECTED_CITATION)
    if not check.grounded:
        return Decision(Reason.NO_GROUNDED_CITATION)
    return Decision(None)
