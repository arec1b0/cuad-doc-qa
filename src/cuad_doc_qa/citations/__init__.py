"""Verify claimed citations against the contract text."""

from cuad_doc_qa.citations.grounding import (
    FUZZY_THRESHOLD,
    MIN_FUZZY_CHARS,
    CitationCheck,
    GroundedCitation,
    check_citations,
    ground,
)

__all__ = [
    "FUZZY_THRESHOLD",
    "MIN_FUZZY_CHARS",
    "CitationCheck",
    "GroundedCitation",
    "check_citations",
    "ground",
]
