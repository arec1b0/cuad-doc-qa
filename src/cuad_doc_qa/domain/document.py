"""A source contract as published by CUAD."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    """One contract.

    `text` is kept exactly as published. CUAD answer offsets index into it, so any
    normalization (whitespace, line endings, Unicode) would invalidate the gold spans.
    """

    doc_id: str
    text: str
