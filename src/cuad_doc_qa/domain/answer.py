"""A generated answer and the citations it claims, before any verification."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Citation:
    """A quote the model claims to have copied verbatim from chunk `chunk_id`."""

    chunk_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class Answer:
    """`found` is the model's own claim that the sources contain the requested clause."""

    found: bool
    text: str
    citations: tuple[Citation, ...]
