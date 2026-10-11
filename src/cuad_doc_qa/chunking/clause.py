"""Clause-aware chunking: cut contracts at paragraph and clause boundaries within a size budget.

Boundaries are chosen in this order of preference:
1. Paragraph breaks (blank lines). A clause heading always starts a new chunk.
2. Sentence breaks, only inside a paragraph longer than the budget.
3. The last whitespace before the budget, only inside a sentence longer than the budget.
4. A hard cut at the budget, only inside a run of text with no whitespace.

Chunks never overlap and never begin or end with whitespace, so every non-whitespace
character of the document belongs to exactly one chunk.
"""

import re
from collections.abc import Iterator

from cuad_doc_qa.domain import Chunk, Document

DEFAULT_MAX_CHARS = 1500

_BLANK_LINE = re.compile(r"\n[^\S\n]*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(\"])")
_CLAUSE_START = re.compile(
    r"(?:ARTICLE|Article|SECTION|Section)\s+[0-9IVXLC]+\b"  # ARTICLE IV, Section 3
    r"|(?:\d{1,2}|[IVX]{1,4})\.\s"  # 7. Term, IV. Payment (not 7.1)
)
_CAPS_HEADING = re.compile(r"[A-Z][A-Z &,'-]{2,79}\.?")  # RECITALS, TERMINATION.


def chunk_document(document: Document, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Split `document` into ordered, non-overlapping chunks of at most `max_chars` characters."""
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")

    text = document.text
    spans: list[tuple[int, int]] = []
    for start, end in _segments(text, max_chars):
        fits = bool(spans) and end - spans[-1][0] <= max_chars
        if fits and not _is_clause_start(text, start, end):
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))

    return [
        Chunk(
            chunk_id=f"{document.doc_id}::{index:04d}",
            doc_id=document.doc_id,
            start=start,
            end=end,
            text=text[start:end],
        )
        for index, (start, end) in enumerate(spans)
    ]


def _segments(text: str, max_chars: int) -> Iterator[tuple[int, int]]:
    for start, end in _paragraphs(text):
        if end - start <= max_chars:
            yield start, end
            continue
        for sentence_start, sentence_end in _sentences(text, start, end):
            yield from _hard_split(text, sentence_start, sentence_end, max_chars)


def _paragraphs(text: str) -> Iterator[tuple[int, int]]:
    previous = 0
    for match in _BLANK_LINE.finditer(text):
        start, end = _trim(text, previous, match.start())
        if start < end:
            yield start, end
        previous = match.end()
    start, end = _trim(text, previous, len(text))
    if start < end:
        yield start, end


def _sentences(text: str, start: int, end: int) -> Iterator[tuple[int, int]]:
    previous = start
    for match in _SENTENCE_BREAK.finditer(text, start, end):
        yield previous, match.start()
        previous = match.end()
    yield previous, end


def _hard_split(text: str, start: int, end: int, max_chars: int) -> Iterator[tuple[int, int]]:
    while end - start > max_chars:
        cut = start + max_chars
        while cut > start and not text[cut].isspace():
            cut -= 1
        if cut == start:
            cut = start + max_chars
        yield _trim(text, start, cut)
        start, end = _trim(text, cut, end)
    yield start, end


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _is_clause_start(text: str, start: int, end: int) -> bool:
    return bool(_CLAUSE_START.match(text, start) or _CAPS_HEADING.fullmatch(text, start, end))
