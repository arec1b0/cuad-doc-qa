"""Ground claimed citations in the contract: locate each quote inside the chunk it cites.

A grounded citation carries document offsets, and its `text` is always the contract's own text
at those offsets, never the model's quote. A quote that cannot be located is rejected.

Matching runs on a normalized form of both strings: curly quotes and dashes made plain,
whitespace runs collapsed to one space. CUAD contracts are full of doubled spaces and hard line
breaks that a model does not reproduce. Normalized text keeps a map back to original offsets,
so normalization never shifts a span. Tiers, first hit wins:

1. Verbatim: the normalized quote occurs in the normalized chunk.
2. Verbatim ignoring case: the same, both sides lowercased. Tier 1 runs first so that a short
   quote such as a party name lands on the occurrence written the way it was quoted.
3. Fuzzy: if the lowercased quote has at least MIN_FUZZY_CHARS characters, the best window
   found by rapidfuzz partial_ratio_alignment, accepted at FUZZY_THRESHOLD or above.
   Both constants were set before any measurement and are not tuned.

Within a tier the first occurrence wins.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from rapidfuzz import fuzz

from cuad_doc_qa.domain import Answer, Chunk, Citation

FUZZY_THRESHOLD = 90.0
MIN_FUZZY_CHARS = 20

_PLAIN = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-"})  # fmt: skip

_Located = tuple[int, int, bool, float]


@dataclass(frozen=True, slots=True)
class GroundedCitation:
    """`text` equals the contract text at [start, end); `fuzzy` is False for verbatim matches."""

    citation: Citation
    doc_id: str
    start: int
    end: int
    text: str
    fuzzy: bool
    score: float


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """Every claimed citation lands in exactly one of the two tuples, in claim order."""

    grounded: tuple[GroundedCitation, ...]
    rejected: tuple[Citation, ...]


def check_citations(answer: Answer, chunks: Iterable[Chunk]) -> CitationCheck:
    """Ground every citation of `answer` in its own chunk; a citation of an absent chunk fails."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    grounded: list[GroundedCitation] = []
    rejected: list[Citation] = []
    for citation in answer.citations:
        chunk = by_id.get(citation.chunk_id)
        located = ground(citation, chunk) if chunk is not None else None
        if located is None:
            rejected.append(citation)
        else:
            grounded.append(located)
    return CitationCheck(grounded=tuple(grounded), rejected=tuple(rejected))


def ground(citation: Citation, chunk: Chunk) -> GroundedCitation | None:
    """Locate `citation.quote` inside `chunk`, or None if it is not there."""
    located = _locate(citation.quote, chunk.text)
    if located is None:
        return None
    start, end, fuzzy, score = located
    return GroundedCitation(
        citation=citation,
        doc_id=chunk.doc_id,
        start=chunk.start + start,
        end=chunk.start + end,
        text=chunk.text[start:end],
        fuzzy=fuzzy,
        score=score,
    )


def _locate(quote: str, text: str) -> _Located | None:
    """(start, end, fuzzy, score) of `quote` in `text`, as offsets into `text`."""
    needle, haystack, origin = "", "", []
    for lower in (False, True):
        needle, _ = _normalize(quote, lower)
        if not needle:
            return None
        haystack, origin = _normalize(text, lower)
        position = haystack.find(needle)
        if position >= 0:
            return _span(text, origin, position, position + len(needle), False, 100.0)

    if not MIN_FUZZY_CHARS <= len(needle) <= len(haystack):
        return None
    alignment = fuzz.partial_ratio_alignment(needle, haystack, score_cutoff=FUZZY_THRESHOLD)
    if alignment is None:
        return None
    return _span(text, origin, alignment.dest_start, alignment.dest_end, True, alignment.score)


def _span(
    text: str, origin: list[int], start: int, end: int, fuzzy: bool, score: float
) -> _Located | None:
    first, last = origin[start], origin[end - 1] + 1
    while first < last and text[first].isspace():
        first += 1
    while last > first and text[last - 1].isspace():
        last -= 1
    return (first, last, fuzzy, score) if first < last else None


def _normalize(text: str, lower: bool) -> tuple[str, list[int]]:
    """Normalized text and, for each of its characters, the index it came from in `text`."""
    chars: list[str] = []
    origin: list[int] = []
    space_at: int | None = None
    for index, char in enumerate(text):
        if char.isspace():
            if chars and space_at is None:
                space_at = index
            continue
        if space_at is not None:
            chars.append(" ")
            origin.append(space_at)
            space_at = None
        plain = char.translate(_PLAIN)
        for normalized in plain.lower() if lower else plain:
            chars.append(normalized)
            origin.append(index)
    return "".join(chars), origin
