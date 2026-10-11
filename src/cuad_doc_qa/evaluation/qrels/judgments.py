"""Relevance judgments: which chunks answer each (contract, category) question."""

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from cuad_doc_qa.domain import Annotation, Chunk


@dataclass(frozen=True, slots=True)
class Judgment:
    """`relevant` holds the ids of chunks that overlap a gold span; empty means unanswerable."""

    query_id: str
    doc_id: str
    category: str
    relevant: frozenset[str]

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


def build_judgments(annotations: Iterable[Annotation], chunks: Iterable[Chunk]) -> list[Judgment]:
    """One judgment per annotation, in annotation order.

    Raises ValueError if a gold span overlaps no chunk, because that would silently turn
    an answerable question into an unanswerable one.
    """
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)
    index: dict[str, tuple[list[Chunk], list[int]]] = {}
    for doc_id, doc_chunks in by_doc.items():
        ordered = sorted(doc_chunks, key=lambda c: c.start)
        index[doc_id] = (ordered, [c.start for c in ordered])

    judgments: list[Judgment] = []
    for annotation in annotations:
        ordered, starts = index.get(annotation.doc_id, ([], []))
        relevant: set[str] = set()
        for start, end in annotation.spans:
            hits = _overlapping(ordered, starts, start, end)
            if not hits:
                raise ValueError(
                    f"{annotation.query_id}: gold span [{start}, {end}) overlaps no chunk"
                )
            relevant.update(hit.chunk_id for hit in hits)
        judgments.append(
            Judgment(
                query_id=annotation.query_id,
                doc_id=annotation.doc_id,
                category=annotation.category,
                relevant=frozenset(relevant),
            )
        )
    return judgments


def _overlapping(ordered: list[Chunk], starts: list[int], start: int, end: int) -> list[Chunk]:
    i = bisect_right(starts, start) - 1
    if i < 0 or ordered[i].end <= start:
        i += 1
    hits: list[Chunk] = []
    while i < len(ordered) and ordered[i].start < end:
        hits.append(ordered[i])
        i += 1
    return hits
