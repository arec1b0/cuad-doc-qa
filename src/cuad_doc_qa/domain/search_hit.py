"""One ranked retrieval result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchHit:
    """`score` is comparable only within one ranking: higher means more relevant."""

    chunk_id: str
    score: float
