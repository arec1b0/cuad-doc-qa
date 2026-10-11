"""A contiguous slice of a document, addressed by character offsets."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Chunk:
    """`text` equals `document.text[start:end]`; offsets let citations point at the source."""

    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end:
            raise ValueError(f"{self.chunk_id}: invalid span [{self.start}, {self.end})")
        if len(self.text) != self.end - self.start:
            raise ValueError(f"{self.chunk_id}: text length does not match span")
