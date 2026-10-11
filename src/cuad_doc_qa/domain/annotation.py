"""Expert annotation of one clause category in one contract."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Annotation:
    """Gold spans for one (contract, category) pair.

    `spans` are sorted, unique `(start, end)` offsets into the contract text.
    An empty tuple means the clause is absent from the contract.
    """

    doc_id: str
    category: str
    spans: tuple[tuple[int, int], ...]

    @property
    def query_id(self) -> str:
        return f"{self.doc_id}__{self.category}"
