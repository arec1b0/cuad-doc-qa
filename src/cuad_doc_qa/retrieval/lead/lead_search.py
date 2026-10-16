"""Put a contract's opening chunks first for questions about its name, parties and dates.

A contract states its name, its parties and its dates at the top, in a title and a preamble that
share few words with CUAD's questions, so BM25 ranks them low. On the dev split, putting the
first chunks ahead of BM25's hits raises recall@10 by more than 0.05 for exactly four categories
(LEAD_CATEGORIES); for every other category it changes recall@10 by 0.013 at most and usually
lowers it. DEFAULT_LEAD maximized mean recall@10 over those four categories on dev among
1, 2, 3, 4, 5, 7 and 10 lead chunks. The commit that set both records the measurement.
"""

from collections import defaultdict
from collections.abc import Iterable

from cuad_doc_qa.domain import Chunk, Search, SearchHit

LEAD_CATEGORIES = ("Agreement Date", "Document Name", "Effective Date", "Parties")
DEFAULT_LEAD = 7


class LeadChunkSearch:
    """A Search that ranks a contract's first `lead` chunks ahead of the first stage's hits.

    Only queries in `queries` are changed, and only within one contract (`doc_id` given); any
    other query gets the first stage's hits unchanged.
    """

    def __init__(
        self,
        first_stage: Search,
        chunks: Iterable[Chunk],
        queries: Iterable[str],
        lead: int = DEFAULT_LEAD,
    ) -> None:
        if lead < 1:
            raise ValueError(f"lead must be positive, got {lead}")
        self._first_stage = first_stage
        self._queries = frozenset(queries)
        by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            by_doc[chunk.doc_id].append(chunk)
        self._leading = {
            doc_id: [chunk.chunk_id for chunk in sorted(doc_chunks, key=lambda c: c.start)[:lead]]
            for doc_id, doc_chunks in by_doc.items()
        }

    def __call__(self, query: str, k: int = 10, doc_id: str | None = None) -> list[SearchHit]:
        """The lead chunks in document order, then the first stage's other hits; at most `k`.

        Lead chunks carry the first stage's top score, so the top score that the abstention gate
        reads is unchanged. A query the first stage finds nothing for still gets nothing.
        """
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        hits = self._first_stage(query, k, doc_id)
        if query not in self._queries or doc_id is None or not hits:
            return hits
        leading = self._leading.get(doc_id, [])
        lead_hits = [SearchHit(chunk_id=chunk_id, score=hits[0].score) for chunk_id in leading]
        rest = [hit for hit in hits if hit.chunk_id not in set(leading)]
        return (lead_hits + rest)[:k]
