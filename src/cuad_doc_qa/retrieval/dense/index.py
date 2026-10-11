"""Dense retrieval: exact cosine similarity over in-memory chunk embeddings.

Brute force is deliberate: a contract holds at most a few hundred chunks, so an approximate
nearest-neighbour index would add a dependency and nondeterminism for no measurable gain.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Self

import numpy as np
from numpy.typing import NDArray

from cuad_doc_qa.domain import Chunk, SearchHit
from cuad_doc_qa.retrieval.dense.embedder import Embedder, Vectors

logger = logging.getLogger(__name__)


class DenseIndex:
    def __init__(
        self,
        embedder: Embedder,
        chunk_ids: list[str],
        matrix: Vectors,
        doc_rows: dict[str, NDArray[np.intp]],
    ) -> None:
        self._embedder = embedder
        self._chunk_ids = chunk_ids
        self._matrix = matrix
        self._doc_rows = doc_rows
        self._all_rows = np.arange(len(chunk_ids))
        rank = {chunk_id: position for position, chunk_id in enumerate(sorted(chunk_ids))}
        self._id_rank = np.array([rank[chunk_id] for chunk_id in chunk_ids], dtype=np.intp)

    @classmethod
    def build(cls, chunks: Iterable[Chunk], embedder: Embedder) -> Self:
        """Embed every chunk. Raises ValueError on duplicate ids or a malformed embedding matrix."""
        started = time.perf_counter()
        chunk_list = list(chunks)
        chunk_ids = [chunk.chunk_id for chunk in chunk_list]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("duplicate chunk id")

        matrix = embedder.embed_documents([chunk.text for chunk in chunk_list])
        if matrix.ndim != 2 or matrix.shape[0] != len(chunk_list):
            raise ValueError(f"embedder returned shape {matrix.shape} for {len(chunk_list)} chunks")

        rows: dict[str, list[int]] = defaultdict(list)
        for row, chunk in enumerate(chunk_list):
            rows[chunk.doc_id].append(row)
        doc_rows = {doc_id: np.array(indices, dtype=np.intp) for doc_id, indices in rows.items()}

        logger.info(
            "Embedded %d chunks (dim %d) in %.1fs",
            len(chunk_list),
            matrix.shape[1],
            time.perf_counter() - started,
        )
        return cls(embedder, chunk_ids, _normalize(matrix), doc_rows)

    def search(self, query: str, k: int = 10, doc_id: str | None = None) -> list[SearchHit]:
        """Top-`k` chunks by cosine similarity, best first; one contract only if `doc_id` is set.

        A blank query returns no hits. Ties are broken by chunk id for reproducible rankings.
        """
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        if not query.strip():
            return []
        rows = self._all_rows if doc_id is None else self._doc_rows.get(doc_id)
        if rows is None or len(rows) == 0:
            return []

        query_vector = _normalize(self._embedder.embed_query(query)[np.newaxis, :])[0]
        scores = self._matrix[rows] @ query_vector
        order = np.lexsort((self._id_rank[rows], -scores))[:k]
        return [SearchHit(chunk_id=self._chunk_ids[rows[i]], score=float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self._chunk_ids)


def _normalize(matrix: Vectors) -> Vectors:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, np.finfo(np.float32).tiny)).astype(np.float32)
