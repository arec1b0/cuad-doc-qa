"""BM25 lexical retrieval over chunks, backed by an in-memory SQLite FTS5 table.

The index is rebuilt from chunks on every run rather than persisted, so it can never drift
from the chunking that produced the relevance judgments.

Text is tokenized by FTS5 `porter unicode61`: Unicode-aware, case-folded, diacritics removed,
Porter-stemmed (terminate, terminates and termination share one term). BM25 uses FTS5's fixed
parameters k1 = 1.2 and b = 0.75.
"""

import logging
import re
import sqlite3
import time
from collections.abc import Iterable
from types import TracebackType
from typing import Self

from cuad_doc_qa.domain import Chunk, SearchHit

logger = logging.getLogger(__name__)

_TERM = re.compile(r"\w+")


class Bm25Index:
    """Search with `search()`; release the connection with `close()` or a `with` block."""

    def __init__(self, connection: sqlite3.Connection, size: int) -> None:
        self._connection = connection
        self._size = size

    @classmethod
    def build(cls, chunks: Iterable[Chunk]) -> Self:
        """Index every chunk. Raises ValueError on duplicate chunk ids."""
        started = time.perf_counter()
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE chunks USING fts5("
                "text, chunk_id UNINDEXED, doc_id UNINDEXED, tokenize = 'porter unicode61')"
            )
        except sqlite3.OperationalError as error:
            connection.close()
            raise RuntimeError(f"SQLite {sqlite3.sqlite_version} lacks FTS5: {error}") from error

        seen: set[str] = set()
        rows: list[tuple[str, str, str]] = []
        for chunk in chunks:
            if chunk.chunk_id in seen:
                connection.close()
                raise ValueError(f"{chunk.chunk_id}: duplicate chunk id")
            seen.add(chunk.chunk_id)
            rows.append((chunk.text, chunk.chunk_id, chunk.doc_id))
        with connection:
            connection.executemany(
                "INSERT INTO chunks (text, chunk_id, doc_id) VALUES (?, ?, ?)", rows
            )

        logger.info("Indexed %d chunks in %.2fs", len(rows), time.perf_counter() - started)
        return cls(connection, len(rows))

    def search(self, query: str, k: int = 10, doc_id: str | None = None) -> list[SearchHit]:
        """Top-`k` chunks for `query`, best first; restricted to one contract if `doc_id` is set.

        The query is reduced to its words and matched with OR semantics, so FTS5 operators,
        quotes and punctuation in the input are never interpreted. A query without words
        returns no hits. Ties are broken by chunk id for reproducible rankings.
        """
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        terms = dict.fromkeys(term.lower() for term in _TERM.findall(query))
        if not terms:
            return []

        match = " OR ".join(f'"{term}"' for term in terms)
        sql = "SELECT chunk_id, -bm25(chunks) FROM chunks WHERE chunks MATCH ?"
        params: list[str | int] = [match]
        if doc_id is not None:
            sql += " AND doc_id = ?"
            params.append(doc_id)
        sql += " ORDER BY bm25(chunks), chunk_id LIMIT ?"
        params.append(k)

        rows = self._connection.execute(sql, params).fetchall()
        return [SearchHit(chunk_id=chunk_id, score=score) for chunk_id, score in rows]

    def __len__(self) -> int:
        return self._size

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
