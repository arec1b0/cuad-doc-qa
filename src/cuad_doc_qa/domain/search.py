"""The retrieval port: every retriever and every composition of retrievers has this shape."""

from collections.abc import Callable

from cuad_doc_qa.domain.search_hit import SearchHit

Search = Callable[[str, int, str | None], list[SearchHit]]
"""(query, k, doc_id) -> at most k hits, best first; doc_id restricts results to one contract."""
