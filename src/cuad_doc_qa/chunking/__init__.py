"""Split documents into retrievable chunks that keep their source offsets."""

from cuad_doc_qa.chunking.clause import DEFAULT_MAX_CHARS, chunk_document

__all__ = ["DEFAULT_MAX_CHARS", "chunk_document"]
