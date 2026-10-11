"""Core value types shared across modules."""

from cuad_doc_qa.domain.annotation import Annotation
from cuad_doc_qa.domain.chunk import Chunk
from cuad_doc_qa.domain.document import Document
from cuad_doc_qa.domain.search_hit import SearchHit

__all__ = ["Annotation", "Chunk", "Document", "SearchHit"]
