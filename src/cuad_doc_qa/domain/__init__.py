"""Core value types shared across modules."""

from cuad_doc_qa.domain.annotation import Annotation
from cuad_doc_qa.domain.answer import Answer, Citation
from cuad_doc_qa.domain.chunk import Chunk
from cuad_doc_qa.domain.document import Document
from cuad_doc_qa.domain.search import Search
from cuad_doc_qa.domain.search_hit import SearchHit

__all__ = ["Annotation", "Answer", "Chunk", "Citation", "Document", "Search", "SearchHit"]
