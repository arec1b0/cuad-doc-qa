"""Dense retrieval: cosine similarity over sentence embeddings."""

from cuad_doc_qa.retrieval.dense.embedder import Embedder, FastEmbedEmbedder
from cuad_doc_qa.retrieval.dense.index import DenseIndex

__all__ = ["DenseIndex", "Embedder", "FastEmbedEmbedder"]
