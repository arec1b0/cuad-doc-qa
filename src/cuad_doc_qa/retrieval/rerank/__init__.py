"""Second-stage reranking of retrieved candidates with a cross-encoder."""

from cuad_doc_qa.retrieval.rerank.reranked_search import DEFAULT_DEPTH, RerankedSearch
from cuad_doc_qa.retrieval.rerank.scorer import FastEmbedCrossEncoder, PairScorer

__all__ = ["DEFAULT_DEPTH", "FastEmbedCrossEncoder", "PairScorer", "RerankedSearch"]
