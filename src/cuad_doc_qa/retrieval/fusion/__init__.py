"""Hybrid retrieval: combine several rankings into one."""

from cuad_doc_qa.retrieval.fusion.rrf import (
    DEFAULT_DEPTH,
    RRF_K,
    FusedSearch,
    reciprocal_rank_fusion,
)

__all__ = ["DEFAULT_DEPTH", "RRF_K", "FusedSearch", "reciprocal_rank_fusion"]
