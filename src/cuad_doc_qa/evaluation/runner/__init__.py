"""Run retrieval evaluations and write reports."""

from cuad_doc_qa.evaluation.runner.baseline import run_bm25_baseline
from cuad_doc_qa.evaluation.runner.report import write_report
from cuad_doc_qa.evaluation.runner.retrieval import (
    K,
    RetrievalEvaluation,
    RetrievalScores,
    evaluate_retrieval,
)

__all__ = [
    "K",
    "RetrievalEvaluation",
    "RetrievalScores",
    "evaluate_retrieval",
    "run_bm25_baseline",
    "write_report",
]
