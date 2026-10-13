"""Run retrieval evaluations and write reports."""

from cuad_doc_qa.evaluation.runner.baseline import run_bm25_baseline
from cuad_doc_qa.evaluation.runner.experiment import RETRIEVERS, run_experiment, sample_contracts
from cuad_doc_qa.evaluation.runner.report import write_report
from cuad_doc_qa.evaluation.runner.retrieval import (
    K,
    RetrievalEvaluation,
    RetrievalScores,
    evaluate_retrieval,
)

__all__ = [
    "K",
    "RETRIEVERS",
    "RetrievalEvaluation",
    "RetrievalScores",
    "evaluate_retrieval",
    "run_bm25_baseline",
    "run_experiment",
    "sample_contracts",
    "write_report",
]
