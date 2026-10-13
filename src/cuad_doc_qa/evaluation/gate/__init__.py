"""Fail a change that makes retrieval worse than its baseline."""

from cuad_doc_qa.evaluation.gate.regression import (
    DEFAULT_TOLERANCES,
    MetricCheck,
    compare,
    load_report,
)

__all__ = ["DEFAULT_TOLERANCES", "MetricCheck", "compare", "load_report"]
