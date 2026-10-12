"""Decide when to withhold an answer: a retrieval gate before generation, rules after it."""

from cuad_doc_qa.abstention.calibration import calibrate_bm25_gate
from cuad_doc_qa.abstention.gate import (
    MIN_ANSWERABLE,
    TARGET_ANSWERABLE_RECALL,
    Sample,
    ScoreGate,
    calibrate,
    load_gate,
    save_gate,
)
from cuad_doc_qa.abstention.policy import Decision, Reason, decide

__all__ = [
    "MIN_ANSWERABLE",
    "TARGET_ANSWERABLE_RECALL",
    "Decision",
    "Reason",
    "Sample",
    "ScoreGate",
    "calibrate",
    "calibrate_bm25_gate",
    "decide",
    "load_gate",
    "save_gate",
]
