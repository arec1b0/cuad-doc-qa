"""Gold relevance judgments and frozen evaluation splits."""

from cuad_doc_qa.evaluation.qrels.judgments import Judgment, build_judgments
from cuad_doc_qa.evaluation.qrels.splits import load_split

__all__ = ["Judgment", "build_judgments", "load_split"]
