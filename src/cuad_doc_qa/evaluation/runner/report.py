"""Write an evaluation as a deterministic JSON report.

Byte-identical output for identical inputs (sorted keys, fixed rounding, LF line endings,
no timestamps), so a regenerated report diffs cleanly against the committed one.
"""

import json
from collections.abc import Mapping
from pathlib import Path

from cuad_doc_qa.evaluation.runner.retrieval import RetrievalEvaluation


def write_report(
    path: Path, evaluation: RetrievalEvaluation, provenance: Mapping[str, object]
) -> None:
    payload = {
        "provenance": dict(provenance),
        "queries": {
            "answerable": evaluation.overall.n,
            "unanswerable_skipped": evaluation.unanswerable_skipped,
        },
        "metrics": evaluation.overall.as_dict(),
        "per_category": {
            category: scores.as_dict() for category, scores in evaluation.per_category.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
