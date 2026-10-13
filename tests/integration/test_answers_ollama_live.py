"""cuad-qa evaluate-answers with a real local Ollama model on one sampled dev contract.

41 questions, so expect a few minutes. Needs a running Ollama server with the model named in
CUAD_QA_OLLAMA_MODEL already pulled, and CUAD in data/raw. Checks completion, not quality.
"""

import json
import os
from pathlib import Path

import pytest

from cuad_doc_qa.cli import main

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.ollama
def test_evaluate_answers_completes_on_one_contract(tmp_path: Path) -> None:
    if not os.environ.get("CUAD_QA_OLLAMA_MODEL"):
        pytest.fail("set CUAD_QA_OLLAMA_MODEL to a pulled Ollama model, e.g. qwen2.5:7b")
    out = tmp_path / "answers.json"

    code = main(
        [
            "--log-level",
            "WARNING",
            "evaluate-answers",
            "--sample",
            "1",
            "--out",
            str(out),
            "--data-dir",
            str(REPO / "data" / "raw"),
            "--gate",
            str(REPO / "configs" / "bm25-gate.json"),
        ]
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert report["questions"]["total"] == 41
    assert report["provenance"]["model"]["digest"]
