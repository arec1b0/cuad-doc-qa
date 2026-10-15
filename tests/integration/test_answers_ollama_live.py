"""cuad-qa evaluate-answers with a real local Ollama model on one sampled dev contract.

41 questions, 31 of which pass the gate and reach the model: minutes on a GPU, over an hour on
CPU (set CUAD_QA_OLLAMA_TIMEOUT if one call exceeds the default 300 s). Needs a running Ollama
server with the model named in CUAD_QA_OLLAMA_MODEL already pulled, and CUAD in data/raw.
Checks that every question completes without an error, not answer quality.
"""

import json
import os
from pathlib import Path

import pytest

from cuad_doc_qa.cli import main

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.ollama
def test_evaluate_answers_completes_on_one_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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

    assert code == 0, capsys.readouterr().err
    log = out.with_name("answers.questions.jsonl").read_text(encoding="utf-8").splitlines()
    errors = [json.loads(line)["error"] for line in log[1:] if json.loads(line)["error"]]
    assert errors == []
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["questions"]["total"] == 41
    assert report["provenance"]["model"]["digest"]
