"""cuad-qa ask end to end with a real local Ollama model on a real CUAD contract.

Needs a running Ollama server with the model named in CUAD_QA_OLLAMA_MODEL already pulled,
and the CUAD archive in data/raw. Checks that the command completes, not answer quality.
On CPU one call takes minutes: set CUAD_QA_OLLAMA_TIMEOUT if the default 300 s is too short.
"""

import os
from pathlib import Path

import pytest

from cuad_doc_qa.cli import main

REPO = Path(__file__).resolve().parents[2]
CONTRACT = "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"


@pytest.mark.ollama
def test_ask_completes_with_a_decision(capsys: pytest.CaptureFixture[str]) -> None:
    if not os.environ.get("CUAD_QA_OLLAMA_MODEL"):
        pytest.fail("set CUAD_QA_OLLAMA_MODEL to a pulled Ollama model, e.g. qwen2.5:7b")

    code = main(
        [
            "--log-level",
            "WARNING",
            "ask",
            "--contract",
            CONTRACT,
            "--category",
            "Governing Law",
            "--data-dir",
            str(REPO / "data" / "raw"),
            "--gate",
            str(REPO / "configs" / "bm25-gate.json"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "\ndecision: " in captured.out
