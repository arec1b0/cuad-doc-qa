"""run_experiment end to end on one sampled real dev contract, for the composed retrievers.

Needs internet access on first run: CUAD in data/raw, ONNX models in artifacts/models.
"""

import json
from pathlib import Path

import pytest

from cuad_doc_qa.evaluation.runner import run_experiment

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "data" / "eval" / "splits" / "dev.txt"


@pytest.mark.network
@pytest.mark.parametrize("retriever", ["hybrid", "rerank"])
def test_report_records_the_sample_and_every_model(tmp_path: Path, retriever: str) -> None:
    out = tmp_path / "report.json"

    evaluation = run_experiment(
        retriever,
        DEV,
        out,
        sample=1,
        seed=0,
        data_dir=REPO / "data" / "raw",
        models_dir=REPO / "artifacts" / "models",
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    provenance = report["provenance"]
    assert provenance["sample"] == {"contracts": 1, "seed": 0}
    assert provenance["retriever"]["name"] == retriever
    models = (
        [provenance["retriever"]["model"]]
        if retriever == "rerank"
        else [part["model"] for part in provenance["retriever"]["parts"] if "model" in part]
    )
    assert models and all(model["revision"] for model in models)
    assert report["metrics"]["n"] == evaluation.overall.n > 0
