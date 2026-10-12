"""calibrate_bm25_gate end to end on three real dev contracts.

Needs the CUAD archive: downloaded on first run, cached in data/raw (gitignored).
"""

import json
from pathlib import Path

import pytest

from cuad_doc_qa.abstention import calibrate_bm25_gate, load_gate

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "data" / "eval" / "splits" / "dev.txt"


@pytest.mark.network
def test_calibrates_and_stores_a_gate_for_a_split(tmp_path: Path) -> None:
    split = tmp_path / "three.txt"
    split.write_text(
        "\n".join(DEV.read_text(encoding="utf-8").splitlines()[:3]) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    out = tmp_path / "gate.json"

    gate = calibrate_bm25_gate(
        split, out, data_dir=REPO / "data" / "raw", target=1.0, min_answerable=1
    )

    assert load_gate(out) == gate
    assert gate.thresholds
    provenance = json.loads(out.read_text(encoding="utf-8"))["provenance"]
    assert provenance["questions"] == 3 * 41
    assert provenance["target_answerable_recall"] == 1.0
