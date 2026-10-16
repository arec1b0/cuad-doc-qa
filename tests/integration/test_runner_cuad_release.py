"""run_bm25_baseline end to end on real CUAD data, restricted to three contracts for speed.

Needs the CUAD archive: downloaded on first run, cached in data/raw (gitignored).
"""

import hashlib
import json
from pathlib import Path

import pytest

from cuad_doc_qa.evaluation.runner import run_bm25_baseline
from cuad_doc_qa.ingestion import download, load_annotations
from cuad_doc_qa.retrieval.lead import LEAD_CATEGORIES

REPO = Path(__file__).resolve().parents[2]
DATA_RAW = REPO / "data" / "raw"
TEST_SPLIT = REPO / "data" / "eval" / "splits" / "test.txt"


@pytest.mark.network
def test_writes_a_report_for_the_questions_of_the_split(tmp_path: Path) -> None:
    contracts = TEST_SPLIT.read_text(encoding="utf-8").splitlines()[:3]
    split = tmp_path / "three.txt"
    split.write_text("\n".join(contracts) + "\n", encoding="utf-8", newline="\n")
    expected = [a for a in load_annotations(download(DATA_RAW)) if a.doc_id in contracts]

    run_bm25_baseline(split, tmp_path / "report.json", data_dir=DATA_RAW)

    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["queries"] == {
        "answerable": sum(1 for a in expected if a.spans),
        "unanswerable_skipped": sum(1 for a in expected if not a.spans),
    }
    assert report["provenance"]["split_sha256"] == hashlib.sha256(split.read_bytes()).hexdigest()
    assert all(0.0 <= value <= 1.0 for key, value in report["metrics"].items() if key != "n")
    assert sum(scores["n"] for scores in report["per_category"].values()) == report["metrics"]["n"]


@pytest.mark.network
def test_lead_chunks_change_only_the_lead_categories(tmp_path: Path) -> None:
    dev = REPO / "data" / "eval" / "splits" / "dev.txt"
    split = tmp_path / "three.txt"
    split.write_text(
        "\n".join(dev.read_text(encoding="utf-8").splitlines()[:3]) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    run_bm25_baseline(split, tmp_path / "bm25.json", data_dir=DATA_RAW)
    run_bm25_baseline(split, tmp_path / "lead.json", data_dir=DATA_RAW, lead=True)

    bm25, lead = (
        json.loads((tmp_path / name).read_text(encoding="utf-8"))
        for name in ("bm25.json", "lead.json")
    )
    changed = {
        category
        for category, scores in lead["per_category"].items()
        if scores != bm25["per_category"][category]
    }
    assert changed <= set(LEAD_CATEGORIES) and changed
    assert lead["provenance"]["run"] == "bm25-lead-baseline"
    assert lead["queries"] == bm25["queries"]
