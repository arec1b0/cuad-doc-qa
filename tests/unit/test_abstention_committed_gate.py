"""The committed gate was calibrated on today's dev split, data and chunking."""

import hashlib
import json
from pathlib import Path

from cuad_doc_qa.chunking import DEFAULT_MAX_CHARS
from cuad_doc_qa.ingestion import CUAD_SHA256

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "configs" / "bm25-gate.json"
DEV = REPO / "data" / "eval" / "splits" / "dev.txt"


def test_gate_provenance_matches_the_repository() -> None:
    provenance = json.loads(GATE.read_text(encoding="utf-8"))["provenance"]

    assert provenance["split"] == "dev.txt"
    assert provenance["split_sha256"] == hashlib.sha256(DEV.read_bytes()).hexdigest()
    assert provenance["archive_sha256"] == CUAD_SHA256
    assert provenance["chunking"] == {"strategy": "clause", "max_chars": DEFAULT_MAX_CHARS}
