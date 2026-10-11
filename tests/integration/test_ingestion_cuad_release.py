"""The real CUAD v1 release: pinned download and full parse.

Needs internet access on first run. The archive is cached in data/raw (gitignored) and reused.
"""

from pathlib import Path

import pytest

from cuad_doc_qa.ingestion import download, load_documents

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.network
def test_release_parses_into_510_contracts() -> None:
    documents = load_documents(download(DATA_RAW))

    assert len(documents) == 510
    assert len({d.doc_id for d in documents}) == 510
    assert sum(len(d.text) for d in documents) == 26_807_133
    assert documents[0].doc_id == "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
