"""cuad-qa download against the real pinned release.

Needs the CUAD archive: downloaded on first run, cached in data/raw (gitignored).
"""

from pathlib import Path

import pytest

from cuad_doc_qa.cli import main

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@pytest.mark.network
def test_download_verifies_the_archive(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--log-level", "WARNING", "download", "--data-dir", str(DATA_RAW)]) == 0
    assert capsys.readouterr().out == f"archive: {(DATA_RAW / 'cuad_v1_data.zip').as_posix()}\n"
