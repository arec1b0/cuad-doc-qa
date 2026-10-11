"""Smoke test: the package installs, imports, and reports the version declared in pyproject."""

import tomllib
from pathlib import Path

import cuad_doc_qa

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    assert cuad_doc_qa.__version__ == declared
