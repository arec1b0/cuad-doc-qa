"""Frozen splits and judgments against the real CUAD v1 release.

Needs the CUAD archive: downloaded on first run, cached in data/raw (gitignored).
"""

from pathlib import Path

import pytest

from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.evaluation.qrels import build_judgments, load_split
from cuad_doc_qa.ingestion import download, load_annotations, load_documents, load_official_split

REPO = Path(__file__).resolve().parents[2]
DATA_RAW = REPO / "data" / "raw"
SPLITS = REPO / "data" / "eval" / "splits"


@pytest.mark.network
def test_frozen_splits_equal_the_official_cuad_split() -> None:
    official = load_official_split(download(DATA_RAW))

    assert load_split(SPLITS / "dev.txt") == official["train"]
    assert load_split(SPLITS / "test.txt") == official["test"]


@pytest.mark.network
def test_every_gold_span_lands_in_a_chunk() -> None:
    archive = download(DATA_RAW)
    chunks = [chunk for document in load_documents(archive) for chunk in chunk_document(document)]

    judgments = build_judgments(load_annotations(archive), chunks)

    assert len(judgments) == 20_910
    assert sum(judgment.answerable for judgment in judgments) == 6_702
