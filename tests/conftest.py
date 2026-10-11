"""Shared test fixtures."""

import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from cuad_doc_qa.domain import Chunk, Document

PartitionCheck = Callable[[Document, list[Chunk], int], None]
ArchiveFactory = Callable[..., Path]


@pytest.fixture
def assert_partition() -> PartitionCheck:
    """Chunks are exact, ordered, disjoint slices within budget that cover all non-whitespace."""

    def check(document: Document, chunks: list[Chunk], max_chars: int) -> None:
        previous_end = 0
        for chunk in chunks:
            assert chunk.doc_id == document.doc_id
            assert chunk.text == document.text[chunk.start : chunk.end]
            assert chunk.start >= previous_end
            assert chunk.end - chunk.start <= max_chars
            assert not chunk.text[0].isspace() and not chunk.text[-1].isspace()
            previous_end = chunk.end
        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
        assert "".join("".join(c.text.split()) for c in chunks) == "".join(document.text.split())

    return check


@pytest.fixture
def make_cuad_archive(tmp_path: Path) -> ArchiveFactory:
    """Build a zip with the CUAD v1 member layout: full set plus official train/test titles."""

    def make(entries: list[dict], train: tuple[str, ...] = (), test: tuple[str, ...] = ()) -> Path:
        path = tmp_path / "cuad.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("CUADv1.json", json.dumps({"version": "aok_v1.0", "data": entries}))
            for member, titles in (("train_separate_questions.json", train), ("test.json", test)):
                data = [{"title": title, "paragraphs": []} for title in titles]
                zf.writestr(member, json.dumps({"version": "aok_v1.0", "data": data}))
        return path

    return make
