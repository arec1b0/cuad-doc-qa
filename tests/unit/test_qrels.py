"""build_judgments and load_split."""

from pathlib import Path

import pytest

from cuad_doc_qa.domain import Annotation, Chunk
from cuad_doc_qa.evaluation.qrels import build_judgments, load_split

TEXT = "AAAA BBBB CCCC"
CHUNKS = [
    Chunk(chunk_id=f"D::{i:04d}", doc_id="D", start=start, end=end, text=TEXT[start:end])
    for i, (start, end) in enumerate([(0, 4), (5, 9), (10, 14)])
]


@pytest.mark.parametrize(
    ("spans", "relevant"),
    [
        (((5, 9),), {"D::0001"}),
        (((2, 7),), {"D::0000", "D::0001"}),
        (((3, 5),), {"D::0000"}),
        (((0, 4), (10, 14)), {"D::0000", "D::0002"}),
        ((), set()),
    ],
    ids=["inside-one", "crossing-two", "ends-at-next-start", "two-spans", "absent-clause"],
)
def test_relevant_chunks_are_those_overlapping_a_gold_span(
    spans: tuple[tuple[int, int], ...], relevant: set[str]
) -> None:
    [judgment] = build_judgments([Annotation("D", "Term", spans)], reversed(CHUNKS))

    assert judgment.query_id == "D__Term"
    assert judgment.relevant == relevant
    assert judgment.answerable is bool(relevant)


@pytest.mark.parametrize(
    "annotation",
    [Annotation("D", "Term", ((4, 5),)), Annotation("E", "Term", ((0, 1),))],
    ids=["whitespace-gap", "document-without-chunks"],
)
def test_gold_span_outside_every_chunk_is_an_error(annotation: Annotation) -> None:
    with pytest.raises(ValueError, match="overlaps no chunk"):
        build_judgments([annotation], CHUNKS)


def test_load_split_reads_exact_ids(tmp_path: Path) -> None:
    path = tmp_path / "test.txt"
    path.write_bytes("A-EX-10\nCafe\N{COMBINING ACUTE ACCENT} Agreement\n".encode())

    assert load_split(path) == {"A-EX-10", "Cafe\N{COMBINING ACUTE ACCENT} Agreement"}


@pytest.mark.parametrize(
    ("content", "message"),
    [("A\n\nB\n", "blank line"), ("A\nB\nA\n", "duplicate contract id")],
)
def test_load_split_rejects_malformed_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "split.txt"
    path.write_bytes(content.encode())

    with pytest.raises(ValueError, match=message):
        load_split(path)
