"""load_annotations and load_official_split: CUAD labels and split membership."""

from collections.abc import Callable
from pathlib import Path

import pytest

from cuad_doc_qa.domain import Annotation
from cuad_doc_qa.ingestion import load_annotations, load_official_split

ArchiveFactory = Callable[..., Path]

DOC = "A-EX-10-LICENSE AGREEMENT"
CONTEXT = "Governing law: Delaware. Term: five years."


def _qa(category: str, *answers: tuple[str, int], impossible: bool | None = None) -> dict:
    return {
        "id": f"{DOC}__{category}",
        "question": f'Highlight the parts (if any) of this contract related to "{category}"',
        "answers": [{"text": text, "answer_start": start} for text, start in answers],
        "is_impossible": not answers if impossible is None else impossible,
    }


def _entry(*qas: dict) -> dict:
    return {"title": DOC, "paragraphs": [{"context": CONTEXT, "qas": list(qas)}]}


def test_parses_present_and_absent_clauses(make_cuad_archive: ArchiveFactory) -> None:
    archive = make_cuad_archive(
        [
            _entry(
                _qa("Governing Law", ("Delaware", 15), ("Delaware", 15)),
                _qa("Term", ("five years", 31), ("Delaware", 15)),
                _qa("Non-Compete"),
            )
        ]
    )

    annotations = load_annotations(archive)

    assert annotations == [
        Annotation(doc_id=DOC, category="Governing Law", spans=((15, 23),)),
        Annotation(doc_id=DOC, category="Term", spans=((15, 23), (31, 41))),
        Annotation(doc_id=DOC, category="Non-Compete", spans=()),
    ]
    assert annotations[0].query_id == f"{DOC}__Governing Law"


@pytest.mark.parametrize(
    ("qa", "message"),
    [
        (_qa("Governing Law", ("Delaware", 14)), "answer does not match"),
        (_qa("Governing Law", ("", 3)), "answer does not match"),
        (_qa("Governing Law", ("Delaware", 15), impossible=True), "is_impossible disagrees"),
        ({**_qa("Term"), "id": "OTHER CONTRACT__Term"}, "question id is not"),
        ({**_qa("Term"), "id": f"{DOC}__"}, "question id is not"),
    ],
    ids=["offset-mismatch", "empty-answer", "impossible-flag", "foreign-id", "missing-category"],
)
def test_rejects_inconsistent_labels(
    make_cuad_archive: ArchiveFactory, qa: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_annotations(make_cuad_archive([_entry(qa)]))


def test_official_split_lists_train_and_test_titles(make_cuad_archive: ArchiveFactory) -> None:
    archive = make_cuad_archive([], train=("A", "B"), test=("C",))

    assert load_official_split(archive) == {
        "train": frozenset({"A", "B"}),
        "test": frozenset({"C"}),
    }
