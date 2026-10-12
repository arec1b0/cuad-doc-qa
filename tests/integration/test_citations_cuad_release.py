"""Grounding every expert-annotated CUAD span that lies inside one chunk.

Gold span texts stand in for model quotes: verbatim, whitespace-collapsed, and with one word
shortened by a character. When the same wording occurs more than once in a chunk, the first
occurrence wins, so a few groundings land on a near-identical passage instead of the gold one.
Needs the CUAD archive: downloaded on first run, cached in data/raw.
"""

import random
from bisect import bisect_right
from pathlib import Path

import pytest
from rapidfuzz import fuzz

from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.citations import ground
from cuad_doc_qa.domain import Citation
from cuad_doc_qa.ingestion import download, load_annotations, load_documents

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def _squash(text: str) -> str:
    return " ".join(text.lower().split())


@pytest.mark.network
def test_gold_spans_ground_in_their_chunk() -> None:
    archive = download(DATA_RAW)
    documents = {document.doc_id: document for document in load_documents(archive)}
    chunks = {doc_id: chunk_document(document) for doc_id, document in documents.items()}
    rng = random.Random(0)
    inside = same_span = typo_quotes = typo_overlapping = 0
    elsewhere: list[float] = []

    for annotation in load_annotations(archive):
        doc_chunks = chunks[annotation.doc_id]
        starts = [chunk.start for chunk in doc_chunks]
        text = documents[annotation.doc_id].text
        for start, end in annotation.spans:
            chunk = doc_chunks[bisect_right(starts, start) - 1]
            if end > chunk.end:
                continue
            inside += 1
            gold = text[start:end]
            stripped = (
                start + len(gold) - len(gold.lstrip()),
                end - len(gold) + len(gold.rstrip()),
            )

            for quote in (gold, " ".join(gold.split())):
                grounded = ground(Citation(chunk.chunk_id, quote), chunk)
                assert grounded is not None and not grounded.fuzzy, annotation.query_id
                assert grounded.text == text[grounded.start : grounded.end]
                assert _squash(grounded.text) == _squash(gold), annotation.query_id
            same_span += (grounded.start, grounded.end) == stripped

            words = gold.split()
            if len(" ".join(words)) >= 40:
                index = rng.randrange(len(words))
                words[index] = words[index][:-1] or words[index] + "x"
                typo = ground(Citation(chunk.chunk_id, " ".join(words)), chunk)
                assert typo is not None, annotation.query_id
                typo_quotes += 1
                if typo.start < stripped[1] and stripped[0] < typo.end:
                    typo_overlapping += 1
                else:
                    elsewhere.append(fuzz.ratio(_squash(typo.text), _squash(gold)))

    assert inside == 13_505
    assert same_span == 13_372
    assert (typo_quotes, typo_overlapping) == (9_948, 9_921)
    assert min(elsewhere) >= 90.0
