"""Where the committed qwen2.5:7b run loses answerable questions, recomputed from its log.

BM25 is deterministic, so the retrieval of every logged question is replayed over all 510
contracts, as in the run. The replay must reproduce each logged top score; then each answerable
miss is attributed to retrieval (no expert span in the top k chunks) or to a later stage.
docs/FAILURES.md cites the result. Needs CUAD in data/raw.
"""

from collections import Counter
from pathlib import Path

import pytest

from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.evaluation.answers import is_correct, read_question_log
from cuad_doc_qa.ingestion import download, load_annotations, load_documents, load_questions
from cuad_doc_qa.retrieval.lexical import Bm25Index

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "reports" / "answers" / "dev-sample10-seed0-qwen2.5-7b.questions.jsonl"
DOC = (REPO / "docs" / "FAILURES.md").read_text(encoding="utf-8")

EXPECTED = {
    "correct": 56,
    "not_retrieved": 38,
    "low_retrieval_score": 5,
    "not_found": 18,
    "rejected_citation": 4,
    "answer": 5,
}
ROWS = {
    "correct": "| Correct: a shown citation overlaps an expert span | 56 |",
    "not_retrieved": "| No expert span in the 10 retrieved chunks | 38 |",
    "low_retrieval_score": "| Retrieved, withheld by the gate | 5 |",
    "not_found": "| Retrieved, the model answers that the clause is absent | 18 |",
    "rejected_citation": "| Retrieved, a quote cannot be located | 4 |",
    "answer": "| Retrieved, the answer cites other text | 5 |",
}


@pytest.mark.network
def test_answerable_misses_are_attributed_to_retrieval_or_a_later_stage() -> None:
    config, records = read_question_log(LOG)
    assert config is not None
    archive = download(REPO / "data" / "raw")
    questions = load_questions(archive)
    gold = {(a.doc_id, a.category): a.spans for a in load_annotations(archive)}
    chunks = [chunk for document in load_documents(archive) for chunk in chunk_document(document)]
    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    stages: Counter[str] = Counter()
    with Bm25Index.build(chunks) as bm25:
        for record in records.values():
            hits = bm25.search(questions[record["category"]], config["k"], record["doc_id"])
            assert round(hits[0].score, 6) == record["top_score"], record["question_id"]
            if not record["answerable"]:
                continue
            spans = gold[(record["doc_id"], record["category"])]
            retrieved = any(
                by_id[hit.chunk_id].start < end and start < by_id[hit.chunk_id].end
                for hit in hits
                for start, end in spans
            )
            if is_correct(record):
                stages["correct"] += 1
            elif not retrieved:
                stages["not_retrieved"] += 1
            else:
                stages[record["decision"]] += 1

    assert stages == EXPECTED
    for stage, row in ROWS.items():
        assert row in DOC, stage
    answerable = sum(stages.values())
    misses = answerable - stages["correct"]
    later = misses - stages["not_retrieved"]
    assert f"{stages['not_retrieved']} of {misses} misses" in DOC
    assert f"{stages['not_found']} of {misses} misses" in DOC
    assert f"all later stages together ({later})" in DOC
    assert f"at most {stages['correct'] + later} of {answerable}" in DOC
