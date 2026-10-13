"""answer_eval on one real dev contract: real BM25, committed gate, real gold spans, fake model.

The fake model quotes the first 80 characters of the first source, so the test checks that every
CUAD question is asked, logged and counted once, not answer quality. Needs CUAD in data/raw.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from cuad_doc_qa.evaluation.answers import answer_eval
from cuad_doc_qa.ingestion import download, load_annotations
from cuad_doc_qa.pipeline import DEFAULT_GATE, cuad_session

REPO = Path(__file__).resolve().parents[2]
DATA_RAW = REPO / "data" / "raw"
CONTRACT = "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"


class QuotingChat:
    def complete(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        first_source = re.search(r"\[S1\]\n(.*)", messages[1]["content"], re.DOTALL)
        assert first_source is not None
        quote = first_source.group(1)[:80]
        return json.dumps(
            {"found": True, "answer": "Quoted.", "citations": [{"source": "S1", "quote": quote}]}
        )


@pytest.mark.network
def test_every_question_of_the_contract_is_scored_once(tmp_path: Path) -> None:
    gold = {
        (annotation.doc_id, annotation.category): annotation.spans
        for annotation in load_annotations(download(DATA_RAW))
    }
    report_path = tmp_path / "answers.json"

    with cuad_session(QuotingChat(), 8192, DATA_RAW, REPO / DEFAULT_GATE) as session:
        report = answer_eval(session, gold, [CONTRACT], report_path, {"run": "test"})

    expected_answerable = sum(1 for (doc, _), spans in gold.items() if doc == CONTRACT and spans)
    assert report["questions"] == {
        "total": 41,
        "answerable": expected_answerable,
        "unanswerable": 41 - expected_answerable,
    }
    outcomes = report["outcomes"]
    counted = sum(n for o in outcomes.values() for key, n in o.items() if key != "correct")
    assert counted == 41
    assert report["metrics"]["error_rate"] == 0.0
    assert report["grounding"]["rejected"] == 0
