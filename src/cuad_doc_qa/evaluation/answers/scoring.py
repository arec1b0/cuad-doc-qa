"""Score end-to-end answers against CUAD's expert spans.

CUAD answers are spans of the contract, so correctness is judged through citations: a shown
answer is correct when at least one of its grounded citations overlaps an expert span of its
question. The answer's prose is not judged.

A record is a JSON-ready dict, one per question:
  question_id, doc_id, category, answerable, gold_spans,
  decision        "answer", an abstention reason, or "error"
  top_score       best retrieval score, or None
  claimed, grounded, fuzzy, rejected    citation counts of the generated answer
  citations       shown citations: start, end, fuzzy, score, overlaps_gold, text
  gold_spans_hit  expert spans overlapped by a shown citation
  answer          the model's answer text, or None
  seconds         stage timings
  error           error message, or None
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import quantiles
from typing import Any

from cuad_doc_qa.pipeline import PipelineResult

Record = dict[str, Any]
ANSWER = "answer"
ERROR = "error"


def score_question(
    question_id: str, result: PipelineResult, gold: Sequence[tuple[int, int]]
) -> Record:
    shown = not result.decision.abstain
    grounded = result.check.grounded if result.check is not None else ()
    citations = [
        {
            "start": cited.start,
            "end": cited.end,
            "fuzzy": cited.fuzzy,
            "score": round(cited.score, 2),
            "overlaps_gold": any(_overlap((cited.start, cited.end), span) for span in gold),
            "text": cited.text,
        }
        for cited in (grounded if shown else ())
    ]
    hit = sum(any(_overlap((c["start"], c["end"]), span) for c in citations) for span in gold)
    return {
        "question_id": question_id,
        "doc_id": result.doc_id,
        "category": result.category,
        "answerable": bool(gold),
        "gold_spans": len(gold),
        "decision": ANSWER if shown else str(result.decision.reason),
        "top_score": round(result.retrieved[0].score, 6) if result.retrieved else None,
        "claimed": len(result.answer.citations) if result.answer is not None else 0,
        "grounded": len(grounded),
        "fuzzy": sum(cited.fuzzy for cited in grounded),
        "rejected": len(result.check.rejected) if result.check is not None else 0,
        "citations": citations,
        "gold_spans_hit": hit,
        "answer": result.answer.text if result.answer is not None else None,
        "seconds": {step: round(value, 3) for step, value in result.seconds.items()},
        "error": None,
    }


def error_record(
    question_id: str, doc_id: str, category: str, gold: Sequence[tuple[int, int]], error: Exception
) -> Record:
    return {
        "question_id": question_id,
        "doc_id": doc_id,
        "category": category,
        "answerable": bool(gold),
        "gold_spans": len(gold),
        "decision": ERROR,
        "top_score": None,
        "claimed": 0,
        "grounded": 0,
        "fuzzy": 0,
        "rejected": 0,
        "citations": [],
        "gold_spans_hit": 0,
        "answer": None,
        "seconds": {},
        "error": f"{type(error).__name__}: {error}",
    }


def is_correct(record: Mapping[str, Any]) -> bool:
    return (
        record["decision"] == ANSWER
        and record["answerable"]
        and any(c["overlaps_gold"] for c in record["citations"])
    )


def aggregate(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Report sections: question counts, outcomes, metrics, grounding, latency, per category."""
    rows = list(records)
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    shown = [r for r in rows if r["decision"] == ANSWER]
    withheld = [r for r in rows if r["decision"] not in (ANSWER, ERROR)]
    correct = [r for r in rows if is_correct(r)]
    shown_citations = [c for r in shown for c in r["citations"]]
    withheld_unanswerable = [r for r in withheld if not r["answerable"]]
    shown_unanswerable = [r for r in shown if not r["answerable"]]
    generated = [r for r in rows if "generation" in r["seconds"]]

    per_category: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts = per_category[row["category"]]
        counts["answerable" if row["answerable"] else "unanswerable"] += 1
        counts["shown"] += row["decision"] == ANSWER
        counts["correct"] += is_correct(row)
        counts["errors"] += row["decision"] == ERROR

    return {
        "questions": {
            "total": len(rows),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
        },
        "outcomes": {
            "answerable": _outcomes(answerable),
            "unanswerable": _outcomes(unanswerable),
        },
        "metrics": {
            "answer_recall": _ratio(len(correct), len(answerable)),
            "answer_precision": _ratio(len(correct), len(shown)),
            "false_answer_rate": _ratio(len(shown_unanswerable), len(unanswerable)),
            "abstention_recall": _ratio(len(withheld_unanswerable), len(unanswerable)),
            "abstention_precision": _ratio(len(withheld_unanswerable), len(withheld)),
            "citation_precision": _ratio(
                sum(c["overlaps_gold"] for c in shown_citations), len(shown_citations)
            ),
            "gold_span_recall": _ratio(
                sum(r["gold_spans_hit"] for r in answerable),
                sum(r["gold_spans"] for r in answerable),
            ),
            "error_rate": _ratio(sum(r["decision"] == ERROR for r in rows), len(rows)),
        },
        "grounding": {
            key: sum(r[key] for r in rows) for key in ("claimed", "grounded", "fuzzy", "rejected")
        },
        "latency_seconds": _latency([r["seconds"]["total"] for r in generated]),
        "per_category": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(per_category.items())
        },
    }


def _outcomes(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(row["decision"] for row in rows)
    counts["correct"] = sum(is_correct(row) for row in rows)
    return dict(sorted(counts.items()))


def _latency(values: Sequence[float]) -> dict[str, float | None]:
    if len(values) < 2:
        only = round(values[0], 2) if values else None
        return {"p50": only, "p95": only}
    cuts = quantiles(values, n=20, method="inclusive")
    return {"p50": round(cuts[9], 2), "p95": round(cuts[18], 2)}


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]
