"""Score a retriever against relevance judgments, overall and per clause category."""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean

from cuad_doc_qa.domain import SearchHit
from cuad_doc_qa.evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from cuad_doc_qa.evaluation.qrels import Judgment

K = 10

Search = Callable[[str, int, str | None], list[SearchHit]]


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """Mean metrics over `n` answerable questions."""

    n: int
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float

    def as_dict(self) -> dict[str, float | int]:
        """Report form: metric names as in the literature, values rounded to 4 decimals."""
        return {
            "n": self.n,
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr@10": round(self.mrr_at_10, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
        }


@dataclass(frozen=True, slots=True)
class RetrievalEvaluation:
    overall: RetrievalScores
    per_category: dict[str, RetrievalScores]
    unanswerable_skipped: int


def evaluate_retrieval(
    search: Search, judgments: Iterable[Judgment], questions: Mapping[str, str]
) -> RetrievalEvaluation:
    """Run each answerable question through `search` and score the top `K` hits.

    Each question is searched only inside its own contract: CUAD questions ask about
    "this contract" and never name it, so corpus-wide retrieval would be ill-posed.
    Unanswerable questions are counted and skipped; they are scored as abstention, not here.
    """
    per_query: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    skipped = 0
    for judgment in judgments:
        if not judgment.answerable:
            skipped += 1
            continue
        hits = search(questions[judgment.category], K, judgment.doc_id)
        ranking = [hit.chunk_id for hit in hits]
        relevant = judgment.relevant
        per_query[judgment.category].append(
            (
                recall_at_k(ranking, relevant, 5),
                recall_at_k(ranking, relevant, 10),
                reciprocal_rank(ranking, relevant),
                ndcg_at_k(ranking, relevant, 10),
            )
        )

    if not per_query:
        raise ValueError("no answerable judgments to evaluate")
    every_query = [row for rows in per_query.values() for row in rows]
    return RetrievalEvaluation(
        overall=_mean(every_query),
        per_category={category: _mean(rows) for category, rows in sorted(per_query.items())},
        unanswerable_skipped=skipped,
    )


def _mean(rows: list[tuple[float, float, float, float]]) -> RetrievalScores:
    recall_5, recall_10, rr, ndcg = (fmean(column) for column in zip(*rows, strict=True))
    return RetrievalScores(
        n=len(rows), recall_at_5=recall_5, recall_at_10=recall_10, mrr_at_10=rr, ndcg_at_10=ndcg
    )
