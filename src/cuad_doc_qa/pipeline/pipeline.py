"""Answer one question about one contract: retrieve, gate, generate, ground, decide."""

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from cuad_doc_qa.abstention import Decision, Reason, ScoreGate, decide
from cuad_doc_qa.citations import CitationCheck, check_citations
from cuad_doc_qa.domain import Answer, Chunk, Search, SearchHit
from cuad_doc_qa.generation import Generator

logger = logging.getLogger(__name__)

DEFAULT_K = 10


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """`answer` and `check` are None when the retrieval gate withheld the question."""

    question: str
    doc_id: str
    category: str | None
    retrieved: tuple[SearchHit, ...]
    answer: Answer | None
    check: CitationCheck | None
    decision: Decision
    seconds: dict[str, float] = field(default_factory=dict)


class Pipeline:
    """Every step is injected, so retrieval, gate and model can be swapped and faked in tests."""

    def __init__(
        self,
        search: Search,
        chunks: Iterable[Chunk],
        generator: Generator,
        gate: ScoreGate,
        k: int = DEFAULT_K,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        self._search = search
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._generator = generator
        self._gate = gate
        self._k = k

    def ask(self, question: str, doc_id: str, category: str | None = None) -> PipelineResult:
        """`category` names the CUAD clause the question asks about; it enables the gate."""
        started = time.perf_counter()
        hits = tuple(self._search(question, self._k, doc_id))
        retrieved_at = time.perf_counter()
        seconds = {"retrieval": retrieved_at - started}

        top_score = hits[0].score if hits else None
        if not self._gate.passes(category, top_score):
            result = PipelineResult(
                question=question,
                doc_id=doc_id,
                category=category,
                retrieved=hits,
                answer=None,
                check=None,
                decision=Decision(Reason.LOW_RETRIEVAL_SCORE),
                seconds=seconds | {"total": retrieved_at - started},
            )
            self._log(result, top_score)
            return result

        context = [self._chunks[hit.chunk_id] for hit in hits]
        answer = self._generator.answer(question, context)
        generated_at = time.perf_counter()
        check = check_citations(answer, context)
        finished = time.perf_counter()
        seconds |= {
            "generation": generated_at - retrieved_at,
            "grounding": finished - generated_at,
            "total": finished - started,
        }
        result = PipelineResult(
            question=question,
            doc_id=doc_id,
            category=category,
            retrieved=hits,
            answer=answer,
            check=check,
            decision=decide(answer, check),
            seconds=seconds,
        )
        self._log(result, top_score)
        return result

    @staticmethod
    def _log(result: PipelineResult, top_score: float | None) -> None:
        logger.info(
            "doc=%r category=%r top_score=%s decision=%s total=%.2fs",
            result.doc_id,
            result.category,
            f"{top_score:.4f}" if top_score is not None else None,
            result.decision.reason or "answer",
            result.seconds["total"],
        )
