"""Retrieval score gate: abstain before generation when the best chunk scores too low.

Thresholds are per clause category. Raw BM25 scores grow with query length, and every CUAD
category asks one fixed question, so scores compare across contracts within a category but not
across categories (measured on dev: pooled AUROC 0.600, per-category median 0.789).

A category's threshold is the highest score that still keeps `target` of its answerable dev
questions. Categories with fewer than `min_answerable` answerable dev questions, and questions
without a category, are never gated.
"""

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

TARGET_ANSWERABLE_RECALL = 0.95
MIN_ANSWERABLE = 20
_DECIMALS = 6


@dataclass(frozen=True, slots=True)
class Sample:
    category: str
    answerable: bool
    top_score: float | None
    """Best retrieval score for the question; None when retrieval returned nothing."""


@dataclass(frozen=True, slots=True)
class ScoreGate:
    thresholds: Mapping[str, float]

    def passes(self, category: str | None, top_score: float | None) -> bool:
        """False when retrieval found nothing or the category's threshold is not reached."""
        if top_score is None:
            return False
        threshold = self.thresholds.get(category) if category is not None else None
        return threshold is None or top_score >= threshold


def calibrate(
    samples: Iterable[Sample],
    target: float = TARGET_ANSWERABLE_RECALL,
    min_answerable: int = MIN_ANSWERABLE,
) -> ScoreGate:
    """Per-category thresholds from labelled samples; floored to 6 decimals so storing them
    never turns a passing score into a failing one."""
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target must be in (0, 1], got {target}")
    scores: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        if sample.answerable:
            scores[sample.category].append(
                sample.top_score if sample.top_score is not None else -math.inf
            )

    thresholds: dict[str, float] = {}
    for category, values in scores.items():
        if len(values) < min_answerable:
            continue
        values.sort()
        threshold = values[math.floor((1.0 - target) * len(values))]
        if math.isfinite(threshold):
            thresholds[category] = math.floor(threshold * 10**_DECIMALS) / 10**_DECIMALS
    return ScoreGate(thresholds=dict(sorted(thresholds.items())))


def save_gate(path: Path, gate: ScoreGate, provenance: Mapping[str, object]) -> None:
    """Deterministic JSON: sorted keys, LF line endings, no timestamps."""
    payload = {"provenance": dict(provenance), "thresholds": dict(gate.thresholds)}
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def load_gate(path: Path) -> ScoreGate:
    thresholds = json.loads(path.read_text(encoding="utf-8"))["thresholds"]
    return ScoreGate(thresholds={str(k): float(v) for k, v in thresholds.items()})
