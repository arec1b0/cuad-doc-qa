"""Retrieval regression gate: compare a candidate report with a baseline report.

Two reports are comparable only if they evaluate the same questions: same split file, same
archive, same k and the same number of answerable questions. Retriever, chunking and code
version may differ; that difference is what the gate judges.

Report metrics are rounded to 4 decimals, so drops are compared at that precision: a drop
exactly equal to the tolerance passes.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TOLERANCES: dict[str, float] = {"recall@10": 0.02}
_MUST_MATCH = ("split", "split_sha256", "archive_sha256", "k")


@dataclass(frozen=True, slots=True)
class MetricCheck:
    metric: str
    baseline: float
    candidate: float
    tolerance: float

    @property
    def drop(self) -> float:
        return round(self.baseline - self.candidate, 4)

    @property
    def regressed(self) -> bool:
        return self.drop > self.tolerance


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    tolerances: Mapping[str, float] = DEFAULT_TOLERANCES,
) -> list[MetricCheck]:
    """One check per gated metric, in tolerance order.

    Raises ValueError if the reports are not comparable or a gated metric is missing.
    """
    for field in _MUST_MATCH:
        expected, actual = baseline["provenance"].get(field), candidate["provenance"].get(field)
        if expected != actual:
            raise ValueError(f"reports are not comparable: {field} is {expected!r} vs {actual!r}")
    if baseline["metrics"]["n"] != candidate["metrics"]["n"]:
        raise ValueError(
            "reports are not comparable: "
            f"n is {baseline['metrics']['n']} vs {candidate['metrics']['n']}"
        )

    checks: list[MetricCheck] = []
    for metric, tolerance in tolerances.items():
        if tolerance < 0:
            raise ValueError(f"{metric}: tolerance must be non-negative, got {tolerance}")
        if metric not in baseline["metrics"] or metric not in candidate["metrics"]:
            raise ValueError(f"metric {metric!r} is not in both reports")
        checks.append(
            MetricCheck(
                metric=metric,
                baseline=float(baseline["metrics"][metric]),
                candidate=float(candidate["metrics"][metric]),
                tolerance=tolerance,
            )
        )
    return checks
