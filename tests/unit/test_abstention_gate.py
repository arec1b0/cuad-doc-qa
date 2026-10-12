"""ScoreGate: calibration, the pass rule, and storage."""

from pathlib import Path

import pytest

from cuad_doc_qa.abstention import Sample, ScoreGate, calibrate, load_gate, save_gate


def _samples(category: str, answerable: bool, scores: list[float | None]) -> list[Sample]:
    return [Sample(category, answerable, score) for score in scores]


def test_threshold_keeps_the_target_share_of_answerable_questions() -> None:
    samples = (
        _samples("Governing Law", True, [float(s) for s in range(20, 0, -1)])
        + _samples("Governing Law", False, [0.5, 30.0])
        + _samples("Non-Compete", True, [1.0, 2.0, 3.0])
    )

    gate = calibrate(samples, target=0.95, min_answerable=20)

    assert gate.thresholds == {"Governing Law": 2.0}
    kept = sum(gate.passes("Governing Law", float(s)) for s in range(1, 21))
    assert kept == 19


def test_missing_retrieval_counts_as_a_miss_and_never_sets_the_threshold() -> None:
    samples = _samples("Term", True, [None, None, 5.0, 6.0])

    assert calibrate(samples, target=0.5, min_answerable=1).thresholds == {"Term": 5.0}
    assert calibrate(samples, target=1.0, min_answerable=1).thresholds == {}


def test_thresholds_are_floored_to_six_decimals() -> None:
    gate = calibrate(_samples("Term", True, [1.23456789]), target=1.0, min_answerable=1)

    assert gate.thresholds == {"Term": 1.234567}
    assert gate.passes("Term", 1.23456789)


def test_rejects_an_invalid_target() -> None:
    with pytest.raises(ValueError, match="target must be in"):
        calibrate([], target=0.0)


@pytest.mark.parametrize(
    ("category", "score", "passes"),
    [
        ("Governing Law", 2.0, True),
        ("Governing Law", 1.999999, False),
        ("Governing Law", None, False),
        ("Ungated Category", 0.0, True),
        (None, 0.0, True),
        (None, None, False),
    ],
    ids=["at-threshold", "below", "no-hits", "ungated-category", "free-form", "free-form-no-hits"],
)
def test_pass_rule(category: str | None, score: float | None, passes: bool) -> None:
    assert ScoreGate({"Governing Law": 2.0}).passes(category, score) is passes


def test_storage_round_trip_is_deterministic(tmp_path: Path) -> None:
    gate = ScoreGate({"Term": 1.5, "Governing Law": 2.0})
    first, second = tmp_path / "a" / "gate.json", tmp_path / "b" / "gate.json"

    save_gate(first, gate, {"split": "dev.txt", "target_answerable_recall": 0.95})
    save_gate(second, ScoreGate(dict(reversed(list(gate.thresholds.items())))), {
        "target_answerable_recall": 0.95, "split": "dev.txt"})  # fmt: skip

    assert first.read_bytes() == second.read_bytes()
    assert b"\r" not in first.read_bytes()
    assert load_gate(first) == gate
