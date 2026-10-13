"""sample_contracts and run_experiment's input guards."""

from pathlib import Path

import pytest

from cuad_doc_qa.evaluation.runner import run_experiment, sample_contracts

CONTRACTS = frozenset(f"C{i:03d}" for i in range(50))


def test_sample_is_reproducible_and_drawn_from_sorted_ids() -> None:
    first = sample_contracts(CONTRACTS, 5, seed=0)

    assert first == sample_contracts(frozenset(reversed(sorted(CONTRACTS))), 5, seed=0)
    assert len(first) == 5 and first <= CONTRACTS
    assert sample_contracts(CONTRACTS, 5, seed=1) != first


def test_zero_means_every_contract() -> None:
    assert sample_contracts(CONTRACTS, 0, seed=0) == CONTRACTS


@pytest.mark.parametrize("size", [-1, 51])
def test_rejects_impossible_sample_sizes(size: int) -> None:
    with pytest.raises(ValueError, match="sample must be between 0 and 50"):
        sample_contracts(CONTRACTS, size, seed=0)


@pytest.mark.parametrize(
    ("retriever", "split", "message"),
    [
        ("bm42", "dev.txt", "unknown retriever 'bm42'"),
        ("bm25", "test.txt", "experiments run on dev.txt only, got test.txt"),
    ],
)
def test_guards_run_before_any_work(
    tmp_path: Path, retriever: str, split: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_experiment(retriever, tmp_path / split, tmp_path / "r.json", sample=1)
    assert not (tmp_path / "r.json").exists()
