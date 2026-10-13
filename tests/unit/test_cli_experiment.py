"""cuad-qa experiment: argument handling and default report paths. The runner is faked."""

from pathlib import Path
from typing import Any

import pytest

from cuad_doc_qa.cli import app, main
from cuad_doc_qa.evaluation.runner import RetrievalEvaluation, RetrievalScores

SCORES = RetrievalScores(n=4, recall_at_5=0.25, recall_at_10=0.5, mrr_at_10=0.125, ndcg_at_10=0.3)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    recorded: list[dict[str, Any]] = []

    def fake(retriever: str, split_path: Path, report_path: Path, **kwargs: Any) -> Any:
        recorded.append({"retriever": retriever, "split": split_path, "out": report_path, **kwargs})
        return RetrievalEvaluation(overall=SCORES, per_category={}, unanswerable_skipped=0)

    monkeypatch.setattr(app, "run_experiment", fake)
    return recorded


def test_defaults_sample_40_contracts_with_seed_0(
    calls: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["experiment", "--retriever", "hybrid"]) == 0

    assert calls == [
        {
            "retriever": "hybrid",
            "split": Path("data/eval/splits/dev.txt"),
            "out": Path("reports/experiments/dev-sample40-seed0-hybrid.json"),
            "sample": 40,
            "seed": 0,
            "data_dir": Path("data/raw"),
            "models_dir": Path("artifacts/models"),
        }
    ]
    assert capsys.readouterr().out == (
        "report: reports/experiments/dev-sample40-seed0-hybrid.json\n"
        "n=4 recall@5=0.2500 recall@10=0.5000 mrr@10=0.1250 ndcg@10=0.3000\n"
    )


def test_sample_zero_names_the_report_after_the_whole_split(calls: list[dict[str, Any]]) -> None:
    main(["experiment", "--retriever", "rerank", "--sample", "0", "--seed", "7"])

    assert calls[0]["out"] == Path("reports/experiments/dev-all-rerank.json")
    assert (calls[0]["sample"], calls[0]["seed"]) == (0, 7)


def test_unknown_retriever_exits_2(calls: list[dict[str, Any]]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["experiment", "--retriever", "splade"])

    assert exit_info.value.code == 2
    assert calls == []
