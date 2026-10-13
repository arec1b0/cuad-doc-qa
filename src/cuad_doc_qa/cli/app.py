"""`cuad-qa` command line: thin wrappers over the library, run from the repository root.

Exit codes: 0 success (an abstention is a success), 1 handled failure (bad data, checksum,
network, missing FTS5, Ollama unreachable or invalid reply, reports not comparable),
2 invalid usage (argparse), 3 retrieval regression found by `gate`.
"""

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from cuad_doc_qa import __version__
from cuad_doc_qa.evaluation.gate import DEFAULT_TOLERANCES, compare, load_report
from cuad_doc_qa.evaluation.runner import (
    RETRIEVERS,
    RetrievalScores,
    run_bm25_baseline,
    run_experiment,
)
from cuad_doc_qa.generation import DEFAULT_BASE_URL, OllamaChat
from cuad_doc_qa.ingestion import ChecksumMismatchError, download
from cuad_doc_qa.pipeline import DEFAULT_GATE, DEFAULT_K, PipelineResult, cuad_session

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
SPLITS_DIR = Path("data/eval/splits")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("artifacts/models")
MODEL_ENV = "CUAD_QA_OLLAMA_MODEL"
EXIT_REGRESSION = 3


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        return args.handler(args)
    except (ChecksumMismatchError, OSError, RuntimeError, ValueError) as error:
        logger.debug("Command failed", exc_info=True)
        print(f"error: {error}", file=sys.stderr)
        return 1


def _download(args: argparse.Namespace) -> int:
    print(f"archive: {download(args.data_dir).as_posix()}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    out = args.out or REPORTS_DIR / f"bm25-baseline-{args.split}.json"
    evaluation = run_bm25_baseline(SPLITS_DIR / f"{args.split}.txt", out, data_dir=args.data_dir)
    _print_scores(out, evaluation.overall)
    return 0


def _experiment(args: argparse.Namespace) -> int:
    size = f"sample{args.sample}-seed{args.seed}" if args.sample else "all"
    out = args.out or REPORTS_DIR / "experiments" / f"dev-{size}-{args.retriever}.json"
    evaluation = run_experiment(
        args.retriever,
        SPLITS_DIR / "dev.txt",
        out,
        sample=args.sample,
        seed=args.seed,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
    )
    _print_scores(out, evaluation.overall)
    return 0


def _print_scores(report: Path, scores: RetrievalScores) -> None:
    print(f"report: {report.as_posix()}")
    print(
        f"n={scores.n} recall@5={scores.recall_at_5:.4f} recall@10={scores.recall_at_10:.4f} "
        f"mrr@10={scores.mrr_at_10:.4f} ndcg@10={scores.ndcg_at_10:.4f}"
    )


def _ask(args: argparse.Namespace) -> int:
    chat = OllamaChat(args.model, base_url=args.ollama_url)
    with cuad_session(chat, chat.num_ctx, args.data_dir, args.gate, args.k) as session:
        doc_id = session.resolve_contract(args.contract)
        if args.category is not None and args.category not in session.questions:
            raise ValueError(f"unknown category {args.category!r}; see CUAD's 41 categories")
        question = session.questions[args.category] if args.category else args.question
        result = session.pipeline.ask(question, doc_id, args.category)
    print(_render(result))
    return 0


def _gate(args: argparse.Namespace) -> int:
    tolerances = dict(args.tolerance) if args.tolerance else DEFAULT_TOLERANCES
    checks = compare(load_report(args.baseline), load_report(args.candidate), tolerances)
    for check in checks:
        verdict = "REGRESSION" if check.regressed else "ok"
        print(
            f"{check.metric}: {check.baseline:.4f} -> {check.candidate:.4f} "
            f"(drop {check.drop:+.4f}, tolerance {check.tolerance:.4f}) {verdict}"
        )
    return EXIT_REGRESSION if any(check.regressed for check in checks) else 0


def _tolerance(value: str) -> tuple[str, float]:
    metric, separator, number = value.partition("=")
    try:
        if not separator or not metric:
            raise ValueError
        return metric, float(number)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected METRIC=VALUE, got {value!r}") from None


def _render(result: PipelineResult) -> str:
    lines = [f"contract: {result.doc_id}", f"question: {result.question}"]
    if result.decision.abstain:
        lines.append(f"decision: abstain ({result.decision.reason})")
    elif result.answer is not None and result.check is not None:
        lines.append("decision: answer")
        lines.append(f"answer: {result.answer.text}")
        for number, cited in enumerate(result.check.grounded, start=1):
            match = f"fuzzy {cited.score:.1f}" if cited.fuzzy else "verbatim"
            lines.append(f"[{number}] chars {cited.start}-{cited.end} ({match}): {cited.text}")
    timings = ", ".join(f"{step} {value:.2f}s" for step, value in result.seconds.items())
    lines.append(f"time: {timings}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cuad-qa", description="Document Q&A with verifiable citations over CUAD."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    download_cmd = commands.add_parser("download", help="fetch and verify the pinned CUAD archive")
    download_cmd.add_argument("--data-dir", type=Path, default=DATA_DIR)
    download_cmd.set_defaults(handler=_download)

    evaluate_cmd = commands.add_parser(
        "evaluate", help="evaluate BM25 retrieval on a frozen split and write the report"
    )
    evaluate_cmd.add_argument("--split", choices=["dev", "test"], required=True)
    evaluate_cmd.add_argument("--data-dir", type=Path, default=DATA_DIR)
    evaluate_cmd.add_argument(
        "--out", type=Path, help="report path (default: reports/bm25-baseline-<split>.json)"
    )
    evaluate_cmd.set_defaults(handler=_evaluate)

    experiment_cmd = commands.add_parser(
        "experiment", help="evaluate one retriever on a seeded sample of dev contracts"
    )
    experiment_cmd.add_argument("--retriever", choices=RETRIEVERS, required=True)
    experiment_cmd.add_argument(
        "--sample", type=int, default=40, help="dev contracts to sample; 0 uses all 408"
    )
    experiment_cmd.add_argument("--seed", type=int, default=0)
    experiment_cmd.add_argument("--data-dir", type=Path, default=DATA_DIR)
    experiment_cmd.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    experiment_cmd.add_argument(
        "--out",
        type=Path,
        help="report path (default: reports/experiments/dev-sample<N>-seed<S>-<retriever>.json)",
    )
    experiment_cmd.set_defaults(handler=_experiment)

    ask_cmd = commands.add_parser(
        "ask", help="answer one question about one contract with verified citations"
    )
    ask_cmd.add_argument("--contract", required=True, help="contract id or a unique prefix of it")
    query = ask_cmd.add_mutually_exclusive_group(required=True)
    query.add_argument(
        "--category", help="CUAD clause category: asks CUAD's question and applies the gate"
    )
    query.add_argument("--question", help="free-form question; the gate does not apply")
    model = os.environ.get(MODEL_ENV)
    ask_cmd.add_argument(
        "--model", default=model, required=model is None, help=f"Ollama model (env {MODEL_ENV})"
    )
    ask_cmd.add_argument("--ollama-url", default=DEFAULT_BASE_URL)
    ask_cmd.add_argument("--k", type=int, default=DEFAULT_K, help="chunks given to the model")
    ask_cmd.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    ask_cmd.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ask_cmd.set_defaults(handler=_ask)

    gate_cmd = commands.add_parser(
        "gate", help="fail (exit 3) when a candidate report regresses against a baseline"
    )
    gate_cmd.add_argument("--baseline", type=Path, required=True)
    gate_cmd.add_argument("--candidate", type=Path, required=True)
    gate_cmd.add_argument(
        "--tolerance",
        type=_tolerance,
        action="append",
        metavar="METRIC=VALUE",
        help="largest allowed drop; repeatable; replaces the default recall@10=0.02",
    )
    gate_cmd.set_defaults(handler=_gate)
    return parser
