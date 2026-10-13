"""`cuad-qa` command line: thin wrappers over the library, run from the repository root.

Exit codes: 0 success (an abstention is a success), 1 handled failure (bad data, checksum,
network, missing FTS5, Ollama unreachable or invalid reply), 2 invalid usage (argparse).
"""

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from cuad_doc_qa import __version__
from cuad_doc_qa.evaluation.runner import run_bm25_baseline
from cuad_doc_qa.generation import DEFAULT_BASE_URL, OllamaChat
from cuad_doc_qa.ingestion import ChecksumMismatchError, download
from cuad_doc_qa.pipeline import DEFAULT_GATE, DEFAULT_K, PipelineResult, cuad_session

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
SPLITS_DIR = Path("data/eval/splits")
REPORTS_DIR = Path("reports")
MODEL_ENV = "CUAD_QA_OLLAMA_MODEL"


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
    scores = evaluation.overall
    print(f"report: {out.as_posix()}")
    print(
        f"n={scores.n} recall@5={scores.recall_at_5:.4f} recall@10={scores.recall_at_10:.4f} "
        f"mrr@10={scores.mrr_at_10:.4f} ndcg@10={scores.ndcg_at_10:.4f}"
    )
    return 0


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
    return parser
