"""`cuad-qa` command line: thin wrappers over the library, run from the repository root.

Exit codes: 0 success, 1 handled failure (bad data, checksum, network, missing FTS5),
2 invalid usage (argparse).
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from cuad_doc_qa import __version__
from cuad_doc_qa.evaluation.runner import run_bm25_baseline
from cuad_doc_qa.ingestion import ChecksumMismatchError, download

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/raw")
SPLITS_DIR = Path("data/eval/splits")
REPORTS_DIR = Path("reports")


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
    return parser
