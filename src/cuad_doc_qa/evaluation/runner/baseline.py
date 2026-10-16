"""BM25 baseline: the full pipeline from the pinned CUAD archive to a retrieval report.

The index always covers all 510 contracts, so BM25 corpus statistics do not depend on
which split is evaluated; only the questions are restricted to the split. With `lead`, the
questions of LEAD_CATEGORIES get their contract's first chunks ahead of BM25's hits.
"""

import hashlib
import logging
import sqlite3
import time
from pathlib import Path

from cuad_doc_qa import __version__
from cuad_doc_qa.chunking import DEFAULT_MAX_CHARS, chunk_document
from cuad_doc_qa.evaluation.qrels import build_judgments, load_split
from cuad_doc_qa.evaluation.runner.report import write_report
from cuad_doc_qa.evaluation.runner.retrieval import K, RetrievalEvaluation, evaluate_retrieval
from cuad_doc_qa.ingestion import (
    CUAD_SHA256,
    download,
    load_annotations,
    load_documents,
    load_questions,
)
from cuad_doc_qa.retrieval.lead import DEFAULT_LEAD, LEAD_CATEGORIES, LeadChunkSearch
from cuad_doc_qa.retrieval.lexical import Bm25Index

logger = logging.getLogger(__name__)


def run_bm25_baseline(
    split_path: Path, report_path: Path, data_dir: Path = Path("data/raw"), lead: bool = False
) -> RetrievalEvaluation:
    """Evaluate BM25 on the questions of one split and write the report to `report_path`.

    With `lead`, LeadChunkSearch puts DEFAULT_LEAD chunks first for LEAD_CATEGORIES; every other
    question is ranked exactly as without it.
    """
    started = time.perf_counter()
    archive = download(data_dir)
    split = load_split(split_path)
    chunks = [chunk for document in load_documents(archive) for chunk in chunk_document(document)]
    judgments = [
        judgment
        for judgment in build_judgments(load_annotations(archive), chunks)
        if judgment.doc_id in split
    ]

    questions = load_questions(archive)
    with Bm25Index.build(chunks) as index:
        search = (
            LeadChunkSearch(
                index.search,
                chunks,
                [questions[category] for category in LEAD_CATEGORIES],
                lead=DEFAULT_LEAD,
            )
            if lead
            else index.search
        )
        evaluation = evaluate_retrieval(search, judgments, questions)

    retriever = "bm25, sqlite fts5, porter unicode61"
    if lead:
        retriever += f"; first {DEFAULT_LEAD} chunks lead for {', '.join(LEAD_CATEGORIES)}"
    write_report(
        report_path,
        evaluation,
        provenance={
            "run": "bm25-lead-baseline" if lead else "bm25-baseline",
            "package_version": __version__,
            "archive_sha256": CUAD_SHA256,
            "split": split_path.name,
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "chunking": {"strategy": "clause", "max_chars": DEFAULT_MAX_CHARS},
            "retriever": retriever,
            "scope": "contract",
            "queries": "cuad question text, verbatim",
            "k": K,
        },
    )
    logger.info(
        "Wrote %s in %.1fs (SQLite %s)",
        report_path,
        time.perf_counter() - started,
        sqlite3.sqlite_version,
    )
    return evaluation
