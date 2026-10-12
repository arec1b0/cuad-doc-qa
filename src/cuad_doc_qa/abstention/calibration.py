"""Calibrate the BM25 score gate on one split and store it with its provenance.

The index covers all 510 contracts, exactly as in the BM25 baseline, because BM25 scores depend
on corpus statistics: thresholds are only valid for the index they were measured on.
"""

import hashlib
import logging
import time
from pathlib import Path

from cuad_doc_qa import __version__
from cuad_doc_qa.abstention.gate import (
    MIN_ANSWERABLE,
    TARGET_ANSWERABLE_RECALL,
    Sample,
    ScoreGate,
    calibrate,
    save_gate,
)
from cuad_doc_qa.chunking import DEFAULT_MAX_CHARS, chunk_document
from cuad_doc_qa.evaluation.qrels import build_judgments, load_split
from cuad_doc_qa.ingestion import (
    CUAD_SHA256,
    download,
    load_annotations,
    load_documents,
    load_questions,
)
from cuad_doc_qa.retrieval.lexical import Bm25Index

logger = logging.getLogger(__name__)


def calibrate_bm25_gate(
    split_path: Path,
    out_path: Path,
    data_dir: Path = Path("data/raw"),
    target: float = TARGET_ANSWERABLE_RECALL,
    min_answerable: int = MIN_ANSWERABLE,
) -> ScoreGate:
    """Score every question of the split (answerable or not) and write the gate to `out_path`."""
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

    samples: list[Sample] = []
    with Bm25Index.build(chunks) as index:
        for judgment in judgments:
            hits = index.search(questions[judgment.category], 1, judgment.doc_id)
            top_score = hits[0].score if hits else None
            samples.append(Sample(judgment.category, judgment.answerable, top_score))

    gate = calibrate(samples, target, min_answerable)
    save_gate(
        out_path,
        gate,
        provenance={
            "run": "bm25-gate",
            "package_version": __version__,
            "archive_sha256": CUAD_SHA256,
            "split": split_path.name,
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "chunking": {"strategy": "clause", "max_chars": DEFAULT_MAX_CHARS},
            "retriever": "bm25, sqlite fts5, porter unicode61, all 510 contracts",
            "scope": "contract",
            "queries": "cuad question text, verbatim",
            "target_answerable_recall": target,
            "min_answerable": min_answerable,
            "questions": len(samples),
        },
    )
    logger.info(
        "Calibrated %d of %d categories on %d questions in %.0fs",
        len(gate.thresholds),
        len(questions),
        len(samples),
        time.perf_counter() - started,
    )
    return gate
