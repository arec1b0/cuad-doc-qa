"""Retrieval experiments: one report per retriever on a seeded sample of dev contracts.

Only the dev split is accepted, so experiments can never tune on test. Every retriever sees the
same questions: the sample is drawn from the sorted contract ids with `random.Random(seed)`.

BM25 indexes all 510 contracts, as in the baseline. Dense embeds only the sampled contracts:
cosine similarity does not depend on other documents and search is scoped to one contract.

BM25 reports are byte-reproducible. Dense and rerank scores come from ONNX Runtime, whose CPU
kernels may differ in the last float bits across machines, so a rare tie can rank differently.
"""

import hashlib
import logging
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from cuad_doc_qa import __version__
from cuad_doc_qa.chunking import DEFAULT_MAX_CHARS, chunk_document
from cuad_doc_qa.domain import Chunk, Search
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
from cuad_doc_qa.retrieval.dense import DenseIndex, FastEmbedEmbedder
from cuad_doc_qa.retrieval.fusion import DEFAULT_DEPTH as FUSION_DEPTH
from cuad_doc_qa.retrieval.fusion import RRF_K, FusedSearch
from cuad_doc_qa.retrieval.lead import DEFAULT_LEAD, LEAD_CATEGORIES, LeadChunkSearch
from cuad_doc_qa.retrieval.lexical import Bm25Index
from cuad_doc_qa.retrieval.rerank import DEFAULT_DEPTH as RERANK_DEPTH
from cuad_doc_qa.retrieval.rerank import FastEmbedCrossEncoder, RerankedSearch

logger = logging.getLogger(__name__)

RETRIEVERS = ("bm25", "bm25-lead", "dense", "hybrid", "rerank")
DEV_SPLIT = "dev.txt"


def sample_contracts(contracts: frozenset[str], size: int, seed: int) -> frozenset[str]:
    """`size` contracts drawn reproducibly; 0 means all of them."""
    if size == 0:
        return contracts
    if not 0 < size <= len(contracts):
        raise ValueError(f"sample must be between 0 and {len(contracts)}, got {size}")
    return frozenset(random.Random(seed).sample(sorted(contracts), size))


def run_experiment(
    retriever: str,
    split_path: Path,
    report_path: Path,
    sample: int,
    seed: int = 0,
    data_dir: Path = Path("data/raw"),
    models_dir: Path = Path("artifacts/models"),
) -> RetrievalEvaluation:
    """Evaluate one retriever on the sampled dev contracts and write its report."""
    if retriever not in RETRIEVERS:
        raise ValueError(f"unknown retriever {retriever!r}; expected one of {RETRIEVERS}")
    if split_path.name != DEV_SPLIT:
        raise ValueError(f"experiments run on {DEV_SPLIT} only, got {split_path.name}")

    started = time.perf_counter()
    archive = download(data_dir)
    contracts = sample_contracts(load_split(split_path), sample, seed)
    chunks = [chunk for document in load_documents(archive) for chunk in chunk_document(document)]
    judgments = [
        judgment
        for judgment in build_judgments(load_annotations(archive), chunks)
        if judgment.doc_id in contracts
    ]

    questions = load_questions(archive)
    with _retriever(retriever, chunks, contracts, questions, models_dir) as (search, description):
        evaluation = evaluate_retrieval(search, judgments, questions)

    write_report(
        report_path,
        evaluation,
        provenance={
            "run": f"experiment-{retriever}",
            "package_version": __version__,
            "archive_sha256": CUAD_SHA256,
            "split": split_path.name,
            "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "sample": {"contracts": len(contracts), "seed": seed if sample else None},
            "chunking": {"strategy": "clause", "max_chars": DEFAULT_MAX_CHARS},
            "retriever": description,
            "scope": "contract",
            "queries": "cuad question text, verbatim",
            "k": K,
        },
    )
    logger.info("Wrote %s in %.0fs", report_path, time.perf_counter() - started)
    return evaluation


@contextmanager
def _retriever(
    name: str,
    chunks: Sequence[Chunk],
    contracts: frozenset[str],
    questions: Mapping[str, str],
    models_dir: Path,
) -> Iterator[tuple[Search, dict[str, Any]]]:
    bm25_description = {"name": "bm25", "index": "sqlite fts5, porter unicode61, 510 contracts"}
    with Bm25Index.build(chunks) as bm25:
        if name == "bm25":
            yield bm25.search, bm25_description
            return
        if name == "bm25-lead":
            queries = [questions[category] for category in LEAD_CATEGORIES]
            yield (
                LeadChunkSearch(bm25.search, chunks, queries, lead=DEFAULT_LEAD),
                {
                    "name": "bm25-lead",
                    "first_stage": bm25_description,
                    "lead": DEFAULT_LEAD,
                    "categories": list(LEAD_CATEGORIES),
                },
            )
            return
        if name == "rerank":
            scorer = FastEmbedCrossEncoder(cache_dir=models_dir)
            yield (
                RerankedSearch(bm25.search, scorer, chunks, depth=RERANK_DEPTH),
                {
                    "name": "rerank",
                    "first_stage": bm25_description,
                    "depth": RERANK_DEPTH,
                    "model": _model(TextCrossEncoder, FastEmbedCrossEncoder.MODEL, models_dir),
                },
            )
            return

        sampled = [chunk for chunk in chunks if chunk.doc_id in contracts]
        dense = DenseIndex.build(sampled, FastEmbedEmbedder(cache_dir=models_dir))
        dense_description = {
            "name": "dense",
            "similarity": "cosine, exact",
            "model": _model(TextEmbedding, FastEmbedEmbedder.MODEL, models_dir),
        }
        if name == "dense":
            yield dense.search, dense_description
            return
        yield (
            FusedSearch([bm25.search, dense.search], depth=FUSION_DEPTH, rrf_k=RRF_K),
            {
                "name": "hybrid",
                "fusion": "reciprocal rank",
                "depth": FUSION_DEPTH,
                "rrf_k": RRF_K,
                "parts": [bm25_description, dense_description],
            },
        )


def _model(registry: Any, name: str, models_dir: Path) -> dict[str, str | None]:
    """Model name, the Hugging Face repo fastembed downloads it from, and the cached revision."""
    [entry] = [model for model in registry.list_supported_models() if model["model"] == name]
    repo = entry["sources"]["hf"]
    ref = models_dir / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    return {
        "name": name,
        "source": repo,
        "revision": ref.read_text(encoding="utf-8").strip() if ref.exists() else None,
        "runtime": f"fastembed {version('fastembed')}",
    }
