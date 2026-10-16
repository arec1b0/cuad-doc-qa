"""Run the full pipeline on a seeded sample of dev contracts and write an answer report.

Every CUAD question of every sampled contract goes through retrieval, the gate, the model,
grounding and the abstention rules. Each scored question is appended to
`<report>.questions.jsonl` at once, so an interrupted run resumes where it stopped. The log's
first line holds the run configuration, and resuming under a different one is refused.

A model or reply error is recorded for its question and retried on the next run.
MAX_CONSECUTIVE_ERRORS errors in a row abort the run: that means Ollama is down, not one bad
reply. Model output is not byte-reproducible across machines, and the report records latency.
"""

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cuad_doc_qa import __version__
from cuad_doc_qa.evaluation.answers.scoring import (
    ERROR,
    Record,
    aggregate,
    error_record,
    score_question,
)
from cuad_doc_qa.evaluation.qrels import load_split
from cuad_doc_qa.evaluation.runner import sample_contracts
from cuad_doc_qa.generation import OllamaChat
from cuad_doc_qa.generation.prompt import SYSTEM_PROMPT
from cuad_doc_qa.ingestion import CUAD_SHA256, download, load_annotations
from cuad_doc_qa.pipeline import CuadSession, cuad_session
from cuad_doc_qa.retrieval.lead import DEFAULT_LEAD, LEAD_CATEGORIES

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 3
DEV_SPLIT = "dev.txt"
Gold = Mapping[tuple[str, str], tuple[tuple[int, int], ...]]


def run_answer_eval(
    chat: OllamaChat,
    split_path: Path,
    report_path: Path,
    sample: int,
    seed: int,
    k: int,
    gate_path: Path,
    data_dir: Path = Path("data/raw"),
) -> dict[str, Any]:
    """Evaluate the pipeline with `chat` on the sampled dev contracts; returns the report."""
    if split_path.name != DEV_SPLIT:
        raise ValueError(f"answer evaluation runs on {DEV_SPLIT} only, got {split_path.name}")
    digest = chat.digest()
    if digest is None:
        raise RuntimeError(f"model {chat.model!r} is not pulled; run: ollama pull {chat.model}")

    archive = download(data_dir)
    contracts = sorted(sample_contracts(load_split(split_path), sample, seed))
    gold = {
        (annotation.doc_id, annotation.category): annotation.spans
        for annotation in load_annotations(archive)
    }
    config = {
        "run": "answers",
        "package_version": __version__,
        "archive_sha256": CUAD_SHA256,
        "split": split_path.name,
        "split_sha256": _sha256(split_path.read_bytes()),
        "sample": {"contracts": len(contracts), "seed": seed if sample else None},
        "model": {"name": chat.model, "digest": digest, "options": chat.options},
        "prompt_sha256": _sha256(SYSTEM_PROMPT.encode("utf-8")),
        "retriever": (
            "bm25, sqlite fts5, porter unicode61, 510 contracts; "
            f"first {DEFAULT_LEAD} chunks lead for {', '.join(LEAD_CATEGORIES)}"
        ),
        "k": k,
        "gate": {"path": gate_path.as_posix(), "sha256": _sha256(gate_path.read_bytes())},
    }
    with cuad_session(chat, chat.num_ctx, data_dir, gate_path, k) as session:
        return answer_eval(session, gold, contracts, report_path, config)


def answer_eval(
    session: CuadSession,
    gold: Gold,
    contracts: Sequence[str],
    report_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Ask every category of every contract, logging as it goes, then write the report."""
    log_path = report_path.with_name(report_path.stem + ".questions.jsonl")
    records = _resume(log_path, config)
    pending = [
        (doc_id, category)
        for doc_id in contracts
        for category in sorted(session.questions)
        if records.get(f"{doc_id}__{category}", {}).get("decision", ERROR) == ERROR
    ]
    finished = sum(record["decision"] != ERROR for record in records.values())
    logger.info("%d questions to ask, %d already done", len(pending), finished)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started, consecutive_errors = time.perf_counter(), 0
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        if log.tell() == 0:
            log.write(json.dumps({"config": config}, sort_keys=True) + "\n")
        for number, (doc_id, category) in enumerate(pending, start=1):
            question_id = f"{doc_id}__{category}"
            spans = gold[(doc_id, category)]
            try:
                result = session.pipeline.ask(session.questions[category], doc_id, category)
                record = score_question(question_id, result, spans)
                consecutive_errors = 0
            except (RuntimeError, ValueError) as error:
                record = error_record(question_id, doc_id, category, spans, error)
                consecutive_errors += 1
                logger.warning("%s: %s", question_id, record["error"])
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            records[question_id] = record
            if number % 25 == 0:
                logger.info("%d/%d questions in %.0fs", number, len(pending), _since(started))
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(
                    f"aborted after {consecutive_errors} consecutive errors, last: "
                    f"{record['error']}; finished questions are kept in {log_path.as_posix()}"
                )

    report = {"provenance": dict(config), **aggregate(records.values())}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    report_path.write_text(text, encoding="utf-8", newline="\n")
    return report


def read_question_log(log_path: Path) -> tuple[dict[str, Any] | None, dict[str, Record]]:
    """The run configuration and the latest record per question of a question log.

    A retried question appears more than once; its last record is the one that counts. An empty
    log has no configuration yet. Raises ValueError if the first line holds no configuration.
    """
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None, {}
    header = json.loads(lines[0])
    if "config" not in header:
        raise ValueError(f"{log_path.as_posix()} is not a question log: no config on line 1")
    records: dict[str, Record] = {}
    for line in lines[1:]:
        record = json.loads(line)
        records[record["question_id"]] = record
    return header["config"], records


def _resume(log_path: Path, config: Mapping[str, Any]) -> dict[str, Record]:
    """Latest record per question from an earlier run with the same configuration."""
    if not log_path.exists():
        return {}
    logged, records = read_question_log(log_path)
    if logged is None:
        return {}
    if logged != json.loads(json.dumps(config)):
        raise ValueError(
            f"{log_path.as_posix()} was written with a different configuration; "
            "delete it or choose another --out"
        )
    return records


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _since(started: float) -> float:
    return time.perf_counter() - started
