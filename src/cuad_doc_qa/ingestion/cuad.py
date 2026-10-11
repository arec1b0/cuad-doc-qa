"""CUAD v1 source: pinned, checksum-verified download and parsing of contracts and labels."""

import hashlib
import json
import logging
import os
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cuad_doc_qa.domain import Annotation, Document

logger = logging.getLogger(__name__)

CUAD_COMMIT = "67faa0e6023b04fcaae6cc09497ab00e5d63a2a2"
CUAD_URL = f"https://raw.githubusercontent.com/TheAtticusProject/cuad/{CUAD_COMMIT}/data.zip"
CUAD_SHA256 = "f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a"
ARCHIVE_NAME = "cuad_v1_data.zip"

_FULL_MEMBER = "CUADv1.json"
_SPLIT_MEMBERS = {"train": "train_separate_questions.json", "test": "test.json"}
_CHUNK_BYTES = 1 << 20
_TIMEOUT_SECONDS = 60


class ChecksumMismatchError(Exception):
    """The downloaded archive does not match the pinned SHA-256."""


def download(dest_dir: Path, url: str = CUAD_URL, sha256: str = CUAD_SHA256) -> Path:
    """Fetch the CUAD archive into `dest_dir` and return its path.

    A cached archive with the expected checksum is reused without network access.
    The archive appears under its final name only after its checksum has passed.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / ARCHIVE_NAME

    if target.exists():
        if _sha256_of(target) == sha256:
            logger.info("Reusing verified archive %s", target)
            return target
        logger.warning("Cached archive %s fails checksum; downloading again", target)

    fd, tmp_name = tempfile.mkstemp(prefix=".download-", suffix=".zip", dir=dest_dir)
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response,
        ):
            while chunk := response.read(_CHUNK_BYTES):
                digest.update(chunk)
                out.write(chunk)
        actual = digest.hexdigest()
        if actual != sha256:
            raise ChecksumMismatchError(f"{url}: expected sha256 {sha256}, got {actual}")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)

    logger.info("Downloaded %s to %s", url, target)
    return target


def load_documents(archive: Path) -> list[Document]:
    """Parse every contract in the archive, in published order.

    Raises ValueError if the archive breaks a CUAD v1 layout assumption this parser relies on.
    """
    documents = [
        Document(doc_id=doc_id, text=paragraph["context"])
        for doc_id, paragraph in _entries(archive)
    ]
    logger.info("Loaded %d documents from %s", len(documents), archive)
    return documents


def load_annotations(archive: Path) -> list[Annotation]:
    """Parse the gold spans of every (contract, category) pair, including absent clauses.

    Raises ValueError if a question id, an answer offset or the is_impossible flag is inconsistent.
    """
    annotations: list[Annotation] = []
    for doc_id, paragraph in _entries(archive):
        context = paragraph["context"]
        prefix = f"{doc_id}__"
        for qa in paragraph["qas"]:
            qa_id = qa["id"]
            if not qa_id.startswith(prefix) or len(qa_id) == len(prefix):
                raise ValueError(f"{qa_id!r}: question id is not '<title>__<category>'")
            spans: set[tuple[int, int]] = set()
            for answer in qa["answers"]:
                start, text = answer["answer_start"], answer["text"]
                if not text or context[start : start + len(text)] != text:
                    raise ValueError(f"{qa_id!r}: answer does not match the text at offset {start}")
                spans.add((start, start + len(text)))
            if qa["is_impossible"] != (not spans):
                raise ValueError(f"{qa_id!r}: is_impossible disagrees with the answers")
            annotations.append(
                Annotation(doc_id=doc_id, category=qa_id[len(prefix) :], spans=tuple(sorted(spans)))
            )

    logger.info("Loaded %d annotations from %s", len(annotations), archive)
    return annotations


def load_official_split(archive: Path) -> dict[str, frozenset[str]]:
    """Contract ids of CUAD's own split, keyed "train" and "test"."""
    with zipfile.ZipFile(archive) as zf:
        return {
            name: frozenset(entry["title"] for entry in json.loads(zf.read(member))["data"])
            for name, member in _SPLIT_MEMBERS.items()
        }


def _entries(archive: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with zipfile.ZipFile(archive) as zf:
        payload = json.loads(zf.read(_FULL_MEMBER))

    seen: set[str] = set()
    for entry in payload["data"]:
        doc_id = entry["title"]
        paragraphs = entry["paragraphs"]
        if len(paragraphs) != 1:
            raise ValueError(f"{doc_id!r}: expected 1 paragraph, found {len(paragraphs)}")
        if doc_id in seen:
            raise ValueError(f"{doc_id!r}: duplicate document id")
        seen.add(doc_id)
        yield doc_id, paragraphs[0]


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
