"""Acquire the CUAD v1 release and parse it into documents."""

from cuad_doc_qa.ingestion.cuad import (
    ARCHIVE_NAME,
    CUAD_SHA256,
    CUAD_URL,
    ChecksumMismatchError,
    download,
    load_annotations,
    load_documents,
    load_official_split,
)

__all__ = [
    "ARCHIVE_NAME",
    "CUAD_SHA256",
    "CUAD_URL",
    "ChecksumMismatchError",
    "download",
    "load_annotations",
    "load_documents",
    "load_official_split",
]
