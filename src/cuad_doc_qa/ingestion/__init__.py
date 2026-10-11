"""Acquire the CUAD v1 release and parse it into documents."""

from cuad_doc_qa.ingestion.cuad import (
    ARCHIVE_NAME,
    CUAD_SHA256,
    CUAD_URL,
    ChecksumMismatchError,
    download,
    load_documents,
)

__all__ = [
    "ARCHIVE_NAME",
    "CUAD_SHA256",
    "CUAD_URL",
    "ChecksumMismatchError",
    "download",
    "load_documents",
]
