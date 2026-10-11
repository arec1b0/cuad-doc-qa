"""download: checksum verification, atomic placement and caching, against a local HTTP server."""

import functools
import hashlib
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from cuad_doc_qa.ingestion import ARCHIVE_NAME, ChecksumMismatchError, download

PAYLOAD = b"archive bytes served by the local test server"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
UNREACHABLE_URL = "http://127.0.0.1:9/data.zip"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        return None


@pytest.fixture
def served_url(tmp_path: Path) -> Iterator[str]:
    root = tmp_path / "remote"
    root.mkdir()
    (root / "data.zip").write_bytes(PAYLOAD)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(root))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/data.zip"
    finally:
        server.shutdown()
        server.server_close()


def test_places_only_the_verified_archive(tmp_path: Path, served_url: str) -> None:
    dest = tmp_path / "raw"

    archive = download(dest, url=served_url, sha256=PAYLOAD_SHA256)

    assert archive == dest / ARCHIVE_NAME
    assert archive.read_bytes() == PAYLOAD
    assert [p.name for p in dest.iterdir()] == [ARCHIVE_NAME]


def test_checksum_mismatch_raises_and_leaves_no_files(tmp_path: Path, served_url: str) -> None:
    dest = tmp_path / "raw"

    with pytest.raises(ChecksumMismatchError):
        download(dest, url=served_url, sha256="0" * 64)

    assert list(dest.iterdir()) == []


def test_verified_cache_is_reused_without_network(tmp_path: Path) -> None:
    dest = tmp_path / "raw"
    dest.mkdir()
    (dest / ARCHIVE_NAME).write_bytes(PAYLOAD)

    archive = download(dest, url=UNREACHABLE_URL, sha256=PAYLOAD_SHA256)

    assert archive.read_bytes() == PAYLOAD


def test_corrupt_cache_is_replaced(tmp_path: Path, served_url: str) -> None:
    dest = tmp_path / "raw"
    dest.mkdir()
    (dest / ARCHIVE_NAME).write_bytes(b"truncated")

    archive = download(dest, url=served_url, sha256=PAYLOAD_SHA256)

    assert archive.read_bytes() == PAYLOAD
