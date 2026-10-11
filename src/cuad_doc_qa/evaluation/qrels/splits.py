"""Frozen evaluation splits: contract ids stored one per line in UTF-8 text files."""

from pathlib import Path


def load_split(path: Path) -> frozenset[str]:
    """Read a split file. Raises ValueError on blank lines or duplicate ids."""
    ids = path.read_text(encoding="utf-8").splitlines()
    if any(not doc_id.strip() for doc_id in ids):
        raise ValueError(f"{path}: blank line")
    unique = frozenset(ids)
    if len(unique) != len(ids):
        raise ValueError(f"{path}: duplicate contract id")
    return unique
