"""Text embedders: the contract DenseIndex relies on, and the fastembed implementation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """One row per text, shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """Shape (dim,)."""
        ...


class FastEmbedEmbedder:
    """BAAI/bge-small-en-v1.5 (384 dimensions) on CPU through fastembed's ONNX runtime.

    fastembed serves this model from the quantized export `Qdrant/bge-small-en-v1.5-onnx-Q`,
    not BAAI's original weights. Queries are embedded without the optional BGE instruction.
    The model is downloaded on first use into `cache_dir`.
    """

    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, cache_dir: Path = Path("artifacts/models"), batch_size: int = 64) -> None:
        self._model = TextEmbedding(model_name=self.MODEL, cache_dir=str(cache_dir))
        self._batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        rows = self._model.passage_embed(list(texts), batch_size=self._batch_size)
        return np.asarray(list(rows), dtype=np.float32)

    def embed_query(self, text: str) -> Vectors:
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)
