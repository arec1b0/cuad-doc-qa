"""Cross-encoder scorers: the contract RerankedSearch relies on, and the fastembed one."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from fastembed.rerank.cross_encoder import TextCrossEncoder


class PairScorer(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """One relevance score per text, in input order; higher means more relevant."""
        ...


class FastEmbedCrossEncoder:
    """Xenova/ms-marco-MiniLM-L-6-v2 (22M parameters) on CPU through fastembed's ONNX runtime.

    Trained on MS MARCO web passages, not contracts: domain shift is expected and is what the
    evaluation has to measure. Query and passage are truncated together to 512 tokens.
    The model (80 MB) is downloaded on first use into `cache_dir`.
    """

    MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

    def __init__(self, cache_dir: Path = Path("artifacts/models"), batch_size: int = 32) -> None:
        self._model = TextCrossEncoder(model_name=self.MODEL, cache_dir=str(cache_dir))
        self._batch_size = batch_size

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self._model.rerank(query, list(texts), batch_size=self._batch_size)
        return [float(score) for score in scores]
