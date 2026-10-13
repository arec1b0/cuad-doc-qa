"""The pipeline wired to the CUAD corpus: BM25 over all 510 contracts and the dev-calibrated gate.

The index must cover the same contracts and chunking the gate was calibrated on, or its
thresholds are meaningless; `tests/unit/test_abstention_committed_gate.py` guards the latter.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from cuad_doc_qa.abstention import load_gate
from cuad_doc_qa.chunking import chunk_document
from cuad_doc_qa.generation import ChatModel, Generator
from cuad_doc_qa.ingestion import download, load_documents, load_questions
from cuad_doc_qa.pipeline.pipeline import DEFAULT_K, Pipeline
from cuad_doc_qa.retrieval.lexical import Bm25Index

DEFAULT_GATE = Path("configs/bm25-gate.json")


@dataclass(frozen=True, slots=True)
class CuadSession:
    pipeline: Pipeline
    questions: dict[str, str]
    """CUAD question text per clause category."""
    contracts: tuple[str, ...]

    def resolve_contract(self, name: str) -> str:
        """The contract id equal to `name`, else the only one starting with it."""
        if name in self.contracts:
            return name
        matches = [doc_id for doc_id in self.contracts if doc_id.startswith(name)]
        if len(matches) != 1:
            raise ValueError(f"{name!r} matches {len(matches)} contracts; give a unique prefix")
        return matches[0]


@contextmanager
def cuad_session(
    chat: ChatModel,
    num_ctx: int,
    data_dir: Path = Path("data/raw"),
    gate_path: Path = DEFAULT_GATE,
    k: int = DEFAULT_K,
) -> Iterator[CuadSession]:
    archive = download(data_dir)
    documents = load_documents(archive)
    chunks = [chunk for document in documents for chunk in chunk_document(document)]
    with Bm25Index.build(chunks) as index:
        yield CuadSession(
            pipeline=Pipeline(
                index.search, chunks, Generator(chat, num_ctx=num_ctx), load_gate(gate_path), k
            ),
            questions=load_questions(archive),
            contracts=tuple(document.doc_id for document in documents),
        )
