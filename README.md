# cuad-doc-qa

Document Q&A with verifiable citations over the CUAD contract corpus.

## Requirements

- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 from `.python-version` if it is missing.

## Setup

```powershell
uv sync --locked
```

## Data

[CUAD v1](https://github.com/TheAtticusProject/cuad): 510 commercial contracts with expert clause
annotations, licensed CC BY 4.0. The archive is pinned to commit
`67faa0e6023b04fcaae6cc09497ab00e5d63a2a2` and SHA-256
`f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a`. It is downloaded to
`data/raw/` and never modified.

Evaluation splits are frozen in `data/eval/splits/`, one contract id per line:

- `dev.txt`: CUAD's official train split (408 contracts), used for tuning.
- `test.txt`: CUAD's official test split (102 contracts), used only for reported results.

Each contract has 41 questions, one per clause category. A chunk is relevant to a question if it
overlaps one of the expert-annotated spans; a question with no spans is unanswerable.

## Reports

Retrieval reports live in `reports/` and are byte-reproducible: regenerating one must leave
`git diff` empty. Each question is searched only within its own contract, using CUAD's question
text verbatim. Metrics cover answerable questions; unanswerable ones are counted, not scored.

```powershell
uv run cuad-qa download
uv run cuad-qa evaluate --split test
uv run cuad-qa evaluate --split dev
```

Run commands from the repository root. Exit codes: 0 success, 1 handled failure (bad data,
checksum, network, missing FTS5), 2 invalid usage. `--log-level DEBUG` adds tracebacks.

## Abstention gate

`configs/bm25-gate.json` holds per-category BM25 score thresholds, calibrated on dev to keep 95%
of answerable questions. Questions below their category's threshold are withheld before any LLM
call. Regenerating it scores all 16,728 dev questions (about 15 minutes) and must leave
`git diff` empty:

```powershell
uv run python -c "from pathlib import Path; from cuad_doc_qa.abstention import calibrate_bm25_gate; calibrate_bm25_gate(Path('data/eval/splits/dev.txt'), Path('configs/bm25-gate.json'))"
```

## Ask

Answers one question about one contract through a local [Ollama](https://ollama.com) model.
Contract ids are listed in `data/eval/splits/`; a unique prefix is enough. `--category` asks
CUAD's question for that clause and applies the gate; `--question` asks free text without it.
Every citation printed is the contract's own text at the printed character offsets.

```powershell
$env:CUAD_QA_OLLAMA_MODEL = "qwen2.5:7b"
uv run cuad-qa ask --contract LIMEENERGYCO --category "Governing Law"
uv run cuad-qa ask --contract LIMEENERGYCO --question "Can the distributor sell competing products?"
```

## Checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests that need internet access are deselected by default. They download CUAD (18 MB) into
`data/raw/` and the ONNX models into `artifacts/models/` on first run:

```powershell
uv run pytest -m network
```

Tests against a local [Ollama](https://ollama.com) server are deselected too. They need the
server running and the model pulled:

```powershell
ollama pull qwen2.5:7b
$env:CUAD_QA_OLLAMA_MODEL = "qwen2.5:7b"; uv run pytest -m ollama
```
