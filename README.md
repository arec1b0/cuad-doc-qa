# cuad-doc-qa

[![ci](https://github.com/arec1b0/cuad-doc-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/arec1b0/cuad-doc-qa/actions/workflows/ci.yml)

Document Q&A with verifiable citations over the CUAD contract corpus.

What failed, what was fixed and what is still unmeasured: [docs/FAILURES.md](docs/FAILURES.md).

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
uv run cuad-qa evaluate --split test --retriever bm25-lead
uv run cuad-qa evaluate --split dev --retriever bm25-lead
```

`--retriever bm25-lead` puts each contract's first 7 chunks ahead of BM25's hits for the four
questions about its name, parties and dates (Agreement Date, Document Name, Effective Date,
Parties); every other question is ranked exactly as by BM25.

Run commands from the repository root. Exit codes: 0 success, 1 handled failure (bad data,
checksum, network, missing FTS5), 2 invalid usage. `--log-level DEBUG` adds tracebacks.

## Experiments

Alternative retrievers are compared on the same 40 dev contracts (seed 0), never on test.
Reports land in `reports/experiments/` and record every model with its downloaded revision.
BM25 reports are byte-reproducible; dense and rerank scores come from ONNX Runtime and may differ
in the last float bits across CPUs.

```powershell
uv run cuad-qa experiment --retriever bm25
uv run cuad-qa experiment --retriever bm25-lead
uv run cuad-qa experiment --retriever dense
uv run cuad-qa experiment --retriever hybrid
uv run cuad-qa experiment --retriever rerank
```

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

For the Agreement Date, Document Name, Effective Date and Parties categories, the model sees the
contract's first 7 chunks first, then BM25's hits. A 7B model answers in seconds on a GPU and in
minutes on CPU; `ollama ps` shows which one it runs on. `--timeout SECONDS` (env `CUAD_QA_OLLAMA_TIMEOUT`, default 300) bounds each model reply.

## Answer evaluation

Runs all 41 CUAD questions of a seeded dev sample (default 10 contracts, 410 questions) through
the full pipeline and scores answers against the expert spans: a shown answer is correct when
one of its citations overlaps an expert span. The report records answer recall and precision,
false answers on unanswerable questions, abstention by reason, citation precision and latency.
Every scored question is appended to `<report>.questions.jsonl`, so an interrupted run picks up
where it stopped; three errors in a row abort it. The default sample makes 300 model calls
(the gate withholds the other 110 questions), so on CPU it runs for hours.

`reports/answers/` holds the committed qwen2.5:7b run with its question log. Model output is not
reproducible across machines, so the log is the evidence: a test checks that the report is
recomputed exactly from it and that no question is left in error.

```powershell
$env:CUAD_QA_OLLAMA_MODEL = "qwen2.5:7b"
uv run cuad-qa evaluate-answers --sample 1
uv run cuad-qa evaluate-answers
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
server running and the model pulled. They make 33 model calls; on CPU raise the timeout:

```powershell
ollama pull qwen2.5:7b
$env:CUAD_QA_OLLAMA_MODEL = "qwen2.5:7b"; uv run pytest -m ollama
```

## Regression gate

`cuad-qa gate` compares a candidate report with a baseline on the same split and exits 3 when
dev recall@10 drops by more than 0.02 (`--tolerance METRIC=VALUE` overrides, repeatable):

```powershell
uv run cuad-qa gate --baseline reports/bm25-baseline-dev.json --candidate artifacts/candidate-dev.json
```

CI (`.github/workflows/ci.yml`) runs lint and unit tests on Ubuntu and Windows. On Ubuntu it
also runs the real-data tests, checks that the committed BM25 and bm25-lead dev reports are
reproduced by the code, and gates both against the ones on the base branch. CI never evaluates on
the test split.

## License

Code: MIT, see [LICENSE](LICENSE). CUAD v1 is by The Atticus Project under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); the contract excerpts quoted in
`reports/answers/` come from it under the same license.
