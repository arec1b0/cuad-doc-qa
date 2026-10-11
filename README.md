# cuad-doc-qa

Document Q&A with verifiable citations over the CUAD contract corpus.

## Requirements

- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 from `.python-version` if it is missing.

## Setup

```powershell
uv sync --locked
```

## Checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
