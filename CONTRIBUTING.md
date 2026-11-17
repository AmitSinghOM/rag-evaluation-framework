# Contributing

## Set up

```bash
git clone https://github.com/AmitSinghOM/rag-evaluation-framework.git
cd rag-evaluation-framework
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.11 or newer. The `dev` extra pulls `ranx` (and with it numba,
pandas and scipy); that is the test reference only and never a runtime
dependency.

## Before you push

Run the same gate CI runs, and fix anything red before pushing:

```bash
ruff check . && ruff format --check .
mypy
pytest --cov
```

## Rules

- `dependencies` in `pyproject.toml` stays empty. A test fails if it does not.
- Every metric change must keep `tests/test_ranx_crosscheck.py` green. If a
  definition legitimately differs from ranx, document it in
  `docs/METRIC-DEFINITIONS.md` and assert the documented relationship.
- New behaviour gets a test that fails on the old code. Say so in the PR.
- Add a line under `Unreleased` in `CHANGELOG.md`.
- Scope decisions live in `docs/decisions/`; read the latest ADR before
  proposing new surface area.
- Keep the README honest: no claim the code does not back.
