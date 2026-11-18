# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Nothing has been released or published yet.

## [Unreleased]

### Added

- [R] `rageval run --dataset FILE.jsonl --metrics ... --out ...`: score one
  JSONL file (the format `read_jsonl` reads) instead of `--qrels` plus
  `--run`. Giving both kinds of input, or neither, is a usage error (exit 1);
  an empty or malformed JSONL file is an input error (exit 2). The report
  keeps `schema_version` 1 unchanged: for JSONL, `dataset.qrels.path` and
  `run.path` are the JSONL path, `run.sha256` hashes the file and
  `dataset.qrels.sha256` hashes the judgments alone
  (`FileRef.of_judgments`), so two JSONL files with the same labels and
  different rankings can be compared and gated. Renaming to
  `dataset.source` was rejected because it would break schema 1 readers;
  see `docs/REPORT-SCHEMA.md`.
- [R] `python -m rageval` (`rageval/__main__.py`), the same command line as
  `rageval`; `rageval --version` and `python -m rageval --version` print
  `rageval 0.1.0a0`. A test locks `pyproject.toml`'s version to
  `rageval.__version__`.
- [R] Release workflow `.github/workflows/release.yml`: a `v*` tag builds
  the sdist and wheel (`python -m build`, `twine check --strict`), checks
  the tag equals the package version, uploads to TestPyPI (environment
  `testpypi`) and attaches both files to a GitHub Release. The real PyPI
  job (environment `pypi`) runs only from a manual run on a tag with
  `confirm: publish`; a tag alone never reaches it. Trusted Publishing
  (OIDC) only, no stored token; every action pinned to a commit SHA.
  `tests/test_release.py` locks the pins, the upload jobs and their
  conditions. Not run yet.
- [R] `docs/RELEASING.md`: what runs when, the one-time Trusted Publisher
  and environment setup, how to cut and check a release, and the name
  check behind the rename below.
- [R] CI job `build`: builds and checks both distributions, installs the
  wheel in a clean venv, requires `pip list` to show only
  `rag-evaluation-framework`, checks
  both `--version` forms and runs the toy quickstart, the JSONL run and the
  gate from the installed wheel.
- [R] `.github/dependabot.yml`: weekly updates for GitHub Actions
  (workflows and `action.yml`) and pip (the pinned `dev` extra).
- [R] README: install section says the package is not on PyPI (source or a
  GitHub Release wheel), JSONL quickstart with real output,
  `python -m rageval`.

- [G] `rageval gate --baseline B --candidate C --metric NAME --max-drop X
  [--metric ... --max-drop ...] [--resamples N --seed S --level L]
  [--format table|json] [--summary PATH]`: a non-inferiority regression
  gate (`rageval.gate`, also `rageval.gate()` in the library). Per metric,
  the paired per-query difference (candidate - baseline) over the shared
  queries and its percentile bootstrap CI (`paired_diff_ci`, 10000
  resamples by default); a metric passes when the lower bound, rounded to
  6 places, is above `-max_drop`. Exit 0 when every metric passes, 10 when
  any fails, 1 on a usage error (a `--metric` without its `--max-drop`, a
  metric missing from a report, a max drop not above 0), 2 on an input
  error (unreadable report, other `schema_version`, different qrels, no
  shared query, a missing or non-finite per-query value). Warns on stderr
  under 30 shared queries. `--summary` appends a Markdown table for
  `$GITHUB_STEP_SUMMARY`.
- [G] `docs/GATE.md`: what the gate checks in plain words, why a p-value
  gate is rejected (low power at ~50 queries passes real regressions), how
  to choose `--max-drop` and `--resamples`, and the small-n warning.
- [G] Composite GitHub Action `action.yml` (inputs `baseline`,
  `candidate`, `metrics` as `ndcg@10:0.02,recall@100:0.05`, `resamples`,
  `seed`, `python-version`; output `exit-code`): installs rageval from the
  action's own checkout, runs the gate, writes the job summary, fails the
  step on a failed gate. `.github/workflows/gate-selftest.yml` runs it with
  `uses: ./` on every pull request: one job must pass (identical reports),
  one must fail with exit 10 (degraded candidate).
- [G] `tests/test_gate.py`: every verdict, exit code and usage rule, a
  seeded 200-case check of the invariant (a pass bounds the mean drop by
  `max_drop` plus the CI half-width; an identical candidate always passes),
  and a lock that the README gate output is what a live run prints.
- [G] README `Regression gate` section with real output.

- [A] Loaders (`rageval.loaders`): `read_qrels` (TREC qrels, graded; grade 0
  = judged not relevant), `read_run` (TREC run; ordered by score desc then
  rank asc, repeats kept) and `read_jsonl` (one object per query with
  `retrieved` and `relevant`). A malformed line raises `LoaderError` naming
  the file and 1-based line number; repeated judgments and repeated
  `query_id`s are errors, never silent overwrites.
- [A] `rageval.stats`: `bootstrap_ci` (percentile bootstrap of the mean,
  `random.Random(seed)`, pure stdlib) and `paired_diff_ci` (`b - a` on
  aligned per-query values).
- [A] `evaluate(qrels, run, metrics)` -> `Report`, with metric names
  `ndcg@K`, `recall@K`, `precision@K`, `mrr`, `mrr@K`; a bad name raises
  `ValueError` listing the valid forms. Versioned JSON report
  (`schema_version` 1): dataset and run file SHA-256, query counts,
  duplicate-id count, per-metric mean + seeded bootstrap `ci95`, per-query
  rows, `warnings` for run queries that are not judged. Schema in
  `docs/report.schema.json` (JSON Schema 2020-12), field guide in
  `docs/REPORT-SCHEMA.md`; every emitted report is validated in tests.
- [A] `compare(a, b)` -> paired per-query differences with bootstrap CI for
  every metric in both reports; refuses reports made on different qrels.
- [A] Command line `rageval` (argparse, no dependencies): `rageval run
  --qrels Q --run R --metrics M[,M...] --out FILE [--resamples N --seed S
  --name NAME]` and `rageval compare a.json b.json [--format table|json]`.
  Exit codes: 0 ok, 1 usage, 2 input/format, 3 internal (10 reserved for
  the Phase 2 gate).
- [A] `examples/toy/`: six hand-checked queries, two runs and the same data
  as JSONL; `tests/test_cli.py` asserts every `rageval run` mean agrees with
  `ranx.evaluate` on the same files within 1e-9 and that the README
  quickstart commands still parse. `jsonschema` (+ stubs) added to the
  `dev` extra; runtime dependencies stay empty.
- [A] README Phase-1 quickstart, run live in a fresh venv with the real
  output pasted.
- Graded relevance: `relevant` may be a mapping `id -> grade` as well as a
  set of ids. DCG/NDCG use the grade as gain with the standard
  `1 / log2(rank + 1)` discount; the ideal ranking sorts grades descending.
  Grade `0` means "judged, not relevant". Binary (set) inputs give exactly
  the values they did before, bit for bit, except where duplicates were
  inflating them (see Fixed).
- `mrr(retrieved, relevant, k=None)`: optional cutoff (`MRR@k`).
- Cross-check against `ranx` 0.3.21 (`tests/test_ranx_crosscheck.py`): 256
  queries with graded qrels, judged-non-relevant docs, short lists and
  duplicate ids; every per-query value of ndcg, dcg, recall, precision,
  mrr@k and mrr agrees within 1e-9 (measured 1.1e-16). `ranx` is a
  dev-only dependency.
- `docs/METRIC-DEFINITIONS.md`: the exact rule for every metric and how it
  maps onto ranx.
- `docs/decisions/ADR-0001-v0.1-scope.md`: what v0.1 is and is not.
- Packaging hygiene: `py.typed`, MIT `LICENSE`, `SECURITY.md`,
  `CONTRIBUTING.md`, ruff + mypy (strict) configuration, pytest coverage
  floor, CI on Python 3.11-3.14 (Ubuntu) and 3.13 (macOS), plus a job that
  proves a user install brings no third-party package.
- `load()` keeps a grade mapping as a dict so graded data survives loading.
- [C] SciFact dogfood fixtures in `examples/scifact/`: TREC qrels for the
  BEIR SciFact test split (300 queries, 339 judgements), a deterministic
  pure-Python BM25 run (`bm25.top100.run`, k1=0.9, b=0.4, byte-identical on
  Python 3.11-3.14) and two documented degraded runs (`truncate:20`,
  `shuffle-tail:0`) for later gate demos, plus `MANIFEST.json` with sha256,
  counts, generator commands, source URL and licence.
- [C] `scripts/fetch_scifact.py` (sha256-pinned download, TREC conversion),
  `scripts/bm25_baseline.py` (`--degrade`, `--offline`) and
  `scripts/make_manifest.py`; stdlib only, documented in `scripts/README.md`.
  `tests/test_scifact_fixtures.py` checks every fixture against the manifest
  and smoke-tests the scripts offline on a synthetic corpus. An empty
  `dogfood` extra is reserved for script-only dependencies.

### Changed

- [R] The distribution (pip) name is now `rag-evaluation-framework`, was
  `rageval`: `rageval` is taken on TestPyPI (another publisher, 2023) and
  too similar to `rag-eval` on PyPI. `import rageval`, the `rageval`
  command and `python -m rageval` do not change.
- `requires-python` is now `>=3.11` (was `>=3.9`; 3.9 and 3.10 are
  end-of-life). Version is `0.1.0a0`.
- `k <= 0` now raises `ValueError` for every metric, also when `relevant`
  is empty (previously `recall_at_k` and `ndcg_at_k` returned `0.0` first).

### Fixed

- Duplicate ids in `retrieved` no longer inflate recall, precision, DCG or
  NDCG. The first occurrence counts; later repeats are ignored at their
  position (they earn nothing and do not move later documents up a rank).
  Before this fix `recall_at_k` could exceed `1.0`.
