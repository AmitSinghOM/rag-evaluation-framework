# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Nothing has been released or published yet.

## [Unreleased]

### Added

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

### Changed

- `requires-python` is now `>=3.11` (was `>=3.9`; 3.9 and 3.10 are
  end-of-life). Version is `0.1.0a0`.
- `k <= 0` now raises `ValueError` for every metric, also when `relevant`
  is empty (previously `recall_at_k` and `ndcg_at_k` returned `0.0` first).

### Fixed

- Duplicate ids in `retrieved` no longer inflate recall, precision, DCG or
  NDCG. The first occurrence counts; later repeats are ignored at their
  position (they earn nothing and do not move later documents up a rank).
  Before this fix `recall_at_k` could exceed `1.0`.
