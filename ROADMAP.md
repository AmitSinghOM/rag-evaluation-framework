# rag-evaluation-framework — Build Roadmap

Built in small rounds. Each phase ends with CI green, the README status line updated, and a CHANGELOG entry. Scope and the reasons for it: [docs/decisions/ADR-0001-v0.1-scope.md](./docs/decisions/ADR-0001-v0.1-scope.md).

> Rule: ship small, iterate visibly. Earn scope by finishing. No claim in the README that the code does not back.

## Phase 0 — Metrics core, provably correct  ·  DONE 2026-11-18
**Goal:** Recall@K, Precision@K, MRR, NDCG that agree with a reference implementation on every input.

- [x] Recall@K, Precision@K, MRR, NDCG with hand-computed unit tests
- [x] Duplicate ids count once, at their first position (was inflating recall above 1.0)
- [x] Graded relevance (`id -> grade`) for DCG/NDCG; grade 0 = judged not relevant
- [x] `mrr(..., k=)` cutoff
- [x] Cross-check against ranx on 256 synthetic queries, agreement within 1e-9 (dev-only dependency)
- [x] `docs/METRIC-DEFINITIONS.md`
- [x] Hygiene: ruff, mypy strict, coverage floor, CI 3.11-3.14 + macOS, LICENSE, SECURITY, CONTRIBUTING, CHANGELOG, ADR-0001

## Phase 1 — Loaders, report, CLI
**Goal:** Evaluate a real run file against real labels from the command line and get a report you can diff.

- [x] Loaders: JSONL (`{"query_id", "retrieved": [...]}` plus labels), TREC qrels, TREC run files
- [x] Versioned JSON report: per-query scores, means, bootstrap confidence intervals
- [x] `rageval run` and `rageval compare`
- [x] Quickstart in the README, run live exactly as written
- [ ] Repository public

## Phase 2 — Non-inferiority gate + GitHub Action
**Goal:** Block a retrieval regression in CI without passing real regressions at ~50 queries.

- [x] `rageval gate`: fail unless the lower CI bound of (candidate − baseline) per metric is above `−max_drop`
- [x] Exit codes documented and tested
- [x] GitHub Action wrapping the gate

## Phase 3 — Grounding check, EXPERIMENTAL
**Goal:** Measure, not claim. A grounding signal is only described in the README once its calibration report exists.

- [x] Lexical-support baseline
- [x] Optional judge (Ollama or a free-tier API), clearly optional
- [x] Labelling pack (60 items, two hidden conditions) generated from the SciFact BM25 run with a local 3B model
- [ ] Calibration report against ≥ 50 human-written labels from a BEIR SciFact dogfood run
- [ ] README wording taken from the report, labelled experimental

## Phase 4 — Release
**Goal:** v0.1 tagged; PyPI only on an explicit go and preferably after one named consumer exists.

- [x] Release workflow, CI build-and-install check and Dependabot written ([docs/RELEASING.md](./docs/RELEASING.md))
- [x] Distribution name `rag-evaluation-framework` (`rageval` is taken on TestPyPI and too close to `rag-eval` on PyPI; import name and command stay `rageval`)
- [x] Release workflow proven on TestPyPI (`v0.1.0a0` rehearsal, 2026-11-18)
- [ ] v0.1.0 tag and GitHub release
- [ ] PyPI publish (decision deferred; see ADR-0001)

## Definition of done (every phase)
1. Tests pass, CI green on every supported interpreter.
2. README status line updated; no unbacked claims.
3. CHANGELOG entry.
