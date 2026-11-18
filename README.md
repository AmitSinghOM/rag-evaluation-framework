# rag-evaluation-framework

**Measures retrieval quality for RAG systems: Recall@K, Precision@K, MRR, NDCG with graded relevance, in pure Python with no runtime dependencies, and every value cross-checked against [ranx](https://github.com/AmenRa/ranx).**

> **Status (2026-11-18): Phase 0 metrics corrected and reference-checked; Phase 1 loaders, versioned report with bootstrap CIs and the `rageval run` / `rageval compare` CLI done; Phase 2 non-inferiority regression gate (`rageval gate`) and its GitHub Action done; release workflow written (a tag uploads to TestPyPI only) but not yet run; nothing published.**
> The package is not on PyPI or TestPyPI. Install from source (below). The repository is private until v0.1.
> Scope and the reasons for it: [docs/decisions/ADR-0001-v0.1-scope.md](./docs/decisions/ADR-0001-v0.1-scope.md).

## What works today

- `recall_at_k`, `precision_at_k`, `mrr` (with optional `k`), `dcg_at_k`, `ndcg_at_k` over ordered id lists.
- Binary relevance (a set of ids) or graded relevance (a mapping `id -> grade`; grade 0 = judged not relevant).
- Duplicate ids in a ranking count once, at their first position; later repeats are wasted slots and cannot inflate a score.
- Every rule is written down in [docs/METRIC-DEFINITIONS.md](./docs/METRIC-DEFINITIONS.md) and locked by a test that scores 256 synthetic queries through both rageval and ranx and requires agreement within 1e-9.
- Loaders for TREC qrels, TREC run files and a JSONL dataset format (`read_qrels`, `read_run`, `read_jsonl`); a malformed line fails with the file and line number.
- `rageval run`: a versioned JSON report (`schema_version` 1, [docs/REPORT-SCHEMA.md](./docs/REPORT-SCHEMA.md), [docs/report.schema.json](./docs/report.schema.json)) with per-query scores, means and seeded bootstrap 95% confidence intervals; `rageval compare`: paired per-query differences with their CI, refusing reports made on different qrels. Means agree with `ranx.evaluate` on the same files within 1e-9.
- `rageval gate`: a non-inferiority regression gate for CI. Per metric it fails (exit 10) unless the lower bound of the paired bootstrap CI of candidate minus baseline is above `-max_drop`; a composite GitHub Action (`action.yml`) wraps it. See [Regression gate](#regression-gate) and [docs/GATE.md](./docs/GATE.md).
- `rageval run --dataset data.jsonl`: one JSONL file (judgments and ranking per query) instead of a TREC qrels file plus a run file. `python -m rageval` works the same as `rageval`.
- A release workflow ([docs/RELEASING.md](./docs/RELEASING.md)): a `v*` tag builds the sdist and wheel, uploads them to TestPyPI only and attaches them to a GitHub Release; the real PyPI is reached only by a manual run on a tag with `confirm: publish`. Uploads use Trusted Publishing (OIDC), so no token is stored anywhere. It has not been run yet. CI builds the wheel, installs it in a clean venv and runs the quickstart below on every change.
- Typed (`py.typed`), mypy strict, ruff, a 95% branch-coverage floor enforced in CI, CI on Python 3.11-3.14 (Ubuntu) and 3.13 (macOS).
- A tiny in-memory loader (`load`) that turns `(query, retrieved, relevant)` triples into records.

Not yet: a grounding check, a published package. See [ROADMAP.md](./ROADMAP.md).

## Install

rageval is not on PyPI yet. Install from source, or, once a version is tagged, from the wheel attached to its [GitHub Release](https://github.com/AmitSinghOM/rag-evaluation-framework/releases) (none is tagged yet). The package you install is named `rag-evaluation-framework`; you import `rageval` and run `rageval`.

```bash
pip install git+https://github.com/AmitSinghOM/rag-evaluation-framework.git
# or, from a release:  pip install rag_evaluation_framework-<version>-py3-none-any.whl
```

Python 3.11 or newer. The install brings no third-party package. Check it with `rageval --version` or `python -m rageval --version` (prints `rageval 0.1.0a0`).

## Quickstart (Phase 1)

From a clone of this repository, in a fresh virtual environment. The toy fixtures are six judged queries and two runs, small enough to check by hand ([examples/toy/README.md](./examples/toy/README.md)).

```bash
pip install .
rageval run --qrels examples/toy/qrels.txt --run examples/toy/run_a.txt --metrics ndcg@10,recall@100,mrr --out report_a.json
rageval run --qrels examples/toy/qrels.txt --run examples/toy/run_b.txt --metrics ndcg@10,recall@100,mrr --out report_b.json
rageval compare report_a.json report_b.json
```

Real output of those lines (Python 3.13.12, run while writing this section; `pip list` showed `rag-evaluation-framework==0.1.0a0` and nothing else):

```text
$ rageval run --qrels examples/toy/qrels.txt --run examples/toy/run_a.txt --metrics ndcg@10,recall@100,mrr --out report_a.json
warning: query 'q99' is in the run but not in the qrels; ignored
metric          mean  ci95                  n
ndcg@10       0.4890  [0.1116, 0.8223]      6
recall@100    0.5833  [0.1667, 0.9167]      6
mrr           0.4583  [0.0833, 0.7500]      6
wrote report_a.json (6 queries)
$ rageval run --qrels examples/toy/qrels.txt --run examples/toy/run_b.txt --metrics ndcg@10,recall@100,mrr --out report_b.json
metric          mean  ci95                  n
ndcg@10       0.8636  [0.7406, 0.9866]      6
recall@100    1.0000  [1.0000, 1.0000]      6
mrr           0.8333  [0.6667, 1.0000]      6
wrote report_b.json (6 queries)
$ rageval compare report_a.json report_b.json
metric             run_a         run_b    diff (b-a)  ci95
ndcg@10         0.488957      0.863597     +0.374640  [+0.010254, +0.762443]
recall@100      0.583333      1.000000     +0.416667  [+0.083333, +0.833333]
mrr             0.458333      0.833333     +0.375000  [-0.083333, +0.750000]
n_queries: 6
```

`report_a.json` starts:

```json
{
  "schema_version": 1,
  "rageval_version": "0.1.0a0",
  "created_at": "2026-11-17T21:18:37Z",
  "dataset": {
    "qrels": {
      "path": "examples/toy/qrels.txt",
      "sha256": "e3905f1c83c664baeb122a9c6aed441ed0a218edf5d4bd7a884f8816958144d4"
    },
    "n_queries": 6,
    "n_judged_relevant": 9
  },
```

Metric names: `ndcg@K`, `recall@K`, `precision@K`, `mrr`, `mrr@K`. Options: `--resamples N` (1000), `--seed S` (0), `--name NAME`; `compare --format json`. Exit codes: 0 ok, 1 usage, 2 input or format error, 3 internal, 10 regression gate failed. A query in the qrels that the run never answers scores 0; a query in the run that is not judged is ignored and listed under `warnings`. Six queries give the wide intervals above by design; the point of the CI is that on a real query set it tells you whether a difference is noise.

### One JSONL file instead of two TREC files

`--dataset` reads one object per query, `{"query_id": ..., "query": ..., "retrieved": [...], "relevant": {id: grade} or [ids]}` (format: [docs/REPORT-SCHEMA.md](./docs/REPORT-SCHEMA.md#input-formats)). Give either `--dataset` or `--qrels` with `--run`; mixing them is a usage error (exit 1). `examples/toy/dataset.jsonl` holds the same data as `qrels.txt` plus `run_b.txt`, so it scores the same:

```bash
rageval run --dataset examples/toy/dataset.jsonl --metrics ndcg@10,recall@100,mrr --out report_j.json
python -m rageval --version
```

```text
$ rageval run --dataset examples/toy/dataset.jsonl --metrics ndcg@10,recall@100,mrr --out report_j.json
metric          mean  ci95                  n
ndcg@10       0.8636  [0.7406, 0.9866]      6
recall@100    1.0000  [1.0000, 1.0000]      6
mrr           0.8333  [0.6667, 1.0000]      6
wrote report_j.json (6 queries)
$ python -m rageval --version
rageval 0.1.0a0
```

The report layout is unchanged (`schema_version` 1). For a JSONL input, `dataset.qrels.path` and `run.path` are both the JSONL path; `dataset.qrels.sha256` hashes the judgments only, so two JSONL files with the same labels and different rankings can be compared and gated. `compare` and `gate` refuse to pair a JSONL report with a TREC-file report, because their hashes differ.

## Regression gate

`rageval gate` fails a CI job when a candidate is worse than the baseline by more than a drop you accept. Per metric it takes the per-query difference (candidate − baseline), bootstraps its mean, and passes only when the interval's **lower bound** is above `-max_drop`: the candidate must show it is not worse. Too little evidence is a fail, not a pass. Why this and not a p-value, and how to pick `--max-drop`: [docs/GATE.md](./docs/GATE.md).

With the two Quickstart reports (Python 3.13.12, same fresh venv):

```text
$ rageval gate --baseline report_a.json --candidate report_b.json --metric ndcg@10 --max-drop 0.02 --metric recall@100 --max-drop 0.05
warning: only 6 shared queries (< 30): the CI is wide, so the gate fails changes it cannot show to be harmless; see docs/GATE.md
metric       baseline  candidate  diff (c-b)  ci95                     threshold  verdict  n
ndcg@10      0.488957   0.863597   +0.374640  [-0.018236, +0.738854]   -0.020000  PASS     6
recall@100   0.583333   1.000000   +0.416667  [+0.083333, +0.750000]   -0.050000  PASS     6
gate: PASS (2 of 2 metrics pass; candidate run_b vs baseline run_a)
$ echo $?
0
$ rageval gate --baseline report_b.json --candidate report_a.json --metric ndcg@10 --max-drop 0.02 --metric recall@100 --max-drop 0.05
warning: only 6 shared queries (< 30): the CI is wide, so the gate fails changes it cannot show to be harmless; see docs/GATE.md
metric       baseline  candidate  diff (c-b)  ci95                     threshold  verdict  n
ndcg@10      0.863597   0.488957   -0.374640  [-0.738854, +0.018236]   -0.020000  FAIL     6
recall@100   1.000000   0.583333   -0.416667  [-0.750000, -0.083333]   -0.050000  FAIL     6
gate: FAIL (0 of 2 metrics pass; candidate run_a vs baseline run_b)
$ echo $?
10
```

`ndcg@10` passes the first run by only 0.002 although the candidate is far better: six queries leave a wide interval. In the second run its interval contains 0, so a significance test would have passed that 0.37 drop; the gate fails it. Options: `--resamples N` (10000), `--seed S` (0), `--level L` (0.95), `--format table|json`, `--summary PATH` (appends a Markdown table, for `$GITHUB_STEP_SUMMARY`). Exit codes: 0 pass, 10 a metric failed, 1 usage, 2 input (for example reports made on different qrels).

As a GitHub Action (`action.yml` at the repository root; it sets up Python, installs rageval from the action's own checkout and writes the job summary):

```yaml
- uses: AmitSinghOM/rag-evaluation-framework@<commit-sha>  # private until v0.1: inside this repo use ./
  with:
    baseline: reports/baseline.json
    candidate: reports/candidate.json
    metrics: ndcg@10:0.02,recall@100:0.05
```

The table `--summary` wrote for the failing run above (the action sends it to the job summary, with a heading, the bootstrap settings and any warning):

| metric | baseline | candidate | diff | 95% CI | threshold | verdict | n |
|---|---:|---:|---:|---|---:|---|---:|
| ndcg@10 | 0.863597 | 0.488957 | -0.374640 | [-0.738854, +0.018236] | -0.020000 | **FAIL** | 6 |
| recall@100 | 1.000000 | 0.583333 | -0.416667 | [-0.750000, -0.083333] | -0.050000 | **FAIL** | 6 |

`.github/workflows/gate-selftest.yml` runs the action on every pull request: one job must pass (identical reports) and one must fail with exit 10 (a degraded candidate).

## Use

```python
from rageval import ndcg_at_k, recall_at_k, mrr

retrieved = ["p7", "p2", "p2", "p9"]  # your retriever's top-k, best first
relevant = {"p2": 3, "p9": 1, "p4": 2}  # graded labels; or a plain set for binary

recall_at_k(retrieved, relevant, k=4)  # 2/3: p2 and p9 found, p4 missed
ndcg_at_k(retrieved, relevant, k=4)  # graded gain, log2 discount, repeat of p2 ignored
mrr(retrieved, relevant, k=10)  # 0.5: first relevant passage at rank 2
```

## Why this exists

Most RAG systems are tuned by vibes. This makes retrieval quality measurable: load a labelled dataset, run your retriever, and get real IR metrics you can trust, because they are checked against an independent implementation rather than asserted.

## The assistant pattern this framework measures

The framework exists to score a **grounded** question-answering assistant: retrieve the most relevant passages from a trusted corpus, answer only from them, cite what was used, and route low-confidence answers to a human. The point is deflecting repetitive operational queries without hallucinating — an unsupported answer is worse than "I don't know."

```
question
  → embed query
  → vector search top-k passages (hybrid BM25 + vector recommended)
  → rerank / dedupe
  → prompt: "answer ONLY from these passages, cite them, else say you don't know"
  → confidence score (max retrieval score + LLM self-check)
  → [>= threshold] answer + citations
    [<  threshold] escalate to human verification
```

Four invariants, in priority order:

1. **Retrieve-then-answer.** The LLM sees only retrieved passages, never the raw corpus. Answer quality is bounded by retrieval quality — which is why this framework measures retrieval first.
2. **Confidence gate + human-in-the-loop.** Below threshold the assistant returns "needs human verification" with the top passages attached, never a guess. This is the single biggest trust lever. Calibrate the threshold on a labelled set of in-corpus *and* out-of-corpus questions.
3. **Citations are mandatory.** Every answer names its source passages. No citation → treat as ungrounded.
4. **Knowledge-refresh lifecycle.** Re-index on a schedule with retained snapshots so a bad refresh can be rolled back; sample questions weekly to catch drift.

Worked example of the gate doing its job:

```
Q: "How do I roll back a bad orders backfill?"
Retrieved: [0.88] runbook/backfill.md#rollback, [0.71] runbook/backfill.md#verify
→ answered, with both sources cited.

Q: "What's the team's headcount budget?"        ← not in corpus
max retrieval score 0.12 < threshold
→ "I don't have a grounded source for that — routing to human verification."
```

### Retrieval tuning defaults

| Concern | Default | Why |
|---|---|---|
| Chunk size | 300–500 tokens, ~15% overlap, split on semantic boundaries | Too large dilutes retrieval; carry source metadata on every chunk — it becomes the citation |
| Search | Hybrid BM25 + vector, fused (e.g. reciprocal-rank fusion) | Hybrid beats pure-vector on exact-term queries (error codes, IDs) that embeddings blur |
| Top-k | Retrieve 8–12, rerank, keep 4–6 | Dedupe near-identical chunks so the prompt isn't wasted on repeats |
| Confidence gate | Tune on a labelled set | Below → human review, don't answer |
| Refresh cadence | Weekly, snapshot + retention | So a regressing index can be rolled back |
| Embedding model | One model per index; re-embed the whole corpus on change | Mixed-model vectors are not comparable |

Fix retrieval before prompt-tuning: if the right passage is never retrieved, no prompt change can recover the answer. Do not tune by feel — run the IR metrics in this framework against a labelled query set before and after every change.

## Development

```bash
pip install -e ".[dev]"      # pulls ranx (numba, pandas, scipy) for the cross-check only
ruff check . && ruff format --check . && mypy && pytest --cov
```

See [CONTRIBUTING.md](./CONTRIBUTING.md), [SECURITY.md](./SECURITY.md) and [CHANGELOG.md](./CHANGELOG.md).

## License

MIT. See [LICENSE](./LICENSE).
