# rag-evaluation-framework

**Measures retrieval quality for RAG systems: Recall@K, Precision@K, MRR, NDCG with graded relevance, in pure Python with no runtime dependencies, and every value cross-checked against [ranx](https://github.com/AmenRa/ranx).**

> **Status (2026-11-18): Phase 0 metrics corrected and reference-checked; Phase 1 loaders, versioned report with bootstrap CIs and the `rageval run` / `rageval compare` CLI done; nothing published.**
> The package is not on PyPI. Install from source (below). The repository is private until v0.1.
> Scope and the reasons for it: [docs/decisions/ADR-0001-v0.1-scope.md](./docs/decisions/ADR-0001-v0.1-scope.md).

## What works today

- `recall_at_k`, `precision_at_k`, `mrr` (with optional `k`), `dcg_at_k`, `ndcg_at_k` over ordered id lists.
- Binary relevance (a set of ids) or graded relevance (a mapping `id -> grade`; grade 0 = judged not relevant).
- Duplicate ids in a ranking count once, at their first position; later repeats are wasted slots and cannot inflate a score.
- Every rule is written down in [docs/METRIC-DEFINITIONS.md](./docs/METRIC-DEFINITIONS.md) and locked by a test that scores 256 synthetic queries through both rageval and ranx and requires agreement within 1e-9.
- Loaders for TREC qrels, TREC run files and a JSONL dataset format (`read_qrels`, `read_run`, `read_jsonl`); a malformed line fails with the file and line number.
- `rageval run`: a versioned JSON report (`schema_version` 1, [docs/REPORT-SCHEMA.md](./docs/REPORT-SCHEMA.md), [docs/report.schema.json](./docs/report.schema.json)) with per-query scores, means and seeded bootstrap 95% confidence intervals; `rageval compare`: paired per-query differences with their CI, refusing reports made on different qrels. Means agree with `ranx.evaluate` on the same files within 1e-9.
- Typed (`py.typed`), mypy strict, ruff, a 95% branch-coverage floor enforced in CI, CI on Python 3.11-3.14 (Ubuntu) and 3.13 (macOS).
- A tiny in-memory loader (`load`) that turns `(query, retrieved, relevant)` triples into records.

Not yet: a regression gate with an exit code, a GitHub Action, a grounding check, JSONL input on the command line. See [ROADMAP.md](./ROADMAP.md).

## Install (from source)

```bash
pip install git+https://github.com/AmitSinghOM/rag-evaluation-framework.git
```

Python 3.11 or newer. The install brings no third-party package.

## Quickstart (Phase 1)

From a clone of this repository, in a fresh virtual environment. The toy fixtures are six judged queries and two runs, small enough to check by hand ([examples/toy/README.md](./examples/toy/README.md)).

```bash
pip install .
rageval run --qrels examples/toy/qrels.txt --run examples/toy/run_a.txt --metrics ndcg@10,recall@100,mrr --out report_a.json
rageval run --qrels examples/toy/qrels.txt --run examples/toy/run_b.txt --metrics ndcg@10,recall@100,mrr --out report_b.json
rageval compare report_a.json report_b.json
```

Real output of those lines (Python 3.13.12, run while writing this section; `pip list` showed `rageval==0.1.0a0` and nothing else):

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

Metric names: `ndcg@K`, `recall@K`, `precision@K`, `mrr`, `mrr@K`. Options: `--resamples N` (1000), `--seed S` (0), `--name NAME`; `compare --format json`. Exit codes: 0 ok, 1 usage, 2 input or format error, 3 internal. A query in the qrels that the run never answers scores 0; a query in the run that is not judged is ignored and listed under `warnings`. Six queries give the wide intervals above by design; the point of the CI is that on a real query set it tells you whether a difference is noise.

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
