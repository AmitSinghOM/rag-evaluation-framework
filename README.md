# rag-evaluation-framework

**Measures retrieval quality for RAG systems: Recall@K, Precision@K, MRR, NDCG with graded relevance, in pure Python with no runtime dependencies, and every value cross-checked against [ranx](https://github.com/AmenRa/ranx).**

> **Status (2026-11-18): Phase 0 metrics corrected and reference-checked; Phase 1 (loaders, report, CLI) in progress; nothing published.**
> The package is not on PyPI. Install from source (below). The repository is private until v0.1.
> Scope and the reasons for it: [docs/decisions/ADR-0001-v0.1-scope.md](./docs/decisions/ADR-0001-v0.1-scope.md).

## What works today

- `recall_at_k`, `precision_at_k`, `mrr` (with optional `k`), `dcg_at_k`, `ndcg_at_k` over ordered id lists.
- Binary relevance (a set of ids) or graded relevance (a mapping `id -> grade`; grade 0 = judged not relevant).
- Duplicate ids in a ranking count once, at their first position; later repeats are wasted slots and cannot inflate a score.
- Every rule is written down in [docs/METRIC-DEFINITIONS.md](./docs/METRIC-DEFINITIONS.md) and locked by a test that scores 256 synthetic queries through both rageval and ranx and requires agreement within 1e-9.
- Typed (`py.typed`), mypy strict, ruff, a 95% branch-coverage floor enforced in CI, CI on Python 3.11-3.14 (Ubuntu) and 3.13 (macOS).
- A tiny in-memory loader (`load`) that turns `(query, retrieved, relevant)` triples into records.

Not yet: file loaders, a report, a CLI, a regression gate, a grounding check. See [ROADMAP.md](./ROADMAP.md).

## Install (from source)

```bash
pip install git+https://github.com/AmitSinghOM/rag-evaluation-framework.git
```

Python 3.11 or newer. The install brings no third-party package.

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
