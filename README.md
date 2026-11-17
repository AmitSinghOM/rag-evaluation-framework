# rag-evaluation-framework

**A framework for measuring retrieval quality — Recall@K, Precision@K, MRR, NDCG — and detecting hallucination in generated answers.**

> **Status: planned.** Part of the AI Platform Engineer blueprint. Phase 0 skeleton only (the four IR metrics in `rageval/metrics.py`, 9 tests). Build order: **AgentOS first**, then earn the right to start the next one by finishing the last.

## Why this exists

Most RAG systems are tuned by vibes. This makes retrieval quality measurable: load a labelled dataset, run your retriever, and get real IR metrics plus a grounding check on generated answers.

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

## Architecture (planned)

```
Dataset Loader → Retriever → Evaluator (Recall@K, Precision@K, MRR, NDCG) → Hallucination Detector → Score Dashboard
```

## Components

- Dataset loader (queries + relevance labels)
- Pluggable retriever interface
- IR metrics: Recall@K, Precision@K, MRR, NDCG
- Hallucination / grounding detector
- Score dashboard + run comparison

## Tech stack

Python · NumPy · a vector store (FAISS/Chroma) · pytest · small dashboard

See [ROADMAP.md](./ROADMAP.md) for the phased build plan.
