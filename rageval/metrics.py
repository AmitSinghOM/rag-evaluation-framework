"""Pure-Python information-retrieval metrics.

No third-party dependencies (no NumPy) — Phase 0 of the
rag-evaluation-framework: measurable IR metrics over labelled data.

Conventions
-----------
- ``retrieved``: an ordered list of document ids (best-ranked first).
- ``relevant``: a set/list of relevant document ids (order irrelevant).
- ``k``: cutoff rank; only the top-``k`` retrieved ids are considered.
"""

from __future__ import annotations

from math import log2
from typing import Hashable, Iterable, List, Sequence, Set

DocId = Hashable


def _as_set(relevant: Iterable[DocId]) -> Set[DocId]:
    """Normalise the relevant collection to a set."""
    return set(relevant)


def _top_k(retrieved: Sequence[DocId], k: int) -> List[DocId]:
    """Return the first ``k`` retrieved ids, validating ``k``."""
    if k <= 0:
        raise ValueError("k must be a positive integer")
    return list(retrieved[:k])


def recall_at_k(retrieved: Sequence[DocId], relevant: Iterable[DocId], k: int) -> float:
    """Fraction of relevant documents found within the top ``k``.

    Returns 0.0 when there are no relevant documents.
    """
    relevant_set = _as_set(relevant)
    if not relevant_set:
        return 0.0
    top = _top_k(retrieved, k)
    hits = sum(1 for doc in top if doc in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(retrieved: Sequence[DocId], relevant: Iterable[DocId], k: int) -> float:
    """Fraction of the top ``k`` retrieved documents that are relevant.

    The denominator is ``k`` (fixed cutoff), matching the standard P@k.
    """
    relevant_set = _as_set(relevant)
    top = _top_k(retrieved, k)
    hits = sum(1 for doc in top if doc in relevant_set)
    return hits / k


def mrr(retrieved: Sequence[DocId], relevant: Iterable[DocId]) -> float:
    """Reciprocal rank of the first relevant document.

    Returns 0.0 if no relevant document appears in ``retrieved``.
    """
    relevant_set = _as_set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[DocId], relevant: Iterable[DocId], k: int) -> float:
    """Discounted cumulative gain at ``k`` with binary relevance.

    Uses the standard discount ``1 / log2(rank + 1)`` for each relevant hit.
    """
    relevant_set = _as_set(relevant)
    top = _top_k(retrieved, k)
    total = 0.0
    for rank, doc in enumerate(top, start=1):
        if doc in relevant_set:
            total += 1.0 / log2(rank + 1)
    return total


def ndcg_at_k(retrieved: Sequence[DocId], relevant: Iterable[DocId], k: int) -> float:
    """Normalised DCG at ``k`` with binary relevance.

    The ideal DCG assumes as many relevant docs as possible are ranked first,
    capped by both ``k`` and the number of relevant documents. Returns 0.0
    when there are no relevant documents.
    """
    relevant_set = _as_set(relevant)
    if not relevant_set:
        return 0.0
    dcg = dcg_at_k(retrieved, relevant_set, k)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
