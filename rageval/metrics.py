"""Pure-Python information-retrieval metrics.

No third-party dependencies (no NumPy). Every definition here is
cross-checked against ``ranx`` in ``tests/test_ranx_crosscheck.py``; the
exact rules, and where they differ from other tools, are written down in
``docs/METRIC-DEFINITIONS.md``.

Conventions
-----------
- ``retrieved``: an ordered sequence of document ids (best-ranked first).
- ``relevant``: either a set/list of relevant ids (binary relevance, every
  listed id has grade 1) or a mapping ``id -> grade`` (graded relevance).
  In a mapping, an id is relevant only if its grade is greater than zero;
  grade ``0`` means "judged, not relevant" as in TREC qrels and does not
  count towards recall, precision, MRR or the ideal ranking.
- ``k``: cutoff rank; only the top-``k`` retrieved ids are considered.
- Duplicates: a document id that appears more than once in ``retrieved``
  counts only at its first position. Later repeats are ignored at their
  position: they contribute no hit and no gain, and they do not move the
  documents after them up a rank. A retriever that returns the same
  passage twice therefore pays for the wasted slot (precision@k keeps the
  denominator ``k``) and cannot inflate recall or DCG.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from math import log2
from typing import TypeVar

DocId = Hashable
D = TypeVar("D", bound=Hashable)
# Binary relevance (any iterable of ids) or graded relevance (id -> grade).
Relevance = Iterable[D] | Mapping[D, float]


def _grades(relevant: Relevance[D]) -> dict[D, float]:
    """Normalise ``relevant`` to ``{id: grade}`` keeping only grades > 0."""
    if isinstance(relevant, Mapping):
        return {doc: float(grade) for doc, grade in relevant.items() if grade > 0}
    return dict.fromkeys(relevant, 1.0)


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be a positive integer")


def _top_k(retrieved: Sequence[D], k: int) -> list[D]:
    """Return the first ``k`` retrieved ids, validating ``k``."""
    _check_k(k)
    return list(retrieved[:k])


def _hits(top: Sequence[D], grades: Mapping[D, float]) -> int:
    """Number of distinct relevant ids in ``top`` (a repeat is not a second hit)."""
    return sum(1 for doc in set(top) if doc in grades)


def recall_at_k(retrieved: Sequence[D], relevant: Relevance[D], k: int) -> float:
    """Fraction of relevant documents found within the top ``k``.

    Returns 0.0 when there are no relevant documents.
    """
    grades = _grades(relevant)
    top = _top_k(retrieved, k)
    if not grades:
        return 0.0
    return _hits(top, grades) / len(grades)


def precision_at_k(retrieved: Sequence[D], relevant: Relevance[D], k: int) -> float:
    """Fraction of the top ``k`` retrieved documents that are relevant.

    The denominator is ``k`` (fixed cutoff), matching the standard P@k,
    also when fewer than ``k`` documents were retrieved.
    """
    grades = _grades(relevant)
    top = _top_k(retrieved, k)
    return _hits(top, grades) / k


def mrr(retrieved: Sequence[D], relevant: Relevance[D], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant document.

    With ``k`` given, only the top-``k`` retrieved ids are searched
    (``MRR@k``). Returns 0.0 if no relevant document appears.
    """
    grades = _grades(relevant)
    top = retrieved if k is None else _top_k(retrieved, k)
    for rank, doc in enumerate(top, start=1):
        if doc in grades:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[D], relevant: Relevance[D], k: int) -> float:
    """Discounted cumulative gain at ``k``.

    Gain is the relevance grade (1 for binary relevance) and the discount
    is the standard ``1 / log2(rank + 1)`` (Jarvelin and Kekalainen, 2002).
    Only the first occurrence of an id earns its gain.
    """
    grades = _grades(relevant)
    top = _top_k(retrieved, k)
    total = 0.0
    seen: set[D] = set()
    for rank, doc in enumerate(top, start=1):
        if doc in seen:
            continue
        seen.add(doc)
        if doc in grades:
            total += grades[doc] / log2(rank + 1)
    return total


def ndcg_at_k(retrieved: Sequence[D], relevant: Relevance[D], k: int) -> float:
    """Normalised DCG at ``k``.

    The ideal ranking lists the relevant documents by descending grade,
    capped at ``k``. Returns 0.0 when there are no relevant documents.
    """
    grades = _grades(relevant)
    _check_k(k)
    if not grades:
        return 0.0
    dcg = dcg_at_k(retrieved, grades, k)
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(gain / log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg
