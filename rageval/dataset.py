"""Tiny in-memory dataset loader for Phase 0.

A dataset is a list of ``(query, retrieved, relevant)`` triples. ``load``
normalises each triple into a :class:`Record` so the evaluator can iterate
over labelled examples without caring about the source format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, List, Sequence, Set, Tuple

DocId = Hashable
Triple = Tuple[str, Sequence[DocId], Iterable[DocId]]


@dataclass
class Record:
    """A single labelled evaluation example."""

    query: str
    retrieved: List[DocId]
    relevant: Set[DocId] = field(default_factory=set)


def load(rows: Iterable[Triple]) -> List[Record]:
    """Build a list of :class:`Record` from ``(query, retrieved, relevant)`` triples.

    Retrieved order is preserved; relevant ids are collected into a set.
    """
    records: List[Record] = []
    for row in rows:
        query, retrieved, relevant = row
        records.append(
            Record(
                query=query,
                retrieved=list(retrieved),
                relevant=set(relevant),
            )
        )
    return records
