"""Tiny in-memory dataset loader for Phase 0.

A dataset is a list of ``(query, retrieved, relevant)`` triples. ``load``
normalises each triple into a :class:`Record` so the evaluator can iterate
over labelled examples without caring about the source format.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .metrics import DocId, Relevance

Triple = tuple[str, Sequence[DocId], Relevance[DocId]]


@dataclass
class Record:
    """A single labelled evaluation example.

    ``relevant`` is a set of ids for binary relevance or an ``id -> grade``
    mapping for graded relevance (see ``rageval.metrics``).
    """

    query: str
    retrieved: list[DocId]
    relevant: set[DocId] | dict[DocId, float] = field(default_factory=set)


def load(rows: Iterable[Triple]) -> list[Record]:
    """Build a list of :class:`Record` from ``(query, retrieved, relevant)`` triples.

    Retrieved order is preserved (duplicates included; the metrics ignore
    repeats). A mapping of grades is kept as a dict so graded relevance
    survives loading; any other collection of ids becomes a set.
    """
    records: list[Record] = []
    for row in rows:
        query, retrieved, relevant = row
        grades: set[DocId] | dict[DocId, float]
        if isinstance(relevant, Mapping):
            grades = dict(relevant)
        else:
            grades = set(relevant)
        records.append(Record(query=query, retrieved=list(retrieved), relevant=grades))
    return records
