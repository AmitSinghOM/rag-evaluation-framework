"""Read TREC qrels, TREC run files and the rageval JSONL dataset format.

Formats (also in ``docs/REPORT-SCHEMA.md``):

- qrels: ``qid 0 docid grade``; whitespace-separated; ``grade`` is an int
  ``>= 0`` and ``0`` means "judged, not relevant". The second field is
  unused and not checked, as in ``trec_eval``.
- run: ``qid Q0 docid rank score tag``; ``rank`` int, ``score`` float.
  Per query the documents are ordered by score descending, then rank
  ascending, then file order. Repeated doc ids are KEPT: the metrics ignore
  a repeat at its position and the report counts them.
- JSONL: one object per line,
  ``{"query_id": str, "query": str|null, "retrieved": [docid, ...],
  "relevant": {docid: grade} | [docid, ...]}``. A list of relevant ids means
  every id has grade ``1``. Ids must be strings.

Any line the reader cannot use raises :class:`LoaderError` naming the file
and the 1-based line number. Blank lines are skipped. A loader never drops
or overwrites data silently: a repeated ``(qid, docid)`` judgment in qrels
and a repeated ``query_id`` in JSONL are errors.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from math import isfinite, isnan
from pathlib import Path

Qrels = dict[str, dict[str, float]]
"""``qid -> {docid: grade}``; a grade of ``0`` is judged-not-relevant."""

Run = dict[str, list[str]]
"""``qid -> [docid, ...]`` best first; may contain repeats."""


class LoaderError(ValueError):
    """A file could not be read as the format expected (``path:line: reason``)."""

    def __init__(self, path: str | Path, line_no: int, reason: str) -> None:
        super().__init__(f"{path}:{line_no}: {reason}")
        self.path = str(path)
        self.line_no = line_no
        self.reason = reason


def _lines(path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield ``(line_no, stripped_line)`` for every non-blank line.

    UTF-8 with or without a BOM (Windows editors write one; ``str.strip`` does
    not remove it, so it would silently become part of the first query id).
    Any other encoding is a :class:`LoaderError` naming the file.
    """
    try:
        with Path(path).open(encoding="utf-8-sig") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.strip()
                if line:
                    yield line_no, line
    except UnicodeDecodeError as exc:
        raise LoaderError(
            path, 0, f"not UTF-8 ({exc.reason} at byte {exc.start}); re-save the file as UTF-8"
        ) from None


def read_qrels(path: str | Path) -> Qrels:
    """Read a TREC qrels file into ``qid -> {docid: grade}``."""
    qrels: Qrels = {}
    for line_no, line in _lines(path):
        fields = line.split()
        if len(fields) != 4:
            raise LoaderError(
                path, line_no, f"expected 4 fields 'qid 0 docid grade', got {len(fields)}"
            )
        qid, _, docid, grade_text = fields
        try:
            grade = int(grade_text)
        except ValueError:
            raise LoaderError(
                path, line_no, f"grade must be an integer, got {grade_text!r}"
            ) from None
        if grade < 0:
            raise LoaderError(path, line_no, f"grade must be >= 0, got {grade}")
        judged = qrels.setdefault(qid, {})
        if docid in judged:
            raise LoaderError(path, line_no, f"duplicate judgment for query {qid!r} doc {docid!r}")
        judged[docid] = float(grade)
    return qrels


def read_run(path: str | Path) -> Run:
    """Read a TREC run file into ``qid -> [docid, ...]`` best first, repeats kept."""
    entries: dict[str, list[tuple[float, int, str]]] = {}
    for line_no, line in _lines(path):
        fields = line.split()
        if len(fields) != 6:
            raise LoaderError(
                path, line_no, f"expected 6 fields 'qid Q0 docid rank score tag', got {len(fields)}"
            )
        qid, _, docid, rank_text, score_text, _ = fields
        try:
            rank = int(rank_text)
        except ValueError:
            raise LoaderError(
                path, line_no, f"rank must be an integer, got {rank_text!r}"
            ) from None
        try:
            score = float(score_text)
        except ValueError:
            raise LoaderError(
                path, line_no, f"score must be a number, got {score_text!r}"
            ) from None
        if isnan(score):
            raise LoaderError(path, line_no, "score must not be NaN")
        entries.setdefault(qid, []).append((score, rank, docid))
    run: Run = {}
    for qid, docs in entries.items():
        # Stable sort: equal (score, rank) keep file order.
        docs.sort(key=lambda entry: (-entry[0], entry[1]))
        run[qid] = [docid for _, _, docid in docs]
    return run


def _ids(path: str | Path, line_no: int, value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(doc, str) for doc in value):
        raise LoaderError(path, line_no, f"{field!r} must be a list of string ids")
    return list(value)


def _grades(path: str | Path, line_no: int, value: object) -> dict[str, float]:
    if isinstance(value, list):
        return dict.fromkeys(_ids(path, line_no, value, "relevant"), 1.0)
    if not isinstance(value, dict):
        raise LoaderError(
            path, line_no, "'relevant' must be a list of ids or an object {id: grade}"
        )
    grades: dict[str, float] = {}
    for docid, grade in value.items():
        # json.loads accepts the non-standard literals NaN and Infinity; neither is a grade.
        if (
            isinstance(grade, bool)
            or not isinstance(grade, int | float)
            or not isfinite(grade)
            or grade < 0
        ):
            raise LoaderError(
                path, line_no, f"grade for doc {docid!r} must be a finite number >= 0"
            )
        grades[docid] = float(grade)
    return grades


def read_jsonl(path: str | Path) -> tuple[Qrels, Run]:
    """Read a JSONL dataset into ``(qrels, run)`` keyed by ``query_id``."""
    qrels: Qrels = {}
    run: Run = {}
    for line_no, line in _lines(path):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LoaderError(path, line_no, f"invalid JSON: {exc.msg}") from None
        if not isinstance(obj, dict):
            raise LoaderError(path, line_no, "expected a JSON object")
        qid = obj.get("query_id")
        if not isinstance(qid, str):
            raise LoaderError(path, line_no, "'query_id' must be a string")
        if qid in qrels:
            raise LoaderError(path, line_no, f"duplicate query_id {qid!r}")
        if "retrieved" not in obj or "relevant" not in obj:
            raise LoaderError(path, line_no, "object needs 'retrieved' and 'relevant'")
        run[qid] = _ids(path, line_no, obj["retrieved"], "retrieved")
        qrels[qid] = _grades(path, line_no, obj["relevant"])
    return qrels, run
