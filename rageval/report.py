"""The versioned evaluation report (``schema_version`` 1) and report comparison.

The JSON layout is fixed by ``docs/report.schema.json`` and explained field
by field in ``docs/REPORT-SCHEMA.md``. Floats are rounded to 6 decimal
places when written. Two reports made with the same inputs and seed are
byte-identical apart from ``created_at``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .stats import mean, paired_diff_ci

SCHEMA_VERSION = 1
PLACES = 6


class ReportError(ValueError):
    """A report file has the wrong shape, or two reports cannot be compared."""


@dataclass(frozen=True)
class FileRef:
    """Where an input came from. ``None`` when evaluated from in-memory data."""

    path: str | None
    sha256: str | None

    @classmethod
    def of(cls, path: str | Path) -> FileRef:
        """Reference a file by the path as given plus the SHA-256 of its bytes."""
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return cls(path=str(path), sha256=digest)

    @classmethod
    def of_judgments(cls, path: str | Path, qrels: Mapping[str, Mapping[str, float]]) -> FileRef:
        """Reference the judgments read from ``path`` by the SHA-256 of their canonical JSON.

        For a JSONL dataset, which holds the judgments and a run in one file:
        two files with the same judgments and different rankings get the same
        hash, so ``compare`` and ``gate`` accept their reports.
        """
        canonical = json.dumps(qrels, sort_keys=True, separators=(",", ":"))
        return cls(path=str(path), sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest())


@dataclass(frozen=True)
class DatasetInfo:
    qrels: FileRef
    n_queries: int
    n_judged_relevant: int


@dataclass(frozen=True)
class RunInfo:
    name: str | None
    path: str | None
    sha256: str | None
    n_queries_in_run: int
    duplicate_ids: int


@dataclass(frozen=True)
class Bootstrap:
    resamples: int
    seed: int


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    ci95: tuple[float, float]
    n: int
    bootstrap: Bootstrap


@dataclass(frozen=True)
class Report:
    rageval_version: str
    created_at: str
    dataset: DatasetInfo
    run: RunInfo
    metrics: dict[str, MetricSummary]
    per_query: dict[str, dict[str, float]]
    warnings: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """The JSON object, in file order, floats rounded to 6 places."""
        return {
            "schema_version": self.schema_version,
            "rageval_version": self.rageval_version,
            "created_at": self.created_at,
            "dataset": {
                "qrels": {"path": self.dataset.qrels.path, "sha256": self.dataset.qrels.sha256},
                "n_queries": self.dataset.n_queries,
                "n_judged_relevant": self.dataset.n_judged_relevant,
            },
            "run": {
                "name": self.run.name,
                "path": self.run.path,
                "sha256": self.run.sha256,
                "n_queries_in_run": self.run.n_queries_in_run,
                "duplicate_ids": self.run.duplicate_ids,
            },
            "metrics": {
                name: {
                    "mean": round(summary.mean, PLACES),
                    "ci95": [round(summary.ci95[0], PLACES), round(summary.ci95[1], PLACES)],
                    "n": summary.n,
                    "bootstrap": {
                        "resamples": summary.bootstrap.resamples,
                        "seed": summary.bootstrap.seed,
                    },
                }
                for name, summary in self.metrics.items()
            },
            "per_query": {
                qid: {name: round(value, PLACES) for name, value in scores.items()}
                for qid, scores in self.per_query.items()
            },
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Any) -> Report:
        """Rebuild a report from its JSON object; ``ReportError`` on a wrong shape."""
        try:
            if data["schema_version"] != SCHEMA_VERSION:
                raise ReportError(
                    f"unsupported schema_version {data['schema_version']!r}; "
                    f"this rageval reads {SCHEMA_VERSION}"
                )
            dataset, run = data["dataset"], data["run"]
            return cls(
                rageval_version=data["rageval_version"],
                created_at=data["created_at"],
                dataset=DatasetInfo(
                    qrels=FileRef(dataset["qrels"]["path"], dataset["qrels"]["sha256"]),
                    n_queries=dataset["n_queries"],
                    n_judged_relevant=dataset["n_judged_relevant"],
                ),
                run=RunInfo(
                    name=run["name"],
                    path=run["path"],
                    sha256=run["sha256"],
                    n_queries_in_run=run["n_queries_in_run"],
                    duplicate_ids=run["duplicate_ids"],
                ),
                metrics={
                    name: MetricSummary(
                        mean=summary["mean"],
                        ci95=(summary["ci95"][0], summary["ci95"][1]),
                        n=summary["n"],
                        bootstrap=Bootstrap(
                            summary["bootstrap"]["resamples"], summary["bootstrap"]["seed"]
                        ),
                    )
                    for name, summary in data["metrics"].items()
                },
                per_query={qid: dict(scores.items()) for qid, scores in data["per_query"].items()},
                warnings=list(data.get("warnings", [])),
            )
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise ReportError(f"not a rageval report: missing or malformed {exc!r}") from None

    @classmethod
    def read(cls, path: str | Path) -> Report:
        """Load a report file; ``ReportError`` if it is not valid JSON of the right shape."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReportError(f"{path}: invalid JSON: {exc.msg}") from None
        return cls.from_dict(data)


@dataclass(frozen=True)
class MetricDiff:
    mean_a: float
    mean_b: float
    diff: float  # mean over queries of (b - a)
    ci95: tuple[float, float]


@dataclass(frozen=True)
class Comparison:
    a: RunInfo
    b: RunInfo
    qrels_sha256: str | None
    n_queries: int
    bootstrap: Bootstrap
    metrics: dict[str, MetricDiff]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "a": {"name": self.a.name, "path": self.a.path, "sha256": self.a.sha256},
            "b": {"name": self.b.name, "path": self.b.path, "sha256": self.b.sha256},
            "qrels_sha256": self.qrels_sha256,
            "n_queries": self.n_queries,
            "bootstrap": {"resamples": self.bootstrap.resamples, "seed": self.bootstrap.seed},
            "metrics": {
                name: {
                    "a": round(d.mean_a, PLACES),
                    "b": round(d.mean_b, PLACES),
                    "diff": round(d.diff, PLACES),
                    "ci95": [round(d.ci95[0], PLACES), round(d.ci95[1], PLACES)],
                }
                for name, d in self.metrics.items()
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def to_table(self) -> str:
        """Fixed-width text: one row per metric, difference is ``b - a``."""
        name_a = self.a.name or "a"
        name_b = self.b.name or "b"
        width = max(len("metric"), *(len(name) for name in self.metrics))
        header = f"{'metric':<{width}}  {name_a:>12}  {name_b:>12}  {'diff (b-a)':>12}  ci95"
        lines = [header]
        for name, d in self.metrics.items():
            lo, hi = d.ci95
            lines.append(
                f"{name:<{width}}  {d.mean_a:>12.6f}  {d.mean_b:>12.6f}  {d.diff:>+12.6f}"
                f"  [{lo:+.6f}, {hi:+.6f}]"
            )
        lines.append(f"n_queries: {self.n_queries}")
        return "\n".join(lines) + "\n"


def per_query_values(report: Report, which: str, name: str, qids: list[str]) -> list[float]:
    """The per-query column for ``name``; ``ReportError`` on a gap or a non-finite value.

    The file is untrusted JSON: Python's json accepts NaN and Infinity, and a
    hand-edited row can hold anything, so the type is checked as well as
    finiteness. Shared by ``compare`` and ``gate`` so both refuse the same files.
    """
    values: list[float] = []
    for qid in qids:
        value: object = report.per_query[qid].get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ReportError(
                f"{which} report: per_query[{qid!r}][{name!r}] is {value!r}, not a finite number"
            )
        values.append(float(value))
    return values


def compare(a: Report, b: Report) -> Comparison:
    """Paired comparison of two reports on the same qrels (``b - a`` per metric).

    Resamples the paired per-query differences with report ``a``'s bootstrap
    settings. Refuses with ``ReportError`` when the qrels SHA-256 differ,
    when the per-query ids differ, or when no metric is in both reports.
    """
    if a.dataset.qrels.sha256 != b.dataset.qrels.sha256:
        raise ReportError(
            "reports were made on different qrels: "
            f"{a.dataset.qrels.sha256} != {b.dataset.qrels.sha256}"
        )
    if a.per_query.keys() != b.per_query.keys():
        raise ReportError("reports do not score the same query ids")
    shared = [name for name in a.metrics if name in b.metrics]
    if not shared:
        raise ReportError("no metric is present in both reports")
    bootstrap = next(iter(a.metrics.values())).bootstrap
    qids = list(a.per_query)
    metrics: dict[str, MetricDiff] = {}
    for name in shared:
        values_a = per_query_values(a, "a", name, qids)
        values_b = per_query_values(b, "b", name, qids)
        diffs = [y - x for x, y in zip(values_a, values_b, strict=True)]
        metrics[name] = MetricDiff(
            mean_a=a.metrics[name].mean,
            mean_b=b.metrics[name].mean,
            diff=mean(diffs),
            ci95=paired_diff_ci(
                values_a, values_b, resamples=bootstrap.resamples, seed=bootstrap.seed
            ),
        )
    return Comparison(
        a=a.run,
        b=b.run,
        qrels_sha256=a.dataset.qrels.sha256,
        n_queries=len(qids),
        bootstrap=bootstrap,
        metrics=metrics,
    )
