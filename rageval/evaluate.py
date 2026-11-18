"""Score a run against qrels for a list of named metrics and build a :class:`Report`.

Metric names: ``ndcg@K``, ``recall@K``, ``precision@K``, ``mrr`` and
``mrr@K`` (lowercase, ``K`` a positive integer).

Rules (``docs/REPORT-SCHEMA.md``): every query in the qrels is scored and
counted in ``n_queries``; a query missing from the run scores ``0`` for
every metric; a query in the run but not in the qrels is ignored and named
in ``warnings``. Per-query rows follow qrels order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime

from .loaders import Qrels, Run
from .metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from .report import (
    Bootstrap,
    DatasetInfo,
    FileRef,
    MetricSummary,
    Report,
    RunInfo,
)
from .stats import bootstrap_ci, mean

_METRIC_FORMS = "ndcg@K, recall@K, precision@K, mrr, mrr@K (K a positive integer)"
_AT_K: dict[str, Callable[[Sequence[str], Mapping[str, float], int], float]] = {
    "ndcg": ndcg_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr,
}


def parse_metric(name: str) -> tuple[str, int | None]:
    """``"ndcg@10" -> ("ndcg", 10)``, ``"mrr" -> ("mrr", None)``; ``ValueError`` otherwise."""
    base, at, k_text = name.partition("@")
    if base not in _AT_K or (at and not k_text.isdigit()) or (not at and base != "mrr"):
        raise ValueError(f"unknown metric {name!r}; valid forms: {_METRIC_FORMS}")
    if not at:
        return base, None
    k = int(k_text)
    if k < 1:
        raise ValueError(f"unknown metric {name!r}; valid forms: {_METRIC_FORMS}")
    return base, k


def canonical_metric(name: str) -> str:
    """``"ndcg@010" -> "ndcg@10"``: the key a report and ``rageval gate --metric`` agree on."""
    base, k = parse_metric(name)
    return base if k is None else f"{base}@{k}"


def _score(name: str, retrieved: Sequence[str], grades: Mapping[str, float]) -> float:
    base, k = parse_metric(name)
    if k is None:
        return mrr(retrieved, grades)
    return _AT_K[base](retrieved, grades, k)


def evaluate(
    qrels: Qrels,
    run: Run,
    metrics: Sequence[str],
    *,
    resamples: int = 1000,
    seed: int = 0,
    qrels_ref: FileRef | None = None,
    run_ref: FileRef | None = None,
    run_name: str | None = None,
) -> Report:
    """Score ``run`` against ``qrels`` for ``metrics`` and summarise with bootstrap CIs.

    ``qrels_ref`` and ``run_ref`` record where the inputs came from; they are
    ``None`` fields in the report when evaluating in-memory data.
    """
    names = list(dict.fromkeys(canonical_metric(name) for name in metrics))
    if not names:
        raise ValueError("at least one metric is required")
    if not qrels:
        raise ValueError("qrels contain no queries")
    per_query: dict[str, dict[str, float]] = {}
    for qid, grades in qrels.items():
        retrieved = run.get(qid, [])
        per_query[qid] = {name: _score(name, retrieved, grades) for name in names}
    warnings = [
        f"query {qid!r} is in the run but not in the qrels; ignored"
        for qid in run
        if qid not in qrels
    ]
    bootstrap = Bootstrap(resamples=resamples, seed=seed)
    summaries: dict[str, MetricSummary] = {}
    for name in names:
        values = [scores[name] for scores in per_query.values()]
        summaries[name] = MetricSummary(
            mean=mean(values),
            ci95=bootstrap_ci(values, resamples=resamples, seed=seed),
            n=len(values),
            bootstrap=bootstrap,
        )
    from . import __version__  # at call time: the package is fully imported by then

    return Report(
        rageval_version=__version__,
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        dataset=DatasetInfo(
            qrels=qrels_ref or FileRef(None, None),
            n_queries=len(qrels),
            n_judged_relevant=sum(1 for g in qrels.values() for grade in g.values() if grade > 0),
        ),
        run=RunInfo(
            name=run_name,
            path=run_ref.path if run_ref else None,
            sha256=run_ref.sha256 if run_ref else None,
            n_queries_in_run=len(run),
            duplicate_ids=sum(len(docs) - len(set(docs)) for docs in run.values()),
        ),
        metrics=summaries,
        per_query=per_query,
        warnings=warnings,
    )
