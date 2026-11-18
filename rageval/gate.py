"""Non-inferiority regression gate: is the candidate no worse than the baseline?

For each gated metric, take the paired per-query difference
``candidate - baseline`` over the queries both reports score, and its
percentile bootstrap interval (:func:`rageval.stats.paired_diff_ci`). The
metric passes when the interval's lower bound is above ``-max_drop``, that
is, when the data show the candidate loses less than ``max_drop``. A
candidate that cannot show this, because it really is worse or because the
query set is too small to tell, fails. ``docs/GATE.md`` explains the choice.

The verdict is taken on the lower bound rounded to 6 places, the precision
of the report files, so it always agrees with the printed numbers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .report import PLACES, SCHEMA_VERSION, Report, ReportError, RunInfo, per_query_values
from .stats import mean, paired_diff_ci

SMALL_N = 30  # below this many shared queries the gate warns that the CI is wide


@dataclass(frozen=True)
class MetricVerdict:
    baseline: float  # mean over the shared queries
    candidate: float
    diff: float  # mean over the shared queries of (candidate - baseline)
    ci: tuple[float, float]
    max_drop: float
    passed: bool
    n: int


@dataclass(frozen=True)
class GateResult:
    baseline: RunInfo
    candidate: RunInfo
    qrels_sha256: str | None
    n_queries: int
    resamples: int
    seed: int
    level: float
    metrics: dict[str, MetricVerdict]

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.metrics.values())

    @property
    def warnings(self) -> list[str]:
        if self.n_queries >= SMALL_N:
            return []
        return [
            f"only {self.n_queries} shared queries (< {SMALL_N}): the CI is wide, so the "
            "gate fails changes it cannot show to be harmless; see docs/GATE.md"
        ]

    def _ci_label(self) -> str:
        return f"ci{self.level * 100:g}"

    def to_dict(self) -> dict[str, Any]:
        def ref(run: RunInfo) -> dict[str, str | None]:
            return {"name": run.name, "path": run.path, "sha256": run.sha256}

        return {
            "schema_version": SCHEMA_VERSION,
            "verdict": "pass" if self.passed else "fail",
            "baseline": ref(self.baseline),
            "candidate": ref(self.candidate),
            "qrels_sha256": self.qrels_sha256,
            "n_queries": self.n_queries,
            "bootstrap": {"resamples": self.resamples, "seed": self.seed, "level": self.level},
            "metrics": {
                name: {
                    "baseline": round(v.baseline, PLACES),
                    "candidate": round(v.candidate, PLACES),
                    "diff": round(v.diff, PLACES),
                    "ci": [round(v.ci[0], PLACES), round(v.ci[1], PLACES)],
                    "max_drop": v.max_drop,
                    "verdict": "pass" if v.passed else "fail",
                    "n": v.n,
                }
                for name, v in self.metrics.items()
            },
            "warnings": self.warnings,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    def _footer(self) -> str:
        passed = sum(v.passed for v in self.metrics.values())
        return (
            f"gate: {'PASS' if self.passed else 'FAIL'} ({passed} of {len(self.metrics)} "
            f"metrics pass; candidate {self.candidate.name or '?'} vs baseline "
            f"{self.baseline.name or '?'})"
        )

    def to_table(self) -> str:
        """Fixed-width text: one row per metric; a row passes when ``ci[0] > threshold``."""
        width = max(len("metric"), *(len(name) for name in self.metrics))
        lines = [
            f"{'metric':<{width}}  {'baseline':>9}  {'candidate':>9}  {'diff (c-b)':>10}  "
            f"{self._ci_label():<22}  {'threshold':>10}  verdict  n"
        ]
        for name, v in self.metrics.items():
            ci = f"[{v.ci[0]:+.6f}, {v.ci[1]:+.6f}]"
            lines.append(
                f"{name:<{width}}  {v.baseline:>9.6f}  {v.candidate:>9.6f}  {v.diff:>+10.6f}  "
                f"{ci:<22}  {-v.max_drop:>+10.6f}  {'PASS' if v.passed else 'FAIL':<7}  {v.n}"
            )
        lines.append(self._footer())
        return "\n".join(lines) + "\n"

    def to_markdown(self) -> str:
        """A Markdown table for ``$GITHUB_STEP_SUMMARY``."""
        lines = [
            f"### rageval gate: {'PASS' if self.passed else 'FAIL'}",
            "",
            f"| metric | baseline | candidate | diff | {self.level * 100:g}% CI | threshold "
            "| verdict | n |",
            "|---|---:|---:|---:|---|---:|---|---:|",
        ]
        for name, v in self.metrics.items():
            verdict = "PASS" if v.passed else "**FAIL**"
            lines.append(
                f"| {name} | {v.baseline:.6f} | {v.candidate:.6f} | {v.diff:+.6f} "
                f"| [{v.ci[0]:+.6f}, {v.ci[1]:+.6f}] | {-v.max_drop:+.6f} | {verdict} | {v.n} |"
            )
        lines += [
            "",
            f"Candidate {self.candidate.name or '?'} against baseline {self.baseline.name or '?'}"
            f", {self.n_queries} shared queries, {self.resamples} paired bootstrap resamples, "
            f"seed {self.seed}. A metric passes when the lower CI bound of "
            "(candidate - baseline) is above its threshold (-max drop).",
        ]
        lines += [f"\n> warning: {w}" for w in self.warnings]
        return "\n".join(lines) + "\n"


def gate(
    baseline: Report,
    candidate: Report,
    max_drops: Mapping[str, float],
    *,
    resamples: int = 10000,
    seed: int = 0,
    level: float = 0.95,
) -> GateResult:
    """Check each metric in ``max_drops`` for non-inferiority of ``candidate``.

    ``ValueError`` (a usage error) when ``max_drops`` is empty, a max drop is
    not a finite number above 0, or a metric is missing from either report.
    ``ReportError`` (an input error) when the reports were made on different
    qrels, share no query, or hold a missing or non-finite per-query value.
    """
    if not max_drops:
        raise ValueError("the gate needs at least one metric with its max drop")
    for name, max_drop in max_drops.items():
        if not (math.isfinite(max_drop) and max_drop > 0):
            raise ValueError(f"max drop for {name!r} must be a finite number above 0: {max_drop}")
    for which, report in (("baseline", baseline), ("candidate", candidate)):
        missing = [name for name in max_drops if name not in report.metrics]
        if missing:
            raise ValueError(
                f"{which} report has no metric {missing[0]!r}; "
                f"it has: {', '.join(report.metrics) or 'none'}"
            )
    if baseline.dataset.qrels.sha256 != candidate.dataset.qrels.sha256:
        raise ReportError(
            "reports were made on different qrels: "
            f"{baseline.dataset.qrels.sha256} != {candidate.dataset.qrels.sha256}"
        )
    qids = [qid for qid in baseline.per_query if qid in candidate.per_query]
    if not qids:
        raise ReportError("the two reports share no query id")
    verdicts: dict[str, MetricVerdict] = {}
    for name, max_drop in max_drops.items():
        base = per_query_values(baseline, "baseline", name, qids)
        cand = per_query_values(candidate, "candidate", name, qids)
        lo, hi = paired_diff_ci(base, cand, resamples=resamples, seed=seed, level=level)
        verdicts[name] = MetricVerdict(
            baseline=mean(base),
            candidate=mean(cand),
            diff=mean([y - x for x, y in zip(base, cand, strict=True)]),
            ci=(lo, hi),
            max_drop=max_drop,
            passed=round(lo, PLACES) > -max_drop,
            n=len(qids),
        )
    return GateResult(
        baseline=baseline.run,
        candidate=candidate.run,
        qrels_sha256=baseline.dataset.qrels.sha256,
        n_queries=len(qids),
        resamples=resamples,
        seed=seed,
        level=level,
        metrics=verdicts,
    )
