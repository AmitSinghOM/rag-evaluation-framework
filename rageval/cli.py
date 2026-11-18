"""Command line: ``rageval run``, ``rageval compare`` and ``rageval gate``.

Exit codes: ``0`` ok, ``1`` usage or argument error, ``2`` input or format
error (unreadable file, malformed line, qrels mismatch), ``3`` internal
error, ``10`` the regression gate failed on at least one metric.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from .evaluate import evaluate, parse_metric
from .gate import gate
from .loaders import LoaderError, read_jsonl, read_qrels, read_run
from .report import FileRef, Report, ReportError, compare

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INPUT = 2
EXIT_INTERNAL = 3
EXIT_GATE_FAILED = 10


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error; this project reserves 2 for input errors."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


class _MetricAction(argparse.Action):
    """``--metric NAME`` opens a gate rule; the ``--max-drop X`` right after it closes it."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        rules: list[tuple[str, float | None]] = list(getattr(namespace, self.dest) or [])
        rules.append((cast(str, values), None))
        setattr(namespace, self.dest, rules)


class _MaxDropAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        rules: list[tuple[str, float | None]] = list(getattr(namespace, self.dest) or [])
        if not rules or rules[-1][1] is not None:
            raise argparse.ArgumentError(self, "each --max-drop must follow its own --metric")
        rules[-1] = (rules[-1][0], cast(float, values))
        setattr(namespace, self.dest, rules)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="rageval", description="Retrieval-quality evaluation for RAG systems.")
    from . import __version__  # at call time: avoids a cycle with rageval/__init__.py

    parser.add_argument("--version", action="version", version=f"rageval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{run,compare,gate}")

    run = sub.add_parser(
        "run",
        help="score a run against judgments (TREC files or one JSONL file), write a JSON report",
        description=(
            "Score a TREC run file against TREC qrels, or a JSONL dataset that holds both, "
            "and write a JSON report."
        ),
    )
    source = run.add_argument_group(
        "input", "either --qrels with --run (two TREC files) or --dataset (one JSONL file)"
    )
    source.add_argument("--qrels", metavar="Q", help="TREC qrels: qid 0 docid grade")
    source.add_argument("--run", metavar="R", help="TREC run: qid Q0 docid rank score tag")
    source.add_argument(
        "--dataset",
        metavar="D",
        help='JSONL, one query per line: {"query_id", "retrieved", "relevant"}',
    )
    run.add_argument(
        "--metrics",
        required=True,
        metavar="M[,M...]",
        help="comma-separated: ndcg@K, recall@K, precision@K, mrr, mrr@K",
    )
    run.add_argument("--out", required=True, metavar="FILE", help="where to write the report")
    run.add_argument(
        "--resamples", type=int, default=1000, metavar="N", help="bootstrap resamples (1000)"
    )
    run.add_argument("--seed", type=int, default=0, metavar="S", help="bootstrap seed (0)")
    run.add_argument(
        "--name", metavar="NAME", help="run name in the report (default: run or dataset file stem)"
    )
    run.set_defaults(func=_run)

    cmp_ = sub.add_parser(
        "compare",
        help="paired comparison of two reports made on the same qrels",
        description="Paired comparison of two reports made on the same qrels (b - a).",
    )
    cmp_.add_argument("a", help="baseline report")
    cmp_.add_argument("b", help="candidate report")
    cmp_.add_argument("--format", choices=("table", "json"), default="table")
    cmp_.set_defaults(func=_compare)

    gate_ = sub.add_parser(
        "gate",
        help="fail (exit 10) unless the candidate is no worse than the baseline by max drop",
        description=(
            "Non-inferiority gate. Per metric, a percentile bootstrap CI of the paired "
            "per-query difference (candidate - baseline); the metric passes when the CI's "
            "lower bound is above -max-drop. Exit 0 when every metric passes, 10 when any "
            "fails. See docs/GATE.md."
        ),
    )
    gate_.add_argument("--baseline", required=True, metavar="FILE", help="baseline report")
    gate_.add_argument("--candidate", required=True, metavar="FILE", help="candidate report")
    gate_.add_argument(
        "--metric",
        dest="rules",
        action=_MetricAction,
        metavar="NAME",
        help="metric to gate, e.g. ndcg@10; repeatable, each followed by its --max-drop",
    )
    gate_.add_argument(
        "--max-drop",
        dest="rules",
        action=_MaxDropAction,
        type=float,
        metavar="X",
        help="largest drop of the metric's mean you accept, e.g. 0.02 (must be > 0)",
    )
    gate_.add_argument(
        "--resamples", type=int, default=10000, metavar="N", help="bootstrap resamples (10000)"
    )
    gate_.add_argument("--seed", type=int, default=0, metavar="S", help="bootstrap seed (0)")
    gate_.add_argument(
        "--level", type=float, default=0.95, metavar="L", help="two-sided CI level (0.95)"
    )
    gate_.add_argument("--format", choices=("table", "json"), default="table")
    gate_.add_argument(
        "--summary",
        metavar="PATH",
        help="also append a Markdown table to PATH (for $GITHUB_STEP_SUMMARY)",
    )
    gate_.set_defaults(func=_gate)
    return parser


def _metric_lines(report: Report) -> list[str]:
    width = max(len("metric"), *(len(name) for name in report.metrics))
    lines = [f"{'metric':<{width}}  {'mean':>8}  {'ci95':<20}  n"]
    for name, summary in report.metrics.items():
        lo, hi = summary.ci95
        ci = f"[{lo:.4f}, {hi:.4f}]"
        lines.append(f"{name:<{width}}  {summary.mean:>8.4f}  {ci:<20}  {summary.n}")
    return lines


def _run(args: argparse.Namespace) -> int:
    metrics = [name.strip() for name in args.metrics.split(",") if name.strip()]
    if not metrics:
        raise ValueError("--metrics needs at least one metric name")
    for name in metrics:
        parse_metric(name)
    if args.resamples < 1:
        raise ValueError("--resamples must be >= 1")
    if args.dataset is not None:
        if args.qrels is not None or args.run is not None:
            raise ValueError("give --dataset alone, or --qrels with --run, not both")
        qrels, run = read_jsonl(args.dataset)
        if not qrels:
            raise LoaderError(args.dataset, 0, "dataset has no queries")
        qrels_ref, run_ref = FileRef.of_judgments(args.dataset, qrels), FileRef.of(args.dataset)
        source = args.dataset
    elif args.qrels is None or args.run is None:
        raise ValueError("needs --qrels Q with --run R, or --dataset D")
    else:
        qrels = read_qrels(args.qrels)
        run = read_run(args.run)
        if not qrels:
            raise LoaderError(args.qrels, 0, "qrels file has no judgments")
        qrels_ref, run_ref = FileRef.of(args.qrels), FileRef.of(args.run)
        source = args.run
    report = evaluate(
        qrels,
        run,
        metrics,
        resamples=args.resamples,
        seed=args.seed,
        qrels_ref=qrels_ref,
        run_ref=run_ref,
        run_name=args.name or Path(source).stem,
    )
    report.write(args.out)
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print("\n".join(_metric_lines(report)))
    print(f"wrote {args.out} ({report.dataset.n_queries} queries)")
    return EXIT_OK


def _compare(args: argparse.Namespace) -> int:
    result = compare(Report.read(args.a), Report.read(args.b))
    sys.stdout.write(result.to_table() if args.format == "table" else result.to_json())
    return EXIT_OK


def _gate(args: argparse.Namespace) -> int:
    max_drops: dict[str, float] = {}
    for name, max_drop in args.rules or []:
        if max_drop is None:
            raise ValueError(f"--metric {name} needs a --max-drop right after it")
        if name in max_drops:
            raise ValueError(f"--metric {name} is given more than once")
        max_drops[name] = max_drop
    if not max_drops:
        raise ValueError("gate needs at least one --metric NAME --max-drop X")
    result = gate(
        Report.read(args.baseline),
        Report.read(args.candidate),
        max_drops,
        resamples=args.resamples,
        seed=args.seed,
        level=args.level,
    )
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    sys.stdout.write(result.to_table() if args.format == "table" else result.to_json())
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as fh:
            fh.write(result.to_markdown())
    return EXIT_OK if result.passed else EXIT_GATE_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse handled --help/--version or a usage error itself
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        result: int = args.func(args)
        return result
    except (LoaderError, ReportError, OSError, UnicodeDecodeError) as exc:
        print(f"rageval: error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except ValueError as exc:
        print(f"rageval: error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as exc:  # last resort: one line on stderr, never a traceback dump
        print(f"rageval: internal error: {exc!r}", file=sys.stderr)
        return EXIT_INTERNAL
