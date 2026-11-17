"""Command line: ``rageval run`` and ``rageval compare``.

Exit codes: ``0`` ok, ``1`` usage or argument error, ``2`` input or format
error (unreadable file, malformed line, qrels mismatch), ``3`` internal
error. ``10`` is reserved for the Phase 2 regression gate.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from .evaluate import evaluate, parse_metric
from .loaders import LoaderError, read_qrels, read_run
from .report import FileRef, Report, ReportError, compare

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INPUT = 2
EXIT_INTERNAL = 3


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error; this project reserves 2 for input errors."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="rageval", description="Retrieval-quality evaluation for RAG systems.")
    from . import __version__  # at call time: avoids a cycle with rageval/__init__.py

    parser.add_argument("--version", action="version", version=f"rageval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{run,compare}")

    run = sub.add_parser(
        "run",
        help="score a TREC run file against TREC qrels and write a JSON report",
        description="Score a TREC run file against TREC qrels and write a JSON report.",
    )
    run.add_argument("--qrels", required=True, metavar="Q", help="TREC qrels: qid 0 docid grade")
    run.add_argument(
        "--run", required=True, metavar="R", help="TREC run: qid Q0 docid rank score tag"
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
        "--name", metavar="NAME", help="run name in the report (default: run file stem)"
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
    qrels = read_qrels(args.qrels)
    run = read_run(args.run)
    if not qrels:
        raise LoaderError(args.qrels, 0, "qrels file has no judgments")
    report = evaluate(
        qrels,
        run,
        metrics,
        resamples=args.resamples,
        seed=args.seed,
        qrels_ref=FileRef.of(args.qrels),
        run_ref=FileRef.of(args.run),
        run_name=args.name or Path(args.run).stem,
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
