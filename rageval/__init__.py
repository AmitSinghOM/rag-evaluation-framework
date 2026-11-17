"""rageval: retrieval-quality evaluation (metrics core, loaders, report, CLI)."""

from .dataset import Record, load
from .evaluate import evaluate, parse_metric
from .loaders import LoaderError, Qrels, Run, read_jsonl, read_qrels, read_run
from .metrics import (
    DocId,
    Relevance,
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .report import Comparison, Report, ReportError, compare
from .stats import bootstrap_ci, paired_diff_ci

__version__ = "0.1.0a0"

__all__ = [
    "Comparison",
    "DocId",
    "LoaderError",
    "Qrels",
    "Record",
    "Relevance",
    "Report",
    "ReportError",
    "Run",
    "__version__",
    "bootstrap_ci",
    "compare",
    "dcg_at_k",
    "evaluate",
    "load",
    "mrr",
    "ndcg_at_k",
    "paired_diff_ci",
    "parse_metric",
    "precision_at_k",
    "read_jsonl",
    "read_qrels",
    "read_run",
    "recall_at_k",
]
