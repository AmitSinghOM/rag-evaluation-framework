"""rageval: retrieval-quality evaluation (Phase 0: metrics core, reference-checked)."""

from .dataset import Record, load
from .metrics import (
    DocId,
    Relevance,
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__version__ = "0.1.0a0"

__all__ = [
    "DocId",
    "Record",
    "Relevance",
    "__version__",
    "dcg_at_k",
    "load",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
