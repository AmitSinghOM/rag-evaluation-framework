"""rageval — retrieval-quality evaluation framework (Phase 0: metrics core)."""

from .dataset import Record, load
from .metrics import (
    dcg_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__version__ = "0.0.1"

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "dcg_at_k",
    "load",
    "Record",
]
