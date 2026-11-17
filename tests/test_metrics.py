"""Phase 0 metric tests with hand-computed expected values."""

from math import log2

from rageval import (
    dcg_at_k,
    load,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# Primary fixture: 3 relevant docs, one of which (d6) is never retrieved.
RETRIEVED = ["d1", "d2", "d3", "d4", "d5"]
RELEVANT = {"d2", "d4", "d6"}


def test_recall_at_k():
    # top-3 = d1,d2,d3 -> only d2 relevant -> 1/3
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == 1 / 3
    # top-5 -> d2,d4 relevant -> 2/3
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == 2 / 3


def test_precision_at_k():
    # 1 hit in top-3 -> 1/3
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == 1 / 3
    # 2 hits in top-5 -> 2/5
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == 2 / 5


def test_mrr():
    # first relevant (d2) at rank 2 -> 1/2
    assert mrr(RETRIEVED, RELEVANT) == 1 / 2


def test_mrr_no_hit():
    assert mrr(["x", "y"], {"z"}) == 0.0


def test_dcg_at_k():
    # relevant hits at ranks 2 and 4
    expected = 1.0 / log2(3) + 1.0 / log2(5)
    assert dcg_at_k(RETRIEVED, RELEVANT, 5) == expected


def test_ndcg_at_k():
    dcg = 1.0 / log2(3) + 1.0 / log2(5)
    # ideal: 3 relevant docs ranked first (capped at k=5)
    idcg = 1.0 / log2(2) + 1.0 / log2(3) + 1.0 / log2(4)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == dcg / idcg


def test_perfect_ranking():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, 2) == 1.0
    assert precision_at_k(retrieved, relevant, 2) == 1.0
    assert mrr(retrieved, relevant) == 1.0
    assert ndcg_at_k(retrieved, relevant, 3) == 1.0
    assert ndcg_at_k(retrieved, relevant, 2) == 1.0


def test_empty_relevant():
    assert recall_at_k(RETRIEVED, [], 3) == 0.0
    assert ndcg_at_k(RETRIEVED, [], 3) == 0.0
    assert mrr(RETRIEVED, []) == 0.0


def test_dataset_loader():
    rows = [
        ("q1", ["d1", "d2"], ["d2"]),
        ("q2", ["a", "b", "c"], {"a", "c"}),
    ]
    records = load(rows)
    assert len(records) == 2
    assert records[0].query == "q1"
    assert records[0].retrieved == ["d1", "d2"]
    assert records[0].relevant == {"d2"}
    # metrics run over a loaded record
    r = records[1]
    assert precision_at_k(r.retrieved, r.relevant, 3) == 2 / 3
    assert mrr(r.retrieved, r.relevant) == 1.0
