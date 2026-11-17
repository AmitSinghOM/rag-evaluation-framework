"""Phase 0 metric tests with hand-computed expected values."""

from math import log2

import pytest

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


def test_recall_at_k() -> None:
    # top-3 = d1,d2,d3 -> only d2 relevant -> 1/3
    assert recall_at_k(RETRIEVED, RELEVANT, 3) == 1 / 3
    # top-5 -> d2,d4 relevant -> 2/3
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == 2 / 3


def test_precision_at_k() -> None:
    # 1 hit in top-3 -> 1/3
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == 1 / 3
    # 2 hits in top-5 -> 2/5
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == 2 / 5


def test_mrr() -> None:
    # first relevant (d2) at rank 2 -> 1/2
    assert mrr(RETRIEVED, RELEVANT) == 1 / 2


def test_mrr_no_hit() -> None:
    assert mrr(["x", "y"], {"z"}) == 0.0


def test_dcg_at_k() -> None:
    # relevant hits at ranks 2 and 4
    expected = 1.0 / log2(3) + 1.0 / log2(5)
    assert dcg_at_k(RETRIEVED, RELEVANT, 5) == expected


def test_ndcg_at_k() -> None:
    dcg = 1.0 / log2(3) + 1.0 / log2(5)
    # ideal: 3 relevant docs ranked first (capped at k=5)
    idcg = 1.0 / log2(2) + 1.0 / log2(3) + 1.0 / log2(4)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == dcg / idcg


def test_perfect_ranking() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, 2) == 1.0
    assert precision_at_k(retrieved, relevant, 2) == 1.0
    assert mrr(retrieved, relevant) == 1.0
    assert ndcg_at_k(retrieved, relevant, 3) == 1.0
    assert ndcg_at_k(retrieved, relevant, 2) == 1.0


def test_empty_relevant() -> None:
    assert recall_at_k(RETRIEVED, [], 3) == 0.0
    assert ndcg_at_k(RETRIEVED, [], 3) == 0.0
    assert mrr(RETRIEVED, []) == 0.0


def test_dataset_loader() -> None:
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


# --- v0.1 rules: duplicates, graded relevance, mrr@k ------------------------


def test_duplicate_ids_count_once_and_keep_positions() -> None:
    # d2 is relevant and returned twice; d4 relevant at rank 4 behind the repeat.
    retrieved = ["d1", "d2", "d2", "d4"]
    relevant = {"d2", "d4"}
    # recall: two distinct relevant docs found, not three hits
    assert recall_at_k(retrieved, relevant, 4) == 1.0
    assert recall_at_k(retrieved, relevant, 3) == 1 / 2
    # precision keeps the denominator k; the wasted slot is a miss
    assert precision_at_k(retrieved, relevant, 4) == 2 / 4
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    # dcg: the repeat earns nothing and d4 stays at rank 4
    assert dcg_at_k(retrieved, relevant, 4) == 1.0 / log2(3) + 1.0 / log2(5)
    assert ndcg_at_k(retrieved, relevant, 4) == dcg_at_k(retrieved, relevant, 4) / (
        1.0 / log2(2) + 1.0 / log2(3)
    )
    # a repeat of a non-relevant doc is also just a wasted slot
    assert recall_at_k(["x", "x", "d2"], relevant, 2) == 0.0
    assert mrr(["x", "x", "d2"], relevant) == 1 / 3


def test_graded_relevance_uses_grade_as_gain() -> None:
    retrieved = ["a", "b", "c", "d"]
    grades = {"a": 1, "c": 3, "d": 0, "z": 2}  # d judged non-relevant, z never retrieved
    assert dcg_at_k(retrieved, grades, 4) == 1.0 / log2(2) + 3.0 / log2(4)
    # ideal ranking: grades 3, 2, 1 -> ranks 1, 2, 3
    idcg = 3.0 / log2(2) + 2.0 / log2(3) + 1.0 / log2(4)
    assert ndcg_at_k(retrieved, grades, 4) == dcg_at_k(retrieved, grades, 4) / idcg
    # ideal is capped at k
    assert ndcg_at_k(retrieved, grades, 2) == (1.0 / log2(2)) / (3.0 / log2(2) + 2.0 / log2(3))
    # grade 0 is not relevant for the set-based metrics either
    assert recall_at_k(retrieved, grades, 4) == 2 / 3
    assert precision_at_k(retrieved, grades, 4) == 2 / 4
    assert mrr(retrieved, grades) == 1.0
    assert mrr(["d", "c"], grades) == 1 / 2


def test_binary_mapping_equals_set() -> None:
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    as_mapping = dict.fromkeys(RELEVANT, 1)
    for k in (1, 3, 5):
        assert recall_at_k(retrieved, as_mapping, k) == recall_at_k(retrieved, RELEVANT, k)
        assert precision_at_k(retrieved, as_mapping, k) == precision_at_k(retrieved, RELEVANT, k)
        assert dcg_at_k(retrieved, as_mapping, k) == dcg_at_k(retrieved, RELEVANT, k)
        assert ndcg_at_k(retrieved, as_mapping, k) == ndcg_at_k(retrieved, RELEVANT, k)
        assert mrr(retrieved, as_mapping, k) == mrr(retrieved, RELEVANT, k)


def test_mrr_at_k() -> None:
    # first relevant (d2) at rank 2
    assert mrr(RETRIEVED, RELEVANT, k=1) == 0.0
    assert mrr(RETRIEVED, RELEVANT, k=2) == 1 / 2
    assert mrr(RETRIEVED, RELEVANT, k=None) == mrr(RETRIEVED, RELEVANT)
    with pytest.raises(ValueError, match="positive"):
        mrr(RETRIEVED, RELEVANT, k=0)


def test_invalid_k_rejected() -> None:
    for fn in (recall_at_k, precision_at_k, dcg_at_k, ndcg_at_k):
        with pytest.raises(ValueError, match="positive"):
            fn(RETRIEVED, RELEVANT, 0)
        with pytest.raises(ValueError, match="positive"):
            fn(RETRIEVED, [], -1)


def test_short_retrieved_list() -> None:
    # fewer than k results: precision denominator stays k, nothing else breaks
    assert precision_at_k(["d2"], RELEVANT, 5) == 1 / 5
    assert recall_at_k(["d2"], RELEVANT, 5) == 1 / 3
    assert ndcg_at_k(["d2"], RELEVANT, 5) == (1.0 / log2(2)) / (
        1.0 / log2(2) + 1.0 / log2(3) + 1.0 / log2(4)
    )
    assert recall_at_k([], RELEVANT, 5) == 0.0
    assert mrr([], RELEVANT, k=5) == 0.0


def test_loader_keeps_grades() -> None:
    records = load([("q", ["a", "b"], {"a": 2, "b": 0})])
    assert records[0].relevant == {"a": 2, "b": 0}
    assert ndcg_at_k(records[0].retrieved, records[0].relevant, 2) == 1.0


def test_version_matches_pyproject() -> None:
    import tomllib
    from pathlib import Path

    import rageval

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert rageval.__version__ == declared
