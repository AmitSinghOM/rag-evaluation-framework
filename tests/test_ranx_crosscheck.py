"""Cross-check every rageval metric against ranx (the reference implementation).

250 synthetic queries with graded qrels (grades 0-4), judged-non-relevant
documents, short result lists and injected duplicate ids are scored by both
libraries; every per-query value must agree within 1e-9.

ranx represents a run as ``{doc_id: score}`` and so cannot hold a duplicate.
rageval's rule is "first occurrence counts, later repeats are ignored at
their position", which in ranx terms means: replace each repeat by a unique
filler id that is in no qrels. The two must then agree exactly. See
``docs/METRIC-DEFINITIONS.md`` for the full mapping.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pytest
from ranx import Qrels, Run, evaluate

from rageval import dcg_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k

N_SYNTHETIC = 250
KS = (1, 3, 5, 10, 20)
TOLERANCE = 1e-9


@dataclass(frozen=True)
class Case:
    qid: str
    retrieved: list[str]  # ordered, may contain repeats
    grades: dict[str, int]  # judged docs, grade 0 = judged not relevant


def _synthetic_cases(seed: int = 20261118) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    for i in range(N_SYNTHETIC):
        corpus = [f"d{j:03d}" for j in range(40)]
        rng.shuffle(corpus)
        n_rel = rng.randint(1, 8)
        n_zero = rng.randint(0, 3)
        grades = {doc: rng.randint(1, 4) for doc in corpus[:n_rel]}
        grades.update(dict.fromkeys(corpus[n_rel : n_rel + n_zero], 0))
        # Retrieve 1..25 distinct docs from the whole corpus (relevant ones
        # included at random), then repeat 0..4 earlier items later on.
        retrieved = rng.sample(corpus, k=rng.randint(1, 25))
        for _ in range(rng.randint(0, 4)):
            src = rng.randrange(len(retrieved))
            retrieved.insert(rng.randrange(src + 1, len(retrieved) + 1), retrieved[src])
        cases.append(Case(f"q{i:03d}", retrieved, grades))
    return cases


EDGE_CASES = [
    # the only relevant doc is returned twice
    Case("edge_twice", ["d001", "d001", "d002"], {"d001": 3}),
    # every slot is the same non-relevant doc
    Case("edge_same_nonrel", ["d005"] * 6, {"d001": 1}),
    # every slot is the same relevant doc
    Case("edge_same_rel", ["d001"] * 6, {"d001": 2, "d002": 1}),
    # result list shorter than every k except 1
    Case("edge_short", ["d002"], {"d001": 4, "d002": 1}),
    # all judged docs are grade 0: nothing is relevant
    Case("edge_all_zero", ["d001", "d002"], {"d001": 0, "d002": 0}),
    # relevant docs never retrieved
    Case("edge_miss", ["d010", "d011", "d010"], {"d001": 2}),
]

CASES = _synthetic_cases() + EDGE_CASES


def _ranx_run(case: Case) -> dict[str, float]:
    """Express the ordered list as a ranx run: repeats become unique fillers."""
    run: dict[str, float] = {}
    seen: set[str] = set()
    n = len(case.retrieved)
    for position, doc in enumerate(case.retrieved):
        key = doc if doc not in seen else f"__repeat_{position}__"
        seen.add(doc)
        run[key] = float(n - position)  # strictly decreasing, no ties
    return run


def _ranx_scores(cases: list[Case], qrels_of: Any) -> dict[str, dict[str, float]]:
    qrels = Qrels({c.qid: qrels_of(c) for c in cases})
    run = Run({c.qid: _ranx_run(c) for c in cases})
    metrics = [f"{m}@{k}" for m in ("ndcg", "dcg", "recall", "precision", "mrr") for k in KS]
    metrics.append("mrr")
    evaluate(qrels, run, metrics, return_mean=False)
    return {m: dict(scores) for m, scores in run.scores.items()}


def _rageval_scores(cases: list[Case], relevant_of: Any) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for c in cases:
        rel = relevant_of(c)
        for k in KS:
            out.setdefault(f"ndcg@{k}", {})[c.qid] = ndcg_at_k(c.retrieved, rel, k)
            out.setdefault(f"dcg@{k}", {})[c.qid] = dcg_at_k(c.retrieved, rel, k)
            out.setdefault(f"recall@{k}", {})[c.qid] = recall_at_k(c.retrieved, rel, k)
            out.setdefault(f"precision@{k}", {})[c.qid] = precision_at_k(c.retrieved, rel, k)
            out.setdefault(f"mrr@{k}", {})[c.qid] = mrr(c.retrieved, rel, k)
        out.setdefault("mrr", {})[c.qid] = mrr(c.retrieved, rel)
    return out


def _max_abs_diff(
    ours: dict[str, dict[str, float]], theirs: dict[str, dict[str, float]]
) -> tuple[float, str]:
    worst, where = 0.0, ""
    for metric, per_query in ours.items():
        for qid, value in per_query.items():
            diff = abs(value - float(theirs[metric][qid]))
            if diff > worst:
                worst, where = diff, f"{metric} {qid}"
    return worst, where


def test_fixture_exercises_the_rules() -> None:
    assert len(CASES) >= 200
    with_repeats = [c for c in CASES if len(set(c.retrieved)) < len(c.retrieved)]
    repeated_relevant = [
        c
        for c in with_repeats
        if any(c.retrieved.count(d) > 1 and c.grades.get(d, 0) > 0 for d in c.retrieved)
    ]
    graded = [c for c in CASES if len({g for g in c.grades.values() if g > 0}) > 1]
    assert len(with_repeats) >= 100
    assert len(repeated_relevant) >= 20
    assert len(graded) >= 100
    assert any(0 in c.grades.values() for c in CASES)
    assert any(len(c.retrieved) < max(KS) for c in CASES)


def test_graded_metrics_agree_with_ranx() -> None:
    theirs = _ranx_scores(CASES, lambda c: c.grades)
    ours = _rageval_scores(CASES, lambda c: c.grades)
    assert ours.keys() == theirs.keys()
    worst, where = _max_abs_diff(ours, theirs)
    assert worst <= TOLERANCE, f"max abs diff {worst!r} at {where}"


def test_binary_metrics_agree_with_ranx() -> None:
    # Binary relevance passed to rageval as a plain set; to ranx as grade-1 qrels.
    def binarised(c: Case) -> dict[str, int]:
        return {d: 1 for d, g in c.grades.items() if g > 0} or {"__none__": 0}

    theirs = _ranx_scores(CASES, binarised)
    ours = _rageval_scores(CASES, lambda c: {d for d, g in c.grades.items() if g > 0})
    worst, where = _max_abs_diff(ours, theirs)
    assert worst <= TOLERANCE, f"max abs diff {worst!r} at {where}"


@pytest.mark.parametrize("k", KS)
def test_precision_denominator_is_k_in_both(k: int) -> None:
    # ranx: hits over min(k, n) retrieved, divided by k. Same as rageval.
    case = Case("short", ["d001"], {"d001": 1})
    theirs = _ranx_scores([case], lambda c: c.grades)[f"precision@{k}"]["short"]
    assert precision_at_k(case.retrieved, case.grades, k) == pytest.approx(theirs, abs=TOLERANCE)
    assert precision_at_k(case.retrieved, case.grades, k) == 1 / k
