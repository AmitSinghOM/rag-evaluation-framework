# Metric definitions

Every number rageval produces follows the rules on this page, and every rule
is checked against [ranx](https://github.com/AmenRa/ranx) 0.3.21 in
`tests/test_ranx_crosscheck.py` (256 queries, 26 metric columns, agreement
within 1e-9; measured maximum difference 1.1e-16). Where ranx cannot express
an input, the mapping used is written down here.

## Inputs

| Name | Type | Meaning |
|---|---|---|
| `retrieved` | ordered sequence of ids | The ranking, best first. May contain repeats. |
| `relevant` (binary) | set or list of ids | Every listed id has grade 1. |
| `relevant` (graded) | mapping `id -> grade` | Grade > 0 is relevant; grade 0 is "judged, not relevant"; it never counts. |
| `k` | positive int | Cutoff; only `retrieved[:k]` is looked at. `k <= 0` raises `ValueError`. |

Notation: `R` = set of relevant ids (grade > 0), `n` = `len(retrieved)`,
`top_k` = `retrieved[:k]`, `hits_k` = number of **distinct** ids in `top_k`
that are in `R`.

## Duplicate ids

A repeat earns nothing at its position and does not shift later documents.

```text
retrieved = [d1, d2, d2, d4]   relevant = {d2, d4}

rank:        1    2    3    4
counts:      -   d2    -   d4        (the second d2 is a wasted slot)

recall@4    = 2 / 2 = 1.0          (not 3 / 2)
precision@4 = 2 / 4                (denominator stays k)
dcg@4       = 1/log2(3) + 1/log2(5) (d4 keeps rank 4)
```

The alternative, dropping repeats and compacting the list, would let a
retriever that returns `[a, a, b]` score `recall@2 = 1` for a document it
ranked third. rageval does not do that.

**ranx mapping.** A ranx run is `{doc_id: score}` and cannot hold a repeat.
The cross-check replaces every repeat by a unique filler id that appears in
no qrels (`__repeat_<position>__`) with strictly decreasing scores. That is
exactly the rule above, so the two libraries must agree to the last bit.

## Metrics

| Metric | rageval | ranx 0.3.21 | Same? |
|---|---|---|---|
| `recall_at_k` | `hits_k / len(R)`; `0.0` if `R` is empty | `recall@k`: hits over `min(k, n)` divided by number of qrels with grade >= `rel_lvl` (default 1) | yes |
| `precision_at_k` | `hits_k / k` (denominator is `k` even when `n < k`) | `precision@k`: hits over `min(k, n)` divided by `k` | yes |
| `mrr(k=None)` | `1 / rank` of the first relevant id, else `0.0` | `mrr` (`k = 0`): same | yes |
| `mrr(k)` | same, searching `top_k` only | `mrr@k`: the code honours `k` via `fix_k` although the docstring says it is ignored | yes |
| `dcg_at_k` | `sum(grade_i / log2(i + 1))` over first occurrences in `top_k` | `dcg@k` (Jarvelin and Kekalainen, linear gain) | yes |
| `ndcg_at_k` | `dcg / idcg`; `idcg` = DCG of grades sorted descending, first `min(len(R), k)`; `0.0` if `R` is empty | `ndcg@k`: `idcg` = DCG of the qrels ranked by grade; `idcg == 0` is replaced by 1 | yes (both give 0.0 for an empty `R`) |

Not provided: `ndcg_burges` (gain `2^grade - 1`). rageval uses the linear
gain only, which is ranx's default `ndcg`.

## Where the definitions could differ, and why they do not in the tests

| Point | rageval | ranx | Test consequence |
|---|---|---|---|
| Relevance threshold | grade `> 0` | grade `>= rel_lvl`, default `1` | Identical for integer grades, which is what the cross-check and TREC qrels use. A float grade of `0.5` would be relevant to rageval and not to ranx at the default `rel_lvl`. |
| Short result list (`n < k`) | `top_k = retrieved[:k]`, so `hits` are counted over `n` items and precision divides by `k` | `fix_k` counts over `min(k, n)`, precision divides by `k` | Same values; `edge_short` and `test_precision_denominator_is_k_in_both` lock it. |
| Empty `R` | `recall = 0.0`, `ndcg = 0.0` | `0.0` for both (`idcg` forced to 1 so `0 / 1`) | Same; `edge_all_zero` locks it. |
| Repeats | first occurrence counts | not representable | filler-id mapping above |

## References

- Jarvelin, K. and Kekalainen, J. (2002). Cumulated gain-based evaluation of
  IR techniques. ACM TOIS 20(4). The DCG discount `1 / log2(rank + 1)`.
- ranx source read for this page: `ranx/metrics/{precision,recall,
  reciprocal_rank,ndcg,hits,common}.py` at version 0.3.21.
