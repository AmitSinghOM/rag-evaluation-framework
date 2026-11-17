# Toy fixtures

Six judged queries, two runs of five documents each, small enough to check
by hand. `tests/test_cli.py` and `tests/test_report.py` assert these numbers
and cross-check every mean with ranx.

| File | Content |
|---|---|
| `qrels.txt` | TREC qrels, graded: q1 `d1=2 d2=1 d3=0`; q2 `d4`; q3 `d5=3 d6=3`; q4 `d7`; q5 `d8 d9`; q6 `d10=2`. 9 relevant judgments, one grade-0 (judged not relevant). |
| `run_a.txt` | "bm25": scores q1-q5, **misses q6** (scores 0) and adds **q99** (not judged; reported as a warning). No repeats, no score ties. |
| `run_b.txt` | "hybrid": scores all six queries. No repeats, no score ties. |
| `dataset.jsonl` | The same data as `qrels.txt` + `run_b.txt` in the JSONL format; `read_jsonl` must give exactly `read_qrels` + `read_run`. |

## Hand check (`L3 = log2(3) = 1.58496`, `L5 = log2(5) = 2.32193`)

Per query, `ndcg@10` / `recall@100` / `mrr` / `precision@3`:

| q | relevant (grade) | run_a ranking | run_a values | run_b ranking | run_b values |
|---|---|---|---|---|---|
| q1 | d1 (2), d2 (1) | d3 d1 d2 d11 d12 | dcg `2/L3 + 1/2 = 1.76186`, idcg `2 + 1/L3 = 2.63093`, ndcg **0.669672**; recall 1; mrr 1/2; p@3 2/3 | d1 d2 d3 d11 d12 | ndcg 1; recall 1; mrr 1; p@3 2/3 |
| q2 | d4 | d4 ... | 1; 1; 1; 1/3 | d13 d4 ... | ndcg `1/L3 = 0.630930`; recall 1; mrr 1/2; p@3 1/3 |
| q3 | d5 (3), d6 (3) | d17 d18 d19 d5 d20 | dcg `3/L5 = 1.29203`, idcg `3 + 3/L3 = 4.89279`, ndcg **0.264068**; recall 1/2; mrr 1/4; p@3 0 | d5 d6 ... | 1; 1; 1; 2/3 |
| q4 | d7 | d21 d22 d23 d24 d25 | 0; 0; 0; 0 | d21 d7 ... | ndcg `1/L3 = 0.630930`; recall 1; mrr 1/2; p@3 1/3 |
| q5 | d8, d9 | d8 d9 ... | dcg `1 + 1/L3` = idcg, ndcg 1; recall 1; mrr 1; p@3 2/3 | d8 d26 d9 ... | dcg `1 + 1/2 = 1.5`, idcg `1 + 1/L3 = 1.63093`, ndcg **0.919721**; recall 1; mrr 1; p@3 2/3 |
| q6 | d10 (2) | (missing) | 0; 0; 0; 0 | d10 ... | 1; 1; 1; 1/3 |

Means over the 6 judged queries:

| metric | run_a | run_b | diff (b − a) |
|---|---|---|---|
| ndcg@10 | `(0.669672 + 1 + 0.264068 + 0 + 1 + 0) / 6 = 0.488957` | `(1 + 0.630930 + 1 + 0.630930 + 0.919721 + 1) / 6 = 0.863597` | +0.374640 |
| recall@100 | `3.5 / 6 = 0.583333` | `1.000000` | +0.416667 |
| mrr | `2.75 / 6 = 0.458333` | `5 / 6 = 0.833333` | +0.375000 |
| precision@3 | `(2/3 + 1/3 + 0 + 0 + 2/3 + 0) / 6 = 0.277778` | `(2/3 + 1/3 + 2/3 + 1/3 + 2/3 + 1/3) / 6 = 0.5` | +0.222222 |

Report facts for run_a: `n_queries 6`, `n_judged_relevant 9`,
`n_queries_in_run 6` (q1-q5 and q99), `duplicate_ids 0`, one warning for q99.
