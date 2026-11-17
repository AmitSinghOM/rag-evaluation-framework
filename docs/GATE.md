# Regression gate (`rageval gate`)

`rageval gate` answers one question for CI: **is the candidate no worse than
the baseline by more than a drop you can accept?** It exits `0` when the
answer is yes for every metric you name and `10` when it is not shown for at
least one. The decision behind this design is in
[ADR-0001](./decisions/ADR-0001-v0.1-scope.md) (item 3).

```bash
rageval gate --baseline base.json --candidate cand.json \
  --metric ndcg@10 --max-drop 0.02 --metric recall@100 --max-drop 0.05
```

## What it checks

For each `--metric NAME --max-drop X` pair:

1. Take the queries both reports score. For each one, `d = candidate - baseline`.
2. Bootstrap the mean of `d`: draw the queries with replacement `--resamples`
   times (seeded by `--seed`) and take the `--level` percentile interval
   (95% by default, two-sided). This is `rageval.stats.paired_diff_ci`, the
   same interval `rageval compare` prints.
3. **PASS** when the interval's lower bound is above `-X`. **FAIL** otherwise.

The gate passes only when every named metric passes.

In plain words: the lower bound is the worst drop the data still allow. If
even that worst case is smaller than the drop you said you can live with, the
candidate passes. If the data cannot rule out a bigger drop, it fails, whether
the candidate really is worse or the query set is too small to tell. This is
called a **non-inferiority** check: the candidate has to show it is not worse.

Two exact rules:

- The lower bound is rounded to 6 places (the precision of the report files)
  before the comparison, so the verdict always matches the printed numbers. A
  drop of exactly `X` on every query fails: the rule is strictly "above".
- `X` must be a finite number above 0. With `X = 0`, a candidate identical to
  the baseline (interval `[0, 0]`) could never pass.

## Why not a p-value gate

The usual alternative asks the opposite question: "is the candidate
*significantly worse*?" and passes unless it can prove harm. Proving harm
needs statistical power, and a retrieval eval set of about 50 queries does not
have much.

A worked example with the normal approximation. Suppose the per-query
differences have a standard deviation of 0.15 (a made-up but plausible value
for nDCG). With 50 queries the standard error of the mean difference is
`0.15 / sqrt(50) = 0.021`. Now let the candidate truly lose 0.03 nDCG:

| Gate | Passes this real 0.03 drop |
|---|---|
| significance test, 5% two-sided | about 71% of the time (power 0.29) |
| non-inferiority, `--max-drop 0.02`, 95% CI | about 0.8% of the time |

So the significance gate lets most real regressions of this size through,
and the fewer queries you have, the more it lets through. The non-inferiority
gate moves the other way: less data means more failures, never more passes.

The toy data shows the same thing at a larger scale. Going from `run_b` to
`run_a` drops `ndcg@10` by 0.37, but with six queries the 95% interval of the
difference is `[-0.74, +0.02]`. It contains 0, so a significance gate would
pass this regression. `rageval gate` fails it.

## The price, and how to choose `--max-drop`

The cost of that safety is false alarms: a harmless change also has to show it
is harmless. In the example above (sd 0.15, 50 queries), a candidate that is
*exactly as good* as the baseline passes `--max-drop 0.02` only about 15% of
the time, because the interval half-width (about 0.042) is wider than 0.02.

For a truly equal candidate to pass about 80% of the time, `--max-drop` has to
be at least about `2.8 * sd / sqrt(n)`. With sd 0.15:

| queries `n` | CI half-width | smallest `--max-drop` that passes an equal candidate ~80% of the time |
|---:|---:|---:|
| 50 | 0.042 | 0.059 |
| 100 | 0.029 | 0.042 |
| 200 | 0.021 | 0.030 |
| 500 | 0.013 | 0.019 |
| 1000 | 0.009 | 0.013 |

How to pick a value for your data:

1. Decide the largest drop you would ship without a second look (for
   example 0.02 nDCG@10). That is the `--max-drop` you *want*.
2. Measure your noise: run `rageval compare` (or the gate) between the
   baseline and a change you believe is harmless. Read the half-width `h` of
   the interval.
3. A `--max-drop` of about `1.4 * h` or more passes an equal candidate most of
   the time. If that is larger than the value from step 1, you need more
   queries, not a looser threshold.

Each metric is checked on its own. Adding metrics adds ways to fail, never
ways to pass, so no multiple-comparison correction is needed for the "do not
pass a regression" side; the cost is more false alarms.

## `--resamples`, `--seed` and `--level`

- `--resamples` defaults to 10000 (`run` defaults to 1000, and `compare` reuses
  the baseline report's setting). The verdict
  hangs on one tail quantile, which needs more draws to be stable than a
  displayed interval does. The cost is small: 1000 queries x 10000 resamples
  took 0.36 s with Python 3.13 on an Apple-silicon laptop.
- `--seed` defaults to 0. The same reports, seed and resamples always give the
  same verdict. If the lower bound is within about 0.002 of the threshold, run
  again with another seed or more resamples; if the verdict flips, the result
  is too close to call: treat it as a fail or add queries.
- `--level` defaults to 0.95. The lower bound of a two-sided 95% interval is a
  one-sided 97.5% bound, the same as a one-sided test at 2.5%.

## Small query sets (the n < 30 warning)

Under 30 shared queries the gate prints a warning on stderr (and in the
`--summary` table). Two reasons:

- The interval is wide, so the gate fails most changes unless `--max-drop` is
  large. That is the intended, safe direction.
- The percentile bootstrap is itself less trustworthy with few queries: it
  resamples the same handful of values, so its interval tends to be too
  *narrow*, which makes the gate less conservative than it looks.

30 is a rule of thumb, not a theorem. Treat a gate on fewer queries as a smoke
test.

## Inputs, output and exit codes

- Both reports must be `schema_version` 1 and made on the same qrels (same
  `dataset.qrels.sha256`). Only queries in both reports are used; `rageval
  run` scores every qrels query, so two reports on the same qrels always share
  all of them. Metrics present in a report but not named with `--metric` are
  ignored.
- `--format table` (default) prints one row per metric: baseline mean,
  candidate mean, mean difference, interval, threshold (`-max_drop`),
  verdict and `n`. `--format json` prints the same as an object with a
  top-level `"verdict": "pass" | "fail"`.
- `--summary PATH` **appends** a Markdown table to `PATH`; point it at
  `$GITHUB_STEP_SUMMARY` in CI.

| Exit | Meaning |
|---|---|
| 0 | every metric passed |
| 10 | at least one metric failed |
| 1 | usage error: a `--metric` without its `--max-drop`, a metric missing from a report, a max drop that is not above 0, `--resamples < 1`, `--level` outside (0, 1) |
| 2 | input error: unreadable report, invalid JSON, `schema_version` not 1, different qrels, no shared query, a missing or non-finite per-query value |

## GitHub Action

`action.yml` at the repository root wraps the gate:

```yaml
- uses: AmitSinghOM/rag-evaluation-framework@<commit-sha>
  with:
    baseline: reports/baseline.json
    candidate: reports/candidate.json
    metrics: ndcg@10:0.02,recall@100:0.05
    # resamples: "10000"   seed: "0"   python-version: "3.13"
```

- It runs `actions/setup-python`, installs rageval from the action's own
  checkout with `pip install` (nothing from PyPI but the build backend), runs
  the gate, appends the table to the job summary, and fails the step on any
  non-zero exit. The step output `exit-code` tells a verdict failure (10)
  from a broken input (1 or 2).
- `setup-python` changes which `python` later steps in the same job see. Run
  the gate as the last step, or set up your own Python again after it.
- The inputs reach the script through environment variables only, so a
  crafted path or metric string cannot inject shell.
- The repository is private until v0.1; until then the action works only
  inside this repository (`uses: ./`).
  `.github/workflows/gate-selftest.yml` uses it that way on every pull request:
  one job must pass (identical reports), one must fail with exit 10 (a
  degraded candidate).
