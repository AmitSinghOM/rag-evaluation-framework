# Report schema (schema_version 1)

`rageval run` writes one JSON file per evaluated run. Its shape is fixed by
[`report.schema.json`](./report.schema.json) (JSON Schema draft 2020-12) and
`tests/test_report.py` validates every emitted report against it. This page
explains the fields and records the choices the round-2 contract left open.

## Example

```json
{
  "schema_version": 1,
  "rageval_version": "0.1.0a0",
  "created_at": "2026-11-18T02:10:00Z",
  "dataset": {
    "qrels": {"path": "examples/toy/qrels.txt", "sha256": "e3905f1c..."},
    "n_queries": 6,
    "n_judged_relevant": 9
  },
  "run": {
    "name": "run_a",
    "path": "examples/toy/run_a.txt",
    "sha256": "f0fc69ca...",
    "n_queries_in_run": 6,
    "duplicate_ids": 0
  },
  "metrics": {
    "ndcg@10": {"mean": 0.488957, "ci95": [0.111612, 0.82229], "n": 6,
                "bootstrap": {"resamples": 1000, "seed": 0}}
  },
  "per_query": {"q1": {"ndcg@10": 0.669672}, "q2": {"ndcg@10": 1.0}},
  "warnings": ["query 'q99' is in the run but not in the qrels; ignored"]
}
```

## Fields

| Field | Meaning |
|---|---|
| `schema_version` | Always `1` for this layout. A reader must refuse any other value (`rageval compare` exits 2). |
| `rageval_version` | `rageval.__version__` that wrote the file. |
| `created_at` | UTC, ISO 8601, second precision, `Z` suffix. The only field that differs between two runs with the same inputs and seed. |
| `dataset.qrels.path` | The qrels path exactly as given on the command line (not resolved, so no machine-specific prefix leaks into a shared report). `null` when `evaluate()` was called on in-memory data. |
| `dataset.qrels.sha256` | SHA-256 of the qrels file bytes; `null` for in-memory data. `compare` uses it to refuse reports made on different judgments. |
| `dataset.n_queries` | Number of queries in the qrels. Every one is scored, including queries the run never answered (they score `0`) and queries whose judgments are all grade `0` (they also score `0`). |
| `dataset.n_judged_relevant` | Number of `(query, doc)` judgments with grade `> 0`. |
| `run.name` | `--name`, else the run file's stem (`run_a.txt` -> `run_a`); `null` for in-memory data with no name. |
| `run.path`, `run.sha256` | As for the qrels. |
| `run.n_queries_in_run` | Distinct query ids in the run file, including ones not in the qrels. |
| `run.duplicate_ids` | Total repeated positions over the whole run file: `sum(len(docs) - len(set(docs)))` per query. A repeat scores nothing (see `METRIC-DEFINITIONS.md`); this counter makes it visible. |
| `metrics.<name>.mean` | Mean of the per-query values over `n` = `dataset.n_queries` queries. |
| `metrics.<name>.ci95` | Percentile bootstrap 95% interval of that mean: `resamples` draws of `n` values with replacement from `random.Random(seed)`, quantiles 0.025 and 0.975 with linear interpolation between order statistics (numpy's default rule). Pure stdlib. `n = 1` gives a zero-width interval. |
| `metrics.<name>.bootstrap` | The `resamples` and `seed` used, so the interval can be reproduced. |
| `per_query` | One row per qrels query, in qrels file order, with every requested metric. Rows for queries missing from the run are all `0.0`. |
| `warnings` | Always present (possibly empty). One line per run query that is not in the qrels; those queries are ignored. |

Floats are rounded to 6 decimal places when written. The agreement with
ranx (`tests/test_cli.py::test_run_means_agree_with_ranx_on_the_same_files`)
is asserted on the computed means within 1e-9 and the file value is asserted
to be their 6-place rounding.

## Metric names

`ndcg@K`, `recall@K`, `precision@K`, `mrr`, `mrr@K`; lowercase, `K` a
positive integer. Anything else is a usage error (exit 1) that lists these
forms. Repeated names are computed once.

## Input formats

- **TREC qrels** `qid 0 docid grade`: four whitespace-separated fields; the
  second is unused and not checked (as in `trec_eval`); `grade` is an int
  `>= 0`, `0` = judged not relevant.
- **TREC run** `qid Q0 docid rank score tag`: six fields; `rank` int,
  `score` float (NaN rejected). Per query the documents are ordered by score
  descending, then rank ascending, then file order. Repeats are kept.
- **JSONL** one object per line:
  `{"query_id": str, "query": str|null, "retrieved": [docid, ...], "relevant": {docid: grade} | [docid, ...]}`.
  A list of relevant ids means grade `1` for each. Ids must be strings;
  grades are numbers `>= 0`. `read_jsonl` returns `(qrels, run)` keyed by
  `query_id`. The CLI reads TREC files only in this round.

Choices where the contract was silent:

| Point | Choice | Why |
|---|---|---|
| Blank lines | skipped | trailing newlines and visual gaps are common; nothing is lost |
| Repeated `(qid, docid)` judgment in qrels | `LoaderError` | silently keeping the last one would hide a data problem; the loaders never drop or overwrite |
| Repeated `query_id` in JSONL | `LoaderError` | same rule |
| Non-string ids in JSONL | `LoaderError` | `1`, `"1"` and `1.0` must not silently become the same (or different) document |
| Unused TREC fields (`0`, `Q0`, `tag`) | not validated | `trec_eval` ignores them; strictness here would reject real files for nothing |
| Qrels file with no judgments | exit 2 | there is nothing to score |

Every `LoaderError` message starts with `path:line_no:` (1-based).

## `rageval compare a.json b.json`

Paired comparison, `b - a`, of every metric present in both reports, over
the per-query rows. Output `--format table` (default):

```text
metric              run_a         run_b    diff (b-a)  ci95
ndcg@10          0.488957      0.863597     +0.374640  [+0.010254, +0.762443]
n_queries: 6
```

`--format json`:

```json
{
  "schema_version": 1,
  "a": {"name": "run_a", "path": "...", "sha256": "..."},
  "b": {"name": "run_b", "path": "...", "sha256": "..."},
  "qrels_sha256": "...",
  "n_queries": 6,
  "bootstrap": {"resamples": 1000, "seed": 0},
  "metrics": {"ndcg@10": {"a": 0.488957, "b": 0.863597, "diff": 0.37464, "ci95": [0.010254, 0.762443]}}
}
```

- `diff` is the mean of the per-query differences; `ci95` is its percentile
  bootstrap interval, resampled with **report a's** `resamples` and `seed`
  (no extra flags; the comparison is as precise as the baseline report).
- Compare works from the files, so it sees 6-place values.
- Refused with exit 2 when: the two `dataset.qrels.sha256` differ (two
  `null`s are equal); the per-query id sets differ; no metric is in both;
  a report is not valid JSON, has another `schema_version`, or lacks a field.
- Metrics present in only one report are left out silently.

## Exit codes (`rageval`)

| Code | Meaning |
|---|---|
| 0 | ok |
| 1 | usage or argument error: missing option, unknown metric name, `--resamples < 1` |
| 2 | input or format error: unreadable or undecodable file, malformed line, empty qrels, qrels mismatch, bad report file |
| 3 | internal error (one line on stderr, no traceback) |
| 10 | reserved for the Phase 2 regression gate |
