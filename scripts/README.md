# scripts/ — SciFact dogfood fixtures

Three stdlib-only scripts that produce the committed fixtures in
[`examples/scifact/`](../examples/scifact/): a real IR test collection (BEIR
SciFact, 300 queries over 5,183 abstracts) plus a deterministic BM25 run and
two deliberately degraded runs. Later rounds use them to demo `rageval run`,
`rageval compare` and the regression gate on real data without any download.

**These fixtures are evaluation data, not training data.** Nothing here is
used to fit a model; they exist so metric values can be checked and compared.

## Regenerate everything

From the repository root (Python 3.11+, no extra packages):

```bash
python scripts/fetch_scifact.py --dest .cache/scifact --trec-qrels examples/scifact/qrels.test.tsv
python scripts/bm25_baseline.py --dest .cache/scifact --out examples/scifact/bm25.top100.run
python scripts/bm25_baseline.py --dest .cache/scifact --degrade truncate:20 \
    --out examples/scifact/bm25-degraded-truncate20.top20.run
python scripts/bm25_baseline.py --dest .cache/scifact --degrade shuffle-tail:0 \
    --out examples/scifact/bm25-degraded-shuffle-tail0.top100.run
python scripts/make_manifest.py
```

The result is byte-identical to what is committed (`MANIFEST.json` holds the
sha256 of every file; `tests/test_scifact_fixtures.py` checks it). Observed on
2026-11-18, Apple M-series laptop, CPython 3.11-3.14: download + unzip 2.8 s
(6.6 s for the raw 2.8 MB transfer the first time), each BM25 run 0.8-0.9 s,
whole pipeline under 10 s. The run hash is the same on all four interpreters.

`.cache/scifact/` keeps the verified zip and the unpacked files; it writes its
own `.gitignore` so it is never committed. Delete it to force a fresh download.

## Source and licence

| Item | Value |
|---|---|
| Download | <https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip> (the URL the [BEIR README](https://github.com/beir-cellar/beir) uses) |
| Pinned sha256 | `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165` (2,816,079 bytes; BEIR's md5 `5f7d1de60b170fc8027bb7898e2efca1`) |
| Members used | `corpus.jsonl` (5,183 docs), `queries.jsonl` (1,109 claims, 300 in the test split), `qrels/test.tsv` (339 judgements, all grade 1) |
| Licence | Read at <https://github.com/allenai/scifact/blob/master/LICENSE.md>: claims and evidence annotations (our queries and qrels) **CC BY 4.0**; abstracts (our corpus) **ODC-By 1.0** (Semantic Scholar S2ORC); SciFact code Apache-2.0. The BEIR zip itself ships no licence file; BEIR's code is Apache-2.0. |
| Attribution | Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020; Thakur et al., *BEIR*, NeurIPS 2021 Datasets and Benchmarks. |

Only the queries and qrels (CC BY 4.0, 5.7 KB) and the BM25 output (document
ids and scores) are committed. The corpus text stays in the local cache.

## What each script does

### `fetch_scifact.py`

Downloads the zip over https (refuses anything else), verifies the pinned
sha256 **before** unpacking, writes the three members under `--dest` and keeps
the zip there so the next run is offline. `--trec-qrels PATH` converts BEIR's
`query-id<TAB>corpus-id<TAB>score` (with header) into TREC qrels
`qid 0 docid grade`, row order preserved. Exit 1 on checksum or I/O failure.

### `bm25_baseline.py`

Pure-Python BM25 over `title + " " + text`, scored for every query id that
appears in `qrels/test.tsv` (in order of first appearance), written as a TREC
run `qid Q0 docid rank score tag` with the top `--k` (default 100) per query.

Fixed choices, so the output never changes between runs or interpreters:

- Tokeniser: lowercase, then each maximal run of `[a-z0-9]` is a token (split
  on non-alphanumerics). No stemming, no stopword list. Query terms are
  deduplicated.
- `k1 = 0.9`, `b = 0.4` (Anserini/BEIR defaults). Lucene idf
  `ln(1 + (N - df + 0.5) / (df + 0.5))`, always positive.
- Terms are summed in sorted order; ties break on document id; scores are
  printed with six decimals.

For reference, scored with `rageval` against `qrels.test.tsv` the baseline
gives nDCG@10 0.6588, Recall@100 0.8886, MRR@10 0.6288 (BEIR reports 0.665
nDCG@10 for Anserini BM25 with stemming and stopwords).

Degradations for gate demos (`--degrade`):

| Flag | Effect | Tag | Measured against the baseline |
|---|---|---|---|
| `truncate:20` | keep ranks 1..20 only | `bm25-degraded-truncate20` | Recall@100 0.8886 -> 0.8252, nDCG@100 0.6853 -> 0.6727; nDCG@10 and MRR@10 unchanged |
| `shuffle-tail:0` | keep ranks 1..10; permute the doc ids at ranks 11..k with `random.Random(0).shuffle` (fresh per query); score column stays in place so the run is still a valid ranking | `bm25-degraded-shuffle-tail0` | nDCG@100 0.6853 -> 0.6806, MRR@100 0.6333 -> 0.6309; nDCG@10, MRR@10 and Recall@100 unchanged |

`--offline` forbids any download (exit 1 with a pointer to `fetch_scifact.py`
when `--dest` is incomplete); without it the dataset is fetched on demand.

### `make_manifest.py`

Writes `examples/scifact/MANIFEST.json`: sha256, byte and line counts, query
counts, per-query depth and tag for every fixture, plus the source URL, source
sha256, licence line and the exact commands above. Sorted keys, no timestamp,
so regenerating an unchanged fixture set changes nothing.

## Use with the CLI

```bash
rageval run --qrels examples/scifact/qrels.test.tsv --run examples/scifact/bm25.top100.run \
    --metrics ndcg@10,ndcg@100,recall@100,mrr@10 --out bm25.json
rageval run --qrels examples/scifact/qrels.test.tsv \
    --run examples/scifact/bm25-degraded-truncate20.top20.run \
    --metrics ndcg@10,ndcg@100,recall@100,mrr@10 --out truncated.json
rageval compare bm25.json truncated.json
```

Observed (2026-11-18, 1000 bootstrap resamples, seed 0): `recall@100`
0.8886 -> 0.8252, diff -0.0633 with CI95 [-0.0920, -0.0393]; `ndcg@10` and
`mrr@10` unchanged. The shuffle-tail run against the baseline gives `ndcg@100`
-0.0047 with CI95 [-0.0070, -0.0027] and no change at cutoff 10.

## Fixture sizes (committed, 2.9 MB total)

| File | Lines | Bytes |
|---|---|---|
| `qrels.test.tsv` | 339 | 5,702 |
| `bm25.top100.run` | 30,000 | 996,193 |
| `bm25-degraded-truncate20.top20.run` | 6,000 | 319,384 |
| `bm25-degraded-shuffle-tail0.top100.run` | 30,000 | 1,686,193 |
