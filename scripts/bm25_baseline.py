"""Deterministic pure-Python BM25 over the SciFact corpus, written as a TREC run.

Reads the BEIR layout produced by ``fetch_scifact.py`` from ``--dest``
(``corpus.jsonl``, ``queries.jsonl``, ``qrels/test.tsv``), scores every query
that appears in the test qrels and writes ``qid Q0 docid rank score tag`` lines,
top ``--k`` (default 100) per query.

Choices, all fixed so the output is byte-identical run to run:

* Tokeniser: lowercase, then every maximal run of ``[a-z0-9]`` is a token
  (split on non-alphanumerics). No stemming, no stopwords. Query terms are
  deduplicated (a word repeated in a claim counts once).
* Document text is ``title + " " + text``.
* BM25 with ``k1 = 0.9``, ``b = 0.4`` (the Anserini/BEIR defaults) and the
  Lucene idf ``ln(1 + (N - df + 0.5) / (df + 0.5))``, which is always positive.
* Ties are broken by document id (string order); scores are written with six
  decimals; terms are summed in sorted order so float rounding is stable.
* ``--degrade truncate:N`` keeps ranks 1..N only (tag ``bm25-degraded-truncateN``).
* ``--degrade shuffle-tail:SEED`` keeps ranks 1..10, permutes the document ids
  at ranks 11..k with ``random.Random(SEED).shuffle`` (fresh per query), and
  keeps the score column in place so the run still reads as a valid ranking
  (tag ``bm25-degraded-shuffle-tailSEED``).

Without ``--offline`` the dataset is fetched when ``--dest`` is missing; with it
the script refuses to touch the network and exits 1 instead. Stdlib only.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
import re
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from fetch_scifact import DEFAULT_DEST, MEMBERS, fetch

K1 = 0.9
B = 0.4
HEAD = 10  # ranks kept untouched by shuffle-tail
TOKEN = re.compile(r"[a-z0-9]+")
Ranked = list[tuple[str, float]]


def tokenize(text: str) -> list[str]:
    """Lowercase, then every maximal run of ``[a-z0-9]`` is a token."""
    return TOKEN.findall(text.lower())


class BM25:
    """Inverted-index BM25 (Robertson/Lucene variant) over a fixed document list."""

    def __init__(self, docs: Sequence[tuple[str, str]], k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self.ids = [doc_id for doc_id, _ in docs]
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, (_, text) in enumerate(docs):
            tokens = tokenize(text)
            self.lengths.append(len(tokens))
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, tf in counts.items():
                self.postings.setdefault(token, []).append((index, tf))
        n = len(docs)
        self.avgdl = (sum(self.lengths) / n) if n else 0.0
        self.idf = {
            term: math.log(1.0 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }

    def search(self, query: str, k: int) -> Ranked:
        """Top ``k`` ``(doc_id, score)`` by score desc, then doc id asc; zero scores excluded."""
        scores: dict[int, float] = {}
        for term in sorted(set(tokenize(query))):
            plist = self.postings.get(term)
            if plist is None:
                continue
            idf = self.idf[term]
            for index, tf in plist:
                norm = self.k1 * (1.0 - self.b + self.b * self.lengths[index] / self.avgdl)
                scores[index] = scores.get(index, 0.0) + idf * (tf * (self.k1 + 1.0)) / (tf + norm)
        best = heapq.nsmallest(k, scores.items(), key=lambda kv: (-kv[1], self.ids[kv[0]]))
        return [(self.ids[index], score) for index, score in best]


def read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row: dict[str, object] = json.loads(line)
                yield row


def load_corpus(dest: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for row in read_jsonl(dest / "corpus.jsonl"):
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")
        docs.append((str(row["_id"]), f"{title} {text}".strip()))
    return docs


def load_test_queries(dest: Path) -> list[tuple[str, str]]:
    """Queries whose id appears in ``qrels/test.tsv``, in order of first appearance there."""
    qids: list[str] = []
    seen: set[str] = set()
    with (dest / "qrels/test.tsv").open(encoding="utf-8") as fh:
        next(fh)  # header: query-id, corpus-id, score
        for line in fh:
            if not line.strip():
                continue
            qid = line.split("\t")[0]
            if qid not in seen:
                seen.add(qid)
                qids.append(qid)
    texts = {str(row["_id"]): str(row["text"]) for row in read_jsonl(dest / "queries.jsonl")}
    missing = [qid for qid in qids if qid not in texts]
    if missing:
        raise ValueError(f"{len(missing)} test qids have no query text, e.g. {missing[0]}")
    return [(qid, texts[qid]) for qid in qids]


def parse_degrade(spec: str | None) -> tuple[str, int] | None:
    if spec is None:
        return None
    mode, sep, value = spec.partition(":")
    if mode not in {"truncate", "shuffle-tail"} or not sep or not value.isdigit():
        raise ValueError("--degrade must be truncate:N or shuffle-tail:SEED (N, SEED integers)")
    return mode, int(value)


def degrade(ranked: Ranked, mode: str, value: int) -> Ranked:
    if mode == "truncate":
        return ranked[:value]
    head, tail = ranked[:HEAD], ranked[HEAD:]
    tail_ids = [doc_id for doc_id, _ in tail]
    random.Random(value).shuffle(tail_ids)  # noqa: S311 - deliberate, seeded, not security
    return head + [(doc_id, score) for doc_id, (_, score) in zip(tail_ids, tail, strict=True)]


def run_tag(spec: tuple[str, int] | None) -> str:
    if spec is None:
        return "bm25"
    mode, value = spec
    return f"bm25-degraded-{mode}{value}"


def write_run(
    out: Path, queries: Sequence[tuple[str, str]], index: BM25, k: int, spec: tuple[str, int] | None
) -> tuple[int, int]:
    """Write the TREC run; return ``(queries, lines)`` written."""
    tag = run_tag(spec)
    lines = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for qid, text in queries:
            ranked = index.search(text, k)
            if spec is not None:
                ranked = degrade(ranked, *spec)
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                fh.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {tag}\n")
                lines += 1
    return len(queries), lines


def dataset_present(dest: Path) -> bool:
    return all((dest / relative).is_file() for relative in MEMBERS.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic pure-Python BM25 baseline over SciFact, as a TREC run."
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="dataset directory")
    parser.add_argument("--out", type=Path, required=True, help="TREC run file to write")
    parser.add_argument("--k", type=int, default=100, help="documents per query (default 100)")
    parser.add_argument("--degrade", default=None, help="truncate:N or shuffle-tail:SEED")
    parser.add_argument(
        "--offline", action="store_true", help="never download; fail if --dest is incomplete"
    )
    args = parser.parse_args(argv)
    try:
        spec = parse_degrade(args.degrade)
        if args.k <= 0:
            raise ValueError("--k must be positive")
        if not dataset_present(args.dest):
            if args.offline:
                raise FileNotFoundError(
                    f"{args.dest} is incomplete and --offline is set; run scripts/fetch_scifact.py"
                )
            fetch(args.dest)
        started = time.perf_counter()
        index = BM25(load_corpus(args.dest))
        queries = load_test_queries(args.dest)
        n_queries, n_lines = write_run(args.out, queries, index, args.k, spec)
    except (OSError, ValueError, KeyError) as exc:
        print(f"bm25_baseline: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    print(
        f"{args.out}: {n_queries} queries, {n_lines} lines, tag {run_tag(spec)},"
        f" {len(index.ids)} docs indexed, {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
