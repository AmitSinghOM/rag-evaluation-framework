"""SciFact dogfood fixtures: manifest integrity, TREC shape, documented degradations.

Nothing here touches the network. The committed fixtures under
``examples/scifact/`` are checked against ``MANIFEST.json``; the scripts are
exercised on a synthetic 20-document corpus built in a temp dir.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import random
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXTURES = ROOT / "examples" / "scifact"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text(encoding="utf-8"))
RUN_NAMES = sorted(name for name in MANIFEST["files"] if name.endswith(".run"))
BASELINE = "bm25.top100.run"
TRUNCATED = "bm25-degraded-truncate20.top20.run"
SHUFFLED = "bm25-degraded-shuffle-tail0.top100.run"
N_QUERIES = 300  # SciFact test split


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` by path (scripts/ is not a package)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))  # bm25_baseline imports fetch_scifact by name
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_run(path: Path) -> dict[str, list[tuple[str, int, float, str]]]:
    """qid -> [(docid, rank, score, tag)] in file order."""
    runs: dict[str, list[tuple[str, int, float, str]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        qid, q0, docid, rank, score, tag = line.split()
        assert q0 == "Q0"
        runs[qid].append((docid, int(rank), float(score), tag))
    return runs


def assert_valid_run(runs: dict[str, list[tuple[str, int, float, str]]], tag: str) -> None:
    assert len(runs) == N_QUERIES
    for qid, rows in runs.items():
        ranks = [rank for _, rank, _, _ in rows]
        assert ranks == list(range(1, len(rows) + 1)), f"{qid}: ranks not contiguous"
        scores = [score for _, _, score, _ in rows]
        assert all(a >= b for a, b in zip(scores, scores[1:], strict=False)), qid
        assert {row[3] for row in rows} == {tag}
        assert len({docid for docid, _, _, _ in rows}) == len(rows), f"{qid}: duplicate docid"


# --- committed fixtures vs manifest -------------------------------------------------------


def test_manifest_lists_exactly_the_committed_fixtures() -> None:
    on_disk = sorted(p.name for p in FIXTURES.iterdir() if p.name != "MANIFEST.json")
    assert on_disk == sorted(MANIFEST["files"])
    assert MANIFEST["source_url"].startswith("https://")
    assert "CC BY 4.0" in MANIFEST["licence"] and "ODC-By 1.0" in MANIFEST["licence"]
    assert sum(info["bytes"] for info in MANIFEST["files"].values()) < 5 * 1024 * 1024


@pytest.mark.parametrize("name", sorted(MANIFEST["files"]))
def test_fixture_matches_manifest(name: str) -> None:
    info = MANIFEST["files"][name]
    data = (FIXTURES / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == info["sha256"]
    assert len(data) == info["bytes"]
    assert data.decode("utf-8").count("\n") == info["lines"]


def test_qrels_has_300_queries_and_binary_grades() -> None:
    qids: set[str] = set()
    grades: set[int] = set()
    for line in (FIXTURES / "qrels.test.tsv").read_text(encoding="utf-8").splitlines():
        qid, zero, docid, grade = line.split()
        assert zero == "0" and docid
        qids.add(qid)
        grades.add(int(grade))
    assert len(qids) == N_QUERIES == MANIFEST["files"]["qrels.test.tsv"]["queries"]
    assert grades <= {0, 1}


def qrels_qids() -> set[str]:
    text = (FIXTURES / "qrels.test.tsv").read_text(encoding="utf-8")
    return {line.split()[0] for line in text.splitlines()}


@pytest.mark.parametrize("name", RUN_NAMES)
def test_run_shape(name: str) -> None:
    info = MANIFEST["files"][name]
    runs = read_run(FIXTURES / name)
    assert_valid_run(runs, info["tag"])
    assert {len(rows) for rows in runs.values()} == {info["per_query"]}
    assert set(runs) == qrels_qids()


def test_truncated_run_is_the_baseline_head() -> None:
    base = read_run(FIXTURES / BASELINE)
    trunc = read_run(FIXTURES / TRUNCATED)
    for qid, rows in trunc.items():
        expected = [(d, r, s, "bm25-degraded-truncate20") for d, r, s, _ in base[qid][:20]]
        assert rows == expected, qid


def test_shuffled_run_is_the_seeded_tail_permutation() -> None:
    base = read_run(FIXTURES / BASELINE)
    shuffled = read_run(FIXTURES / SHUFFLED)
    differs = 0
    for qid, rows in shuffled.items():
        head, tail = base[qid][:10], base[qid][10:]
        tail_ids = [d for d, _, _, _ in tail]
        random.Random(0).shuffle(tail_ids)
        permuted = head + [
            (d, r, s, "bm25") for d, (_, r, s, _) in zip(tail_ids, tail, strict=True)
        ]
        expected = [(d, r, s, "bm25-degraded-shuffle-tail0") for d, r, s, _ in permuted]
        assert rows == expected, qid
        differs += rows != [(d, r, s, rows[0][3]) for d, r, s, _ in base[qid]]
    assert differs == N_QUERIES  # every query's tail actually moved


# --- scripts on a synthetic corpus, offline ----------------------------------------------

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()


def write_synthetic_dataset(dest: Path) -> None:
    """20 docs; doc ``d<i>`` carries the unique title token ``uniq<i>`` plus random filler."""
    rng = random.Random(7)
    (dest / "qrels").mkdir(parents=True)
    with (dest / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for i in range(20):
            doc = {"_id": f"d{i}", "title": f"uniq{i}", "text": " ".join(rng.choices(WORDS, k=15))}
            fh.write(json.dumps(doc) + "\n")
    with (dest / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for q in range(3):
            fh.write(json.dumps({"_id": f"q{q}", "text": f"uniq{q} {WORDS[q]}"}) + "\n")
        fh.write(json.dumps({"_id": "q-train", "text": "not in the test split"}) + "\n")
    with (dest / "qrels" / "test.tsv").open("w", encoding="utf-8") as fh:
        fh.write("query-id\tcorpus-id\tscore\n")
        for q in range(3):
            fh.write(f"q{q}\td{q}\t1\n")


def run_bm25(dest: Path, out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and program path
        [sys.executable, str(SCRIPTS / "bm25_baseline.py"), "--dest", str(dest), "--out", str(out)]
        + ["--offline", *extra],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_bm25_offline_smoke_is_deterministic(tmp_path: Path) -> None:
    dest = tmp_path / "data"
    write_synthetic_dataset(dest)
    first, second = tmp_path / "a.run", tmp_path / "b.run"
    assert run_bm25(dest, first, "--k", "5").returncode == 0
    assert run_bm25(dest, second, "--k", "5").returncode == 0
    assert first.read_bytes() == second.read_bytes()
    runs = read_run(first)
    assert sorted(runs) == ["q0", "q1", "q2"]  # q-train is not in the test qrels
    for qid, rows in runs.items():
        assert [r for _, r, _, _ in rows] == [1, 2, 3, 4, 5]
        scores = [s for _, _, s, _ in rows]
        assert scores == sorted(scores, reverse=True)
        assert {row[3] for row in rows} == {"bm25"}
        # doc d<i> alone carries the query's unique token uniq<i>, so it must rank first.
        assert rows[0][0] == f"d{qid[1:]}"


def test_bm25_offline_degrade_modes(tmp_path: Path) -> None:
    dest = tmp_path / "data"
    write_synthetic_dataset(dest)
    base, trunc, shuf = (tmp_path / n for n in ("base.run", "trunc.run", "shuf.run"))
    assert run_bm25(dest, base, "--k", "15").returncode == 0
    assert run_bm25(dest, trunc, "--k", "15", "--degrade", "truncate:3").returncode == 0
    assert run_bm25(dest, shuf, "--k", "15", "--degrade", "shuffle-tail:1").returncode == 0
    b, t, s = read_run(base), read_run(trunc), read_run(shuf)
    for qid in b:
        assert t[qid] == [(d, r, sc, "bm25-degraded-truncate3") for d, r, sc, _ in b[qid][:3]]
        assert s[qid][:10] == [
            (d, r, sc, "bm25-degraded-shuffle-tail1") for d, r, sc, _ in b[qid][:10]
        ]
        assert sorted(d for d, _, _, _ in s[qid][10:]) == sorted(d for d, _, _, _ in b[qid][10:])
        assert [sc for _, _, sc, _ in s[qid]] == [sc for _, _, sc, _ in b[qid]]


def test_bm25_offline_refuses_missing_dataset(tmp_path: Path) -> None:
    result = run_bm25(tmp_path / "missing", tmp_path / "x.run")
    assert result.returncode == 1
    assert "--offline" in result.stderr and "fetch_scifact" in result.stderr


def test_bm25_rejects_bad_degrade_spec(tmp_path: Path) -> None:
    dest = tmp_path / "data"
    write_synthetic_dataset(dest)
    result = run_bm25(dest, tmp_path / "x.run", "--degrade", "reverse:1")
    assert result.returncode == 1 and "--degrade must be" in result.stderr


# --- fetch_scifact pieces that need no network -------------------------------------------


def test_fetch_verify_rejects_wrong_sha256() -> None:
    fetch_scifact = load_script("fetch_scifact")
    with pytest.raises(fetch_scifact.ChecksumMismatch):
        fetch_scifact.verify(b"not the zip")
    fetch_scifact.verify(b"abc", hashlib.sha256(b"abc").hexdigest())


def test_fetch_extract_and_trec_conversion(tmp_path: Path) -> None:
    fetch_scifact = load_script("fetch_scifact")
    beir_qrels = "query-id\tcorpus-id\tscore\n1\t31715818\t1\n3\t14717500\t0\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scifact/corpus.jsonl", '{"_id": "31715818", "text": "x"}\n')
        archive.writestr("scifact/queries.jsonl", '{"_id": "1", "text": "q"}\n')
        archive.writestr("scifact/qrels/test.tsv", beir_qrels)
        archive.writestr("scifact/qrels/train.tsv", "query-id\tcorpus-id\tscore\n")
    dest = tmp_path / "cache"
    written = fetch_scifact.extract(buffer.getvalue(), dest)
    assert sorted(p.relative_to(dest).as_posix() for p in written) == [
        "corpus.jsonl",
        "qrels/test.tsv",
        "queries.jsonl",
    ]
    assert (dest / ".gitignore").read_text() == "*\n"
    assert not (dest / "qrels/train.tsv").exists()
    assert fetch_scifact.to_trec_qrels(beir_qrels) == "1 0 31715818 1\n3 0 14717500 0\n"
    with pytest.raises(ValueError, match="BEIR qrels"):
        fetch_scifact.to_trec_qrels("qid\tdocid\n1\t2\n")
