"""TREC qrels, TREC run and JSONL loaders: typed output, ordering, errors with file:line."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from rageval import LoaderError, read_jsonl, read_qrels, read_run

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------- qrels


def test_graded_qrels_keep_grade_zero_as_judged_not_relevant() -> None:
    qrels = read_qrels(TOY / "qrels.txt")
    assert qrels["q1"] == {"d1": 2.0, "d2": 1.0, "d3": 0.0}
    assert qrels["q3"] == {"d5": 3.0, "d6": 3.0}
    assert list(qrels) == ["q1", "q2", "q3", "q4", "q5", "q6"]  # file order
    assert all(isinstance(g, float) for judged in qrels.values() for g in judged.values())


def test_qrels_skips_blank_lines_and_ignores_the_unused_second_field(tmp_path: Path) -> None:
    path = _write(tmp_path, "q.txt", "\nq1 0 d1 1\n\n   \nq1 7 d2 2\n")
    assert read_qrels(path) == {"q1": {"d1": 1.0, "d2": 2.0}}


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("q1 0 d1", "expected 4 fields"),
        ("q1 0 d1 1 extra", "expected 4 fields"),
        ("q1 0 d1 one", "grade must be an integer"),
        ("q1 0 d1 1.5", "grade must be an integer"),
        ("q1 0 d1 -1", "grade must be >= 0"),
    ],
)
def test_malformed_qrels_line_names_file_and_line(tmp_path: Path, line: str, reason: str) -> None:
    path = _write(tmp_path, "bad.txt", f"q1 0 d0 1\n{line}\n")
    with pytest.raises(LoaderError, match=reason) as info:
        read_qrels(path)
    assert str(info.value).startswith(f"{path}:2: ")
    assert (info.value.path, info.value.line_no) == (str(path), 2)


def test_duplicate_judgment_is_an_error_not_an_overwrite(tmp_path: Path) -> None:
    path = _write(tmp_path, "dup.txt", "q1 0 d1 1\nq1 0 d1 0\n")
    with pytest.raises(LoaderError, match=r"dup.txt:2: duplicate judgment for query 'q1' doc 'd1'"):
        read_qrels(path)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_qrels(tmp_path / "nope.txt")


# ---------------------------------------------------------------- run


def test_run_is_ordered_by_score_desc_then_rank_asc(tmp_path: Path) -> None:
    # File order is deliberately scrambled; scores decide, then rank on ties.
    path = _write(
        tmp_path,
        "run.txt",
        "q1 Q0 low 3 1.0 t\nq1 Q0 tie_r2 2 5.0 t\nq1 Q0 top 1 9.0 t\nq1 Q0 tie_r1 1 5.0 t\n",
    )
    assert read_run(path) == {"q1": ["top", "tie_r1", "tie_r2", "low"]}


def test_full_tie_keeps_file_order(tmp_path: Path) -> None:
    path = _write(tmp_path, "run.txt", "q1 Q0 b 1 1.0 t\nq1 Q0 a 1 1.0 t\nq1 Q0 c 1 1.0 t\n")
    assert read_run(path) == {"q1": ["b", "a", "c"]}


def test_run_keeps_duplicate_doc_ids(tmp_path: Path) -> None:
    path = _write(tmp_path, "run.txt", "q1 Q0 d1 1 3.0 t\nq1 Q0 d1 2 2.0 t\nq1 Q0 d2 3 1.0 t\n")
    assert read_run(path) == {"q1": ["d1", "d1", "d2"]}


def test_run_may_hold_queries_absent_from_qrels() -> None:
    run = read_run(TOY / "run_a.txt")
    qrels = read_qrels(TOY / "qrels.txt")
    assert "q99" in run and "q99" not in qrels
    assert "q6" in qrels and "q6" not in run
    assert run["q3"] == ["d17", "d18", "d19", "d5", "d20"]
    assert all(isinstance(doc, str) for docs in run.values() for doc in docs)


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("q1 Q0 d1 1 2.0", "expected 6 fields"),
        ("q1 Q0 d1 first 2.0 t", "rank must be an integer"),
        ("q1 Q0 d1 1 high t", "score must be a number"),
        ("q1 Q0 d1 1 nan t", "score must not be NaN"),
    ],
)
def test_malformed_run_line_names_file_and_line(tmp_path: Path, line: str, reason: str) -> None:
    path = _write(tmp_path, "bad.txt", f"q1 Q0 d0 1 3.0 t\nq1 Q0 d0 2 2.0 t\n{line}\n")
    with pytest.raises(LoaderError, match=reason) as info:
        read_run(path)
    assert str(info.value).startswith(f"{path}:3: ")


# ---------------------------------------------------------------- JSONL


def test_jsonl_matches_the_trec_files_for_the_same_data() -> None:
    qrels, run = read_jsonl(TOY / "dataset.jsonl")
    assert qrels == read_qrels(TOY / "qrels.txt")
    assert run == read_run(TOY / "run_b.txt")


def test_jsonl_list_of_relevant_ids_means_grade_one(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "d.jsonl",
        '{"query_id": "a", "query": null, "retrieved": ["x", "x"], "relevant": ["x", "y"]}\n',
    )
    qrels, run = read_jsonl(path)
    assert qrels == {"a": {"x": 1.0, "y": 1.0}}
    assert run == {"a": ["x", "x"]}  # repeats kept


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("{not json", "invalid JSON"),
        ("[1, 2]", "expected a JSON object"),
        ('{"query_id": 1, "retrieved": [], "relevant": []}', "'query_id' must be a string"),
        ('{"query_id": "b", "retrieved": []}', "needs 'retrieved' and 'relevant'"),
        ('{"query_id": "b", "retrieved": [1], "relevant": []}', "'retrieved' must be a list"),
        ('{"query_id": "b", "retrieved": [], "relevant": "x"}', "'relevant' must be a list"),
        ('{"query_id": "b", "retrieved": [], "relevant": {"x": -1}}', "must be a number >= 0"),
        ('{"query_id": "b", "retrieved": [], "relevant": {"x": true}}', "must be a number >= 0"),
        ('{"query_id": "b", "retrieved": [], "relevant": [2]}', "'relevant' must be a list"),
        ('{"query_id": "a", "retrieved": [], "relevant": []}', "duplicate query_id 'a'"),
    ],
)
def test_malformed_jsonl_line_names_file_and_line(tmp_path: Path, line: str, reason: str) -> None:
    good = '{"query_id": "a", "query": "q", "retrieved": ["x"], "relevant": {"x": 2}}'
    path = _write(tmp_path, "bad.jsonl", f"{good}\n{line}\n")
    with pytest.raises(LoaderError, match=reason) as info:
        read_jsonl(path)
    assert str(info.value).startswith(f"{path}:2: ")


# ---------------------------------------------------------------- encoding


def test_a_utf8_bom_is_not_part_of_the_first_query_id(tmp_path: Path) -> None:
    # Windows editors prepend U+FEFF; str.strip() does not remove it, so on the
    # unfixed code the first qrels query became '\ufeffq1', the run's 'q1' was
    # "not in the qrels" and the query silently scored 0.
    qrels = tmp_path / "q.txt"
    qrels.write_bytes(b"\xef\xbb\xbfq1 0 d1 1\nq2 0 d2 1\n")
    run = tmp_path / "r.txt"
    run.write_bytes(b"\xef\xbb\xbfq1 Q0 d1 1 1.0 t\nq2 Q0 d2 1 1.0 t\n")
    jsonl = tmp_path / "d.jsonl"
    jsonl.write_bytes(b'\xef\xbb\xbf{"query_id": "q1", "retrieved": ["d1"], "relevant": ["d1"]}\n')
    assert list(read_qrels(qrels)) == ["q1", "q2"]
    assert list(read_run(run)) == ["q1", "q2"]
    assert list(read_jsonl(jsonl)[0]) == ["q1"]


@pytest.mark.parametrize("reader", [read_qrels, read_run, read_jsonl])
def test_a_non_utf8_file_is_a_loader_error_naming_the_file(
    tmp_path: Path, reader: Callable[[Path], object]
) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes(b"q1 0 d\xe9 1\n")
    with pytest.raises(LoaderError, match="not UTF-8") as info:
        reader(path)
    assert str(info.value).startswith(f"{path}:0: ")
