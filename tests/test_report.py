"""evaluate(), the schema_version-1 report and compare(): hand-checked toy numbers."""

from __future__ import annotations

import json
from math import log2
from pathlib import Path

import jsonschema
import pytest

from rageval import (
    Report,
    ReportError,
    compare,
    evaluate,
    parse_metric,
    read_qrels,
    read_run,
)
from rageval.report import FileRef

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy"
SCHEMA = json.loads((ROOT / "docs" / "report.schema.json").read_text(encoding="utf-8"))
METRICS = ["ndcg@10", "recall@100", "mrr", "precision@3"]
L3, L5 = log2(3), log2(5)


def _report(run_file: str) -> Report:
    return evaluate(
        read_qrels(TOY / "qrels.txt"),
        read_run(TOY / run_file),
        METRICS,
        qrels_ref=FileRef.of(TOY / "qrels.txt"),
        run_ref=FileRef.of(TOY / run_file),
        run_name=run_file.removesuffix(".txt"),
    )


# ---------------------------------------------------------------- metric names


@pytest.mark.parametrize(
    ("name", "parsed"),
    [
        ("ndcg@10", ("ndcg", 10)),
        ("recall@100", ("recall", 100)),
        ("precision@3", ("precision", 3)),
        ("mrr", ("mrr", None)),
        ("mrr@5", ("mrr", 5)),
    ],
)
def test_parse_metric_accepts_the_five_forms(name: str, parsed: tuple[str, int | None]) -> None:
    assert parse_metric(name) == parsed


@pytest.mark.parametrize(
    "name", ["NDCG@10", "ndcg", "ndcg@", "ndcg@0", "ndcg@-1", "ndcg@1.5", "map@10", "", "mrr@"]
)
def test_parse_metric_rejects_bad_names_listing_valid_forms(name: str) -> None:
    with pytest.raises(ValueError, match=r"valid forms: ndcg@K, recall@K, precision@K, mrr, mrr@K"):
        parse_metric(name)


# ---------------------------------------------------------------- evaluate


def test_per_query_values_match_the_hand_check() -> None:
    per_query = _report("run_a.txt").per_query
    # q1: d1 (grade 2) at rank 2, d2 (grade 1) at rank 3; ideal is [2, 1].
    assert per_query["q1"]["ndcg@10"] == pytest.approx((2 / L3 + 1 / 2) / (2 + 1 / L3))
    assert per_query["q1"] == pytest.approx(
        {"ndcg@10": 0.669672, "recall@100": 1.0, "mrr": 0.5, "precision@3": 2 / 3}, abs=1e-6
    )
    # q3: one of two grade-3 docs at rank 4.
    assert per_query["q3"]["ndcg@10"] == pytest.approx((3 / L5) / (3 + 3 / L3))
    assert per_query["q3"]["recall@100"] == 0.5
    assert per_query["q3"]["mrr"] == 0.25
    # q6 is judged but missing from run_a: zero for every metric.
    assert per_query["q6"] == {name: 0.0 for name in METRICS}
    assert list(per_query) == ["q1", "q2", "q3", "q4", "q5", "q6"]


def test_means_counts_and_warnings_match_the_hand_check() -> None:
    report = _report("run_a.txt")
    means = {name: summary.mean for name, summary in report.metrics.items()}
    assert means == pytest.approx(
        {
            "ndcg@10": (0.669672 + 1 + 0.264068 + 0 + 1 + 0) / 6,
            "recall@100": 3.5 / 6,
            "mrr": 2.75 / 6,
            "precision@3": (2 / 3 + 1 / 3 + 2 / 3) / 6,
        },
        abs=1e-6,
    )
    assert all(summary.n == 6 for summary in report.metrics.values())
    assert report.dataset.n_queries == 6
    assert report.dataset.n_judged_relevant == 9  # grade-0 d3 does not count
    assert report.run.n_queries_in_run == 6  # q1-q5 and q99
    assert report.run.duplicate_ids == 0
    assert report.warnings == ["query 'q99' is in the run but not in the qrels; ignored"]
    assert report.run.name == "run_a"
    assert report.run.sha256 == FileRef.of(TOY / "run_a.txt").sha256


def test_ci_brackets_the_mean_with_the_recorded_bootstrap_settings() -> None:
    report = evaluate(read_qrels(TOY / "qrels.txt"), read_run(TOY / "run_b.txt"), ["mrr"], seed=3)
    summary = report.metrics["mrr"]
    assert summary.ci95[0] <= summary.mean <= summary.ci95[1]
    assert (summary.bootstrap.resamples, summary.bootstrap.seed) == (1000, 3)


def test_duplicates_are_counted_and_scored_once() -> None:
    qrels = {"q1": {"d1": 1.0, "d2": 1.0}}
    run = {"q1": ["d1", "d1", "d2", "d2", "d2"], "extra": ["d9", "d9"]}
    report = evaluate(qrels, run, ["recall@5", "precision@5"], resamples=10)
    assert report.run.duplicate_ids == 4  # q1: 1 repeat of d1 + 2 of d2; 'extra': 1 repeat of d9
    assert report.per_query["q1"] == {"recall@5": 1.0, "precision@5": 0.4}
    assert report.dataset.qrels == FileRef(None, None)
    assert (report.run.name, report.run.path, report.run.sha256) == (None, None, None)


def test_evaluate_rejects_bad_metric_empty_metrics_and_empty_qrels() -> None:
    qrels = {"q1": {"d1": 1.0}}
    with pytest.raises(ValueError, match="unknown metric 'map@10'"):
        evaluate(qrels, {}, ["ndcg@10", "map@10"])
    with pytest.raises(ValueError, match="at least one metric"):
        evaluate(qrels, {}, [])
    with pytest.raises(ValueError, match="no queries"):
        evaluate({}, {}, ["mrr"])


# ---------------------------------------------------------------- report file


def test_report_json_validates_against_the_published_schema(tmp_path: Path) -> None:
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    validator = jsonschema.Draft202012Validator(SCHEMA)
    for run_file in ("run_a.txt", "run_b.txt"):
        report = _report(run_file)
        report.write(tmp_path / "r.json")
        data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
        validator.validate(data)
        assert data == report.to_dict()
    # In-memory evaluation leaves the file references null, which the schema allows.
    validator.validate(evaluate({"q": {"d": 1.0}}, {"q": ["d"]}, ["mrr"], resamples=5).to_dict())


def test_schema_rejects_an_unknown_metric_name_and_an_extra_key() -> None:
    validator = jsonschema.Draft202012Validator(SCHEMA)
    data = _report("run_b.txt").to_dict()
    data["metrics"]["map@10"] = data["metrics"]["mrr"]
    assert not validator.is_valid(data)
    data = _report("run_b.txt").to_dict()
    data["extra"] = 1
    assert not validator.is_valid(data)


def test_report_floats_are_rounded_to_six_places_and_keys_are_in_contract_order() -> None:
    data = _report("run_a.txt").to_dict()
    assert list(data) == [
        "schema_version",
        "rageval_version",
        "created_at",
        "dataset",
        "run",
        "metrics",
        "per_query",
        "warnings",
    ]
    assert data["metrics"]["ndcg@10"]["mean"] == 0.488957
    assert data["per_query"]["q1"]["ndcg@10"] == 0.669672
    assert data["metrics"]["ndcg@10"]["ci95"] == [
        round(v, 6) for v in _report("run_a.txt").metrics["ndcg@10"].ci95
    ]


def test_same_seed_gives_a_byte_identical_report_apart_from_created_at() -> None:
    first = _report("run_a.txt").to_dict()
    second = _report("run_a.txt").to_dict()
    first.pop("created_at")
    second.pop("created_at")
    assert json.dumps(first) == json.dumps(second)


def test_report_round_trips_through_json(tmp_path: Path) -> None:
    report = _report("run_b.txt")
    report.write(tmp_path / "b.json")
    loaded = Report.read(tmp_path / "b.json")
    assert loaded.to_dict() == report.to_dict()
    assert loaded.metrics["mrr"].ci95 == tuple(report.to_dict()["metrics"]["mrr"]["ci95"])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{not json", "invalid JSON"),
        ('{"schema_version": 2}', "unsupported schema_version 2"),
        ('{"schema_version": 1, "dataset": {}}', "not a rageval report"),
        ("[]", "not a rageval report"),
        ('{"schema_version": 1, "rageval_version": "x", "created_at": "t", "dataset": 3}', "not a"),
    ],
)
def test_reading_a_bad_report_is_a_report_error(tmp_path: Path, text: str, message: str) -> None:
    (tmp_path / "bad.json").write_text(text, encoding="utf-8")
    with pytest.raises(ReportError, match=message):
        Report.read(tmp_path / "bad.json")


# ---------------------------------------------------------------- compare


def test_compare_reports_paired_difference_b_minus_a_with_ci() -> None:
    result = compare(_report("run_a.txt"), _report("run_b.txt"))
    assert list(result.metrics) == METRICS
    recall = result.metrics["recall@100"]
    assert (recall.mean_a, recall.mean_b) == pytest.approx((3.5 / 6, 1.0))
    assert recall.diff == pytest.approx(1.0 - 3.5 / 6)
    assert recall.ci95[0] <= recall.diff <= recall.ci95[1]
    assert result.n_queries == 6
    assert (result.a.name, result.b.name) == ("run_a", "run_b")
    assert result.qrels_sha256 == FileRef.of(TOY / "qrels.txt").sha256
    assert (result.bootstrap.resamples, result.bootstrap.seed) == (1000, 0)


def test_compare_uses_only_metrics_present_in_both_reports() -> None:
    a = evaluate(read_qrels(TOY / "qrels.txt"), read_run(TOY / "run_a.txt"), ["mrr", "ndcg@3"])
    b = evaluate(read_qrels(TOY / "qrels.txt"), read_run(TOY / "run_b.txt"), ["ndcg@3", "recall@5"])
    assert list(compare(a, b).metrics) == ["ndcg@3"]


def test_compare_output_shapes() -> None:
    result = compare(_report("run_a.txt"), _report("run_b.txt"))
    table = result.to_table()
    lines = table.splitlines()
    assert lines[0].split() == ["metric", "run_a", "run_b", "diff", "(b-a)", "ci95"]
    assert lines[2].startswith("recall@100") and "+0.416667" in lines[2]
    assert lines[-1] == "n_queries: 6"
    data = json.loads(result.to_json())
    assert data["schema_version"] == 1
    assert data["metrics"]["recall@100"] == {
        "a": 0.583333,
        "b": 1.0,
        "diff": 0.416667,
        "ci95": [round(v, 6) for v in result.metrics["recall@100"].ci95],
    }
    unnamed = compare(
        evaluate({"q": {"d": 1.0}}, {"q": ["d"]}, ["mrr"], resamples=5),
        evaluate({"q": {"d": 1.0}}, {"q": ["x", "d"]}, ["mrr"], resamples=5),
    )
    assert unnamed.to_table().splitlines()[0].split()[1:3] == ["a", "b"]


def test_compare_refuses_different_qrels_different_queries_or_nothing_shared(
    tmp_path: Path,
) -> None:
    a = _report("run_a.txt")
    other_qrels = tmp_path / "other.txt"
    other_qrels.write_text("q1 0 d1 1\n", encoding="utf-8")
    b = evaluate(
        read_qrels(other_qrels), {"q1": ["d1"]}, METRICS, qrels_ref=FileRef.of(other_qrels)
    )
    with pytest.raises(ReportError, match="different qrels"):
        compare(a, b)
    same_sha_other_queries = Report.from_dict(
        {
            **b.to_dict(),
            "dataset": {**b.to_dict()["dataset"], "qrels": a.to_dict()["dataset"]["qrels"]},
        }
    )
    with pytest.raises(ReportError, match="same query ids"):
        compare(a, same_sha_other_queries)
    only_mrr = evaluate(read_qrels(TOY / "qrels.txt"), read_run(TOY / "run_b.txt"), ["mrr@3"])
    with pytest.raises(ReportError, match="no metric is present in both"):
        compare(evaluate(read_qrels(TOY / "qrels.txt"), {}, ["mrr"]), only_mrr)
    broken = a.to_dict()
    del broken["per_query"]["q4"]["mrr"]
    with pytest.raises(ReportError, match=r"a report: per_query\['q4'\]\['mrr'\] is None"):
        compare(Report.from_dict(broken), _report("run_b.txt"))


@pytest.mark.parametrize(
    ("value", "shown"),
    [(float("nan"), "nan"), (float("inf"), "inf"), ("0.5", "'0.5'"), (True, "True")],
)
def test_compare_refuses_a_non_finite_or_non_numeric_per_query_value(
    value: object, shown: str
) -> None:
    # Same rule as the gate. On the unfixed code a NaN row gave "+nan" with exit 0
    # and a string row crashed with TypeError (exit 3, "internal error").
    a = _report("run_a.txt")
    edited = a.to_dict()
    edited["per_query"]["q2"]["recall@100"] = value
    expected = rf"b report: per_query\['q2'\]\['recall@100'\] is {shown}"
    with pytest.raises(ReportError, match=expected):
        compare(a, Report.from_dict(edited))
