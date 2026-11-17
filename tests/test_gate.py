"""`rageval gate`: the non-inferiority verdict, exit codes, output and the invariant."""

from __future__ import annotations

import json
import random
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from rageval import Report, gate
from rageval.cli import EXIT_GATE_FAILED, EXIT_INPUT, EXIT_OK, EXIT_USAGE, main

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy"
QRELS = str(TOY / "qrels.txt")
METRICS = "ndcg@10,recall@100,mrr,precision@3"
GATED = [
    "--metric",
    "ndcg@10",
    "--max-drop",
    "0.02",
    "--metric",
    "recall@100",
    "--max-drop",
    "0.05",
]


def _run(out: Path, run_file: str | Path, qrels: str | Path = QRELS) -> Path:
    argv = ["run", "--qrels", str(qrels), "--run", str(run_file), "--metrics", METRICS]
    assert main([*argv, "--out", str(out)]) == EXIT_OK
    return out


def _trec_run(path: Path, rankings: dict[str, list[str]]) -> Path:
    lines = [
        f"{qid} Q0 {doc} {rank} {100 - rank} test"
        for qid, docs in rankings.items()
        for rank, doc in enumerate(docs, start=1)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _synthetic(tmp_path: Path, n: int, base_rank: int, cand_rank: int) -> tuple[Path, Path]:
    """``n`` queries, one relevant doc each, placed at ``base_rank`` / ``cand_rank`` (1-based)."""
    qrels = tmp_path / f"qrels_{n}.txt"
    qrels.write_text("".join(f"s{i} 0 rel{i} 1\n" for i in range(n)), encoding="utf-8")

    def ranking(i: int, rank: int) -> list[str]:
        docs = [f"x{i}_{j}" for j in range(4)]
        docs.insert(rank - 1, f"rel{i}")
        return docs

    reports = []
    for label, rank in (("base", base_rank), ("cand", cand_rank)):
        run = _trec_run(
            tmp_path / f"{label}_{n}.txt", {f"s{i}": ranking(i, rank) for i in range(n)}
        )
        reports.append(_run(tmp_path / f"{label}_{n}.json", run, qrels))
    return reports[0], reports[1]


def _gate(
    capsys: pytest.CaptureFixture[str], base: Path, cand: Path, *extra: str
) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(["gate", "--baseline", str(base), "--candidate", str(cand), *extra])
    out, err = capsys.readouterr()
    return code, out, err


def _report_dict(values: dict[str, Any], name: str = "x", qrels_sha: str | None = "q") -> Any:
    """A minimal schema-1 report holding one metric, ``ndcg@10``, per query."""
    return {
        "schema_version": 1,
        "rageval_version": "0.1.0a0",
        "created_at": "2026-11-18T00:00:00Z",
        "dataset": {
            "qrels": {"path": None, "sha256": qrels_sha},
            "n_queries": len(values),
            "n_judged_relevant": len(values),
        },
        "run": {
            "name": name,
            "path": None,
            "sha256": None,
            "n_queries_in_run": len(values),
            "duplicate_ids": 0,
        },
        "metrics": {
            "ndcg@10": {
                "mean": 0.0,
                "ci95": [0.0, 0.0],
                "n": len(values),
                "bootstrap": {"resamples": 1, "seed": 0},
            }
        },
        "per_query": {qid: {"ndcg@10": value} for qid, value in values.items()},
        "warnings": [],
    }


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")  # json.dumps writes NaN as NaN
    return path


# ---------------------------------------------------------------- verdicts


def test_identical_reports_pass_with_a_zero_width_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a1 = _run(tmp_path / "a1.json", TOY / "run_a.txt")
    a2 = _run(tmp_path / "a2.json", TOY / "run_a.txt")
    code, out, err = _gate(capsys, a1, a2, *GATED, "--format", "json")
    assert code == EXIT_OK
    data = json.loads(out)
    assert data["verdict"] == "pass"
    # The reports also hold mrr and precision@3; only the gated metrics are checked.
    assert list(data["metrics"]) == ["ndcg@10", "recall@100"]
    for row in data["metrics"].values():
        assert row["ci"] == [0.0, 0.0] and row["diff"] == 0.0 and row["n"] == 6
        assert row["verdict"] == "pass"
    assert "only 6 shared queries" in err


def test_clearly_degraded_candidate_fails_with_exit_10(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _run(tmp_path / "b.json", TOY / "run_b.txt")
    cand = _run(tmp_path / "a.json", TOY / "run_a.txt")
    code, out, _ = _gate(capsys, base, cand, *GATED)
    assert code == EXIT_GATE_FAILED
    lines = out.splitlines()
    header = "metric baseline candidate diff (c-b) ci95 threshold verdict n"
    assert lines[0].split() == header.split()
    row = lines[2].split()
    assert row[:4] == ["recall@100", "1.000000", "0.583333", "-0.416667"]
    assert row[-3:] == ["-0.050000", "FAIL", "6"]
    assert lines[-1] == "gate: FAIL (0 of 2 metrics pass; candidate run_a vs baseline run_b)"


def test_candidate_improved_on_every_query_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base, cand = _synthetic(tmp_path, 40, base_rank=3, cand_rank=1)
    code, out, err = _gate(capsys, base, cand, *GATED, "--format", "json")
    assert code == EXIT_OK
    ndcg = json.loads(out)["metrics"]["ndcg@10"]
    assert ndcg["baseline"] == 0.5 and ndcg["candidate"] == 1.0
    assert ndcg["ci"] == [0.5, 0.5] and ndcg["verdict"] == "pass"
    assert err == ""  # 40 queries: no small-n warning


def test_small_noisy_change_on_six_queries_fails_because_the_interval_is_wide(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Against run_a this run helps q1 and q3 and hurts q2 and q5: the mean ndcg@10
    # moves by +0.0006, but the per-query swings are +-0.37. With six queries the
    # bootstrap interval of the mean difference is about [-0.17, +0.17], so the data
    # cannot show the drop is under 0.02 and the gate FAILS. This is the intended,
    # conservative behaviour of a non-inferiority gate: the candidate must prove it
    # is not worse; too little evidence is a fail, never a pass. A significance gate
    # ("is it significantly worse?") would pass this and every other change at n=6.
    noisy = _trec_run(
        tmp_path / "noisy.txt",
        {
            "q1": ["d1", "d2", "d3", "d11", "d12"],
            "q2": ["d13", "d4", "d14", "d15", "d16"],
            "q3": ["d17", "d5", "d18", "d19", "d20"],
            "q4": ["d21", "d22", "d23", "d24", "d25"],
            "q5": ["d8", "d26", "d9", "d27", "d28"],
        },
    )
    base = _run(tmp_path / "a.json", TOY / "run_a.txt")
    cand = _run(tmp_path / "noisy.json", noisy)
    code, out, err = _gate(capsys, base, cand, "--metric", "ndcg@10", "--max-drop", "0.02")
    assert code == EXIT_GATE_FAILED
    row = out.splitlines()[1].split()
    assert row[0] == "ndcg@10" and row[-2] == "FAIL"
    assert abs(float(row[3])) < 0.001  # the mean barely moved
    assert float(row[4].strip("[,")) < -0.1  # the lower bound is far below -0.02
    assert "only 6 shared queries" in err


def test_a_drop_of_exactly_max_drop_fails_despite_float_noise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 0.28 - 0.30 is -0.019999999999999962 in binary floating point, just above -0.02.
    # The verdict uses the bound rounded to the report's 6 places, so a drop of exactly
    # max_drop fails (the rule is strict: bound > -max_drop), as the printed table says.
    qids = [f"q{i}" for i in range(40)]
    base = _write(tmp_path / "base.json", _report_dict(dict.fromkeys(qids, 0.3)))
    cand = _write(tmp_path / "cand.json", _report_dict(dict.fromkeys(qids, 0.28)))
    code, out, _ = _gate(capsys, base, cand, "--metric", "ndcg@10", "--max-drop", "0.02")
    assert code == EXIT_GATE_FAILED
    assert "[-0.020000, -0.020000]   -0.020000  FAIL" in out


def test_one_failing_metric_fails_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _run(tmp_path / "b.json", TOY / "run_b.txt")
    cand = _run(tmp_path / "a.json", TOY / "run_a.txt")
    # recall@100 drops by 0.42; allowing a drop of 1.0 makes it pass, mrr still fails.
    argv = ["--metric", "recall@100", "--max-drop", "1.0", "--metric", "mrr", "--max-drop", "0.02"]
    code, out, _ = _gate(capsys, base, cand, *argv, "--format", "json")
    assert code == EXIT_GATE_FAILED
    data = json.loads(out)
    assert data["verdict"] == "fail"
    assert [m["verdict"] for m in data["metrics"].values()] == ["pass", "fail"]


# ---------------------------------------------------------------- determinism and options


def test_same_seed_gives_identical_output_and_options_reach_the_bootstrap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _run(tmp_path / "b.json", TOY / "run_b.txt")
    cand = _run(tmp_path / "a.json", TOY / "run_a.txt")
    argv = [*GATED, "--seed", "3", "--resamples", "500", "--format", "json"]
    first = _gate(capsys, base, cand, *argv)
    assert first == _gate(capsys, base, cand, *argv)
    data = json.loads(first[1])
    assert data["bootstrap"] == {"resamples": 500, "seed": 3, "level": 0.95}
    other_seed = json.loads(_gate(capsys, base, cand, *GATED, "--seed", "4", "--format", "json")[1])
    assert other_seed["metrics"]["ndcg@10"]["ci"] != data["metrics"]["ndcg@10"]["ci"]
    code, table, _ = _gate(capsys, base, cand, *GATED, "--seed", "3", "--level", "0.5")
    assert code == EXIT_GATE_FAILED
    assert table.splitlines()[0].split()[5] == "ci50"


@pytest.mark.parametrize(("n", "warned"), [(29, True), (30, False)])
def test_small_n_warning_below_30_queries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], n: int, warned: bool
) -> None:
    base, cand = _synthetic(tmp_path, n, base_rank=2, cand_rank=2)
    code, _, err = _gate(capsys, base, cand, *GATED)
    assert code == EXIT_OK
    assert (f"warning: only {n} shared queries (< 30)" in err) is warned


def test_summary_appends_a_markdown_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _run(tmp_path / "b.json", TOY / "run_b.txt")
    cand = _run(tmp_path / "a.json", TOY / "run_a.txt")
    summary = tmp_path / "summary.md"
    summary.write_text("earlier step\n", encoding="utf-8")
    code, _, _ = _gate(capsys, base, cand, *GATED, "--summary", str(summary))
    assert code == EXIT_GATE_FAILED
    text = summary.read_text(encoding="utf-8")
    assert text.startswith("earlier step\n### rageval gate: FAIL\n")
    assert "| metric | baseline | candidate | diff | 95% CI | threshold | verdict | n |" in text
    assert "| recall@100 | 1.000000 | 0.583333 | -0.416667 |" in text
    assert "| -0.050000 | **FAIL** | 6 |" in text
    assert "> warning: only 6 shared queries" in text


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--metric", "ndcg@10"], "needs a --max-drop"),
        (["--max-drop", "0.02", "--metric", "ndcg@10"], "must follow its own --metric"),
        (["--metric", "ndcg@10", "--max-drop", "0.02", "--max-drop", "0.1"], "must follow"),
        ([], "at least one --metric"),
        (
            ["--metric", "mrr", "--max-drop", "0.1", "--metric", "mrr", "--max-drop", "0.2"],
            "more than once",
        ),
        (["--metric", "ndcg@5", "--max-drop", "0.02"], "has no metric 'ndcg@5'"),
        (["--metric", "mrr", "--max-drop", "0"], "above 0"),
        (["--metric", "mrr", "--max-drop", "-0.01"], "above 0"),
        (["--metric", "mrr", "--max-drop", "nan"], "above 0"),
        (["--metric", "mrr", "--max-drop", "inf"], "above 0"),
        (["--metric", "mrr", "--max-drop", "abc"], "invalid float"),
        (["--metric", "mrr", "--max-drop", "0.1", "--resamples", "0"], ">= 1"),
        (["--metric", "mrr", "--max-drop", "0.1", "--level", "1.5"], "between 0 and 1"),
    ],
)
def test_usage_errors_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    a = _run(tmp_path / "a.json", TOY / "run_a.txt")
    code, _, err = _gate(capsys, a, a, *extra)
    assert code == EXIT_USAGE
    assert message in err


def test_reports_on_different_qrels_are_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = tmp_path / "other_qrels.txt"
    other.write_text("q1 0 d1 1\nq2 0 d4 1\n", encoding="utf-8")
    a = _run(tmp_path / "a.json", TOY / "run_a.txt")
    b = _run(tmp_path / "b.json", TOY / "run_a.txt", qrels=other)
    code, _, err = _gate(capsys, a, b, *GATED)
    assert code == EXIT_INPUT
    assert "different qrels" in err


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (None, "No such file"),
        ("{not json", "invalid JSON"),
        ({**_report_dict({"q1": 0.5}), "schema_version": 2}, "unsupported schema_version 2"),
        ({"schema_version": 1}, "not a rageval report"),
        (_report_dict({}), "share no query"),  # an empty report
        (_report_dict({"other": 0.5}), "share no query"),
        (_report_dict({"q1": float("nan")}), "is nan, not a finite number"),
        (_report_dict({"q1": float("inf")}), "is inf, not a finite number"),
        (_report_dict({"q1": "0.5"}), "is '0.5', not a finite number"),
        (_report_dict({"q1": True}), "is True, not a finite number"),
        ({**_report_dict({"q1": 0.5}), "per_query": {"q1": {}}}, "is None, not a finite"),
    ],
)
def test_input_errors_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], candidate: Any, message: str
) -> None:
    base = _write(tmp_path / "base.json", _report_dict({"q1": 0.5}))
    cand = tmp_path / "cand.json"
    if isinstance(candidate, str):
        cand.write_text(candidate, encoding="utf-8")
    elif candidate is not None:
        _write(cand, candidate)
    code, _, err = _gate(capsys, base, cand, "--metric", "ndcg@10", "--max-drop", "0.02")
    assert code == EXIT_INPUT
    assert message in err


# ---------------------------------------------------------------- nearby inputs


@pytest.mark.parametrize(
    ("base", "cand", "passed"),
    [
        ({"q1": 0.7}, {"q1": 0.7}, True),  # one query: zero-width interval at the diff
        ({"q1": 0.7}, {"q1": 0.69}, True),  # one query, drop 0.01 < 0.02
        ({"q1": 0.7}, {"q1": 0.6}, False),  # one query, drop 0.1
        ({"q1": 0.0, "q2": 0.0}, {"q1": 0.0, "q2": 0.0}, True),  # all-zero metrics
        ({"q1": 1, "q2": 0}, {"q1": 1, "q2": 0}, True),  # integer values in the file
    ],
)
def test_degenerate_but_valid_inputs(
    base: dict[str, float], cand: dict[str, float], passed: bool
) -> None:
    a = Report.from_dict(_report_dict(base, "a"))
    b = Report.from_dict(_report_dict(cand, "b"))
    result = gate(a, b, {"ndcg@10": 0.02})
    assert result.passed is passed
    assert result.warnings and result.n_queries == len(base)


def test_queries_in_only_one_report_are_left_out() -> None:
    a = Report.from_dict(_report_dict({"q1": 0.5, "q2": 0.5, "q3": 0.9}))
    b = Report.from_dict(_report_dict({"q2": 0.5, "q1": 0.5, "q4": 0.0}))
    verdict = gate(a, b, {"ndcg@10": 0.02}).metrics["ndcg@10"]
    assert (verdict.n, verdict.baseline, verdict.candidate, verdict.ci) == (2, 0.5, 0.5, (0, 0))


def test_library_gate_rejects_an_empty_rule_set() -> None:
    a = Report.from_dict(_report_dict({"q1": 0.5}))
    with pytest.raises(ValueError, match="at least one metric"):
        gate(a, a, {})


def test_invariant_pass_bounds_the_mean_drop_and_identical_always_passes() -> None:
    # Observable invariant: a pass means the observed mean drop is at most max_drop plus
    # the interval's half-width, and a candidate identical to the baseline always passes.
    rng = random.Random(20261118)
    passes = 0
    for case in range(200):
        n = rng.randint(1, 60)
        shift, noise = rng.uniform(-0.15, 0.1), rng.uniform(0.0, 0.3)
        base = {f"q{i}": rng.random() for i in range(n)}
        cand = {q: min(1.0, max(0.0, v + shift + rng.gauss(0, noise))) for q, v in base.items()}
        max_drop = rng.uniform(0.005, 0.1)
        a = Report.from_dict(_report_dict(base))
        b = Report.from_dict(_report_dict(cand))
        same = gate(a, a, {"ndcg@10": max_drop}, resamples=200, seed=case)
        assert same.passed and same.metrics["ndcg@10"].ci == (0.0, 0.0)
        v = gate(a, b, {"ndcg@10": max_drop}, resamples=200, seed=case).metrics["ndcg@10"]
        if v.passed:
            passes += 1
            half_width = (v.ci[1] - v.ci[0]) / 2
            assert -v.diff <= max_drop + half_width + 1e-6, case
    assert 20 < passes < 180  # both verdicts occur, so the check is not vacuous


# ---------------------------------------------------------------- README


def test_readme_gate_output_matches_a_live_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = re.search(r"## Regression gate\n(.*?)\n## ", readme, re.S)
    assert section, "README needs a 'Regression gate' section"
    block = re.search(r"```text\n(.*?)```", section.group(1), re.S)
    assert block
    monkeypatch.chdir(tmp_path)
    for name in ("a", "b"):
        run_file = str(TOY / f"run_{name}.txt")
        argv = ["run", "--qrels", QRELS, "--run", run_file, "--metrics", "ndcg@10,recall@100,mrr"]
        assert main([*argv, "--out", f"report_{name}.json"]) == EXIT_OK
    code: int | None = None
    commands: list[str] = []
    for chunk in re.split(r"^\$ ", block.group(1), flags=re.M)[1:]:
        command, *expected = chunk.splitlines()
        if command == "echo $?":
            assert expected == [str(code)]
            continue
        capsys.readouterr()
        code = main(shlex.split(command)[1:])
        out, err = capsys.readouterr()
        assert err.splitlines() + out.splitlines() == expected, command
        commands.append(command)
    assert len(commands) == 2 and code == EXIT_GATE_FAILED
    # The pasted Markdown table is what --summary writes for the last (failing) command.
    assert main([*shlex.split(commands[-1])[1:], "--summary", "summary.md"]) == EXIT_GATE_FAILED
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8").splitlines()
    pasted = [line for line in section.group(1).splitlines() if line.startswith("| ")]
    assert len(pasted) == 3 and all(line in summary for line in pasted)


# ---------------------------------------------------------------- real entry point


def test_console_script_exits_10_on_a_failed_gate(tmp_path: Path) -> None:
    script = Path(sys.executable).with_name("rageval")
    assert script.exists(), "install the package first: pip install -e '.[dev]'"
    base = _run(tmp_path / "b.json", TOY / "run_b.txt")
    cand = _run(tmp_path / "a.json", TOY / "run_a.txt")
    argv = ["gate", "--baseline", str(base), "--candidate", str(cand), *GATED]
    done = subprocess.run(  # noqa: S603 - fixed program, test-controlled arguments
        [str(script), *argv], capture_output=True, text=True, check=False
    )
    assert done.returncode == EXIT_GATE_FAILED, done.stderr
    assert done.stdout.splitlines()[-1].startswith("gate: FAIL")
