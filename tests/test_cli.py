"""`rageval run` / `rageval compare`: exit codes, output, ranx agreement, README commands."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from rageval import evaluate, read_qrels, read_run
from rageval.cli import EXIT_INPUT, EXIT_INTERNAL, EXIT_OK, EXIT_USAGE, build_parser, main

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy"
QRELS, RUN_A, RUN_B = (str(TOY / name) for name in ("qrels.txt", "run_a.txt", "run_b.txt"))
METRICS = "ndcg@10,recall@100,mrr,precision@3"
TOLERANCE = 1e-9


def _run(out: Path, run_file: str, *extra: str) -> int:
    args = ["run", "--qrels", QRELS, "--run", run_file, "--metrics", METRICS, "--out", str(out)]
    return main([*args, *extra])


# ---------------------------------------------------------------- run


def test_run_writes_a_report_and_prints_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "a.json"
    assert _run(out, RUN_A, "--name", "bm25") == EXIT_OK
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].split() == ["metric", "mean", "ci95", "n"]
    assert re.fullmatch(r"ndcg@10\s+0\.4890\s+\[0\.\d{4}, 0\.\d{4}\]\s+6", lines[1])
    assert lines[-1] == f"wrote {out} (6 queries)"
    assert captured.err == "warning: query 'q99' is in the run but not in the qrels; ignored\n"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["run"]["name"] == "bm25"
    assert data["run"]["path"] == RUN_A
    assert data["metrics"]["recall@100"]["mean"] == 0.583333


def test_run_honours_resamples_seed_and_default_name(tmp_path: Path) -> None:
    out = tmp_path / "b.json"
    assert _run(out, RUN_B, "--resamples", "50", "--seed", "9") == EXIT_OK
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["run"]["name"] == "run_b"
    assert data["metrics"]["mrr"]["bootstrap"] == {"resamples": 50, "seed": 9}
    assert data["warnings"] == []


def test_same_seed_is_byte_identical_apart_from_created_at(tmp_path: Path) -> None:
    assert _run(tmp_path / "1.json", RUN_A) == EXIT_OK
    assert _run(tmp_path / "2.json", RUN_A) == EXIT_OK
    strip = re.compile(r'"created_at": "[^"]*"')
    first = strip.sub("", (tmp_path / "1.json").read_text(encoding="utf-8"))
    second = strip.sub("", (tmp_path / "2.json").read_text(encoding="utf-8"))
    assert first == second


@pytest.mark.parametrize(
    ("argv", "code", "message"),
    [
        ([], EXIT_USAGE, "required"),
        (["run", "--qrels", "q"], EXIT_USAGE, "required"),
        (["bogus"], EXIT_USAGE, "invalid choice"),
        (["compare", "a.json", "b.json", "--format", "xml"], EXIT_USAGE, "invalid choice"),
        (["run", "--qrels", QRELS, "--run", RUN_A, "--metrics", "x", "--out", "o"], 1, "unknown"),
        (["run", "--qrels", QRELS, "--run", RUN_A, "--metrics", ",", "--out", "o"], 1, "at least"),
        (
            [
                "run",
                "--qrels",
                QRELS,
                "--run",
                RUN_A,
                "--metrics",
                "mrr",
                "--out",
                "o",
                "--resamples",
                "0",
            ],
            EXIT_USAGE,
            ">= 1",
        ),
        (
            ["run", "--qrels", "missing.txt", "--run", RUN_A, "--metrics", "mrr", "--out", "o"],
            2,
            "No such",
        ),
        (
            ["run", "--qrels", RUN_A, "--run", RUN_A, "--metrics", "mrr", "--out", "o"],
            2,
            "expected 4",
        ),
        (["compare", "missing_a.json", "missing_b.json"], EXIT_INPUT, "No such"),
    ],
)
def test_exit_codes_for_usage_and_input_errors(
    capsys: pytest.CaptureFixture[str], argv: list[str], code: int, message: str
) -> None:
    assert main(argv) == code
    assert message in capsys.readouterr().err


def test_help_and_version_exit_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == EXIT_OK
    assert "{run,compare,gate}" in capsys.readouterr().out
    assert main(["--version"]) == EXIT_OK
    assert capsys.readouterr().out.startswith("rageval 0.1.0a0")


def test_empty_qrels_file_is_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("\n", encoding="utf-8")
    argv = ["run", "--qrels", str(empty), "--run", RUN_A, "--metrics", "mrr", "--out", "o.json"]
    assert main(argv) == EXIT_INPUT
    assert "no judgments" in capsys.readouterr().err


def test_undecodable_file_is_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    binary = tmp_path / "bin.txt"
    binary.write_bytes(b"q1 0 d1 1\n\xff\xfe\n")
    argv = ["run", "--qrels", str(binary), "--run", RUN_A, "--metrics", "mrr", "--out", "o.json"]
    assert main(argv) == EXIT_INPUT
    assert "rageval: error:" in capsys.readouterr().err


def test_unexpected_exception_is_exit_3_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("kaboom")

    monkeypatch.setattr("rageval.cli.evaluate", boom)
    assert _run(tmp_path / "x.json", RUN_A) == EXIT_INTERNAL
    err = capsys.readouterr().err
    assert "internal error: RuntimeError('kaboom')" in err
    assert "Traceback" not in err


# ---------------------------------------------------------------- compare


def test_compare_table_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert _run(a, RUN_A) == EXIT_OK
    assert _run(b, RUN_B) == EXIT_OK
    capsys.readouterr()
    assert main(["compare", str(a), str(b)]) == EXIT_OK
    table = capsys.readouterr().out.splitlines()
    assert table[0].split()[:3] == ["metric", "run_a", "run_b"]
    assert table[2].split()[:4] == ["recall@100", "0.583333", "1.000000", "+0.416667"]
    assert table[-1] == "n_queries: 6"
    assert main(["compare", str(a), str(b), "--format", "json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert data["metrics"]["mrr"]["diff"] == 0.375
    assert data["a"]["name"] == "run_a" and data["b"]["name"] == "run_b"


def test_compare_refuses_reports_on_different_qrels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    other = tmp_path / "other_qrels.txt"
    other.write_text("q1 0 d1 1\nq2 0 d4 1\n", encoding="utf-8")
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert _run(a, RUN_A) == EXIT_OK
    argv = ["run", "--qrels", str(other), "--run", RUN_A, "--metrics", METRICS, "--out", str(b)]
    assert main(argv) == EXIT_OK
    capsys.readouterr()
    assert main(["compare", str(a), str(b)]) == EXIT_INPUT
    assert "different qrels" in capsys.readouterr().err


# ---------------------------------------------------------------- real entry point


def _console_script() -> Path:
    script = Path(sys.executable).with_name("rageval")
    assert script.exists(), "install the package first: pip install -e '.[dev]'"
    return script


def test_console_script_runs_and_reports_exit_codes(tmp_path: Path) -> None:
    script = _console_script()
    out = tmp_path / "a.json"
    argv = ["run", "--qrels", QRELS, "--run", RUN_A, "--metrics", METRICS, "--out", str(out)]
    done = subprocess.run(  # noqa: S603 - fixed program, test-controlled arguments
        [str(script), *argv], capture_output=True, text=True, check=False, cwd=ROOT
    )
    assert done.returncode == EXIT_OK, done.stderr
    assert done.stdout.splitlines()[0].split() == ["metric", "mean", "ci95", "n"]
    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"] == 1
    bad = subprocess.run(  # noqa: S603
        [str(script), "run", "--qrels", QRELS], capture_output=True, text=True, check=False
    )
    assert bad.returncode == EXIT_USAGE
    missing = subprocess.run(  # noqa: S603
        [str(script), "compare", "nope.json", str(out)], capture_output=True, text=True, check=False
    )
    assert missing.returncode == EXIT_INPUT


# ---------------------------------------------------------------- ranx agreement


@pytest.mark.parametrize("run_file", ["run_a.txt", "run_b.txt"])
def test_run_means_agree_with_ranx_on_the_same_files(tmp_path: Path, run_file: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ranx import Qrels, Run
        from ranx import evaluate as ranx_evaluate

    out = tmp_path / "r.json"
    assert _run(out, str(TOY / run_file)) == EXIT_OK
    written = json.loads(out.read_text(encoding="utf-8"))
    metrics = METRICS.split(",")
    # The file holds 6-place values; the invariant is on the computed means, so recompute
    # them through the library on the same inputs and tie the file to them by rounding.
    report = evaluate(read_qrels(TOY / "qrels.txt"), read_run(TOY / run_file), metrics)
    qrels = Qrels.from_file(QRELS, kind="trec")
    run = Run.from_file(str(TOY / run_file), kind="trec")
    theirs = ranx_evaluate(qrels, run, metrics, make_comparable=True)
    for name in metrics:
        ours = report.metrics[name].mean
        assert abs(ours - float(theirs[name])) <= TOLERANCE, name
        assert written["metrics"][name]["mean"] == round(ours, 6)
    for name, per_query in run.scores.items():
        for qid, value in per_query.items():
            assert abs(report.per_query[qid][name] - float(value)) <= TOLERANCE, (name, qid)


# ---------------------------------------------------------------- README


def test_readme_quickstart_commands_still_parse() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(r"## Quickstart.*?```bash\n(.*?)```", readme, re.S)
    assert block, "README needs a Quickstart section with a bash block"
    commands = [line for line in block.group(1).splitlines() if line.startswith("rageval ")]
    assert len(commands) >= 3, commands
    parser = build_parser()
    for command in commands:
        args = parser.parse_args(shlex.split(command)[1:])
        assert args.command in {"run", "compare"}
        if args.command == "run":
            assert (ROOT / args.qrels).exists() and (ROOT / args.run).exists(), command
