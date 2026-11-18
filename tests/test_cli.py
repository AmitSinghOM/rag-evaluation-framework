"""`rageval run` / `rageval compare`: exit codes, output, ranx agreement, README commands."""

from __future__ import annotations

import importlib
import json
import re
import runpy
import shlex
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from rageval import __version__, evaluate, read_jsonl, read_qrels, read_run
from rageval.cli import EXIT_INPUT, EXIT_INTERNAL, EXIT_OK, EXIT_USAGE, build_parser, main
from rageval.report import FileRef

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy"
QRELS, RUN_A, RUN_B = (str(TOY / name) for name in ("qrels.txt", "run_a.txt", "run_b.txt"))
JSONL = str(TOY / "dataset.jsonl")  # the same data as qrels.txt + run_b.txt
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
        (["run", "--qrels", QRELS, "--metrics", "mrr", "--out", "o"], EXIT_USAGE, "--run R"),
        (["run", "--run", RUN_A, "--metrics", "mrr", "--out", "o"], EXIT_USAGE, "--run R"),
        (["run", "--metrics", "mrr", "--out", "o"], EXIT_USAGE, "or --dataset D"),
        (
            ["run", "--dataset", JSONL, "--qrels", QRELS, "--metrics", "mrr", "--out", "o"],
            EXIT_USAGE,
            "not both",
        ),
        (
            ["run", "--dataset", JSONL, "--run", RUN_A, "--metrics", "mrr", "--out", "o"],
            EXIT_USAGE,
            "not both",
        ),
        (["run", "--dataset", RUN_A, "--metrics", "mrr", "--out", "o"], EXIT_INPUT, "invalid JSON"),
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
    argv = ["run", "--dataset", str(empty), "--metrics", "mrr", "--out", "o.json"]
    assert main(argv) == EXIT_INPUT
    assert "dataset has no queries" in capsys.readouterr().err


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


def test_version_from_the_script_and_from_python_dash_m(tmp_path: Path) -> None:
    for argv in ([str(_console_script())], [sys.executable, "-m", "rageval"]):
        done = subprocess.run(  # noqa: S603 - fixed program, test-controlled arguments
            [*argv, "--version"], capture_output=True, text=True, check=False, cwd=tmp_path
        )
        assert (done.returncode, done.stdout) == (EXIT_OK, f"rageval {__version__}\n"), argv


def test_python_dash_m_runs_main_and_a_plain_import_does_not(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["rageval", "--version"])
    with pytest.raises(SystemExit) as exited:
        runpy.run_module("rageval", run_name="__main__")
    assert exited.value.code == EXIT_OK
    assert capsys.readouterr().out == f"rageval {__version__}\n"
    monkeypatch.delitem(sys.modules, "rageval.__main__", raising=False)
    importlib.import_module("rageval.__main__")
    assert capsys.readouterr().out == ""


def _script(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed program, test-controlled arguments
        [str(_console_script()), *argv], capture_output=True, text=True, check=False, cwd=ROOT
    )


def test_run_dataset_jsonl_gives_the_same_scores_as_the_trec_files(tmp_path: Path) -> None:
    trec, jsonl = tmp_path / "trec.json", tmp_path / "jsonl.json"
    assert _run(trec, RUN_B) == EXIT_OK
    done = _script("run", "--dataset", JSONL, "--metrics", METRICS, "--out", str(jsonl))
    assert done.returncode == EXIT_OK, done.stderr
    assert done.stdout.splitlines()[-1] == f"wrote {jsonl} (6 queries)"
    a = json.loads(trec.read_text(encoding="utf-8"))
    b = json.loads(jsonl.read_text(encoding="utf-8"))
    assert b["metrics"] == a["metrics"] and b["per_query"] == a["per_query"]
    assert b["dataset"]["qrels"] == {
        "path": JSONL,
        "sha256": FileRef.of_judgments(JSONL, read_jsonl(JSONL)[0]).sha256,
    }
    assert (b["run"]["name"], b["run"]["path"]) == ("dataset", JSONL)
    assert b["run"]["sha256"] == FileRef.of(JSONL).sha256
    assert b["warnings"] == []
    inproc = tmp_path / "inproc.json"  # the same run in-process, so coverage sees the path
    assert main(["run", "--dataset", JSONL, "--metrics", METRICS, "--out", str(inproc)]) == 0
    c = json.loads(inproc.read_text(encoding="utf-8"))
    assert {**c, "created_at": None} == {**b, "created_at": None}


def test_jsonl_reports_compare_when_the_judgments_match_and_only_then(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in Path(JSONL).read_text(encoding="utf-8").splitlines()]
    reranked = [{**row, "retrieved": row["retrieved"][::-1]} for row in rows]
    relabelled = [{**row, "relevant": ["d999"]} if i == 0 else row for i, row in enumerate(rows)]
    for name, data in (("reranked", reranked), ("relabelled", relabelled)):
        text = "".join(json.dumps(row) + "\n" for row in data)
        (tmp_path / f"{name}.jsonl").write_text(text, encoding="utf-8")
    for name in ("dataset", "reranked", "relabelled"):
        path = JSONL if name == "dataset" else str(tmp_path / f"{name}.jsonl")
        out = str(tmp_path / f"{name}.r")
        done = _script("run", "--dataset", path, "--metrics", "mrr", "--out", out)
        assert done.returncode == EXIT_OK, done.stderr
    same = _script("compare", f"{tmp_path}/dataset.r", f"{tmp_path}/reranked.r")
    assert same.returncode == EXIT_OK, same.stderr
    other = _script("compare", f"{tmp_path}/dataset.r", f"{tmp_path}/relabelled.r")
    assert (other.returncode, "different qrels" in other.stderr) == (EXIT_INPUT, True)
    assert _run(tmp_path / "trec.r", RUN_B) == EXIT_OK  # same judgments, TREC files
    mixed = _script("compare", f"{tmp_path}/trec.r", f"{tmp_path}/dataset.r")
    assert (mixed.returncode, "different qrels" in mixed.stderr) == (EXIT_INPUT, True)


def test_equivalent_jsonl_labels_hash_the_same(tmp_path: Path) -> None:
    listed, graded = tmp_path / "listed.jsonl", tmp_path / "graded.jsonl"
    row = '{{"query_id": "q1", "retrieved": {}, "relevant": {}}}\n'
    listed.write_text(row.format('["d1", "d4"]', '["d4"]'), encoding="utf-8")
    graded.write_text(row.format('["d4"]', '{"d4": 1}'), encoding="utf-8")
    hashes = {FileRef.of_judgments(p, read_jsonl(p)[0]).sha256 for p in (listed, graded)}
    assert len(hashes) == 1


def test_run_refuses_dataset_mixed_with_trec_files(tmp_path: Path) -> None:
    out = tmp_path / "never.json"
    done = _script(
        "run", "--dataset", JSONL, "--qrels", QRELS, "--metrics", "mrr", "--out", str(out)
    )
    assert done.returncode == EXIT_USAGE
    assert "give --dataset alone, or --qrels with --run, not both" in done.stderr
    assert not out.exists()


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
    section = re.search(r"## Quickstart(.*?)\n## ", readme, re.S)
    assert section, "README needs a Quickstart section"
    blocks = re.findall(r"```bash\n(.*?)```", section.group(1), re.S)
    commands = [
        line for block in blocks for line in block.splitlines() if line.startswith("rageval ")
    ]
    assert len(commands) >= 4, commands
    parser = build_parser()
    for command in commands:
        args = parser.parse_args(shlex.split(command)[1:])
        assert args.command in {"run", "compare"}
        if args.command == "run":
            inputs = [args.dataset] if args.dataset else [args.qrels, args.run]
            assert all((ROOT / path).exists() for path in inputs), command
    assert any("--dataset" in command for command in commands)
