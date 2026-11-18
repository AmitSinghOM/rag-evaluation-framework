"""EXPERIMENTAL grounding check: lexical rule, judges at the urllib boundary, pack, calibrate.

No network and no Ollama: every HTTP call is replaced at ``rageval.grounding.urlopen``.
"""

from __future__ import annotations

import io
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from rageval import grounding
from rageval.cli import EXIT_INPUT, EXIT_OK, EXIT_USAGE, main
from rageval.grounding import (
    ConfigError,
    JudgeError,
    OllamaJudge,
    OpenAICompatibleJudge,
    PackItem,
    calibrate,
    fake_generator,
    lexical_support,
    parse_verdict,
    prepare_pack,
    read_key,
    read_pack,
    read_sheet_labels,
    score_pack,
)
from rageval.loaders import LoaderError

ROOT = Path(__file__).resolve().parents[1]
WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron".split()

# ------------------------------------------------------------------ lexical rule


def test_sentence_copied_from_a_passage_scores_one() -> None:
    passage = "Aspirin lowers the risk of a second heart attack in adults."
    result = lexical_support("Aspirin lowers the risk of a second heart attack.", [passage])
    assert result.score == 1.0
    assert result.n_sentences == 1
    assert result.sentences[0].best_passage == 0


def test_half_the_content_tokens_supported_scores_half() -> None:
    # content tokens: aspirin, lowers, risk, stroke  (the, of are stopwords)
    result = lexical_support("Aspirin lowers the risk of stroke.", ["aspirin risk"])
    assert result.score == 0.5
    assert result.sentences[0].n_tokens == 4


def test_unsupported_sentence_scores_zero() -> None:
    assert lexical_support("Mars has quantum tides.", ["Aspirin lowers risk."]).score == 0.0


def test_empty_answer_scores_zero_with_no_sentences() -> None:
    result = lexical_support("", ["a passage"])
    assert result.score == 0.0
    assert result.sentences == []


def test_no_passages_scores_zero_per_sentence() -> None:
    result = lexical_support("Aspirin lowers risk. It is cheap.", [])
    assert result.score == 0.0
    assert [s.best_passage for s in result.sentences] == [None, None]


def test_support_must_come_from_one_single_passage() -> None:
    # Tokens split across two passages do not add up: the best single passage gives 0.5.
    assert lexical_support("aspirin stroke", ["aspirin", "stroke"]).score == 0.5


def test_aggregate_is_the_mean_and_stopword_only_sentences_are_skipped() -> None:
    result = lexical_support("Aspirin lowers risk. Mars has tides. It is.", ["aspirin lowers risk"])
    assert [s.score for s in result.sentences] == [1.0, 0.0]
    assert result.score == 0.5


def test_lexical_is_deterministic() -> None:
    args = ("Aspirin lowers the risk of stroke in adults.", ["Adults: aspirin lowers stroke risk"])
    assert lexical_support(*args) == lexical_support(*args)


# ----------------------------------------------------------------------- judges


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _http_error(request: Any, code: int, body: bytes = b"") -> HTTPError:
    return HTTPError(request.full_url, code, "error", None, io.BytesIO(body))  # type: ignore[arg-type]


def _patch_urlopen(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[Any]:
    calls: list[Any] = []

    def fake_urlopen(request: Any, timeout: float = 0.0) -> Any:
        calls.append((request, timeout))
        return handler(request)

    monkeypatch.setattr(grounding, "urlopen", fake_urlopen)
    return calls


def test_ollama_judge_success_uses_json_mode_and_parses_the_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = {"message": {"content": '{"grounded": true, "confidence": 0.9}'}}
    calls = _patch_urlopen(monkeypatch, lambda req: _Response(json.dumps(reply).encode()))
    verdict = OllamaJudge("qwen2.5:3b").judge("q", "a", ["p1", "p2"])
    assert (verdict.grounded, verdict.confidence) == (True, 0.9)
    assert verdict.raw == reply["message"]["content"]
    request, timeout = calls[0]
    assert request.full_url == "http://localhost:11434/api/chat"
    assert timeout == 60.0
    body = json.loads(request.data)
    assert (body["model"], body["format"], body["stream"]) == ("qwen2.5:3b", "json", False)
    assert "[2] p2" in body["messages"][1]["content"]


def test_ollama_judge_non_json_reply_is_a_judge_error(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = {"message": {"content": "Yes, it is grounded."}}
    _patch_urlopen(monkeypatch, lambda req: _Response(json.dumps(reply).encode()))
    with pytest.raises(JudgeError, match="not JSON"):
        OllamaJudge("m").judge("q", "a", ["p"])


def test_ollama_judge_connection_refused_names_ollama_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(request: Any) -> Any:
        raise URLError(ConnectionRefusedError(61, "Connection refused"))

    _patch_urlopen(monkeypatch, refuse)
    with pytest.raises(JudgeError, match="ollama serve"):
        OllamaJudge("m").judge("q", "a", ["p"])


def test_ollama_judge_missing_model_names_ollama_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    def not_found(request: Any) -> Any:
        raise _http_error(request, 404, b'{"error":"model not found"}')

    _patch_urlopen(monkeypatch, not_found)
    with pytest.raises(JudgeError, match="ollama pull qwen2.5:3b"):
        OllamaJudge("qwen2.5:3b").judge("q", "a", ["p"])


def test_ollama_judge_other_http_error_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, lambda req: (_ for _ in ()).throw(_http_error(req, 500, b"boom")))
    with pytest.raises(JudgeError, match="HTTP 500: boom"):
        OllamaJudge("m").judge("q", "a", ["p"])

    def slow(request: Any) -> Any:
        raise TimeoutError()

    _patch_urlopen(monkeypatch, slow)
    with pytest.raises(JudgeError, match="did not answer within 60 s"):
        OllamaJudge("m").judge("q", "a", ["p"])


def test_ollama_reply_shape_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_urlopen(monkeypatch, lambda req: _Response(b"not json"))
    with pytest.raises(JudgeError, match="non-JSON"):
        OllamaJudge("m").judge("q", "a", ["p"])
    _patch_urlopen(monkeypatch, lambda req: _Response(b"[1]"))
    with pytest.raises(JudgeError, match="not an object"):
        OllamaJudge("m").judge("q", "a", ["p"])
    _patch_urlopen(monkeypatch, lambda req: _Response(b'{"message": {}}'))
    with pytest.raises(JudgeError, match="message.content"):
        OllamaJudge("m").judge("q", "a", ["p"])


def test_non_http_base_url_is_refused() -> None:
    with pytest.raises(ConfigError, match="http"):
        OllamaJudge("m", base_url="file:///etc").judge("q", "a", ["p"])


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"grounded": "yes"}', "boolean"),
        ("[1, 2]", "boolean"),
        ('{"grounded": true, "confidence": "high"}', "not a number"),
        ('{"grounded": true, "confidence": true}', "not a number"),
    ],
)
def test_parse_verdict_rejects_bad_shapes(raw: str, message: str) -> None:
    with pytest.raises(JudgeError, match=message):
        parse_verdict(raw)


def test_parse_verdict_clamps_confidence_and_allows_none() -> None:
    assert parse_verdict('{"grounded": false, "confidence": 7}').confidence == 1.0
    assert parse_verdict('{"grounded": false}').confidence is None


def test_openai_compatible_judge_missing_key_names_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAGEVAL_JUDGE_API_KEY", raising=False)
    judge = OpenAICompatibleJudge(base_url="https://example.invalid/v1", model="m")
    with pytest.raises(ConfigError, match="RAGEVAL_JUDGE_API_KEY"):
        judge.judge("q", "a", ["p"])


def test_openai_compatible_judge_reads_key_from_env_only_and_never_echoes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_JUDGE_KEY", "sk-test-not-a-real-key")
    reply = {"choices": [{"message": {"content": '{"grounded": false, "confidence": 0.8}'}}]}
    calls = _patch_urlopen(monkeypatch, lambda req: _Response(json.dumps(reply).encode()))
    judge = OpenAICompatibleJudge(
        base_url="https://example.invalid/v1/", model="m", api_key_env="MY_JUDGE_KEY"
    )
    verdict = judge.judge("q", "a", ["p"])
    assert (verdict.grounded, verdict.confidence) == (False, 0.8)
    request, _ = calls[0]
    assert request.full_url == "https://example.invalid/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer sk-test-not-a-real-key"
    body = json.loads(request.data)
    assert body["response_format"] == {"type": "json_object"}

    _patch_urlopen(monkeypatch, lambda req: (_ for _ in ()).throw(_http_error(req, 401)))
    with pytest.raises(JudgeError) as info:
        judge.judge("q", "a", ["p"])
    assert "sk-test" not in str(info.value)
    assert "MY_JUDGE_KEY" in str(info.value)


def test_openai_compatible_judge_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGEVAL_JUDGE_API_KEY", "k")
    judge = OpenAICompatibleJudge(base_url="https://example.invalid", model="m")
    _patch_urlopen(monkeypatch, lambda req: _Response(b'{"choices": []}'))
    with pytest.raises(JudgeError, match="choices"):
        judge.judge("q", "a", [])
    _patch_urlopen(monkeypatch, lambda req: (_ for _ in ()).throw(_http_error(req, 429)))
    with pytest.raises(JudgeError, match="HTTP 429"):
        judge.judge("q", "a", [])
    _patch_urlopen(monkeypatch, lambda req: (_ for _ in ()).throw(URLError("no route")))
    with pytest.raises(JudgeError, match="cannot reach"):
        judge.judge("q", "a", [])
    _patch_urlopen(monkeypatch, lambda req: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(JudgeError, match="did not answer"):
        judge.judge("q", "a", [])


# ------------------------------------------------------------------------- pack


def write_synthetic_dataset(dest: Path, n_queries: int = 20) -> dict[str, Path]:
    """20 docs ``d<i>`` with unique token ``uniq<i>``; query ``q<i>`` retrieves d<i> first."""
    rng = random.Random(0)
    dest.mkdir(parents=True, exist_ok=True)
    paths = {name: dest / name for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv", "run")}
    with paths["corpus.jsonl"].open("w", encoding="utf-8") as fh:
        for i in range(n_queries):
            text = f"Uniq{i} is studied here. " + " ".join(rng.choices(WORDS, k=12)) + "."
            fh.write(json.dumps({"_id": f"d{i}", "title": f"Paper {i}", "text": text}) + "\n")
    with paths["queries.jsonl"].open("w", encoding="utf-8") as fh:
        for i in range(n_queries):
            query = {"_id": f"q{i}", "text": f"uniq{i} {WORDS[i % len(WORDS)]}"}
            fh.write(json.dumps(query) + "\n")
        fh.write(json.dumps({"_id": "q-train", "text": "not judged"}) + "\n\n")  # blank ok
    with paths["qrels.tsv"].open("w", encoding="utf-8") as fh:
        for i in range(n_queries):
            fh.write(f"q{i} 0 d{i} 1\n")
    with paths["run"].open("w", encoding="utf-8") as fh:
        for i in range(n_queries):
            for rank, j in enumerate((i, (i + 1) % n_queries, (i + 2) % n_queries), start=1):
                fh.write(f"q{i} Q0 d{j} {rank} {10 - rank}.0 synth\n")
        fh.write("q-train Q0 d0 1 1.0 synth\n")
    return paths


def _prepare(tmp_path: Path, *extra: str, n: int = 20) -> tuple[int, Path, Path, Path]:
    paths = write_synthetic_dataset(tmp_path / "data")
    out = tmp_path / "pack" / "pack.jsonl"
    out.parent.mkdir(exist_ok=True)
    sheet = tmp_path / "pack" / "pack.md"
    key = tmp_path / "pack" / "key" / "pack.key.jsonl"
    args = ["ground", "prepare", "--qrels", str(paths["qrels.tsv"]), "--run", str(paths["run"])]
    args += ["--corpus", str(paths["corpus.jsonl"]), "--queries", str(paths["queries.jsonl"])]
    args += ["--n", str(n), "--seed", "0", "--generator", "fake", "--out", str(out)]
    args += ["--sheet", str(sheet), "--key", str(key), *extra]
    return main(args), out, sheet, key


def test_prepare_pack_invariants(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, sheet, key = _prepare(tmp_path)
    assert code == EXIT_OK
    items = read_pack(out)
    assert len(items) == 20
    assert all(item.label is None and item.notes == "" for item in items)
    assert len({item.item_id for item in items}) == 20
    assert all(len(item.passages) == 3 for item in items)
    conditions = read_key(key)
    assert set(conditions) == {item.item_id for item in items}
    assert sorted(conditions.values()).count("faithful") == 10
    assert sorted(conditions.values()).count("unfaithful") == 10
    key_text = key.read_text(encoding="utf-8")
    for item in items:
        for passage in item.passages:
            assert passage not in key_text
        assert item.answer not in key_text
        assert item.question not in key_text
    text = sheet.read_text(encoding="utf-8")
    assert text.count("## item-") == 20
    assert text.count(grounding.LABEL_LINE) == 20
    assert grounding.LABEL_RULE in text
    readme = (out.parent / "README.md").read_text(encoding="utf-8")
    assert "rageval ground calibrate" in readme
    assert "20 items" in readme
    assert "labels empty" in capsys.readouterr().out


def test_prepare_is_seeded_and_the_seed_changes_the_sample(tmp_path: Path) -> None:
    code_a, out_a, _, key_a = _prepare(tmp_path / "a")
    code_b, out_b, _, key_b = _prepare(tmp_path / "b")
    code_c, out_c, _, _ = _prepare(tmp_path / "c", "--seed", "7")
    assert (code_a, code_b, code_c) == (EXIT_OK, EXIT_OK, EXIT_OK)
    assert out_a.read_bytes() == out_b.read_bytes()
    assert key_a.read_bytes() == key_b.read_bytes()
    assert out_a.read_bytes() != out_c.read_bytes()


def test_prepare_fake_generator_separates_the_conditions_lexically(tmp_path: Path) -> None:
    code, out, _, key = _prepare(tmp_path)
    assert code == EXIT_OK
    conditions = read_key(key)
    scores = score_pack(read_pack(out), None)["scores"]
    faithful = [scores[i]["score"] for i, c in conditions.items() if c == "faithful"]
    unfaithful = [scores[i]["score"] for i, c in conditions.items() if c == "unfaithful"]
    assert min(faithful) == 1.0
    assert max(unfaithful) < 0.5


def test_prepare_default_key_sits_next_to_the_pack(tmp_path: Path) -> None:
    paths = write_synthetic_dataset(tmp_path / "data")
    out = tmp_path / "pack.jsonl"
    args = ["ground", "prepare", "--qrels", str(paths["qrels.tsv"]), "--run", str(paths["run"])]
    args += ["--corpus", str(paths["corpus.jsonl"]), "--queries", str(paths["queries.jsonl"])]
    assert main([*args, "--n", "4", "--out", str(out)]) == EXIT_OK
    assert (tmp_path / "pack.key.jsonl").is_file()
    assert not (tmp_path / "pack.md").exists()
    items = read_pack(out)
    assert all(item.question.startswith("uniq") for item in items)
    assert all(p.startswith("Paper ") for item in items for p in item.passages)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--n", "7"), "even"),
        (("--n", "22"), "exceeds"),
        (("--generator", "gpt"), "ollama:MODEL"),
        (("--generator", "ollama:"), "ollama:MODEL"),
        (("--top", "0"), "--top"),
    ],
)
def test_prepare_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: tuple[str, ...], message: str
) -> None:
    code, _, _, _ = _prepare(tmp_path, *extra)
    assert code == EXIT_USAGE
    assert message in capsys.readouterr().err


def test_prepare_reports_missing_docs_and_bad_queries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = write_synthetic_dataset(tmp_path / "data")
    paths["corpus.jsonl"].write_text('{"_id": "d0", "text": "only one"}\n', encoding="utf-8")
    args = ["ground", "prepare", "--qrels", str(paths["qrels.tsv"]), "--run", str(paths["run"])]
    args += ["--corpus", str(paths["corpus.jsonl"]), "--queries", str(paths["queries.jsonl"])]
    args += ["--n", "4", "--out", str(tmp_path / "p.jsonl")]
    assert main(args) == EXIT_INPUT
    assert "doc ids missing" in capsys.readouterr().err
    paths["queries.jsonl"].write_text('{"_id": "q0", "text": "only q0"}\n', encoding="utf-8")
    assert main(args) == EXIT_INPUT
    assert "query ids missing" in capsys.readouterr().err
    paths["queries.jsonl"].write_text("{bad json\n", encoding="utf-8")
    assert main(args) == EXIT_INPUT
    assert "queries.jsonl:1" in capsys.readouterr().err


def test_prepare_skips_queries_whose_top_passages_hold_no_relevant_doc(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = write_synthetic_dataset(tmp_path / "data")
    lines = paths["run"].read_text(encoding="utf-8").splitlines()
    # q0 now retrieves d1, d2, d3: its relevant d0 is not in the top 3, so q0 is ineligible.
    rewritten = [line.replace("q0 Q0 d0 1", "q0 Q0 d3 1") for line in lines]
    paths["run"].write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    args = ["ground", "prepare", "--qrels", str(paths["qrels.tsv"]), "--run", str(paths["run"])]
    args += ["--corpus", str(paths["corpus.jsonl"]), "--queries", str(paths["queries.jsonl"])]
    args += ["--out", str(tmp_path / "p.jsonl")]
    assert main([*args, "--n", "20"]) == EXIT_USAGE
    assert "19 judged queries whose top-3" in capsys.readouterr().err
    assert main([*args, "--n", "18"]) == EXIT_OK
    assert all(item.query_id != "q0" for item in read_pack(tmp_path / "p.jsonl"))


def test_fake_generator_conditions() -> None:
    assert fake_generator("q", ["First claim. Second."], "faithful") == "First claim."
    assert "quantum" in fake_generator("q", ["First claim."], "unfaithful")


def test_ollama_generator_prompts_differ_by_condition(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = {"message": {"content": "  An answer.  "}}
    calls = _patch_urlopen(monkeypatch, lambda req: _Response(json.dumps(reply).encode()))
    generate = grounding.make_generator("ollama:m")
    assert generate("Q?", ["P1"], "faithful") == "An answer."
    assert generate("Q?", [], "unfaithful") == "An answer."
    first, second = (json.loads(call[0].data) for call in calls)
    assert "[1] P1" in first["messages"][0]["content"]
    assert "ONLY facts stated in the passages" in first["messages"][0]["content"]
    assert "own knowledge" in second["messages"][0]["content"]
    assert "format" not in second


def test_prepare_pack_requires_even_n_in_the_library_too(tmp_path: Path) -> None:
    paths = write_synthetic_dataset(tmp_path / "data")
    with pytest.raises(ValueError, match="even"):
        prepare_pack(
            {"q0": {"d0": 1.0}},
            {"q0": ["d0"]},
            paths["corpus.jsonl"],
            paths["queries.jsonl"],
            n=1,
            seed=0,
            generator=fake_generator,
            generator_name="fake",
        )


def test_read_pack_rejects_bad_rows(tmp_path: Path) -> None:
    pack = tmp_path / "p.jsonl"
    row = {"item_id": "i1", "query_id": "q", "question": "?", "passages": [], "answer": "a"}
    pack.write_text("\n" + json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="repeated item_id"):
        read_pack(pack)
    pack.write_text(json.dumps({**row, "label": "maybe"}) + "\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="label must be one of"):
        read_pack(pack)
    pack.write_text("{}\n\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="not a pack item"):
        read_pack(pack)


def test_read_key_rejects_bad_rows(tmp_path: Path) -> None:
    key = tmp_path / "k.jsonl"
    key.write_text('{"item_id": "i1", "condition": "sideways"}\n', encoding="utf-8")
    with pytest.raises(LoaderError, match="condition must be"):
        read_key(key)
    key.write_text("nope\n\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="not a key record"):
        read_key(key)


# ------------------------------------------------------------------------ score


def test_score_lexical_is_byte_identical_across_runs(tmp_path: Path) -> None:
    code, out, _, _ = _prepare(tmp_path)
    assert code == EXIT_OK
    first, second = tmp_path / "s1.json", tmp_path / "s2.json"
    for path in (first, second):
        assert main(["ground", "score", "--pack", str(out), "--out", str(path)]) == EXIT_OK
    assert first.read_bytes() == second.read_bytes()
    data = json.loads(first.read_text(encoding="utf-8"))
    assert (data["schema_version"], data["status"]) == (1, "experimental")
    assert data["method"] == "lexical"
    assert len(data["scores"]) == 20
    assert all(s["grounded"] is None for s in data["scores"].values())


def test_score_with_a_mocked_ollama_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, out, _, _ = _prepare(tmp_path, n=4)
    assert code == EXIT_OK
    replies = iter(
        [
            '{"grounded": true, "confidence": 0.9}',
            '{"grounded": false, "confidence": 0.75}',
            '{"grounded": true}',
            '{"grounded": false}',
        ]
    )
    _patch_urlopen(
        monkeypatch,
        lambda req: _Response(json.dumps({"message": {"content": next(replies)}}).encode()),
    )
    scores = tmp_path / "scores.json"
    args = ["ground", "score", "--pack", str(out), "--method", "ollama:m", "--out", str(scores)]
    assert main(args) == EXIT_OK
    data = json.loads(scores.read_text(encoding="utf-8"))
    assert data["method"] == "ollama:m"
    values = [(s["score"], s["grounded"]) for s in data["scores"].values()]
    assert values == [(0.9, True), (0.25, False), (1.0, True), (0.0, False)]


def test_score_with_a_mocked_openai_compatible_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, out, _, _ = _prepare(tmp_path, n=2)
    assert code == EXIT_OK
    monkeypatch.setenv("RAGEVAL_JUDGE_API_KEY", "k")
    reply = {"choices": [{"message": {"content": '{"grounded": true, "confidence": 0.6}'}}]}
    _patch_urlopen(monkeypatch, lambda req: _Response(json.dumps(reply).encode()))
    scores = tmp_path / "scores.json"
    args = ["ground", "score", "--pack", str(out), "--method", "openai-compatible"]
    args += ["--base-url", "https://example.invalid/v1", "--model", "m", "--out", str(scores)]
    assert main(args) == EXIT_OK
    assert json.loads(scores.read_text(encoding="utf-8"))["method"] == "openai-compatible:m"


def test_score_judge_errors_exit_2_with_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _, _ = _prepare(tmp_path, n=2)
    assert code == EXIT_OK

    def refuse(request: Any) -> Any:
        raise URLError(ConnectionRefusedError(61, "refused"))

    _patch_urlopen(monkeypatch, refuse)
    args = ["ground", "score", "--pack", str(out), "--method", "ollama:m", "--out", "x.json"]
    assert main(args) == EXIT_INPUT
    assert "ollama serve" in capsys.readouterr().err
    monkeypatch.delenv("RAGEVAL_JUDGE_API_KEY", raising=False)
    args = ["ground", "score", "--pack", str(out), "--method", "openai-compatible"]
    args += ["--base-url", "https://example.invalid", "--model", "m", "--out", "x.json"]
    assert main(args) == EXIT_USAGE
    assert "RAGEVAL_JUDGE_API_KEY" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("method", "message"),
    [
        ("bert", "--method must be"),
        ("openai-compatible", "--base-url and --model"),
        ("ollama:", "--method must be"),
    ],
)
def test_score_method_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], method: str, message: str
) -> None:
    code, out, _, _ = _prepare(tmp_path, n=2)
    assert code == EXIT_OK
    args = ["ground", "score", "--pack", str(out), "--method", method, "--out", "x"]
    assert main(args) == EXIT_USAGE
    assert message in capsys.readouterr().err


def test_score_empty_pack_is_an_input_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    assert main(["ground", "score", "--pack", str(empty), "--out", "x"]) == EXIT_INPUT


# -------------------------------------------------------------------- calibrate


def _labelled_pack(
    tmp_path: Path, n_grounded: int, n_not: int, n_unsure: int = 0, n_blank: int = 0
) -> tuple[Path, Path, Path]:
    """Scores: grounded in [0.6, 1.0], not_grounded in [0.0, 0.4]: best F1 is 1 at ~0.6."""
    rng = random.Random(1)
    items: list[dict[str, Any]] = []
    scores: dict[str, Any] = {}
    key_rows: list[dict[str, str]] = []
    plan = (
        [("grounded", "faithful")] * n_grounded
        + [("not_grounded", "unfaithful")] * n_not
        + [("unsure", "faithful")] * n_unsure
        + [(None, "unfaithful")] * n_blank
    )
    for index, (label, condition) in enumerate(plan, start=1):
        item_id = f"item-{index:03d}"
        row: dict[str, Any] = {"item_id": item_id, "query_id": f"q{index}", "question": "?"}
        items.append({**row, "passages": ["p"], "answer": "a", "label": label, "notes": ""})
        low, high = (0.6, 1.0) if label == "grounded" else (0.0, 0.4)
        score = round(rng.uniform(low, high), 6)
        scores[item_id] = {"score": score, "grounded": None, "detail": []}
        key_rows.append({"item_id": item_id, "condition": condition, "generator": "fake"})
    pack = tmp_path / "pack.jsonl"
    pack.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    scores_path = tmp_path / "scores.json"
    scores_doc = {"schema_version": 1, "status": "experimental", "method": "lexical"}
    scores_path.write_text(json.dumps({**scores_doc, "scores": scores}), encoding="utf-8")
    key = tmp_path / "pack.key.jsonl"
    key.write_text("".join(json.dumps(r) + "\n" for r in key_rows), encoding="utf-8")
    return pack, scores_path, key


def test_calibrate_finds_the_known_threshold_and_reads_the_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack, scores, _ = _labelled_pack(tmp_path, 30, 30, n_unsure=3, n_blank=2)
    out = tmp_path / "calibration.json"
    args = ["ground", "calibrate", "--pack", str(pack), "--scores", str(scores), "--out", str(out)]
    assert main(args) == EXIT_OK
    report = json.loads(out.read_text(encoding="utf-8"))
    assert (report["schema_version"], report["status"]) == (1, "experimental")
    assert report["method"] == "lexical"
    counts = (report["n_items"], report["n_labelled"], report["n_unsure"], report["n_unlabelled"])
    assert counts == (65, 60, 3, 2)
    assert report["n_per_class"] == {"grounded": 30, "not_grounded": 30}
    best = report["best"]
    grounded_scores = [
        s["score"]
        for item_id, s in json.loads(scores.read_text(encoding="utf-8"))["scores"].items()
        if int(item_id.split("-")[1]) <= 30
    ]
    assert best["threshold"] == min(grounded_scores)  # lowest threshold with the best F1
    assert (best["accuracy"], best["precision"], best["recall"], best["f1"]) == (1.0, 1.0, 1.0, 1.0)
    assert best["confusion"] == {"tp": 30, "fp": 0, "fn": 0, "tn": 30}
    assert best["accuracy_ci95"] == [1.0, 1.0]
    hidden = report["hidden_condition"]
    assert hidden["n_with_key"] == 60
    assert hidden["label_agrees_with_condition"] == 1.0
    assert hidden["by_condition"]["faithful"] == {"grounded": 30, "not_grounded": 0}
    assert report["warnings"] == []
    thresholds = [row["threshold"] for row in report["sweep"]]
    assert thresholds == sorted(thresholds) and len(thresholds) == 61
    captured = capsys.readouterr()
    assert "best threshold" in captured.out and "hidden condition on 1.0000" in captured.out


def test_calibrate_refuses_under_30_labelled_items_with_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack, scores, _ = _labelled_pack(tmp_path, 15, 14, n_unsure=10)
    out = tmp_path / "c.json"
    args = ["ground", "calibrate", "--pack", str(pack), "--scores", str(scores), "--out", str(out)]
    assert main(args) == EXIT_INPUT
    assert "only 29 labelled items" in capsys.readouterr().err
    assert not out.exists()


def test_calibrate_warns_under_50_and_ignores_unsure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pack, scores, key = _labelled_pack(tmp_path, 20, 15, n_unsure=5)
    key.unlink()  # no key anywhere: hidden_condition is null
    out = tmp_path / "c.json"
    args = ["ground", "calibrate", "--pack", str(pack), "--scores", str(scores), "--out", str(out)]
    assert main(args) == EXIT_OK
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n_labelled"] == 35 and report["n_unsure"] == 5
    assert report["hidden_condition"] is None
    assert "only 35 labelled items (< 50)" in capsys.readouterr().err


def test_calibrate_finds_the_key_in_a_key_subfolder_or_takes_it_explicitly(tmp_path: Path) -> None:
    pack, scores, key = _labelled_pack(tmp_path, 20, 20)
    (tmp_path / "key").mkdir()
    key.rename(tmp_path / "key" / "pack.key.jsonl")
    out = tmp_path / "c.json"
    args = ["ground", "calibrate", "--pack", str(pack), "--scores", str(scores), "--out", str(out)]
    assert main(args) == EXIT_OK
    assert json.loads(out.read_text(encoding="utf-8"))["hidden_condition"]["n_with_key"] == 40
    explicit = tmp_path / "elsewhere.key.jsonl"
    rows = (tmp_path / "key" / "pack.key.jsonl").read_text(encoding="utf-8").splitlines()
    explicit.write_text("\n".join(rows[:10]) + "\n\n", encoding="utf-8")  # trailing blank line
    assert main([*args, "--key", str(explicit)]) == EXIT_OK
    assert json.loads(out.read_text(encoding="utf-8"))["hidden_condition"]["n_with_key"] == 10


def test_calibrate_takes_labels_from_the_sheet(tmp_path: Path) -> None:
    pack, scores, _ = _labelled_pack(tmp_path, 20, 20)
    items = read_pack(pack)
    unlabelled = tmp_path / "blank.jsonl"
    unlabelled.write_text(
        "".join(json.dumps({**i.to_dict(), "label": None}) + "\n" for i in items), encoding="utf-8"
    )
    sheet_lines = ["# sheet", ""]
    for item in items:
        value = item.label if item.item_id != "item-001" else "grounded | not_grounded | unsure"
        sheet_lines += [f"## {item.item_id} (query_id x)", "", f"label: {value}", "notes: fine", ""]
    sheet_lines += ["## item-040 (query_id x)", "", "label:", ""]  # empty = undecided
    sheet = tmp_path / "pack.md"
    sheet.write_text("\n".join(sheet_lines), encoding="utf-8")
    labels = read_sheet_labels(sheet)
    assert labels["item-002"] == "grounded"
    assert "item-001" not in labels
    out = tmp_path / "c.json"
    args = ["ground", "calibrate", "--pack", str(unlabelled), "--scores", str(scores)]
    assert main([*args, "--sheet", str(sheet), "--out", str(out)]) == EXIT_OK
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n_labelled"] == 39 and report["best"]["f1"] == 1.0
    sheet.write_text("## item-003\nlabel: kinda\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="item-003: label must be"):
        read_sheet_labels(sheet)
    sheet.write_text("## item-004\nno label line\n", encoding="utf-8")
    assert read_sheet_labels(sheet) == {}


def test_calibrate_input_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pack, scores, _ = _labelled_pack(tmp_path, 20, 20)
    out = tmp_path / "c.json"
    scores.write_text('{"schema_version": 1, "method": "lexical", "scores": {}}', encoding="utf-8")
    args = ["ground", "calibrate", "--pack", str(pack), "--scores", str(scores), "--out", str(out)]
    assert main(args) == EXIT_USAGE
    assert "no score for labelled items" in capsys.readouterr().err
    scores.write_text('{"schema_version": 2, "method": "x", "scores": {}}', encoding="utf-8")
    assert main(args) == EXIT_INPUT
    assert "schema_version" in capsys.readouterr().err
    scores.write_text("{", encoding="utf-8")
    assert main(args) == EXIT_INPUT
    assert "invalid JSON" in capsys.readouterr().err
    scores.write_text('{"schema_version": 1}', encoding="utf-8")
    assert main(args) == EXIT_INPUT
    assert "not a rageval ground scores file" in capsys.readouterr().err


def test_calibrate_library_rejects_labels_for_unknown_items() -> None:
    items = [PackItem("i1", "q", "?", ["p"], "a", "grounded")]
    with pytest.raises(ValueError, match="not in the pack"):
        calibrate(items, {"i1": 1.0}, method="lexical", sheet_labels={"i9": "grounded"})


def test_calibrate_one_empty_class_warns_and_key_may_cover_only_some_items() -> None:
    items = [PackItem(f"i{i}", "q", "?", ["p"], "a", "grounded") for i in range(30)]
    scores = {f"i{i}": 0.9 for i in range(30)}
    report = calibrate(items, scores, method="lexical", key={"i0": "faithful"})
    assert any("one class has no labelled item" in w for w in report["warnings"])
    assert report["best"]["confusion"] == {"tp": 30, "fp": 0, "fn": 0, "tn": 0}
    assert report["hidden_condition"]["n_with_key"] == 1
    report = calibrate(items, scores, method="lexical", key={"zz": "faithful"})
    assert report["hidden_condition"]["label_agrees_with_condition"] is None


def test_confusion_metrics_on_empty_counts_are_zero() -> None:
    empty = grounding.Confusion(0, 0, 0, 0)
    assert (empty.accuracy, empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 0.0, 0.0)
    assert grounding.sweep_thresholds([]) == [(0.0, empty)]


# ----------------------------------------------------------------- isolation


def test_run_compare_gate_paths_never_import_grounding() -> None:
    probe = (
        "import sys, rageval, rageval.cli; "
        "from rageval.cli import main; main(['--version']); "
        "print('rageval.grounding' in sys.modules)"
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter, fixed program text
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert result.stdout.strip().endswith("False")


def test_ground_help_lists_the_three_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ground", "--help"]) == EXIT_OK
    out = capsys.readouterr().out
    assert all(word in out for word in ("prepare", "score", "calibrate", "EXPERIMENTAL"))
