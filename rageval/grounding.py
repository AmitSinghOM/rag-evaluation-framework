"""EXPERIMENTAL grounding check: lexical support, judges, labelling pack, calibration.

Nothing here is a hallucination detector until the calibration report in
``docs/GROUNDING.md`` says what it measures. The module is never imported by
``rageval run``, ``rageval compare`` or ``rageval gate``; the CLI loads it
only for ``rageval ground``. Standard library only.

Three parts:

- :func:`lexical_support`: the fraction of a sentence's content tokens that
  one single passage contains, averaged over sentences. The exact rule is in
  ``docs/GROUNDING.md``.
- :class:`Judge` implementations that ask a model whether an answer is
  grounded: :class:`OllamaJudge` (local, ``urllib`` against the Ollama HTTP
  API) and :class:`OpenAICompatibleJudge` (Groq, Gemini and other
  ``/chat/completions`` servers; the key is read from an environment
  variable only).
- The labelling pack (:func:`prepare_pack`), scoring (:func:`score_pack`) and
  the calibration report (:func:`calibrate`) behind ``rageval ground``.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .loaders import LoaderError, Qrels, Run
from .stats import bootstrap_ci

SCHEMA_VERSION = 1
STATUS = "experimental"
PLACES = 6
TIMEOUT_SECONDS = 60.0
TOP_PASSAGES = 3
MIN_LABELLED = 30  # calibrate refuses below this
SMALL_LABELLED = 50  # and warns below this
LABELS = ("grounded", "not_grounded", "unsure")
CONDITIONS = ("faithful", "unfaithful")


class ConfigError(ValueError):
    """A judge is missing a setting; the message names the environment variable."""


class JudgeError(RuntimeError):
    """The model server could not be used; the message names the fix."""


class TooFewLabelsError(ValueError):
    """Fewer than :data:`MIN_LABELLED` decided labels; no calibration report is written."""


# ------------------------------------------------------------------ lexical support

# Function words only. Anything not in this set is a content token.
STOPWORDS = frozenset(
    """
    a an the and or but nor so yet of in on at to for from by with without as into onto
    over under than then that this these those there here it its is are was were be been
    being am do does did doing have has had having will would shall should can could may
    might must not no yes if else when where while which who whom whose what why how
    all any both each few more most other some such only own same too very s t just
    also about above after again against before below between during up down out off
    further once because until through we you he she they i me him her them my your his
    their our mine yours hers theirs ours us
    """.split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def tokenize(text: str) -> list[str]:
    """Lowercase, then every maximal run of ``[a-z0-9]``; the BM25 script's rule."""
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    """Unique tokens of ``text`` that are not stopwords."""
    return {token for token in tokenize(text) if token not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    """Split on whitespace after ``.``, ``!`` or ``?`` and on newlines; drop blanks."""
    return [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]


@dataclass(frozen=True)
class SentenceSupport:
    sentence: str
    score: float  # fraction of the sentence's content tokens found in its best passage
    n_tokens: int  # content tokens in the sentence
    best_passage: int | None  # index into ``passages``; None when no passage was given


@dataclass(frozen=True)
class SupportResult:
    score: float  # mean of the sentence scores; 0.0 when no sentence has a content token
    sentences: list[SentenceSupport]
    method: str = "lexical"

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)


def lexical_support(answer: str, passages: Sequence[str]) -> SupportResult:
    """How much of ``answer`` is lexically present in one of ``passages``.

    Per sentence: ``|content(sentence) & content(passage)| / |content(sentence)|``
    for the passage that maximises it. Sentences with no content token are
    skipped. The aggregate is the mean over the scored sentences, ``0.0`` when
    there are none (empty answer) and ``0.0`` per sentence when ``passages`` is
    empty. Deterministic; no model, no network.
    """
    passage_tokens = [content_tokens(passage) for passage in passages]
    sentences: list[SentenceSupport] = []
    for sentence in split_sentences(answer):
        tokens = content_tokens(sentence)
        if not tokens:
            continue
        best_score, best_index = 0.0, None
        for index, available in enumerate(passage_tokens):
            score = len(tokens & available) / len(tokens)
            if best_index is None or score > best_score:
                best_score, best_index = score, index
        sentences.append(SentenceSupport(sentence, best_score, len(tokens), best_index))
    aggregate = sum(s.score for s in sentences) / len(sentences) if sentences else 0.0
    return SupportResult(score=aggregate, sentences=sentences)


# ------------------------------------------------------------------------- judges


@dataclass(frozen=True)
class Verdict:
    grounded: bool
    confidence: float | None  # the model's own 0..1 estimate when it gives one
    raw: str  # the model's text, kept for the record


class Judge(Protocol):
    """Anything that can say whether ``answer`` is grounded in ``passages``."""

    @property
    def name(self) -> str: ...

    def judge(self, question: str, answer: str, passages: Sequence[str]) -> Verdict: ...


JUDGE_SYSTEM = (
    "You are a strict fact checker. You will see a question, some passages and an answer. "
    'Reply with JSON only, of the form {"grounded": true or false, "confidence": 0.0 to 1.0}. '
    "grounded is true only when every factual claim in the answer is supported by the "
    "passages. Knowledge from outside the passages does not count as support."
)
JUDGE_USER = "Question:\n{question}\n\nPassages:\n{passages}\n\nAnswer:\n{answer}\n\nJSON:"


def render_judge_prompt(question: str, answer: str, passages: Sequence[str]) -> str:
    numbered = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)) or "(none)"
    return JUDGE_USER.format(question=question, passages=numbered, answer=answer)


def parse_verdict(raw: str) -> Verdict:
    """Read ``{"grounded": bool, "confidence": number|null}`` out of the model's text."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"judge reply is not JSON ({exc.msg}): {raw[:200]!r}") from None
    if not isinstance(data, dict) or not isinstance(data.get("grounded"), bool):
        raise JudgeError(f"judge reply has no boolean 'grounded': {raw[:200]!r}")
    confidence = data.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise JudgeError(f"judge 'confidence' is not a number: {raw[:200]!r}")
        confidence = min(1.0, max(0.0, float(confidence)))
    return Verdict(grounded=data["grounded"], confidence=confidence, raw=raw)


class _HTTPStatus(Exception):
    """A non-2xx reply, with its body already read and the response closed."""

    def __init__(self, code: int, detail: str) -> None:
        super().__init__(f"HTTP {code}: {detail}")
        self.code = code
        self.detail = detail


class _NoRedirect(HTTPRedirectHandler):
    """Refuse every redirect, so a request (and its Authorization header) never leaves
    the host named in ``base_url``. urllib would otherwise follow a 301/302/303 to any
    host, as a GET, with the Bearer key still attached."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None  # urllib then raises HTTPError for the 3xx itself


urlopen = build_opener(_NoRedirect).open


def _post_json(
    url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: float
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON object; ``http(s)`` only.

    Raises :class:`_HTTPStatus` for a non-2xx reply, ``JudgeError`` for a
    redirect (never followed), ``URLError`` when the server cannot be reached
    and ``TimeoutError`` when it does not answer.
    """
    if not url.startswith(("http://", "https://")):
        raise ConfigError(f"base_url must start with http:// or https://, got {url!r}")
    body = json.dumps(payload).encode("utf-8")
    request = Request(  # noqa: S310 - scheme checked above
        url, data=body, method="POST", headers={"Content-Type": "application/json", **headers}
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - scheme checked above
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        with exc:  # HTTPError is a file-like temp wrapper; close it or 3.14 warns
            detail = exc.read().decode("utf-8", "replace")[:200]
            location = exc.headers.get("Location", "?") if exc.headers is not None else "?"
        if 300 <= exc.code < 400:
            raise JudgeError(
                f"{url} redirected (HTTP {exc.code}) to {location}; redirects are not followed, "
                "so point --base-url at the final address"
            ) from None
        raise _HTTPStatus(exc.code, detail) from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JudgeError(f"{url} returned non-JSON ({exc.msg}): {text[:200]!r}") from None
    if not isinstance(data, dict):
        raise JudgeError(f"{url} returned JSON that is not an object: {text[:200]!r}")
    return data


def ollama_chat(
    model: str,
    messages: Sequence[Mapping[str, str]],
    *,
    base_url: str = "http://localhost:11434",
    json_mode: bool = False,
    timeout: float = TIMEOUT_SECONDS,
) -> str:
    """One non-streaming ``/api/chat`` call; errors name the command that fixes them."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": False,
        "options": {"temperature": 0},
    }
    if json_mode:
        payload["format"] = "json"
    url = base_url.rstrip("/") + "/api/chat"
    try:
        data = _post_json(url, payload, {}, timeout)
    except _HTTPStatus as exc:
        if exc.code == 404:
            raise JudgeError(
                f"Ollama at {base_url} has no model {model!r}: run `ollama pull {model}`"
            ) from None
        raise JudgeError(f"Ollama at {base_url} answered HTTP {exc.code}: {exc.detail}") from None
    except URLError as exc:
        raise JudgeError(
            f"cannot reach Ollama at {base_url} ({exc.reason}): start it with `ollama serve`"
        ) from None
    except TimeoutError:
        raise JudgeError(
            f"Ollama at {base_url} did not answer within {timeout:g} s; try a smaller model"
        ) from None
    try:
        content = data["message"]["content"]
    except (KeyError, TypeError):
        raise JudgeError(f"Ollama reply has no message.content: {json.dumps(data)[:200]}") from None
    return str(content)


@dataclass(frozen=True)
class OllamaJudge:
    """Judge backed by a local Ollama server (``ollama serve``), JSON mode, 60 s timeout."""

    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = TIMEOUT_SECONDS

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def judge(self, question: str, answer: str, passages: Sequence[str]) -> Verdict:
        raw = ollama_chat(
            self.model,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": render_judge_prompt(question, answer, passages)},
            ],
            base_url=self.base_url,
            json_mode=True,
            timeout=self.timeout,
        )
        return parse_verdict(raw)


@dataclass(frozen=True)
class OpenAICompatibleJudge:
    """Judge for any ``/chat/completions`` server (Groq, Gemini, OpenAI, vLLM, LM Studio).

    The API key is read from the environment variable ``api_key_env`` when a
    judgement is made, never from a flag or a file, and it is never written
    into an error message.
    """

    base_url: str
    model: str
    api_key_env: str = "RAGEVAL_JUDGE_API_KEY"
    timeout: float = TIMEOUT_SECONDS

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ConfigError(
                f"environment variable {self.api_key_env} is not set; export it with the API "
                "key for the judge server (the key is read from the environment only)"
            )
        return key

    def judge(self, question: str, answer: str, passages: Sequence[str]) -> Verdict:
        headers = {"Authorization": f"Bearer {self._key()}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": render_judge_prompt(question, answer, passages)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        url = self.base_url.rstrip("/") + "/chat/completions"
        try:
            data = _post_json(url, payload, headers, self.timeout)
        except _HTTPStatus as exc:
            if exc.code in (401, 403):
                raise JudgeError(
                    f"{url} rejected the key (HTTP {exc.code}); check {self.api_key_env}"
                ) from None
            raise JudgeError(f"{url} answered HTTP {exc.code}: {exc.detail}") from None
        except URLError as exc:
            raise JudgeError(f"cannot reach {url} ({exc.reason})") from None
        except TimeoutError:
            raise JudgeError(f"{url} did not answer within {self.timeout:g} s") from None
        try:
            raw = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise JudgeError(
                f"{url} reply has no choices[0].message.content: {json.dumps(data)[:200]}"
            ) from None
        return parse_verdict(str(raw))


# ------------------------------------------------------------------- generators

Generator = Callable[[str, Sequence[str], str], str]
"""``generator(question, passages, condition) -> answer``; ``passages`` is empty when the
condition is ``unfaithful`` (the model is asked to answer from its own knowledge)."""

GENERATE_FAITHFUL = (
    "Below are passages and a scientific claim. In two or three sentences, summarise what the "
    "passages say that relates to the claim. Use ONLY facts stated in the passages and keep "
    "their terms. Do not add outside knowledge, and do not comment on whether the passages "
    "address the claim.\n\nClaim:\n{question}\n\nPassages:\n{passages}\n\nSummary:"
)
GENERATE_UNFAITHFUL = (
    "Using your own knowledge, state in two or three specific sentences what is known about "
    "the question or claim below. Give concrete facts, numbers or mechanisms. Do not say that "
    "you lack information.\n\nQuestion or claim:\n{question}\n\nAnswer:"
)


def ollama_generator(model: str, base_url: str = "http://localhost:11434") -> Generator:
    """Answers from a local Ollama model under the two hidden conditions."""

    def generate(question: str, passages: Sequence[str], condition: str) -> str:
        if condition == "faithful":
            numbered = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
            prompt = GENERATE_FAITHFUL.format(question=question, passages=numbered)
        else:
            prompt = GENERATE_UNFAITHFUL.format(question=question)
        return ollama_chat(model, [{"role": "user", "content": prompt}], base_url=base_url).strip()

    return generate


def fake_generator(question: str, passages: Sequence[str], condition: str) -> str:
    """Canned answers for tests: the first passage's first sentence, or unrelated text."""
    if condition == "faithful" and passages:
        return split_sentences(passages[0])[0]
    return f"Studies report that {question.rstrip('.')} is explained by quantum tides on Mars."


def make_generator(spec: str, base_url: str = "http://localhost:11434") -> Generator:
    """``fake`` or ``ollama:MODEL``."""
    if spec == "fake":
        return fake_generator
    if spec.startswith("ollama:") and len(spec) > len("ollama:"):
        return ollama_generator(spec[len("ollama:") :], base_url=base_url)
    raise ValueError(f"--generator must be 'fake' or 'ollama:MODEL', got {spec!r}")


# --------------------------------------------------------------------------- pack


@dataclass(frozen=True)
class PackItem:
    item_id: str
    query_id: str
    question: str
    passages: list[str]
    answer: str
    label: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "query_id": self.query_id,
            "question": self.question,
            "passages": list(self.passages),
            "answer": self.answer,
            "label": self.label,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Pack:
    items: list[PackItem]
    conditions: dict[str, str]  # item_id -> hidden condition (the key file)
    generator: str
    seconds: list[float] = field(default_factory=list)  # generation time per item


def read_beir_jsonl(path: str | Path, wanted: set[str] | None = None) -> dict[str, str]:
    """``_id -> text`` (``title. text`` when a title exists) for BEIR corpus or queries JSONL.

    With ``wanted`` only those ids are kept, so a corpus can be streamed.
    """
    out: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                doc_id = str(obj["_id"])
                text = str(obj["text"])
            except (json.JSONDecodeError, KeyError, TypeError):
                raise LoaderError(path, line_no, "expected JSON with '_id' and 'text'") from None
            if wanted is not None and doc_id not in wanted:
                continue
            title = str(obj.get("title") or "").strip()
            out[doc_id] = f"{title.rstrip('.')}. {text}".strip() if title else text
    return out


def prepare_pack(
    qrels: Qrels,
    run: Run,
    corpus_path: str | Path,
    queries_path: str | Path,
    *,
    n: int,
    seed: int,
    generator: Generator,
    generator_name: str,
    top: int = TOP_PASSAGES,
) -> Pack:
    """Sample ``n`` judged queries, take their top passages, generate one answer each.

    Only queries whose top ``top`` retrieved passages hold at least one judged
    relevant document are eligible, so the ``faithful`` condition has evidence
    to be faithful to. Half the items are generated under ``faithful`` (answer
    only from the passages) and half under ``unfaithful`` (no passages; own
    knowledge); the assignment is seeded and hidden in ``Pack.conditions``.
    ``n`` must be even and at most the number of eligible queries.
    """
    if n < 2 or n % 2:
        raise ValueError(f"--n must be an even number >= 2 (equal hidden halves), got {n}")
    judged = sorted(
        qid
        for qid in run
        if qid in qrels and any(qrels[qid].get(doc, 0) > 0 for doc in run[qid][:top])
    )
    if n > len(judged):
        raise ValueError(
            f"--n {n} exceeds the {len(judged)} judged queries whose top-{top} retrieved "
            "passages hold a relevant document"
        )
    rng = random.Random(seed)  # noqa: S311 - sampling, not security
    sampled = rng.sample(judged, n)
    conditions = ["faithful"] * (n // 2) + ["unfaithful"] * (n // 2)
    rng.shuffle(conditions)
    questions = read_beir_jsonl(queries_path, set(sampled))
    missing = [qid for qid in sampled if qid not in questions]
    if missing:
        raise LoaderError(queries_path, 0, f"query ids missing from the file: {missing[:5]}")
    wanted_docs = {doc for qid in sampled for doc in run[qid][:top]}
    corpus = read_beir_jsonl(corpus_path, wanted_docs)
    missing_docs = sorted(wanted_docs - corpus.keys())
    if missing_docs:
        raise LoaderError(corpus_path, 0, f"doc ids missing from the corpus: {missing_docs[:5]}")
    width = len(str(n))
    items: list[PackItem] = []
    key: dict[str, str] = {}
    seconds: list[float] = []
    for index, (qid, condition) in enumerate(zip(sampled, conditions, strict=True), start=1):
        passages = [corpus[doc] for doc in dict.fromkeys(run[qid][:top])]
        started = time.perf_counter()
        shown = passages if condition == "faithful" else []
        answer = generator(questions[qid], shown, condition)
        seconds.append(time.perf_counter() - started)
        item_id = f"item-{index:0{width}d}"
        items.append(PackItem(item_id, qid, questions[qid], passages, answer))
        key[item_id] = condition
    return Pack(items=items, conditions=key, generator=generator_name, seconds=seconds)


LABEL_RULE = (
    "Rule: `grounded` = every factual claim in the answer is supported by the passages shown. "
    "Otherwise `not_grounded`. `unsure` is allowed and is ignored by the calibration. Judge "
    "against the passages only, not against what you know to be true."
)
LABEL_LINE = "label: grounded | not_grounded | unsure"


def pack_to_markdown(pack: Pack) -> str:
    lines = [
        "# rageval grounding labelling sheet",
        "",
        LABEL_RULE,
        "",
        f"Keep exactly one word on each `label:` line. {len(pack.items)} items. "
        "Do not open the key file until every item is labelled.",
        "",
    ]
    for item in pack.items:
        lines += [
            f"## {item.item_id} (query_id {item.query_id})",
            "",
            f"**Question or claim:** {item.question}",
            "",
            "**Passages shown:**",
            "",
        ]
        lines += [f"{i + 1}. {p}\n" for i, p in enumerate(item.passages)]
        lines += [
            "**Answer to label:**",
            "",
            *(f"> {line}" for line in item.answer.splitlines() or [""]),
            "",
            LABEL_LINE,
            "notes:",
            "",
        ]
    return "\n".join(lines)


PACK_README = """# rageval grounding labelling pack

{n} items. About one minute each. Labels are yours; nothing is pre-filled.

## How to label

1. Read `pack.md` (or the same items in `pack.jsonl`). For every item:
   question or claim, the passages the answer was supposed to use, the answer.
2. {rule}
3. Write the label. Either:
   - in `pack.md`: edit the `label:` line so that exactly one word remains,
     e.g. `label: not_grounded`; or
   - in `pack.jsonl`: set `"label"` to `"grounded"`, `"not_grounded"` or `"unsure"`.
   Put anything you want to remember in `notes:`.
4. Do not open `key/` until you have labelled every item. It holds the hidden
   condition each answer was generated under and would bias the labels.

## Handing it back

Run, from the repository root, either of

```bash
rageval ground calibrate --pack pack.jsonl --scores scores.json --out calibration.json
rageval ground calibrate --pack pack.jsonl --sheet pack.md --scores scores.json \
    --out calibration.json
```

`scores.json` comes from `rageval ground score --pack pack.jsonl --method lexical --out scores.json`
(or a judge method). The report refuses fewer than {min_labelled} labelled items and warns
under {small}. `unsure` rows are left out. See `docs/GROUNDING.md` in the repository.
"""


def write_pack(pack: Pack, out: str | Path, sheet: str | Path | None, key: str | Path) -> None:
    """Write ``pack.jsonl``, the Markdown sheet, ``README.md`` beside the pack, and the key."""
    out_path = Path(out)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in pack.items:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    if sheet is not None:
        Path(sheet).write_text(pack_to_markdown(pack), encoding="utf-8")
    (out_path.parent / "README.md").write_text(
        PACK_README.format(
            n=len(pack.items), rule=LABEL_RULE, min_labelled=MIN_LABELLED, small=SMALL_LABELLED
        ),
        encoding="utf-8",
    )
    key_path = Path(key)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with key_path.open("w", encoding="utf-8") as fh:
        for item_id, condition in pack.conditions.items():
            record = {"item_id": item_id, "condition": condition, "generator": pack.generator}
            fh.write(json.dumps(record) + "\n")


def read_pack(path: str | Path) -> list[PackItem]:
    items: list[PackItem] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                item = PackItem(
                    item_id=str(obj["item_id"]),
                    query_id=str(obj["query_id"]),
                    question=str(obj["question"]),
                    passages=[str(p) for p in obj["passages"]],
                    answer=str(obj["answer"]),
                    label=obj.get("label"),
                    notes=str(obj.get("notes") or ""),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                raise LoaderError(path, line_no, "not a pack item") from None
            if item.label is not None and item.label not in LABELS:
                raise LoaderError(
                    path, line_no, f"label must be one of {LABELS}, got {item.label!r}"
                )
            if item.item_id in seen:
                raise LoaderError(path, line_no, f"repeated item_id {item.item_id!r}")
            seen.add(item.item_id)
            items.append(item)
    return items


def read_key(path: str | Path) -> dict[str, str]:
    """``item_id -> condition`` from ``pack.key.jsonl``."""
    key: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                item_id, condition = str(obj["item_id"]), str(obj["condition"])
            except (json.JSONDecodeError, KeyError, TypeError):
                raise LoaderError(path, line_no, "not a key record") from None
            if condition not in CONDITIONS:
                raise LoaderError(path, line_no, f"condition must be one of {CONDITIONS}")
            key[item_id] = condition
    return key


_SECTION = re.compile(r"^## (item-\S+)", re.MULTILINE)
_LABEL = re.compile(r"^label:\s*(.*?)\s*$", re.MULTILINE)


def read_sheet_labels(path: str | Path) -> dict[str, str]:
    """Labels from a filled ``pack.md``: ``item_id -> label`` for every decided line.

    A ``label:`` line still holding the template (``a | b | c``) or empty is
    undecided and skipped. Any other value that is not one of the three labels
    is an error naming the item.
    """
    text = Path(path).read_text(encoding="utf-8")
    labels: dict[str, str] = {}
    matches = list(_SECTION.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.end() : end]
        found = _LABEL.search(section)
        if found is None:
            continue
        value = found.group(1)
        if not value or "|" in value:
            continue
        if value not in LABELS:
            raise LoaderError(path, 0, f"{match.group(1)}: label must be one of {LABELS}")
        labels[match.group(1)] = value
    return labels


# -------------------------------------------------------------------------- score


@dataclass(frozen=True)
class ItemScore:
    score: float  # lexical aggregate, or the judge's confidence (1/0 when it gives none)
    grounded: bool | None  # judge verdict; None for lexical
    detail: Any  # per-sentence scores (lexical) or the raw judge reply


def make_judge(method: str, *, base_url: str | None, model: str | None) -> Judge | None:
    """``lexical`` -> None; ``ollama:MODEL`` or ``openai-compatible`` -> a judge."""
    if method == "lexical":
        return None
    if method.startswith("ollama:") and len(method) > len("ollama:"):
        return OllamaJudge(method[len("ollama:") :], base_url=base_url or "http://localhost:11434")
    if method == "openai-compatible":
        if not base_url or not model:
            raise ValueError("--method openai-compatible needs --base-url and --model")
        return OpenAICompatibleJudge(base_url=base_url, model=model)
    raise ValueError(f"--method must be lexical, ollama:MODEL or openai-compatible, got {method!r}")


def score_item(item: PackItem, judge: Judge | None) -> ItemScore:
    if judge is None:
        result = lexical_support(item.answer, item.passages)
        return ItemScore(
            score=result.score,
            grounded=None,
            detail=[round(s.score, PLACES) for s in result.sentences],
        )
    verdict = judge.judge(item.question, item.answer, item.passages)
    if verdict.confidence is None:
        score = 1.0 if verdict.grounded else 0.0
    else:
        # A confident "not grounded" is a low grounding score; a confident "grounded" a high one.
        score = verdict.confidence if verdict.grounded else 1.0 - verdict.confidence
    return ItemScore(score=score, grounded=verdict.grounded, detail=verdict.raw)


def score_pack(items: Iterable[PackItem], judge: Judge | None) -> dict[str, Any]:
    """The ``scores.json`` object. No timestamp, so the lexical method is byte-stable."""
    scores = {item.item_id: score_item(item, judge) for item in items}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "method": judge.name if judge else "lexical",
        "scores": {
            item_id: {
                "score": round(s.score, PLACES),
                "grounded": s.grounded,
                "detail": s.detail,
            }
            for item_id, s in scores.items()
        },
    }


def read_scores(path: str | Path) -> tuple[str, dict[str, float]]:
    """``(method, item_id -> score)`` from a ``scores.json``."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LoaderError(path, exc.lineno, f"invalid JSON: {exc.msg}") from None
    try:
        version = data["schema_version"]
        method = str(data["method"])
        scores = {str(k): float(v["score"]) for k, v in data["scores"].items()}
    except (KeyError, TypeError, ValueError):
        raise LoaderError(path, 0, "not a rageval ground scores file") from None
    if version != SCHEMA_VERSION:
        raise LoaderError(path, 0, f"unsupported schema_version {version!r}")
    return method, scores


# ---------------------------------------------------------------------- calibrate


@dataclass(frozen=True)
class Confusion:
    tp: int  # labelled grounded, predicted grounded
    fp: int  # labelled not_grounded, predicted grounded
    fn: int  # labelled grounded, predicted not grounded
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": None,
            "accuracy": round(self.accuracy, PLACES),
            "precision": round(self.precision, PLACES),
            "recall": round(self.recall, PLACES),
            "f1": round(self.f1, PLACES),
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
        }


def confusion_at(threshold: float, pairs: Sequence[tuple[float, bool]]) -> Confusion:
    """Predict grounded when ``score >= threshold``; ``pairs`` are ``(score, is_grounded)``."""
    tp = fp = fn = tn = 0
    for score, truth in pairs:
        predicted = score >= threshold
        if predicted and truth:
            tp += 1
        elif predicted:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    return Confusion(tp, fp, fn, tn)


def sweep_thresholds(pairs: Sequence[tuple[float, bool]]) -> list[tuple[float, Confusion]]:
    """Every distinct score as a threshold, plus one above the maximum (predict none)."""
    distinct = sorted({score for score, _ in pairs})
    candidates = [*distinct, distinct[-1] + 1.0] if distinct else [0.0]
    return [(t, confusion_at(t, pairs)) for t in candidates]


def calibrate(
    items: Sequence[PackItem],
    scores: Mapping[str, float],
    *,
    method: str,
    sheet_labels: Mapping[str, str] | None = None,
    key: Mapping[str, str] | None = None,
    resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """The calibration report; ``ValueError`` below :data:`MIN_LABELLED` labelled items.

    Labels come from the pack rows, overridden by ``sheet_labels`` when given.
    ``unsure`` and unlabelled items are left out. The best threshold is the one
    with the highest F1 (ties: the lowest threshold). The accuracy CI resamples
    per-item correctness with :func:`rageval.stats.bootstrap_ci`.
    """
    labels: dict[str, str] = {i.item_id: i.label for i in items if i.label is not None}
    if sheet_labels:
        labels.update(sheet_labels)
    unknown = sorted(set(labels) - {i.item_id for i in items})
    if unknown:
        raise ValueError(f"labels for items that are not in the pack: {unknown[:5]}")
    n_unsure = sum(1 for v in labels.values() if v == "unsure")
    decided = {k: v for k, v in labels.items() if v != "unsure"}
    missing_scores = sorted(set(decided) - set(scores))
    if missing_scores:
        raise ValueError(f"scores.json has no score for labelled items: {missing_scores[:5]}")
    n_labelled = len(decided)
    warnings: list[str] = []
    if n_labelled < MIN_LABELLED:
        raise TooFewLabelsError(
            f"only {n_labelled} labelled items (grounded/not_grounded); at least "
            f"{MIN_LABELLED} are needed for a calibration report, {SMALL_LABELLED} recommended"
        )
    if n_labelled < SMALL_LABELLED:
        warnings.append(
            f"only {n_labelled} labelled items (< {SMALL_LABELLED}): the estimates are rough; "
            "see docs/GROUNDING.md"
        )
    item_ids = sorted(decided)
    pairs = [(scores[i], decided[i] == "grounded") for i in item_ids]
    sweep = sweep_thresholds(pairs)
    best_threshold, best = max(sweep, key=lambda tc: (round(tc[1].f1, 9), -tc[0]))
    correct = [1.0 if (s >= best_threshold) == truth else 0.0 for s, truth in pairs]
    ci = bootstrap_ci(correct, resamples=resamples, seed=seed)
    n_per_class = {
        "grounded": sum(1 for v in decided.values() if v == "grounded"),
        "not_grounded": sum(1 for v in decided.values() if v == "not_grounded"),
    }
    if not all(n_per_class.values()):
        warnings.append("one class has no labelled item; precision/recall are not meaningful")
    hidden: dict[str, Any] | None = None
    if key is not None:
        with_key = [i for i in item_ids if i in key]
        by_condition = {
            c: {
                "grounded": sum(1 for i in with_key if key[i] == c and decided[i] == "grounded"),
                "not_grounded": sum(
                    1 for i in with_key if key[i] == c and decided[i] == "not_grounded"
                ),
            }
            for c in CONDITIONS
        }
        expected = {"faithful": "grounded", "unfaithful": "not_grounded"}
        agree = sum(1 for i in with_key if expected[key[i]] == decided[i])
        hidden = {
            "n_with_key": len(with_key),
            "label_agrees_with_condition": round(agree / len(with_key), PLACES)
            if with_key
            else None,
            "by_condition": by_condition,
        }
    best_dict = best.to_dict()
    best_dict["threshold"] = round(best_threshold, PLACES)
    best_dict["accuracy_ci95"] = [round(ci[0], PLACES), round(ci[1], PLACES)]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "method": method,
        "positive_class": "grounded",
        "rule": "predict grounded when score >= threshold; threshold chosen for best F1",
        "n_items": len(items),
        "n_labelled": n_labelled,
        "n_unsure": n_unsure,
        "n_unlabelled": len(items) - n_labelled - n_unsure,
        "n_per_class": n_per_class,
        "best": best_dict,
        "sweep": [{**c.to_dict(), "threshold": round(t, PLACES)} for t, c in sweep],
        "hidden_condition": hidden,
        "bootstrap": {"resamples": resamples, "seed": seed},
        "warnings": warnings,
    }
