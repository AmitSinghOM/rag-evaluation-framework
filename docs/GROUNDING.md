# Grounding check (EXPERIMENTAL)

**EXPERIMENTAL.** Nothing in this document is a claim that rageval detects
hallucinations. It describes two signals, a labelling pack and a calibration
command. Until the calibration report in the last section has been produced
from human labels, the README says only that this experiment exists. See
[ADR-0001](./decisions/ADR-0001-v0.1-scope.md), item 4, and
[ROADMAP.md](../ROADMAP.md) Phase 3.

Everything here is `rageval.grounding` plus `rageval ground`. The module is
standard library only and is never imported by `rageval run`, `compare` or
`gate` (locked by `tests/test_grounding.py::test_run_compare_gate_paths_never_import_grounding`).

## The reference point

[HHEM-2.1-Open](https://huggingface.co/vectara/hallucination_evaluation_model),
Vectara's open hallucination detector, reports **44.8% F1 on RAGTruth-Summ**
in its own model card (quoted in
[vectara/hallucination-leaderboard#128](https://github.com/vectara/hallucination-leaderboard/issues/128)).
That is a trained model built for the task. A lexical overlap score will be
weaker and a 3B local judge has no published number on this data. So the
rule is: measure, never claim. The calibration command exists to put a
number next to each method on labels Amit wrote himself.

## Signal 1: lexical support (`lexical_support`)

What it is: how much of the answer's vocabulary appears in one retrieved
passage. The exact rule:

1. **Sentences.** Split the answer on whitespace after `.`, `!` or `?` and
   on newlines. Drop blanks.
2. **Tokens.** Lowercase; every maximal run of `[a-z0-9]` is a token (the same
   tokeniser as `scripts/bm25_baseline.py`). Drop a fixed stopword list
   (`rageval.grounding.STOPWORDS`, about 120 function words). What is left are
   the sentence's **content tokens**, as a set.
3. **Per sentence.** For each passage, `|content(sentence) ∩ content(passage)|
   / |content(sentence)|`. The sentence's score is the **maximum over single
   passages**; tokens spread across two passages do not add up. A sentence
   with no content token is skipped.
4. **Aggregate.** The mean of the sentence scores. `0.0` when the answer has
   no scored sentence, and `0.0` per sentence when there is no passage.

Deterministic: the same inputs give byte-identical `scores.json`
(`test_score_lexical_is_byte_identical_across_runs`).

What it is **not**: it does not know negation (`"X does not cause Y"` scores
the same as `"X causes Y"`), numbers (`"3 hours"` vs `"30 hours"` share
the token `hours` only), paraphrase, or entailment. A copied sentence with
one invented number scores near 1. An answer that abstains ("the passages
do not address this") scores low although it claims nothing. It is a
baseline to beat, included so that a judge has something to be compared
against.

## Signal 2: a judge (`Judge`, `OllamaJudge`, `OpenAICompatibleJudge`)

A judge is anything with `judge(question, answer, passages) -> Verdict`, where
`Verdict` is `grounded: bool`, `confidence: float | None` (the model's own
0..1 estimate when it gives one) and `raw: str` (the model's text, kept).

Both built-in judges send the same fixed prompt (`JUDGE_SYSTEM`,
`JUDGE_USER` in the module: "every factual claim in the answer is supported
by the passages; outside knowledge does not count") at temperature 0 and ask
for JSON `{"grounded": bool, "confidence": number}`.

- `OllamaJudge(model, base_url="http://localhost:11434")`: local, free,
  `urllib` against `/api/chat` with `format: json`, 60 s timeout. Errors name
  the fix: `ollama serve` when the server is unreachable, `ollama pull MODEL`
  on a 404.
- `OpenAICompatibleJudge(base_url, model, api_key_env="RAGEVAL_JUDGE_API_KEY")`:
  any `/chat/completions` server (Groq and Gemini free tiers, OpenAI, vLLM,
  LM Studio). The key is read from the named environment variable at call
  time, never from a flag or a file, never written into an error message. A
  missing variable raises `ConfigError` naming it.

For `rageval ground score`, a verdict becomes one number: the confidence
when `grounded` is true, `1 - confidence` when false, and `1.0`/`0.0` when
the model gives no confidence. A judge is **optional**: `rageval run`,
`compare` and `gate` never need one, and CI runs no model.

## The labelling pack (`rageval ground prepare`)

```bash
rageval ground prepare --qrels examples/scifact/qrels.test.tsv \
    --run examples/scifact/bm25.top100.run \
    --corpus .cache/scifact/corpus.jsonl --queries .cache/scifact/queries.jsonl \
    --n 60 --seed 0 --generator ollama:qwen2.5:3b \
    --out pack/pack.jsonl --sheet pack/pack.md --key pack/key/pack.key.jsonl
```

(`scripts/fetch_scifact.py` downloads the corpus and queries to
`.cache/scifact/`.)

What it does, in order:

1. Takes the judged queries whose **top 3** retrieved passages (`--top`) hold
   at least one relevant document, so the faithful condition has evidence to
   use. On the SciFact BM25 run that is 211 of 300 queries.
2. Samples `n` of them with `random.Random(seed)`; `n` must be even.
3. Assigns half the items to the hidden condition `faithful` and half to
   `unfaithful`, shuffled with the same seed.
4. Generates **one answer per item** with the local model:
   - `faithful`: the model sees the three passages and is told to summarise
     what they say about the claim, using only facts stated in them.
   - `unfaithful`: the model sees **no passages** and is told to state what
     it knows about the claim, with concrete facts. The pack still shows the
     labeller the three retrieved passages, so an answer that happens to
     agree with them can still be labelled `grounded`. The condition is a
     generation setting, not a truth label.
5. Writes `pack.jsonl` (`item_id, query_id, question, passages[], answer,
   label: null, notes: ""`), the Markdown sheet, a `README.md` next to the
   pack, and the **key** (`item_id, condition, generator`) in a separate file.

The key holds no question, passage or answer text
(`test_prepare_pack_invariants`). Keep it closed until every item is
labelled; the condition would bias the labels.

`--generator fake` returns canned answers (the first sentence of the first
passage, or an unrelated sentence) and is what the tests use.

## Scoring (`rageval ground score`)

```bash
rageval ground score --pack pack/pack.jsonl --method lexical --out scores.lexical.json
rageval ground score --pack pack/pack.jsonl --method ollama:qwen2.5:3b --out scores.qwen3b.json
RAGEVAL_JUDGE_API_KEY=... rageval ground score --pack pack/pack.jsonl \
    --method openai-compatible --base-url https://api.groq.com/openai/v1 \
    --model llama-3.1-8b-instant --out scores.groq.json
```

`scores.json`: `schema_version: 1`, `status: "experimental"`, `method`, and
per item `score` (0..1), `grounded` (judge verdict, `null` for lexical) and
`detail` (per-sentence lexical scores, or the judge's raw reply).

## The calibration report (`rageval ground calibrate`)

```bash
rageval ground calibrate --pack pack/pack.jsonl --scores scores.lexical.json \
    --out calibration.lexical.json            # labels from pack.jsonl
rageval ground calibrate --pack pack/pack.jsonl --sheet pack/pack.md \
    --scores scores.lexical.json --out calibration.lexical.json   # labels from the sheet
```

Labels come from `label` in the pack rows, overridden by any decided
`label:` line in `--sheet`. `unsure` and unlabelled items are left out.
**Exit 2 under 30 labelled items**; a warning under 50.

How to read `calibration.json` (`schema_version: 1`, `status: "experimental"`):

| Field | Meaning |
|---|---|
| `n_labelled`, `n_per_class`, `n_unsure`, `n_unlabelled` | How much evidence there is, and whether both classes are present. With one empty class, precision and recall mean nothing. |
| `best.threshold` | Predict `grounded` when `score >= threshold`. Chosen as the threshold with the highest F1 over every distinct score (ties: the lowest). |
| `best.accuracy`, `best.accuracy_ci95` | Share of labelled items the rule gets right, with a percentile bootstrap interval over per-item correctness (`rageval.stats.bootstrap_ci`, 1000 resamples, seed 0). The interval is the honest number; at n = 60 it is about ±0.1 wide. |
| `best.precision`, `best.recall`, `best.f1`, `best.confusion` | For the positive class `grounded`. `fp` = said grounded, labelled not grounded (the harmful miss for a RAG check). |
| `sweep` | The same metrics at every threshold, so the trade-off is visible. |
| `hidden_condition` | Agreement between the labels and the generation condition, when the key is present: `faithful -> grounded`, `unfaithful -> not_grounded`. Low agreement means the model was faithful without passages or unfaithful with them; it is a check on the pack, not on the method. |

The threshold is tuned on the same labels it is scored on. With 60 items
that is an optimistic estimate; a method's real number comes from a second,
unseen pack. The report says what it measured on this pack, no more.

## Pre-label sanity check (NOT a calibration)

The pack in `~/Downloads/rageval_grounding_labels/` was generated on
2026-11-18 with `qwen2.5:3b` (Q4_K_M, 3.1B parameters) on Ollama 0.40.16,
Apple M5 Pro, seed 0, n = 60: 80.0 s of generation, 1.33 s per item, 38.7 MB
peak RSS for the `rageval` process, 3.37 GB for the loaded model as reported
by Ollama's `/api/ps`.

Before any label existed, `score --method lexical` on that pack, split by the
hidden condition:

| Condition | n | mean | median | q1 | q3 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| faithful | 30 | 0.607 | 0.632 | 0.514 | 0.715 | 0.176 | 0.837 |
| unfaithful | 30 | 0.369 | 0.354 | 0.291 | 0.461 | 0.173 | 0.554 |

The score moves with the condition, which is all this shows: the generator
did produce two kinds of answer and the lexical score sees the difference on
average. The ranges overlap (a threshold that best separates the two
conditions gets 51 of 60 right), and the condition is not a grounding
label: an answer written without passages can still agree with them, and a
"faithful" summary can still slip in a claim the passages do not make. Five
of the thirty faithful answers open with "the passages do not address". The
labels decide; this table does not.

## What would make this more than an experiment

1. The 60 labels, then `calibrate` on the lexical scores and on at least one
   judge.
2. A second pack with a different seed, scored at the first pack's threshold,
   so the number is not tuned on the data it is reported on.
3. Only then a README sentence, taken from the report, still marked
   experimental.
