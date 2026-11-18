# Security policy

## Supported versions

Nothing is released yet. Security fixes land on `main` and ship with the
next release; once releases exist, only the latest minor version is
supported.

## Reporting a vulnerability

Report channel: GitHub private vulnerability reporting for this repository.

**Current state (2026-11-18): not enabled.** The repository is private and
GitHub does not offer private vulnerability reporting on private
repositories without GitHub Advanced Security. It will be switched on in
Settings → Code security when the repository goes public at v0.1; until
then the only readers are collaborators, who should contact the maintainer
directly through GitHub rather than open an issue describing the problem.

Do not open a public issue containing an exploit, a secret, or private data.

Include the minimum needed to reproduce: rageval version, Python version,
operating system, a small synthetic input, expected and observed behaviour,
and the impact.

## Scope

rageval is a pure-Python library and command line with no runtime
dependencies. What it touches:

- **Files you name.** `rageval run`, `compare` and `gate` read the qrels, run,
  JSONL and report files given on the command line and write only the paths
  given with `--out` and `--summary`. Files are read as UTF-8 (a BOM is
  accepted); a malformed line, a non-UTF-8 file, a non-finite grade or a
  report value that is not a finite number is an error that names the file
  (exit 2), never a silent wrong score. Nothing else on disk is read.
- **The network, only in the experiment.** `rageval ground` with a judge or
  generator sends the question, passages and answer to the server you name
  with `--base-url` (a local Ollama by default). `run`, `compare` and `gate`
  never import that module (a test locks this). The API key for an
  OpenAI-compatible server is read from an environment variable at call
  time, is never written to a file, a report or an error message, and
  redirects are not followed, so it is sent only to the host you named.
- **The GitHub Action** installs rageval from its own checkout and passes
  its inputs to the command through environment variables, never through
  `${{ }}` in script text.
- **Dependency and distribution integrity:** `dependencies` stays empty; dev
  tooling (including the `ranx` reference used only in tests) must never
  reach a user install. CI checks this on every change. Release uploads use
  Trusted Publishing (OIDC); no token is stored anywhere.

Hostile ids, grades or report values must raise a clear error, never hang
or exhaust memory.

## Response

A complete report is acknowledged, reproduced with synthetic data, assessed
for affected versions, and fixed with a changelog entry before technical
details are published.
