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

rageval is a pure-Python library with no runtime dependencies. It reads
nothing from disk or the network in this version; it computes metrics over
in-memory ids. Security-relevant areas are therefore small:

- Input handling: unexpected or hostile ids and grades must raise a clear
  error, never hang or exhaust memory.
- Dependency and distribution integrity: `dependencies` stays empty; dev
  tooling (including the `ranx` reference used only in tests) must never
  reach a user install. CI checks this on every change.

Future loaders (Phase 1) will read files and will extend this list.

## Response

A complete report is acknowledged, reproduced with synthetic data, assessed
for affected versions, and fixed with a changelog entry before technical
details are published.
