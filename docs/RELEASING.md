# Releasing rageval

The workflow is `.github/workflows/release.yml`. Uploads use PyPI Trusted
Publishing: GitHub gives the job a short-lived OIDC identity and the index
trusts this repository, workflow and environment. No password or API token
is stored in the repository, its secrets or its environments.

ADR-0001 defers the real PyPI. Until Amit says go, a release means a tag, a
TestPyPI upload and a GitHub Release.

## What runs when

| Trigger | `build` | `testpypi` | `github-release` | `pypi` |
|---|---|---|---|---|
| Push of a `v*` tag | yes, and the tag must equal `v` + the `pyproject.toml` version | yes | yes | **never** |
| Manual run, `confirm` empty (any branch or tag) | yes | yes | no | no |
| Manual run on a `v*` tag, `confirm` = `publish` | yes | no | no | yes |
| Manual run, any other `confirm`, or `publish` on a branch | fails at once | no | no | no |

`build` runs `python -m build` and `twine check --strict` (build 1.6.1,
twine 7.0.0) on Python 3.13 and hands `dist/` to the other jobs as an
artifact. `github-release` creates the release from the tag with both files
attached (`gh release create --verify-tag --generate-notes`, marked
pre-release when the version ends in `aN`, `bN`, `rcN` or `.devN`). Only
`testpypi` and `pypi` get `id-token: write`; only `github-release` gets
`contents: write`. Every action is pinned to a full commit SHA;
`tests/test_release.py` fails if one is not, if a second job can upload, or
if a stored credential appears in a workflow.

CI (`.github/workflows/ci.yml`, job `build`) runs the same build on every
pull request, installs the wheel in a clean venv, requires that `pip list`
shows only `rageval`, checks `rageval --version` and `python -m rageval
--version`, and runs the toy quickstart.

## The project name: `rag-evaluation-framework`

The name people `pip install` is `rag-evaluation-framework` (the
repository's name). The import name `rageval` and the `rageval` command do
not change. Files are `rag_evaluation_framework-<version>.tar.gz` and
`rag_evaluation_framework-<version>-py3-none-any.whl`.

Why not `rageval`, checked 2026-11-18 against the indexes' JSON API:

| Index | `rageval` | `rag-evaluation-framework` |
|---|---|---|
| [TestPyPI](https://test.pypi.org/project/rageval/) | **taken**: version 0.0.1, uploaded 2023-10-18, by another publisher (Zeta Alpha) | not registered |
| [PyPI](https://pypi.org/project/rageval/) | not registered, but [`rag-eval`](https://pypi.org/project/rag-eval/) 0.1.3 exists, and PyPI refuses a new name "too similar" to an existing one once `-`, `_` and `.` are removed ([PyPI help](https://pypi.org/help/#project-name)) | not registered |

`ragevaluationframework`, `rag-evaluation-frame-work` and
`rag-eval-framework` were also not registered on either index. Nothing
reserves a name until the first upload, so check again just before it. If
the name ever changes, change `[project] name` in `pyproject.toml`, the two
`environment.url` lines in `release.yml` (`tests/test_release.py` checks
they match), the wheel file name in the README install line, and the
publisher registration below.

## One-time setup (Amit, in a browser)

1. **TestPyPI Trusted Publisher.** Sign in at <https://test.pypi.org>, open
   *Your account → Publishing → Add a new pending publisher → GitHub* and
   enter:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `rag-evaluation-framework` |
   | Owner | `AmitSinghOM` |
   | Repository name | `rag-evaluation-framework` |
   | Workflow name | `release.yml` |
   | Environment name | `testpypi` |

   A pending publisher does not reserve the name until its first upload; if
   someone else registers the name first, it is invalidated
   ([PyPI docs](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)).
   Register it shortly before the first tag.
2. **GitHub environments.** The first run creates `testpypi` and `pypi` in
   *Settings → Environments* if they do not exist, with no protection rules.
   While the repository is private on a Free, Pro or Team plan GitHub does
   not offer required reviewers
   ([GitHub docs](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments#required-reviewers)),
   so the `confirm` input and the tag check are the only gate on `pypi`.
   When the repository goes public, add yourself as a required reviewer on
   `pypi` and limit both environments to tags matching `v*`.
3. **PyPI, only after the go.** Repeat step 1 on <https://pypi.org> with
   environment name `pypi`.

## Cutting a release

1. On a branch: set the version in both `pyproject.toml` and
   `rageval/__init__.py` (`tests/test_release.py` fails if they differ),
   move the `Unreleased` CHANGELOG entries under the new version and date,
   open a PR, merge it with every check green.
2. From your own terminal, tag the merge commit on `main` with an annotated
   tag `v<version>` (for example `v0.1.0a1`), then push that tag to GitHub.
3. Watch the `release` run: `build`, `testpypi` and `github-release` must
   all be green. A tag that does not match the version fails in `build`
   before anything is uploaded.
4. Check the upload from a clean venv. `--no-deps` keeps pip from pulling
   anything else from TestPyPI, where anyone can upload any name:

   ```bash
   python -m venv /tmp/rageval-check && . /tmp/rageval-check/bin/activate
   pip install --index-url https://test.pypi.org/simple/ --no-deps "rag-evaluation-framework==<version>"
   rageval --version
   ```

5. Real PyPI, only after the go: *Actions → release → Run workflow*, pick
   the tag under *Use workflow from*, type `publish` in `confirm`.

## Rehearsing without a tag

*Actions → release → Run workflow* on a branch with `confirm` empty runs
`build` and `testpypi` only. TestPyPI, like PyPI, never accepts the same
file name twice, even after a delete, so each rehearsal needs a new
pre-release version (`0.1.0a1`, `0.1.0a2`, ...). A used-up version is
normal; move on to the next one.

## Undo

An upload cannot be replaced. On PyPI, *yank* the version (pip then skips it
unless asked for it by exact version) and release a fixed one. A GitHub
Release and its tag can be deleted on GitHub; delete the release first.
