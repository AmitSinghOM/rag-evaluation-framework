"""Locks on the release path: one version, pinned actions, a tag reaches TestPyPI only."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import rageval

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
RELEASE = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
PUBLISH_ACTION = "pypa/gh-action-pypi-publish@"


def _jobs(workflow: str) -> dict[str, str]:
    """``{job_id: job_text}`` for the two-space-indented keys under ``jobs:``."""
    parts = re.split(r"^  ([\w-]+):\n", workflow.split("\njobs:\n", 1)[1], flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


with (ROOT / "pyproject.toml").open("rb") as _fh:
    PROJECT = tomllib.load(_fh)["project"]


def test_version_is_the_same_in_pyproject_and_the_package() -> None:
    # The release workflow checks the tag against pyproject; `--version` prints __version__.
    assert PROJECT["version"] == rageval.__version__


def test_environment_urls_name_the_distribution_in_pyproject() -> None:
    jobs = _jobs(RELEASE)
    assert PROJECT["name"] == "rag-evaluation-framework"  # "rageval" is taken on TestPyPI
    assert f"url: https://test.pypi.org/p/{PROJECT['name']}\n" in jobs["testpypi"]
    assert f"url: https://pypi.org/p/{PROJECT['name']}\n" in jobs["pypi"]


def test_every_action_is_pinned_to_a_full_commit_sha() -> None:
    files = [*WORKFLOWS, ROOT / "action.yml"]
    uses = [
        (path.name, line.split("uses:", 1)[1].strip())
        for path in files
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(("uses:", "- uses:"))
    ]
    assert len(uses) >= 10, uses
    for name, ref in uses:
        if ref.startswith("./"):
            continue
        assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40} # v\d[\w.]*", ref), (name, ref)


def test_a_tag_push_can_reach_testpypi_only() -> None:
    assert re.search(r'^on:\n  push:\n    tags: \["v\*"\]\n  workflow_dispatch:', RELEASE, re.M)
    jobs = _jobs(RELEASE)
    uploads = {job for job, text in jobs.items() if PUBLISH_ACTION in text}
    assert uploads == {"testpypi", "pypi"}
    assert "repository-url: https://test.pypi.org/legacy/" in jobs["testpypi"]
    assert "name: testpypi" in jobs["testpypi"]
    assert "inputs.confirm == ''" in jobs["testpypi"]
    pypi = jobs["pypi"]
    assert "repository-url" not in pypi  # the action's default index is the real PyPI
    assert "name: pypi" in pypi
    condition = re.search(r"if: >-\n(.*?)\n    runs-on:", pypi, re.S)
    assert condition, pypi
    words = " ".join(condition.group(1).split())
    assert words == (
        "github.event_name == 'workflow_dispatch' && inputs.confirm == 'publish' && "
        "startsWith(github.ref, 'refs/tags/v')"
    )


def test_only_the_upload_jobs_get_an_oidc_token() -> None:
    top_level = RELEASE.split("\njobs:\n", 1)[0]
    assert "permissions:\n  contents: read\n" in top_level
    assert "id-token" not in top_level
    with_oidc = {job for job, text in _jobs(RELEASE).items() if "id-token: write" in text}
    assert with_oidc == {"testpypi", "pypi"}


def test_no_workflow_or_action_holds_a_stored_credential() -> None:
    for path in [*WORKFLOWS, ROOT / "action.yml", ROOT / ".github" / "dependabot.yml"]:
        text = path.read_text(encoding="utf-8").lower()
        for needle in ("password:", "secrets.", "api-token:", "api_token"):
            assert needle not in text, (path.name, needle)
        # A PyPI API token is "pypi-" plus a long base64 macaroon.
        assert not re.search(r"pypi-[a-z0-9_-]{20,}", text), path.name


def test_sdist_rules_ship_the_toy_examples_and_no_tests() -> None:
    # The rule file; CI's build job checks the built sdist against the same rules
    # (job `build`, step "The sdist carries examples/toy and no tests").
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
    rules = [line for line in manifest if line and not line.startswith("#")]
    assert rules == ["graft examples/toy", "prune tests"]
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "tar tzf dist/*.tar.gz" in ci and "examples/toy/qrels.txt" in ci
    assert "'^(tests/|examples/scifact/|\\.github/)'" in ci


def test_build_tools_are_pinned_where_dependabot_can_see_them() -> None:
    # Dependabot's pip updater reads *.txt requirements files at the root (and
    # pyproject), not version pins typed into a workflow `run:` step.
    pins = [
        line
        for line in (ROOT / "requirements-release.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert [p.split("==")[0] for p in pins] == ["build", "twine"]
    assert all(re.fullmatch(r"\w+==\d+(\.\d+)+", p) for p in pins), pins
    for name in ("ci.yml", "release.yml"):
        text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
        assert "-r requirements-release.txt" in text, name
        assert not re.search(r"pip install .*\b(build|twine)==", text), name


def test_readme_links_resolve_on_the_pypi_page_too() -> None:
    # PyPI renders the README as-is, so a relative link (./docs/...) points at
    # pypi.org and 404s. Anchors (#section) are fine.
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "](./" not in readme and "](docs/" not in readme
    targets = re.findall(
        r"\]\((https://github\.com/AmitSinghOM/rag-evaluation-framework/blob/main/[^)#]+)", readme
    )
    assert len(targets) >= 10
    for url in targets:  # every linked file exists in the repository
        assert (ROOT / url.split("/blob/main/", 1)[1]).is_file(), url


def test_security_policy_describes_what_the_code_touches() -> None:
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "reads nothing from disk" not in text  # true of Phase 0 only
    for claim in ("--base-url", "redirects are not followed", "environment variable", "--out"):
        assert claim in text, claim
