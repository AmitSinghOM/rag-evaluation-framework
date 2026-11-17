"""Lock: rageval has no runtime dependencies and imports without the scientific stack."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_no_runtime_dependencies() -> None:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)["project"]
    assert project["dependencies"] == []
    assert "ranx" in " ".join(project["optional-dependencies"]["dev"])


def test_import_does_not_load_third_party_modules() -> None:
    probe = (
        "import sys, rageval; "
        "heavy = {'numpy', 'numba', 'pandas', 'scipy', 'ranx'} & set(sys.modules); "
        "print(sorted(heavy))"
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter, fixed program text
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert result.stdout.strip() == "[]"
