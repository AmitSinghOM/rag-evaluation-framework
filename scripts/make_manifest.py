"""Write ``examples/scifact/MANIFEST.json``: sha256, counts and provenance of every fixture.

Run after regenerating the fixtures (see ``scripts/README.md``). The output is
deterministic (sorted keys, no timestamps) so an unchanged fixture set gives an
unchanged manifest. ``tests/test_scifact_fixtures.py`` checks every fixture
against this file. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fetch_scifact import SHA256, URL

FIXTURES_DIR = Path("examples/scifact")
COMMANDS = [
    "python scripts/fetch_scifact.py --dest .cache/scifact"
    " --trec-qrels examples/scifact/qrels.test.tsv",
    "python scripts/bm25_baseline.py --dest .cache/scifact --out examples/scifact/bm25.top100.run",
    "python scripts/bm25_baseline.py --dest .cache/scifact --degrade truncate:20"
    " --out examples/scifact/bm25-degraded-truncate20.top20.run",
    "python scripts/bm25_baseline.py --dest .cache/scifact --degrade shuffle-tail:0"
    " --out examples/scifact/bm25-degraded-shuffle-tail0.top100.run",
    "python scripts/make_manifest.py",
]
LICENCE = (
    "SciFact claims and evidence annotations (queries, qrels): CC BY 4.0; abstracts (corpus):"
    " ODC-By 1.0 (S2ORC); per https://github.com/allenai/scifact/blob/master/LICENSE.md."
    " BEIR packaging code: Apache-2.0 (https://github.com/beir-cellar/beir)."
    " Evaluation data only, not training data."
)


def describe(path: Path) -> dict[str, object]:
    """sha256, line count, distinct qids and (for runs) per-query line count + tag."""
    data = path.read_bytes()
    lines = data.decode("utf-8").splitlines()
    qids: dict[str, int] = {}
    tags: set[str] = set()
    is_run = path.suffix == ".run"
    for line in lines:
        fields = line.split()
        qids[fields[0]] = qids.get(fields[0], 0) + 1
        if is_run:
            tags.add(fields[5])
    info: dict[str, object] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "lines": len(lines),
        "queries": len(qids),
        "format": "trec-run" if is_run else "trec-qrels",
    }
    if is_run:
        per_query = sorted(set(qids.values()))
        info["per_query"] = per_query[0] if len(per_query) == 1 else per_query
        info["tag"] = tags.pop() if len(tags) == 1 else sorted(tags)
    return info


def fixture_files(fixtures_dir: Path) -> list[Path]:
    return sorted(p for p in fixtures_dir.iterdir() if p.suffix in {".run", ".tsv"})


def build(fixtures_dir: Path) -> dict[str, object]:
    return {
        "dataset": "BEIR SciFact, test split (300 queries, 5183 documents, 339 judgements)",
        "source_url": URL,
        "source_sha256": SHA256,
        "licence": LICENCE,
        "generated_with": COMMANDS,
        "files": {p.name: describe(p) for p in fixture_files(fixtures_dir)},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write MANIFEST.json for the SciFact fixtures.")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_DIR)
    args = parser.parse_args(argv)
    manifest = build(args.fixtures)
    out = args.fixtures / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{out}: {len(fixture_files(args.fixtures))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
