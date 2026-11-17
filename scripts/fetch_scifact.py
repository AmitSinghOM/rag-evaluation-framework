"""Download the BEIR SciFact dataset, verify its sha256 and unpack the test split.

Writes ``corpus.jsonl``, ``queries.jsonl`` and ``qrels/test.tsv`` (the BEIR
layout, unchanged bytes) under ``--dest`` (default ``.cache/scifact/``), plus a
``.gitignore`` so the cache never gets committed. The verified zip is kept in
``--dest`` so a second run is offline. ``--trec-qrels PATH`` additionally writes
the test qrels in TREC format (``qid 0 docid grade``), which is what the
committed fixture ``examples/scifact/qrels.test.tsv`` is.

Source, size and licence are documented in ``scripts/README.md``. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
# sha256 of scifact.zip as downloaded on 2026-11-18 (2,816,079 bytes; the BEIR
# README lists md5 5f7d1de60b170fc8027bb7898e2efca1 for the same file).
SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"
# zip member -> relative path under --dest
MEMBERS = {
    "scifact/corpus.jsonl": "corpus.jsonl",
    "scifact/queries.jsonl": "queries.jsonl",
    "scifact/qrels/test.tsv": "qrels/test.tsv",
}
DEFAULT_DEST = Path(".cache/scifact")


class ChecksumMismatch(RuntimeError):
    """The downloaded bytes do not match the pinned sha256."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(data: bytes, expected: str = SHA256) -> None:
    """Raise :class:`ChecksumMismatch` unless ``data`` hashes to ``expected``."""
    actual = sha256_hex(data)
    if actual != expected:
        raise ChecksumMismatch(f"sha256 mismatch: expected {expected}, got {actual}")


def download(url: str = URL) -> bytes:
    """Fetch ``url`` (https only) fully into memory; the file is under 3 MB."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL: {url}")
    with urlopen(url, timeout=120) as response:  # noqa: S310 - scheme checked above
        return bytes(response.read())


def extract(zip_bytes: bytes, dest: Path) -> list[Path]:
    """Write the three test-split members under ``dest``; return the paths written."""
    written: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member, relative in MEMBERS.items():
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
            written.append(target)
    # The cache directory ignores itself, so nothing here can be committed by accident.
    (dest / ".gitignore").write_text("*\n", encoding="utf-8")
    return written


def to_trec_qrels(beir_tsv: str) -> str:
    """Convert BEIR ``query-id<TAB>corpus-id<TAB>score`` (with header) to TREC qrels.

    TREC qrels lines are ``qid 0 docid grade``; row order is preserved.
    """
    lines = beir_tsv.splitlines()
    if not lines or lines[0].split("\t") != ["query-id", "corpus-id", "score"]:
        raise ValueError("not a BEIR qrels file: expected header query-id, corpus-id, score")
    out: list[str] = []
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"line {lineno}: expected 3 tab-separated fields, got {len(parts)}")
        qid, docid, grade = parts
        out.append(f"{qid} 0 {docid} {int(grade)}\n")
    return "".join(out)


def fetch(dest: Path = DEFAULT_DEST, *, url: str = URL, expected: str = SHA256) -> Path:
    """Ensure the verified zip and the three extracted files exist under ``dest``.

    Downloads only when ``dest/scifact.zip`` is absent or fails verification.
    """
    dest.mkdir(parents=True, exist_ok=True)
    cached = dest / "scifact.zip"
    data: bytes | None = None
    if cached.is_file():
        data = cached.read_bytes()
        try:
            verify(data, expected)
        except ChecksumMismatch:
            data = None
    if data is None:
        data = download(url)
        verify(data, expected)
        cached.write_bytes(data)
    extract(data, dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download, verify (sha256) and unpack the BEIR SciFact test split."
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="cache directory")
    parser.add_argument(
        "--trec-qrels",
        type=Path,
        default=None,
        help="also write the test qrels in TREC format (qid 0 docid grade) to this path",
    )
    args = parser.parse_args(argv)
    try:
        dest = fetch(args.dest)
    except (ChecksumMismatch, OSError, ValueError) as exc:
        print(f"fetch_scifact: {exc}", file=sys.stderr)
        return 1
    for relative in MEMBERS.values():
        path = dest / relative
        print(f"{path} ({path.stat().st_size} bytes)")
    if args.trec_qrels is not None:
        text = to_trec_qrels((dest / "qrels/test.tsv").read_text(encoding="utf-8"))
        args.trec_qrels.parent.mkdir(parents=True, exist_ok=True)
        args.trec_qrels.write_text(text, encoding="utf-8")
        print(f"{args.trec_qrels} ({text.count(chr(10))} qrels lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
