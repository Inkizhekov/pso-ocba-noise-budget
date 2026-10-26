"""Refresh editable release checksums while refusing changes to frozen research."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path
import subprocess

from research_integrity import checked_path, digest, is_research_artifact, selected_hashes


ROOT = Path(__file__).resolve().parent
MANIFESTS = ("github_payload_manifest.json", "reviewer_package_manifest.json")
LOGGER = logging.getLogger(__name__)


def logical_digest(root: Path, name: str, compressed: dict[str, dict]) -> str:
    path = checked_path(root, name)
    if path.exists() or name not in compressed:
        return digest(path)
    item = compressed[name]
    source = checked_path(root, item["compressed_path"])
    hasher = hashlib.sha256()
    size = 0
    with gzip.open(source, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            if size > item["restored_bytes"]:
                raise ValueError(f"Expanded data exceed the recorded size: {name}")
            hasher.update(block)
    if size != item["restored_bytes"] or hasher.hexdigest() != item["restored_sha256"]:
        raise ValueError(f"Compressed scientific data changed: {name}")
    return hasher.hexdigest()


def build_manifests(root: Path, additions: set[str]) -> dict[str, dict]:
    manifests = {name: json.loads((root / name).read_text(encoding="utf-8")) for name in MANIFESTS}
    transport = manifests[MANIFESTS[0]]
    compressed = {item["restored_path"]: item for item in transport["compressed_files"]}
    frozen = set()
    for manifest in manifests.values():
        for name, expected in selected_hashes(manifest).items():
            if logical_digest(root, name, compressed) != expected:
                raise ValueError(f"Frozen scientific artifact changed; refresh refused: {name}")
            frozen.add(name)
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True,
    )
    files = set(listing.stdout.decode("utf-8").rstrip("\0").split("\0")) - set(MANIFESTS)
    files.discard("")
    for name in additions:
        if name not in files or not is_research_artifact(name):
            raise ValueError(f"Not a new scientific release file: {name}")
    new_research = {name for name in files if is_research_artifact(name)} - frozen
    if new_research - additions:
        raise ValueError("Explicit --add-research required for new artifacts: " + ", ".join(sorted(new_research - additions)))
    hashes = {name: digest(checked_path(root, name)) for name in sorted(files)}
    logical_hashes = dict(hashes)
    for name, item in compressed.items():
        logical_hashes.pop(item["compressed_path"], None)
        logical_hashes[name] = logical_digest(root, name, compressed)
    transport["sha256"] = hashes
    manifests[MANIFESTS[1]]["sha256"] = dict(sorted(logical_hashes.items()))
    # The two manifests exclude each other to avoid a circular checksum dependency.
    for manifest in manifests.values():
        manifest["unhashed_manifests"] = list(MANIFESTS)
    return manifests


def refresh(root: Path, *, write: bool, additions: set[str] | None = None) -> None:
    expected = build_manifests(root, additions or set())
    changed = [name for name, value in expected.items()
               if json.loads((root / name).read_text(encoding="utf-8")) != value]
    if changed and not write:
        raise ValueError("Release checksums are stale; inspect changes, then run --write: " + ", ".join(changed))
    for name in changed:
        (root / name).write_text(json.dumps(expected[name], indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8", newline="\n")
    LOGGER.info("Frozen research verified; %d release manifests updated", len(changed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Check the current Git release file list and hashes")
    mode.add_argument("--write", action="store_true", help="Refresh code/documentation hashes after review")
    parser.add_argument("--add-research", action="append", default=[], metavar="PATH",
                        help="Explicitly freeze a new scientific artifact; existing artifacts cannot change")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        refresh(ROOT, write=args.write, additions=set(args.add_research))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Manifest refresh failed: {exc}\n")


if __name__ == "__main__":
    main()
