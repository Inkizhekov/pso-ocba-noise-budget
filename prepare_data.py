"""Verify the GitHub payload and restore CSV files compressed for web upload."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import tempfile


ROOT = Path(__file__).resolve().parent
MANIFEST = "github_payload_manifest.json"
CHUNK = 1024 * 1024
LOGGER = logging.getLogger(__name__)


def safe_path(name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
        raise ValueError(f"Unsafe payload path: {name}")
    path = ROOT.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
        raise ValueError(f"Payload path leaves the repository: {name}")
    return path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_transport(manifest: dict) -> None:
    if manifest.get("format") != "ocba-github-web-upload-v1":
        raise ValueError("Unsupported GitHub payload manifest")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("The payload manifest has no file checksums")
    for name, expected in hashes.items():
        path = safe_path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing or changed upload file: {name}")
    LOGGER.info("Verified %d uploaded files", len(hashes))


def restore_one(item: dict) -> None:
    source = safe_path(item["compressed_path"])
    target = safe_path(item["restored_path"])
    expected = item["restored_sha256"]
    expected_size = item["restored_bytes"]
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError("Invalid restored file size")
    if target.exists():
        if target.stat().st_size != expected_size or digest(target) != expected:
            raise ValueError(f"Existing CSV differs; it was not overwritten: {item['restored_path']}")
        LOGGER.info("CSV is already restored and verified: %s", item["restored_path"])
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".restore_", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            hasher = hashlib.sha256()
            size = 0
            with gzip.open(source, "rb") as compressed:
                for chunk in iter(lambda: compressed.read(CHUNK), b""):
                    size += len(chunk)
                    if size > expected_size:
                        raise ValueError("The expanded data exceed the recorded size")
                    hasher.update(chunk)
                    output.write(chunk)
        if size != expected_size or hasher.hexdigest() != expected:
            raise ValueError(f"Restored CSV checksum differs: {item['restored_path']}")
        # Exclusive creation prevents a concurrent restore from replacing a file.
        with temporary.open("rb") as source_stream, target.open("xb") as destination:
            try:
                for chunk in iter(lambda: source_stream.read(CHUNK), b""):
                    destination.write(chunk)
            except BaseException:
                destination.close()
                target.unlink(missing_ok=True)
                raise
        LOGGER.info("Restored and verified %s (%d bytes)", item["restored_path"], size)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check uploaded bytes without restoring data")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
        verify_transport(manifest)
        if not args.verify_only:
            for item in manifest["compressed_files"]:
                restore_one(item)
        LOGGER.info("Payload verification completed")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Data preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
