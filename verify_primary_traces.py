"""Verify release primary-data ZIPs and optionally restore selected job folders."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import BadZipFile, ZipFile

from research_integrity import checked_path, digest


ROOT = Path(__file__).resolve().parent
CHUNK = 8 * 1024 * 1024


def verify_part(path: Path, part: dict, destination: Path | None = None) -> dict:
    if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
        raise ValueError(f"Release asset checksum differs: {path.name}")
    records = {item["path"]: item for item in part["files"]}
    if len(records) != len(part["files"]):
        raise ValueError("Duplicate file in primary-data manifest")
    anchor = destination if destination is not None else ROOT
    for name in records:
        checked_path(anchor, name)
        if not name.startswith("results/"):
            raise ValueError(f"Unexpected primary-data path: {name}")
    restored = total = 0
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(n.casefold() for n in names)) or set(names) != set(records):
            raise ValueError("Archive entries differ from primary-data manifest")
        for info in archive.infolist():
            record = records[info.filename]
            if info.file_size != record["bytes"] or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Invalid primary-data entry: {info.filename}")
            target = checked_path(anchor, info.filename)
            if destination is not None and target.exists():
                if not target.is_file() or digest(target) != record["sha256"]:
                    raise ValueError(f"Existing file differs; not overwritten: {info.filename}")
            temporary = None
            output = None
            try:
                if destination is not None and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = tempfile.NamedTemporaryFile(dir=target.parent, prefix=".primary_", delete=False)
                    temporary = Path(output.name)
                hasher, size = hashlib.sha256(), 0
                with archive.open(info) as incoming:
                    for block in iter(lambda: incoming.read(CHUNK), b""):
                        size += len(block)
                        if size > record["bytes"]:
                            raise ValueError("Expanded primary data exceed recorded size")
                        hasher.update(block)
                        if output is not None:
                            output.write(block)
                if output is not None:
                    output.close()
                    output = None
                if size != record["bytes"] or hasher.hexdigest() != record["sha256"]:
                    raise ValueError(f"Primary file checksum differs: {info.filename}")
                if temporary is not None:
                    created = False
                    try:
                        with temporary.open("rb") as source, target.open("xb") as sink:
                            created = True
                            shutil.copyfileobj(source, sink, CHUNK)
                        restored += 1
                    except BaseException:
                        if created:
                            target.unlink(missing_ok=True)
                        raise
                total += size
            finally:
                if output is not None:
                    output.close()
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return {"asset": path.name, "files_verified": len(records), "bytes_verified": total,
            "files_restored": restored}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Directory of downloaded release ZIP assets")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/primary_traces_manifest.json")
    parser.add_argument("--phase", help="Select one scientific phase; default: all official jobs")
    parser.add_argument("--extract-to", type=Path, help="Restore results/ below this directory; differing files are refused")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("format") != "ocba-primary-traces-v1":
            raise ValueError("Unsupported primary-data manifest")
        if digest(ROOT / "data/job_inventory.csv") != manifest["inventory_sha256"]:
            raise ValueError("Primary traces belong to a different job inventory")
        parts = [p for p in manifest["parts"] if args.phase is None or p["phase"] == args.phase]
        if not parts:
            raise ValueError("No release assets match the selected phase")
        destination = args.extract_to.resolve() if args.extract_to else None
        reports = []
        for part in parts:
            if Path(part["name"]).name != part["name"]:
                raise ValueError("Unsafe release asset name")
            report = verify_part(args.assets / part["name"], part, destination)
            reports.append(report)
            print(json.dumps(report), flush=True)
        print(json.dumps({"parts_verified": len(reports), "files_verified": sum(r["files_verified"] for r in reports),
                          "new_scientific_searches": 0}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
        parser.exit(1, f"Primary-data verification failed: {exc}\n")


if __name__ == "__main__":
    main()
