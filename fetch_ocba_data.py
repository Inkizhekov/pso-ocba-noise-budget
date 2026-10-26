"""Download two small public UCI datasets and retain provenance and hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request
from zipfile import ZipFile
import io

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "ionosphere": {
        "url": "https://archive.ics.uci.edu/static/public/52/ionosphere.zip",
        "member": "ionosphere.data", "rows": 351, "features": 34,
        "doi": "10.24432/C5W01B",
        "citation": "Sigillito, Wing, Hutton and Baker (1989). Ionosphere. UCI.",
    },
    "sonar": {
        "url": "https://archive.ics.uci.edu/static/public/151/connectionist%2Bbench%2Bsonar%2Bmines%2Bvs%2Brocks.zip",
        "member": "sonar.all-data", "rows": 208, "features": 60,
        "doi": "10.24432/C5T01Q",
        "citation": "Sejnowski and Gorman (1988). Sonar, Mines vs. Rocks. UCI.",
    },
}


def main() -> None:
    target = ROOT / "datasets/uci"
    target.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, info in SOURCES.items():
        path = target / f"{name}.csv"
        if not path.exists():
            request = urllib.request.Request(info["url"], headers={"User-Agent": "Scientific-reproducibility-study/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                archive = response.read()
            with ZipFile(io.BytesIO(archive)) as zipped:
                members = [m for m in zipped.namelist() if m.endswith(info["member"])]
                if len(members) != 1:
                    raise ValueError(f"Cannot identify dataset member for {name}")
                data = zipped.read(members[0])
            path.write_bytes(data)
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != info["rows"] or any(len(row.split(",")) != info["features"] + 1 for row in rows):
            raise ValueError(f"Unexpected dataset shape: {name}")
        manifest[name] = {**info, "license": "CC BY 4.0",
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name, len(rows), "rows", manifest[name]["sha256"])
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
