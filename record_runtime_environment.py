"""Record the actual CPU runtime for a current or historical rerun."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys


def record_environment(out: Path) -> Path:
    """Capture observed versions and thread pools without replacing a record."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    # Load the numerical libraries so threadpool_info describes real pools.
    import numpy
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info

    versions = {name: metadata.version(name) for name in (
        "numpy", "scipy", "scikit-learn", "optuna", "pandas", "matplotlib")}
    report = {
        "python": sys.version, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "logical_processors": os.cpu_count(), "libraries": versions,
        "numeric_threads_per_library": threadpool_info(), "gpu_used": False,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "runtime_environment.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(record_environment(args.out.resolve()))


if __name__ == "__main__":
    main()
