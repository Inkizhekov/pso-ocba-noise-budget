"""Recompute the exploratory decomposition on the 16 fixed validation instances."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from paired_mean_inference import simultaneous_mean_intervals
from run_ocba_instance_validation import FUNCTIONS, INSTANCES, METHODS, NOISES


ROOT = Path(__file__).resolve().parent
KEY = ["dataset", "instance", "run", "noise", "method"]
COMPONENTS = ("visit_loss", "selection_loss")
SEED = 21112026


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_cell(folder: Path, completion_hash: str) -> pd.DataFrame:
    marker = folder / "complete.json"
    if digest(marker) != completion_hash:
        raise ValueError(f"Changed completion record: {folder.name}")
    hashes = json.loads(marker.read_text(encoding="utf-8"))["sha256"]
    for name in ("pools.csv.gz", "runs.csv"):
        if digest(folder / name) != hashes[name]:
            raise ValueError(f"Changed trace: {folder.name}/{name}")
    runs = pd.read_csv(folder / "runs.csv", float_precision="round_trip")
    pools = pd.read_csv(folder / "pools.csv.gz", float_precision="round_trip",
                        usecols=["method", "run", "pool_id", "true_score"])
    if (pools.duplicated(["method", "run", "pool_id"]).any()
            or not np.isfinite(pools.true_score).all()):
        raise ValueError(f"Invalid pool records: {folder.name}")
    summary = pools.groupby(["method", "run"]).true_score.agg(["max", "size"])
    chosen = pools.rename(columns={"pool_id": "selected_pool", "true_score": "chosen"})
    rows = runs.join(summary, on=["method", "run"], validate="one_to_one").merge(
        chosen, on=["method", "run", "selected_pool"], validate="one_to_one")
    # Every recorded pool was sampled in these full, untruncated generations.
    if (len(rows) != len(runs) or len(rows) != 90
            or not rows["size"].eq(rows.evaluated_pools).all()
            or not rows.calls.eq(8640).all()):
        raise ValueError(f"Incomplete validation search: {folder.name}")
    np.testing.assert_allclose(rows.chosen, rows.final_score, atol=1e-12, rtol=1e-12)
    rows["visit_loss"] = -rows["max"]
    rows["selection_loss"] = rows["max"] - rows.chosen
    np.testing.assert_allclose(rows[list(COMPONENTS)].sum(axis=1), -rows.final_score,
                               rtol=1e-12, atol=1e-12)
    if (rows[list(COMPONENTS)] < -1e-12).any().any():
        raise ValueError(f"Negative loss component: {folder.name}")
    return rows[KEY + ["seed", "selected_pool", *COMPONENTS]]


def align(rows: pd.DataFrame) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        product(FUNCTIONS, INSTANCES, range(30), NOISES, METHODS), names=KEY)
    data = rows.set_index(KEY, verify_integrity=True)
    if len(data) != len(index) or len(index.difference(data.index)):
        raise ValueError("Unexpected function, instance, seed, noise or method grid")
    data = data.loc[index]
    archive = pd.read_csv(ROOT / "results/ocba_instances_20261022/runs.csv",
                          float_precision="round_trip").set_index(KEY).loc[index]
    np.testing.assert_array_equal(data[["seed", "selected_pool"]],
                                  archive[["seed", "selected_pool"]])
    np.testing.assert_allclose(data[list(COMPONENTS)].sum(axis=1), -archive.final_score,
                               atol=1e-12, rtol=1e-12)
    if not data.groupby(level=KEY[:3]).seed.nunique().eq(1).all():
        raise ValueError("The six method/noise observations must share a seed")
    return data


def infer(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    values = data[list(COMPONENTS)].to_numpy().reshape(16, 30, 2, 3, 2)
    coordinates = list(product(NOISES, METHODS[:2], COMPONENTS))
    paired = np.stack([
        values[:, :, NOISES.index(noise), METHODS.index(control), COMPONENTS.index(term)]
        - values[:, :, NOISES.index(noise), METHODS.index("pso_ocba"), COMPONENTS.index(term)]
        for noise, control, term in coordinates], axis=-1)
    result = simultaneous_mean_intervals(paired, repetitions=99999, seed=SEED)
    table = pd.DataFrame(coordinates, columns=["noise", "control", "component"])
    for name in ("mean", "se", "ci_low", "ci_high"):
        table[name] = result[name]
    meta = {k: result[k] for k in ("critical_max_abs_t", "confidence", "family_size",
                                   "repetitions", "seed")}
    return table, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cells", type=Path, help="The 32 released new-instance job folders")
    inputs.add_argument("--components", type=Path, help="Compact per-search component CSV")
    parser.add_argument("--out", type=Path, required=True, help="A new output directory")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error("--out must name a new directory")
    inventory = pd.read_csv(ROOT / "data/job_inventory.csv")
    official = inventory.loc[inventory.phase.eq("new_instances")]
    if len(official) != 32 or not official.job_id.is_unique:
        raise ValueError("Expected 32 official validation jobs")
    if args.cells is not None:
        rows = pd.concat([read_cell(args.cells / j.job_id, j.completion_sha256)
                          for j in official.itertuples()], ignore_index=True)
    else:
        rows = pd.read_csv(args.components, float_precision="round_trip")
    data = align(rows)
    table, meta = infer(data)
    meta.update(status="exploratory post-hoc fixed-instance diagnostic",
                contrast="control loss minus PSO-OCBA loss; positive favors PSO-OCBA",
                searches=2880, fixed_instances=16, paired_runs_per_instance=30,
                new_searches=0, raw_traces_checked=args.cells is not None)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as scratch:
        finished = Path(scratch) / "result"
        finished.mkdir()
        data.to_csv(finished / "search_components.csv", lineterminator="\n")
        table.to_csv(finished / "component_contrasts.csv", index=False, lineterminator="\n")
        (finished / "analysis.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                encoding="utf-8")
        finished.rename(out)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
