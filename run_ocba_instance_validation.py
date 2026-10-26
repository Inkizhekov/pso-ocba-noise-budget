"""Freeze and run a small independent validation on new benchmark instances."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
from itertools import product
import json
import os
from pathlib import Path
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from ocba_benchmarks import FUNCTIONS, SPANS, raw_loss
from paired_mean_inference import simultaneous_mean_intervals
from record_runtime_environment import record_environment
from robust_ocba_engine import run_search
from run_robust_ocba_study import RobustBenchmark, sha256, source_hashes
from study_statistics import validate_grid


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/ocba_instances_20261022"
INSTANCES = (101, 102)
NOISES = ("gaussian", "lognormal")
METHODS = ("pso_ea", "pso_ocba_rs", "pso_ocba")
CONTRASTS = ("ocba_ea_gaussian", "ocba_ea_lognormal", "noise_interaction",
             "ocba_rs_gaussian", "ocba_rs_lognormal")


class NewInstance(RobustBenchmark):
    """Same function definitions and calibration rule, disjoint design seeds."""

    def __init__(self, name: str, instance: int, noise: str):
        if instance not in INSTANCES or noise not in NOISES:
            raise ValueError("Unknown validation instance or noise")
        super().__init__(name, 2, noise)
        self.instance = instance
        self.design_seed = 1460001 + FUNCTIONS.index(name) * 10007 + instance * 1000003 + 8 * 97
        rng = np.random.default_rng(self.design_seed)
        self.optimum = rng.uniform(0.25, 0.75, 8)
        self.rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        calibration = np.random.default_rng(self.design_seed + 900001).random((4096, 8))
        losses = raw_loss(name, (calibration - self.optimum) @ self.rotation * SPANS[name])
        self.scale = float(np.std(losses, ddof=1))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Invalid new-instance calibration")

    def metadata(self) -> dict:
        return {**super().metadata(), "instance": self.instance}


def jobs(smoke: bool = False) -> list[dict]:
    grid = [{"dataset": f, "instance": i, "noise": z, "eta": 2, "swarm": 12,
             "budget": 8640, "runs": 30} for f, i, z in product(FUNCTIONS, INSTANCES, NOISES)]
    return [{**grid[0], "runs": 1}] if smoke else grid


def cell_id(job: dict) -> str:
    return f"{job['dataset']}_i{job['instance']}_{job['noise']}"


def freeze(out: Path, smoke: bool) -> None:
    if smoke and "smoke" not in out.name:
        raise ValueError("Use a separate smoke directory")
    code = source_hashes()
    code[Path(__file__).name] = sha256(Path(__file__))
    code["paired_mean_inference.py"] = sha256(ROOT / "paired_mean_inference.py")
    design = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "smoke": smoke,
        "status": "new observations; design fixed before validation; question selected after original analysis",
        "source_sha256": code, "functions": list(FUNCTIONS), "instances": list(INSTANCES),
        "noises": list(NOISES), "methods": list(METHODS), "dimension": 8, "eta": 2,
        "swarm": 12, "budget": 8640, "initial_repeats": 3, "generation_multiplier": 6,
        "runs_per_cell": 30, "searches": 2880, "calls": 24883200,
        "rationale": "validate the previously strongest lognormal-vs-Gaussian mean interaction on new transformations; not a full new grid",
        "contrasts": list(CONTRASTS), "bootstrap_repetitions": 99999, "bootstrap_seed": 12222026,
        "inference": "one joint max-t family of five means; 16 fixed function-instance strata with 30 paired seeds each; equal weights; no inference to a population of functions",
        "planned_power": None, "original_primary_families_unchanged": True,
        "new_search_seed_formula": "912700001 + function_index*1000003 + instance*9000011 + run*10003",
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in saved.items() if k != "created_utc"} != {
                k: v for k, v in design.items() if k != "created_utc"}:
            raise ValueError("Validation design or code changed; use another output directory")
    else:
        if any(out.iterdir()):
            raise ValueError("Refusing to freeze a nonempty unversioned directory")
        path.write_text(json.dumps(design, indent=2), encoding="utf-8")
    if not (out / "runtime_environment.json").exists():
        record_environment(out)


def execute(job: dict, output: str) -> dict:
    folder = Path(output) / "cells" / cell_id(job)
    folder.mkdir(parents=True, exist_ok=True)
    completion = folder / "complete.json"
    if completion.exists():
        record = json.loads(completion.read_text(encoding="utf-8"))
        if record["job"] != job or any(sha256(folder / n) != h for n, h in record["sha256"].items()):
            raise ValueError("Completed validation cell changed")
        return record
    task = NewInstance(job["dataset"], job["instance"], job["noise"])
    master = 912700001 + FUNCTIONS.index(task.name) * 1000003 + task.instance * 9000011
    rows, started = [], time.perf_counter()
    with gzip.open(folder / "evaluations.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as ev, \
            gzip.open(folder / "pools.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as po:
        events, pools = csv.writer(ev), csv.writer(po)
        events.writerow(["method", "run", "call", "generation", "pool_id", "seed", "score", "reason"])
        pools.writerow(["method", "run", "pool_id", "generation", "particle", "coordinates", "true_score"])
        for run, method in product(range(job["runs"]), METHODS):
            result = run_search(task, method, master + 10003 * run, 8640,
                                lambda x: events.writerow([method, run, *x]),
                                lambda x: pools.writerow([method, run, *x]), swarm_size=12)
            if result["calls"] != 8640 or result["generations"] != 120:
                raise ValueError("Validation budget or generation mismatch")
            rows.append({**result, **job, "run": run, "master_seed": master})
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "runs.csv", index=False)
    record = {"job": job, "task": task.metadata(), "searches": len(frame),
              "calls": int(frame.calls.sum()), "seconds": time.perf_counter() - started,
              "sha256": {n: sha256(folder / n) for n in ["runs.csv", "evaluations.csv.gz", "pools.csv.gz"]}}
    completion.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def audit(folder_name: str) -> dict:
    """Check every archived call and reconstruct selected means and truth."""
    folder = Path(folder_name)
    meta = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if any(sha256(folder / n) != h for n, h in meta["sha256"].items()):
        raise ValueError("Validation archive changed")
    job = meta["job"]
    task = NewInstance(job["dataset"], job["instance"], job["noise"])
    if task.metadata() != meta["task"]:
        raise ValueError("Validation task metadata changed")
    rows = {(r.method, int(r.run)): r for r in pd.read_csv(folder / "runs.csv").itertuples()}
    pool_ids, coordinates = {key: set() for key in rows}, {}
    with gzip.open(folder / "pools.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key, pool = (row["method"], int(row["run"])), int(row["pool_id"])
            if key not in rows or pool in pool_ids[key]:
                raise ValueError("Unknown search or duplicate pool")
            pool_ids[key].add(pool)
            if pool == rows[key].selected_pool:
                coordinates[key] = np.array(json.loads(row["coordinates"]))
                if abs(task.true_score(coordinates[key]) - float(row["true_score"])) > 1e-11:
                    raise ValueError("Pool oracle score mismatch")
    counts, selected = {key: 0 for key in rows}, {key: [0, 0.] for key in rows}
    with gzip.open(folder / "evaluations.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            key, pool, call = (row[0], int(row[1])), int(row[4]), int(row[2])
            if key not in rows or pool not in pool_ids[key] or call != counts[key] + 1:
                raise ValueError("Invalid call sequence or pool link")
            counts[key] = call
            if pool == rows[key].selected_pool:
                state = selected[key]
                state[0] += 1
                state[1] += (float(row[6]) - state[1]) / state[0]
    for key, row in rows.items():
        if counts[key] != row.budget or not selected[key][0]:
            raise ValueError("Incomplete call budget")
        if abs(selected[key][1] - row.selected_observed_score) > 1e-11:
            raise ValueError("Selected mean mismatch")
        np.testing.assert_allclose(coordinates[key], json.loads(row.selected_coordinates), rtol=0, atol=1e-14)
        if abs(task.true_score(coordinates[key]) - row.final_score) > 1e-11:
            raise ValueError("Final oracle score mismatch")
    return {"cell": cell_id(job), "searches": len(rows), "calls": sum(counts.values()),
            "complete_sha256": sha256(folder / "complete.json")}


def analyze(out: Path, workers: int) -> dict:
    plan = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if plan["smoke"]:
        raise ValueError("Smoke cannot enter validation inference")
    folders = [out / "cells" / cell_id(job) for job in jobs()]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        audits = list(executor.map(audit, map(str, folders)))
    raw = pd.read_csv(out / "runs.csv")
    keys = ["dataset", "instance", "run", "noise", "method"]
    grid = list(product(FUNCTIONS, INSTANCES, range(30), NOISES, METHODS))
    validate_grid(raw, grid, keys)
    index = pd.MultiIndex.from_tuples(grid, names=keys)
    scores = raw.set_index(keys).final_score.reindex(index).to_numpy().reshape(16, 30, 2, 3)
    ea, rs = scores[..., 2] - scores[..., 0], scores[..., 2] - scores[..., 1]
    values = np.stack([ea[..., 0], ea[..., 1], ea[..., 1]-ea[..., 0], rs[..., 0], rs[..., 1]], axis=-1)
    inferred = simultaneous_mean_intervals(values, seed=12222026)
    results = [{"id": name, **{k: float(inferred[k][i]) for k in [
        "mean", "se", "ci_low", "ci_high", "adjusted_p_zero"]}} for i, name in enumerate(CONTRASTS)]
    pd.DataFrame(results).to_csv(out / "mean_contrasts.csv", index=False)
    pd.DataFrame(raw.groupby(["dataset", "instance", "noise", "method"]).final_score.mean()).to_csv(out / "task_means.csv")
    (out / "trace_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    report = {"searches": len(raw), "calls": int(raw.calls.sum()), "audited_cells": len(audits),
              "fixed_function_instance_strata": 16, "family": 5, "contrasts": results,
              "summed_worker_elapsed_hours": sum(json.loads((p / "complete.json").read_text())["seconds"] for p in folders)/3600,
              "critical_max_abs_t": inferred["critical_max_abs_t"], "bootstrap_repetitions": 99999,
              "scope": "conditional on 16 new fixed instances of the same eight function families"}
    (out / "analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--phase", choices=["freeze", "run", "analyze", "all"], default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    out = args.out.resolve()
    freeze(out, args.smoke)
    if args.phase == "freeze":
        return
    if args.phase in {"run", "all"}:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(execute, job, str(out)) for job in jobs(args.smoke)]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                print(f"validation {i}/{len(futures)}; searches={result['searches']}", flush=True)
        pd.concat([pd.read_csv(out / "cells" / cell_id(job) / "runs.csv") for job in jobs(args.smoke)]).to_csv(out / "runs.csv", index=False)
    if args.smoke:
        print(json.dumps(audit(str(out / "cells" / cell_id(jobs(True)[0])))))
    elif args.phase in {"analyze", "all"}:
        analyze(out, args.workers)


if __name__ == "__main__":
    main()
