"""Execute the prospective OCBA extension with resumable per-cell archives.

Protocol is saved before any main-series observations. Normalization and
development/validation function membership are fixed in advance. Worker count
affects wall time only; each numerical worker uses one BLAS thread.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from historical.ocba_20261022.experiment import TabularNNObjective
from ocba_benchmarks import ControlledBenchmark, FUNCTIONS, VALIDATION_FUNCTIONS
from ocba_engine import REAL_METHODS, SYNTHETIC_METHODS, run_search

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results/ocba_extension_20261022"
DATASETS = ("digits", "wine", "breast_cancer", "ionosphere", "sonar")
SPLITS = (4101, 4102, 4103)
DIMENSIONS = (3, 8)
NOISE_LEVELS = (0.0, 0.25, 1.0, 2.0)
BUDGETS = (60, 180, 540)


def source_hashes() -> dict[str, str]:
    names = ["ocba.py", "ocba_engine.py", "ocba_benchmarks.py",
             "run_ocba_study.py", "experiment.py", "study_statistics.py",
             "historical/__init__.py", "historical/ocba_20261022/__init__.py",
             "historical/ocba_20261022/experiment.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def protocol() -> dict:
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes(),
        "synthetic": {"functions": FUNCTIONS, "dimensions": DIMENSIONS,
                      "eta": NOISE_LEVELS, "budgets": BUDGETS,
                      "methods": SYNTHETIC_METHODS, "runs": 30,
                      "validation_functions": VALIDATION_FUNCTIONS,
                      "calibration_size": 4096, "practical_delta": 0.02,
                      "target_regret": 0.1, "target_success_rate": 0.8,
                      "noise": "heteroskedastic Gaussian, std=eta*(0.6+0.8*x0)",
                      "sampling": "common run seed across methods, eta and budgets; different task seeds"},
        "real": {"datasets": DATASETS, "split_seeds": SPLITS, "budgets": [60, 180],
                 "methods": REAL_METHODS, "runs": 6, "epochs": 20,
                 "final_refits": 3, "final_seed_panel": "per run, paired across methods"},
        "ocba": {"implementation": "numerical Gaussian max-min PSO selection model",
                 "reference": "10.1109/TEVC.2016.2592185; models (4),(6)",
                 "swarm_size": 6, "n0": 3, "generation_budget": 36, "increment": 6,
                 "variance_floor": "max(1e-12, max_variance*1e-8)",
                 "gap_floor": "1e-6*max(1, range_of_sample_means)",
                 "initialization": "no personal thresholds before first generation",
                 "tail": "round-robin initial samples; allocate only after >=2 per current pool",
                 "global_samples": "reused; cumulative counts never decrease",
                 "deviations_from_published_experiment": "small n0, swarm and total budgets; fixed coefficients; no reproduction claim for its tables"},
        "inference": {"bootstrap_repeats": 4000, "confidence": 0.95,
                      "primary_comparisons": ["pso_ocba vs pso", "pso_ocba vs pso_ea"],
                      "primary_global_holm_family": "all 384 synthetic primary comparisons",
                      "secondary_holm_family": "five alternatives vs pso inside each synthetic cell",
                      "pointwise_intervals": True,
                      "real_inference": "within each dataset/split/budget; no independent-task claim for overlapping splits"},
        "router": {"algorithm": "DecisionTreeClassifier(max_depth=2,min_samples_leaf=12,random_state=20261022)",
                   "features": ["eta", "log_budget", "dimension"],
                   "label": "cell mean(pso_ocba - pso) >= 0.02",
                   "training": "first six functions, both dimensions, all conditions",
                   "validation": "Levy and Zakharov only, not used in training or hyperparameter selection",
                   "interpretation": "known-noise synthetic policy; not directly applicable when eta is unknown"},
        "timing": "concurrent workers; no speed superiority claims",
        "python": sys.version, "platform": sys.platform,
    }


def make_jobs(phase: str) -> list[dict]:
    if phase == "synthetic":
        return [{"phase": phase, "dataset": function, "dimension": d,
                 "eta": eta, "budget": budget, "runs": 30}
                for function, d, eta, budget in product(FUNCTIONS, DIMENSIONS, NOISE_LEVELS, BUDGETS)]
    return [{"phase": phase, "dataset": dataset, "split_seed": split,
             "dimension": 3, "budget": budget, "runs": 6}
            for dataset, split, budget in product(DATASETS, SPLITS, [60, 180])]


def cell_id(job: dict) -> str:
    noise = str(job.get("eta", "real")).replace(".", "p")
    return (f"{job['dataset']}_d{job['dimension']}_e{noise}"
            f"_s{job.get('split_seed', 0)}_b{job['budget']}")


def load_real(name: str, split: int):
    if name in {"ionosphere", "sonar"}:
        raw = pd.read_csv(ROOT / f"datasets/uci/{name}.csv", header=None)
        x = raw.iloc[:, :-1].to_numpy(float)
        _, y = np.unique(raw.iloc[:, -1].to_numpy(), return_inverse=True)
        task = TabularNNObjective(name, 20, split, data=(x, y))
    else:
        task = TabularNNObjective(name, 20, split)
    return task


def execute_cell(job: dict, out_name: str) -> dict:
    out = Path(out_name) / job["phase"] / cell_id(job)
    out.mkdir(parents=True, exist_ok=True)
    completed = out / "complete.json"
    if completed.is_file():
        return json.loads(completed.read_text(encoding="utf-8"))
    if job["phase"] == "synthetic":
        task = ControlledBenchmark(job["dataset"], job["dimension"], job["eta"])
        master = 20261101 + FUNCTIONS.index(job["dataset"]) * 1000003 + job["dimension"] * 10007
        methods = SYNTHETIC_METHODS
        task_meta = task.metadata()
    else:
        task = load_real(job["dataset"], job["split_seed"])
        master = 20261201 + DATASETS.index(job["dataset"]) * 1000003 + SPLITS.index(job["split_seed"]) * 100003
        methods = REAL_METHODS
        task_meta = {"dataset": task.name, "split_seed": job["split_seed"],
                     "sizes": task.sizes, "epochs": task.epochs}
    rows = []
    started = time.perf_counter()
    with gzip.open(out / "evaluations.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as event_file, \
            gzip.open(out / "pools.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as pool_file:
        events, pools = csv.writer(event_file), csv.writer(pool_file)
        events.writerow(["method", "run", "call", "generation", "pool_id", "seed", "score", "reason"])
        pools.writerow(["method", "run", "pool_id", "generation", "particle", "coordinates", "true_score"])
        for run in range(job["runs"]):
            for method in methods:
                def event_writer(values):
                    events.writerow([method, run, *values])

                def pool_writer(values):
                    pools.writerow([method, run, *values])

                row = run_search(task, method, master + 10003 * run,
                                 job["budget"], event_writer, pool_writer)
                row.update({**job, "run": run, "master_seed": master,
                            "synthetic": task.synthetic,
                            "role": ("validation" if job["dataset"] in VALIDATION_FUNCTIONS
                                     else "discovery") if task.synthetic else "real"})
                rows.append(row)
            pd.DataFrame(rows).to_csv(out / "runs.partial.csv", index=False)
    frame = pd.DataFrame(rows)
    if len(frame) != len(methods) * job["runs"] or not frame.calls.eq(job["budget"]).all():
        raise RuntimeError(f"Incomplete cell {job}")
    frame.to_csv(out / "runs.csv", index=False)
    result = {"id": cell_id(job), "phase": job["phase"], "runs": len(rows),
              "calls": int(frame.calls.sum()), "seconds": time.perf_counter() - started,
              "task": task_meta, "job": job}
    completed.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["synthetic", "real"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out / "protocol.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text(encoding="utf-8"))
        if saved["source_sha256"] != source_hashes():
            raise ValueError("Study sources changed since protocol creation; use a new output directory")
        if saved.get("smoke", False) != args.smoke:
            raise ValueError("Smoke and main studies require separate output directories")
    else:
        design = protocol()
        design["smoke"] = args.smoke
        protocol_path.write_text(json.dumps(design, indent=2), encoding="utf-8")
    jobs = make_jobs(args.phase)
    if args.smoke:
        jobs = [{**jobs[0], "runs": 2}]
    log_path = args.out / f"{args.phase}_progress.jsonl"
    completed = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor, log_path.open("a", encoding="utf-8") as log:
        futures = [executor.submit(execute_cell, job, str(args.out)) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            log.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **result}) + "\n")
            log.flush()
            print(f"{args.phase}: {len(completed)}/{len(jobs)} {result['id']} "
                  f"runs={sum(c['runs'] for c in completed)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    all_runs = pd.concat([pd.read_csv(args.out / args.phase / cell_id(job) / "runs.csv") for job in jobs], ignore_index=True)
    all_runs.to_csv(args.out / f"{args.phase}_runs.csv", index=False)
    print(f"Completed {len(all_runs)} runs, {all_runs.calls.sum()} calls", flush=True)


if __name__ == "__main__":
    main()
