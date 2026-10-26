"""Focused exploratory n0 ablation at fixed total and generation budgets."""
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

from ocba_benchmarks import FUNCTIONS
from record_runtime_environment import record_environment
from robust_ocba_engine import run_search
from run_robust_ocba_study import RobustBenchmark, NOISES, sha256, source_hashes


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/ocba_n0_ablation_20261022"
TASKS = ("ackley", "rosenbrock")
INITIAL_REPEATS = (3, 5, 10)
METHODS = ("pso_ea", "pso_ocba_rs", "pso_ocba")


def design(smoke: bool) -> dict:
    sources = source_hashes()
    sources.update({name: sha256(ROOT / name) for name in [
        "run_n0_ablation.py", "analyze_n0_ablation.py", "paired_mean_inference.py"]})
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory focused ablation; historical results already inspected",
        "smoke": smoke, "source_sha256": sources, "tasks": list(TASKS),
        "task_rationale": "multimodal Ackley and curved-valley Rosenbrock; no best-cell selection",
        "dimension": 8, "eta": 2, "swarms": [12], "budgets": [8640],
        "noise_families": list(NOISES), "initial_repeats": list(INITIAL_REPEATS),
        "generation_multiplier": 12, "methods": list(METHODS), "runs": 30,
        "seed_policy": "reuse the original matched 30 seed blocks for paired historical 6*M bridge",
        "generation_note": "n0=10 is infeasible at 6*M; all n0 arms use 12*M and 60 generations",
        "historical_source": "results/ocba_robustness_20261022/synthetic_runs.csv",
        "historical_source_sha256": sha256(ROOT / "results/ocba_robustness_20261022/synthetic_runs.csv"),
        "new_searches": 2160, "new_calls": 18662400,
        "inference": "72 exploratory joint max-t mean contrasts; fixed two-task strata, 30 shared-seed blocks",
    }


def freeze(out: Path, smoke: bool) -> None:
    if smoke and "smoke" not in out.name:
        raise ValueError("Smoke requires its own smoke directory")
    expected = design(smoke)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in saved.items() if k != "created_utc"} != {
                k: v for k, v in expected.items() if k != "created_utc"}:
            raise ValueError("Ablation protocol or sources changed; use a new output directory")
    else:
        if any(out.iterdir()):
            raise ValueError("Refusing to freeze a nonempty unversioned directory")
        path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    if not (out / "runtime_environment.json").exists():
        record_environment(out)


def jobs(smoke: bool = False) -> list[dict]:
    result = [{"phase": "synthetic", "dataset": name, "noise": noise,
               "n0": n0, "eta": 2, "swarm": 12, "budget": 8640,
               "generation_multiplier": 12, "runs": 30}
              for name, noise, n0 in product(TASKS, NOISES, INITIAL_REPEATS)]
    return [{**result[0], "runs": 1}] if smoke else result


def job_id(job: dict) -> str:
    return f"{job['dataset']}_{job['noise']}_n{job['n0']}_m12_b8640_g12"


def execute(job: dict, output: str) -> dict:
    folder = Path(output) / "synthetic" / job_id(job)
    folder.mkdir(parents=True, exist_ok=True)
    completion = folder / "complete.json"
    if completion.exists():
        saved = json.loads(completion.read_text(encoding="utf-8"))
        if saved["job"] != job or any(sha256(folder / name) != value
                                      for name, value in saved["sha256"].items()):
            raise ValueError("Completed ablation cell changed")
        return saved
    task = RobustBenchmark(job["dataset"], 2, job["noise"])
    master = 30270001 + FUNCTIONS.index(job["dataset"]) * 1000003
    rows, started = [], time.perf_counter()
    with gzip.open(folder / "evaluations.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as ev, \
            gzip.open(folder / "pools.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as po, \
            gzip.open(folder / "initial_variances.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as va:
        events, pools, variances = csv.writer(ev), csv.writer(po), csv.writer(va)
        events.writerow(["method", "run", "call", "generation", "pool_id", "seed", "score", "reason"])
        pools.writerow(["method", "run", "pool_id", "generation", "particle", "coordinates", "true_score"])
        variances.writerow(["method", "run", "pool_id", "n0", "estimated_variance", "true_variance", "ratio"])
        for run, method in product(range(job["runs"]), METHODS):
            states, true_variances, ratios = {}, {}, []

            def pool_writer(values):
                coordinates = np.asarray(json.loads(values[3]))
                true_variances[values[0]] = (2 * (0.6 + 0.8 * coordinates[0])) ** 2
                pools.writerow([method, run, *values])

            def event_writer(values):
                identifier, score = values[2], values[4]
                state = states.setdefault(identifier, [0, 0.0, 0.0])
                state[0] += 1
                delta = score - state[1]
                state[1] += delta / state[0]
                state[2] += delta * (score - state[1])
                if state[0] == job["n0"]:
                    estimate = state[2] / (job["n0"] - 1)
                    ratio = estimate / true_variances[identifier]
                    ratios.append(ratio)
                    variances.writerow([method, run, identifier, job["n0"], estimate,
                                        true_variances[identifier], ratio])
                events.writerow([method, run, *values])

            result = run_search(task, method, master + 10003 * run, job["budget"],
                                event_writer, pool_writer, swarm_size=12,
                                initial_repeats=job["n0"], generation_multiplier=12)
            if result["calls"] != 8640 or result["generations"] != 60 or len(ratios) != 720:
                raise ValueError("Ablation budget, generations or initial sample count mismatch")
            reconstructed = states[result["selected_pool"]][1]
            if abs(reconstructed - result["selected_observed_score"]) > 1e-11:
                raise ValueError("Selected mean cannot be reconstructed")
            rows.append({**result, **job, "run": run, "master_seed": master,
                         "initial_variance_ratio_median": float(np.median(ratios)),
                         "initial_variance_ratio_p99": float(np.quantile(ratios, .99)),
                         "initial_variance_ratio_above_ten": float(np.mean(np.asarray(ratios) > 10)),
                         "initial_variance_count": len(ratios)})
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "runs.csv", index=False)
    names = ["runs.csv", "evaluations.csv.gz", "pools.csv.gz", "initial_variances.csv.gz"]
    record = {"id": job_id(job), "job": job, "task": task.metadata(),
              "searches": len(frame), "calls": int(frame.calls.sum()),
              "seconds": time.perf_counter() - started,
              "sha256": {name: sha256(folder / name) for name in names}}
    completion.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    out = args.out.resolve()
    freeze(out, args.smoke)
    if args.freeze_only:
        print(out / "protocol.json")
        return
    selected = jobs(args.smoke)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute, job, str(out)) for job in selected]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"n0 ablation {index}/{len(selected)}: {result['id']}; "
                  f"{result['searches']} searches, {result['calls']} calls", flush=True)
    frame = pd.concat([pd.read_csv(out / "synthetic" / job_id(job) / "runs.csv")
                       for job in selected], ignore_index=True)
    frame.to_csv(out / "runs.csv", index=False)
    print(f"Completed: {len(frame)} searches; {int(frame.calls.sum())} calls")


if __name__ == "__main__":
    main()
