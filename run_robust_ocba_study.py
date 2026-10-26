"""Run a frozen robustness grid and an independently powered NN confirmation."""
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
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from ocba_benchmarks import ControlledBenchmark, FUNCTIONS
from robust_ocba_engine import run_search
from run_ocba_study import load_real


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results/ocba_current"
NOISES = ("gaussian", "student_t3", "lognormal", "contaminated")
SYNTH_METHODS = ("pso", "pso_ea", "pso_ocba_rs", "pso_ocba")
NN_METHODS = ("pso", "pso_ea", "pso_ocba")
NN_DATASETS = ("ionosphere", "sonar")
NN_SPLITS = (7101, 7102, 7103)
SWARMS = (6, 12, 24)
BUDGETS = (540, 2160, 8640)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    names = ["run_robust_ocba_study.py", "robust_ocba_engine.py", "plan_robust_nn_power.py", "ocba.py",
             "ocba_engine.py", "ocba_benchmarks.py", "experiment.py",
             "run_ocba_study.py", "study_statistics.py",
             "analyze_robust_ocba_study.py", "validate_robust_power.py",
             "record_runtime_environment.py",
             "historical/__init__.py", "historical/ocba_20261022/__init__.py",
             "historical/ocba_20261022/experiment.py"]
    return {name: sha256(ROOT / name) for name in names}


class RobustBenchmark(ControlledBenchmark):
    def __init__(self, name: str, eta: float, noise: str):
        super().__init__(name, 8, eta)
        if noise not in NOISES:
            raise ValueError(f"Unknown noise: {noise}")
        self.noise = noise

    def draw_from_truth(self, x: np.ndarray, truth: float, seed: int) -> float:
        rng = np.random.default_rng(seed)
        if self.noise == "gaussian":
            noise = rng.normal()
        elif self.noise == "student_t3":
            noise = rng.standard_t(3) / np.sqrt(3)
        elif self.noise == "lognormal":
            noise = (np.exp(rng.normal()) - np.exp(0.5)) / np.sqrt(np.e * (np.e - 1))
        else:
            multiplier = 10.0 if rng.random() < 0.05 else 1.0
            noise = multiplier * rng.normal() / np.sqrt(0.95 + 0.05 * 100)
        return truth + self.eta * (0.6 + 0.8 * float(x[0])) * float(noise)

    def metadata(self) -> dict:
        return {**super().metadata(), "noise_family": self.noise,
                "noise_mean": 0, "noise_variance_before_heteroskedastic_factor": 1}


def protocol(smoke: bool = False) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes(),
        "smoke": smoke,
        "synthetic": {
            "functions": FUNCTIONS, "dimension": 8, "swarms": SWARMS,
            "budgets": BUDGETS, "methods": SYNTH_METHODS, "runs": 30,
            "noise_conditions": [["gaussian", 0]] + [[n, eta] for n in NOISES for eta in [1, 2]],
            "noise_distributions": {
                "gaussian": "N(0,1)", "student_t3": "t(df=3)/sqrt(3)",
                "lognormal": "(exp(N(0,1))-exp(1/2))/sqrt(e*(e-1))",
                "contaminated": "(.95*N(0,1)+.05*N(0,100))/sqrt(5.95)",
            },
            "heteroskedastic_factor": "eta*(0.6+0.8*x[0])",
            "primary_family": "1296 two-sided OCBA-vs-PSO/EA comparisons, one global Holm",
            "secondary_family": "648 OCBA-vs-OCBA-RS comparisons, separate global Holm",
            "practical_delta": 0.02,
        },
        "nn": {
            "datasets": NN_DATASETS, "split_seeds": NN_SPLITS,
            "methods": NN_METHODS, "swarm_size": 12, "budget": 540,
            "epochs": 20, "final_refits": 3, "pilot_runs_per_cell": 24,
            "practical_delta_accuracy": 0.01, "target_power": 0.90,
            "global_primary_family_size": 12,
            "contrasts": ["pso_ocba-pso", "pso_ocba-pso_ea"],
            "primary_test": "two-sided paired Student t test on mean test accuracy",
            "power_rule": "worst 95% upper variance bound across 12 pilot contrasts; noncentral t at alpha=.05/12; MC normal, t3 and centered pilot distributions; min 48, multiples 24, max 1536",
            "pilot_is_confirmatory_data": False,
            "main_seeds_disjoint_from_pilot_and_historical": True,
            "scope": "conditional on each fixed dataset/split; splits are not independent datasets",
        },
        "engine": {"n0": 3, "generation_budget": "6*M", "allocation_increment": "M",
                   "historical_default_equivalence_required": True},
        "historical_primary_family_384_unchanged": True,
        "seed_pairing": "shared search seed within each condition; paired final seeds across methods",
        "raw_archives": "all evaluations and pool coordinates, plus selected-mean reconstruction per search",
    }


def ensure_protocol(out: Path, smoke: bool = False) -> None:
    if smoke and "smoke" not in out.name:
        raise ValueError("Smoke requires a separately named smoke folder")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    expected = json.loads(json.dumps(protocol(smoke=smoke)))
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in saved.items() if k != "created_utc"} != {
                k: v for k, v in expected.items() if k != "created_utc"}:
            raise ValueError("Protocol, sources or smoke mode changed; use a new output folder. "
                             "For archived studies use historical/ocba_20261022 scripts.")
    else:
        if any(out.iterdir()):
            raise ValueError("Refusing to freeze a nonempty directory without a protocol")
        path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    runtime = out / "runtime_environment.json"
    if not runtime.exists():
        from record_runtime_environment import record_environment

        record_environment(out)


def jobs(phase: str, out: Path) -> list[dict]:
    if phase == "synthetic":
        conditions = [("gaussian", 0)] + list(product(NOISES, [1, 2]))
        return [{"phase": phase, "dataset": f, "noise": n, "eta": eta,
                 "swarm": m, "budget": b, "runs": 30}
                for f, (n, eta), m, b in product(FUNCTIONS, conditions, SWARMS, BUDGETS)]
    if phase in {"nn_pilot", "nn_main"}:
        count = 24 if phase == "nn_pilot" else int(json.loads(
            (out / "power_plan.json").read_text(encoding="utf-8"))["runs_per_cell"])
        plan_hash = sha256(out / "power_plan.json") if phase == "nn_main" else "pilot"
        return [{"phase": phase, "dataset": f, "split_seed": split,
                 "swarm": 12, "budget": 540, "runs": 1, "run": run,
                 "power_plan_sha256": plan_hash}
                for f, split, run in product(NN_DATASETS, NN_SPLITS, range(count))]
    raise ValueError(phase)


def job_id(job: dict) -> str:
    if job["phase"] == "synthetic":
        return f"{job['dataset']}_{job['noise']}_e{job['eta']}_m{job['swarm']}_b{job['budget']}"
    return f"{job['dataset']}_s{job['split_seed']}_r{job['run']:04d}"


def execute(job: dict, output: str) -> dict:
    folder = Path(output) / job["phase"] / job_id(job)
    folder.mkdir(parents=True, exist_ok=True)
    completion = folder / "complete.json"
    if completion.exists():
        return json.loads(completion.read_text(encoding="utf-8"))
    synthetic = job["phase"] == "synthetic"
    if synthetic:
        task = RobustBenchmark(job["dataset"], job["eta"], job["noise"])
        master = 30270001 + FUNCTIONS.index(job["dataset"]) * 1000003
        methods = SYNTH_METHODS
        runs = range(job["runs"])
        metadata = task.metadata()
    else:
        task = load_real(job["dataset"], job["split_seed"])
        master = (230290001 if job["phase"] == "nn_pilot" else 430290001)
        master += NN_DATASETS.index(job["dataset"]) * 1000003 + NN_SPLITS.index(job["split_seed"]) * 100003
        methods, runs = NN_METHODS, [job["run"]]
        metadata = {"sizes": task.sizes, "epochs": task.epochs}
    started, rows = time.perf_counter(), []
    with gzip.open(folder / "evaluations.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as ev, \
            gzip.open(folder / "pools.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as po:
        events, pools = csv.writer(ev), csv.writer(po)
        events.writerow(["method", "run", "call", "generation", "pool_id", "seed", "score", "reason"])
        pools.writerow(["method", "run", "pool_id", "generation", "particle", "coordinates", "true_score"])
        for run in runs:
            for method in methods:
                observations: dict[int, list[float]] = {}
                calls = 0

                def event_writer(values):
                    nonlocal calls
                    calls += 1
                    state = observations.setdefault(values[2], [0, 0.0])
                    state[0] += 1
                    state[1] += (values[4] - state[1]) / state[0]
                    events.writerow([method, run, *values])

                def pool_writer(values):
                    pools.writerow([method, run, *values])

                result = run_search(task, method, master + 10003 * run,
                                    job["budget"], event_writer, pool_writer,
                                    swarm_size=job["swarm"])
                reconstructed = observations[result["selected_pool"]][1]
                if calls != job["budget"] or abs(reconstructed - result["selected_observed_score"]) > 1e-12:
                    raise ValueError("Budget or selected-mean reconstruction failed")
                rows.append({**result, **job, "run": run, "dimension": 8 if synthetic else 3,
                             "master_seed": master, "reconstructed_mean": reconstructed,
                             "reconstructed_calls": calls})
    frame = pd.DataFrame(rows)
    if len(frame) != job["runs"] * len(methods) or not frame.calls.eq(job["budget"]).all():
        raise ValueError("Incomplete job")
    temporary = folder / "runs.partial.csv"
    frame.to_csv(temporary, index=False)
    temporary.replace(folder / "runs.csv")
    files = ["runs.csv", "evaluations.csv.gz", "pools.csv.gz"]
    result = {"id": job_id(job), "job": job, "task": metadata,
              "searches": len(frame), "calls": int(frame.calls.sum()),
              "seconds": time.perf_counter() - started,
              "sha256": {name: sha256(folder / name) for name in files}}
    completion.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_phase(phase: str, out: Path, workers: int, smoke: bool = False) -> None:
    if smoke and phase not in {"synthetic", "nn_pilot"}:
        raise ValueError("Smoke supports only synthetic or nn_pilot")
    ensure_protocol(out, smoke=smoke)
    if phase == "nn_main":
        plan = json.loads((out / "power_plan.json").read_text(encoding="utf-8"))
        if plan["protocol_sha256"] != sha256(out / "protocol.json"):
            raise ValueError("Power plan refers to a different protocol")
    selected = jobs(phase, out)
    if smoke:
        if "smoke" not in out.name:
            raise ValueError("Smoke requires a separately named smoke folder")
        selected = [{**selected[-1], "runs": 1}]
    started = time.perf_counter()
    done, searches, calls = 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as executor, \
            (out / f"{phase}_progress.jsonl").open("a", encoding="utf-8") as log:
        futures = [executor.submit(execute, job, str(out)) for job in selected]
        for future in as_completed(futures):
            result = future.result()
            done += 1
            searches += result["searches"]
            calls += result["calls"]
            log.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                  "id": result["id"], "searches": result["searches"],
                                  "calls": result["calls"], "seconds": result["seconds"]}) + "\n")
            log.flush()
            if done % (12 if phase == "synthetic" else 24) == 0 or done == len(selected):
                print(f"{phase}: {done}/{len(selected)} jobs; {searches} searches; "
                      f"{calls} calls; {time.perf_counter()-started:.1f}s", flush=True)
    frames = [pd.read_csv(out / phase / job_id(job) / "runs.csv") for job in selected]
    pd.concat(frames, ignore_index=True).to_csv(out / f"{phase}_runs.csv", index=False)
    print(f"Completed {phase}: searches={searches}, calls={calls}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["synthetic", "nn_pilot", "nn_main", "freeze"], required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.phase == "freeze":
        ensure_protocol(args.out, smoke=args.smoke)
        print(f"Frozen {args.out / 'protocol.json'}")
    else:
        run_phase(args.phase, args.out, args.workers, args.smoke)


if __name__ == "__main__":
    main()
