"""Independently validate the frozen NN sample size without changing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from plan_robust_nn_power import paired_t_pvalue
from run_robust_ocba_study import DEFAULT_OUT, sha256


def wilson(successes: int, trials: int, z: float) -> tuple[float, float]:
    p = successes / trials
    center = p + z*z/(2*trials)
    spread = z*np.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))
    denominator = 1+z*z/trials
    return float((center-spread)/denominator), float((center+spread)/denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    plan = json.loads((out / "power_plan.json").read_text(encoding="utf-8"))
    pilot = pd.read_csv(out / "nn_pilot_runs.csv")
    n, sd = plan["runs_per_cell"], plan["worst_upper_sd"]
    rng = np.random.default_rng(80126319)
    models = {
        "normal_upper_sd": lambda shape: rng.normal(0, sd, shape),
        "student_t3_upper_sd": lambda shape: rng.standard_t(3, shape) * sd/np.sqrt(3),
    }
    index = 0
    for _, block in pilot.groupby(["dataset", "split_seed"]):
        pivot = block.pivot(index="run", columns="method", values="final_score")
        for baseline in ["pso", "pso_ea"]:
            values = (pivot.pso_ocba-pivot[baseline]).to_numpy()
            # A resampled empirical law uses population, rather than sample,
            # variance. Match that law exactly to the conservative upper SD.
            std = values.std(ddof=0)
            centered = (values-values.mean())/std if std else np.array([-1.0, 1.0])
            models[f"centered_pilot_{index}"] = lambda shape, v=centered: rng.choice(v, shape)*sd
            index += 1
    repetitions = 50000
    alpha = plan["conservative_per_test_alpha"]
    z = float(norm.ppf(1-0.05/(2*len(models))))
    results = []
    for label, generate in models.items():
        positive, null = 0, 0
        for _ in range(repetitions//1000):
            values = generate((1000, n))
            positive += int(np.count_nonzero(paired_t_pvalue(values+0.01) <= alpha))
            null += int(np.count_nonzero(paired_t_pvalue(values) <= alpha))
        lo, hi = wilson(positive, repetitions, z)
        null_lo, null_hi = wilson(null, repetitions, z)
        results.append({"model": label, "power": positive/repetitions,
                        "simultaneous_mc_power_low": lo, "simultaneous_mc_power_high": hi,
                        "null_rejection_rate": null/repetitions,
                        "simultaneous_mc_null_low": null_lo, "simultaneous_mc_null_high": null_hi})
    report = {"runs_per_cell": n, "mc_repetitions_per_model": repetitions,
              "validation_seed": 80126319, "models": results,
              "power_plan_sha256": sha256(out / "power_plan.json"),
              "pilot_sha256": sha256(out / "nn_pilot_runs.csv"),
              "validator_sha256": sha256(Path(__file__)),
              "main_results_used": False, "sample_size_changed": False,
              "mc_interval_family_confidence": 0.95,
              "empirical_law_variance_standardization_ddof": 0,
              "minimum_power_lower_bound": min(row["simultaneous_mc_power_low"] for row in results),
              "maximum_null_rejection_rate": max(row["null_rejection_rate"] for row in results)}
    (out / "power_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["runs_per_cell", "minimum_power_lower_bound",
                                                 "maximum_null_rejection_rate"]}, indent=2))


if __name__ == "__main__":
    main()
