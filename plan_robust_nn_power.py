"""Freeze the confirmatory NN sample size using only an independent pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, nct, t

from run_robust_ocba_study import DEFAULT_OUT, sha256


def paired_t_pvalue(values: np.ndarray) -> np.ndarray:
    n = values.shape[-1]
    mean, std = values.mean(axis=-1), values.std(axis=-1, ddof=1)
    statistic = np.divide(mean * np.sqrt(n), std, out=np.zeros_like(mean), where=std > 0)
    p = 2 * t.sf(np.abs(statistic), n - 1)
    return np.where(std > 0, p, np.where(mean == 0, 1.0, 0.0))


def lower_wilson(successes: int, trials: int) -> float:
    z = 1.959963984540054
    p = successes / trials
    return float((p + z*z/(2*trials) - z*np.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)))
                 / (1 + z*z/trials))


def plan(out: Path) -> dict:
    if (out / "power_plan.json").exists():
        raise FileExistsError("Power plan is already frozen")
    if (out / "nn_main").exists():
        raise ValueError("Main observations exist before power planning")
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    pilot = pd.read_csv(out / "nn_pilot_runs.csv")
    expected = 2 * 3 * 3 * 24
    if len(pilot) != expected or pilot.duplicated(["dataset", "split_seed", "method", "run"]).any():
        raise ValueError("Incomplete or duplicated pilot")
    alpha, delta = 0.05 / 12, 0.01
    contrasts, centered = [], []
    for (dataset, split), block in pilot.groupby(["dataset", "split_seed"]):
        pivot = block.pivot(index="run", columns="method", values="final_score")
        for baseline in ["pso", "pso_ea"]:
            differences = (pivot.pso_ocba - pivot[baseline]).to_numpy()
            std = float(differences.std(ddof=1))
            upper = std * np.sqrt(23 / chi2.ppf(0.05, 23))
            contrasts.append({"dataset": dataset, "split_seed": int(split),
                              "baseline": baseline, "n": 24, "pilot_sd": std,
                              "upper_sd_normal_model_95": float(upper)})
            centered.append(differences - differences.mean())
    worst_sd = max(row["upper_sd_normal_model_95"] for row in contrasts)
    # All contrasts use a common n. This avoids allocating more runs to a
    # promising pilot effect; no pilot mean enters the sample-size rule.
    candidates = list(range(48, 1537, 24))
    initial = None
    for n in candidates:
        critical = t.ppf(1-alpha/2, n-1)
        noncentrality = delta * np.sqrt(n) / worst_sd
        power = nct.sf(critical, n-1, noncentrality) + nct.cdf(-critical, n-1, noncentrality)
        if power >= 0.90:
            initial = n
            break
    if initial is None:
        raise ValueError("1536 runs do not attain planned power; revise design before main data")
    rng, checks = np.random.default_rng(613271), []
    n = initial
    while True:
        models = []
        generators = {
            "normal_upper_sd": lambda shape: rng.normal(0, worst_sd, shape),
            "student_t3_upper_sd": lambda shape: rng.standard_t(3, shape) * worst_sd / np.sqrt(3),
        }
        for i, values in enumerate(centered):
            scale = max(worst_sd, float(values.std(ddof=1)))
            normalized = values / values.std(ddof=1) if values.std(ddof=1) > 0 else np.array([-1.0, 1.0])
            generators[f"centered_pilot_{i}"] = lambda shape, v=normalized, s=scale: rng.choice(v, shape) * s
        for label, generate in generators.items():
            successes = 0
            for _ in range(10):
                values = generate((1000, n)) + delta
                successes += int(np.count_nonzero(paired_t_pvalue(values) <= alpha))
            lower = lower_wilson(successes, 10000)
            models.append({"model": label, "power": successes/10000,
                           "power_lower_95_wilson": lower})
        checks.append({"n": n, "models": models})
        if min(row["power_lower_95_wilson"] for row in models) >= 0.90:
            break
        n += 24
        if n > 1536:
            raise ValueError("Monte Carlo power criterion not met before maximum n")
    report = {
        "protocol_sha256": sha256(out / "protocol.json"),
        "pilot_runs_sha256": sha256(out / "nn_pilot_runs.csv"),
        "planner_sha256": sha256(Path(__file__)),
        "runs_per_cell": n, "pilot_searches": len(pilot),
        "main_searches": 2 * 3 * 3 * n,
        "main_search_calls": 2 * 3 * 3 * n * 540,
        "main_final_refits": 2 * 3 * 3 * n * 3,
        "family_size": 12, "conservative_per_test_alpha": alpha,
        "delta_accuracy": delta, "target_marginal_power": 0.90,
        "worst_upper_sd": worst_sd, "pilot_contrasts": contrasts,
        "normal_model_initial_n": initial, "monte_carlo_checks": checks,
        "main_data_used_in_planning": False,
        "limitations": "Model-based planning conditional on pilot variance; not guaranteed achieved power or a guarantee of significance; 90% marginal, not joint power.",
    }
    (out / "power_plan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = plan(args.out)
    print(json.dumps({key: result[key] for key in ["runs_per_cell", "main_searches",
                                                 "worst_upper_sd", "main_search_calls"]}, indent=2))


if __name__ == "__main__":
    main()
