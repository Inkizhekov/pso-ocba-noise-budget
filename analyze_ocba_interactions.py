"""Post-hoc paired mean contrasts of noise shape, budget and swarm size."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import product
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, t, ttest_1samp

from ocba_benchmarks import FUNCTIONS
from paired_mean_inference import simultaneous_mean_intervals
from run_robust_ocba_study import NOISES, SWARMS, BUDGETS, sha256
from study_statistics import holm_adjust, validate_grid


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/ocba_robustness_20261022"
OUT = ROOT / "results/ocba_mean_analysis_20261022"
NONNORMAL = NOISES[1:]
KEYS = ["dataset", "noise", "eta", "swarm", "budget", "run", "method"]


def contrast_specs() -> list[dict]:
    specs = [{"id": f"shape_{noise}", "kind": "shape", "noise": noise}
             for noise in NONNORMAL]
    specs += [{"id": f"shape_budget_{noise}_{budget}", "kind": "shape_budget",
               "noise": noise, "budget": budget}
              for noise, budget in product(NONNORMAL, BUDGETS[1:])]
    specs += [{"id": f"shape_swarm_{noise}_{swarm}", "kind": "shape_swarm",
               "noise": noise, "swarm": swarm}
              for noise, swarm in product(NONNORMAL, SWARMS[1:])]
    specs += [{"id": f"shape_eta_{noise}", "kind": "shape_eta", "noise": noise}
              for noise in NONNORMAL]
    specs += [{"id": f"mean_{noise}", "kind": "mean", "noise": noise}
              for noise in NOISES]
    specs += [{"id": "mean_all", "kind": "mean", "noise": "all"}]
    return specs


def paired_tensor(raw: pd.DataFrame) -> np.ndarray:
    """Return function x shared-seed x noise x eta x swarm x budget D."""
    expected = list(product(FUNCTIONS, NOISES, [1, 2], SWARMS, BUDGETS,
                            range(30), ["pso_ocba", "pso_ea"]))
    selected = raw.loc[raw.eta.gt(0) & raw.method.isin(["pso_ocba", "pso_ea"])]
    validate_grid(selected, expected, KEYS)
    pivot = selected.pivot(index=KEYS[:-1], columns="method", values="final_score")
    index = pd.MultiIndex.from_product(
        [FUNCTIONS, range(30), NOISES, [1, 2], SWARMS, BUDGETS],
        names=["dataset", "run", "noise", "eta", "swarm", "budget"],
    )
    difference = (pivot.pso_ocba - pivot.pso_ea).reorder_levels(index.names).reindex(index)
    values = difference.to_numpy().reshape(8, 30, 4, 2, 3, 3)
    if not np.isfinite(values).all():
        raise ValueError("Incomplete paired tensor")
    return values


def make_contrasts(values: np.ndarray) -> np.ndarray:
    specs, columns = contrast_specs(), []
    for spec in specs:
        kind, noise = spec["kind"], spec["noise"]
        if kind == "mean":
            part = values if noise == "all" else values[:, :, NOISES.index(noise)]
            column = part.mean(axis=tuple(range(2, part.ndim)))
        else:
            difference = values[:, :, NOISES.index(noise)] - values[:, :, 0]
            if kind == "shape_budget":
                difference = (difference[..., BUDGETS.index(spec["budget"])]
                              - difference[..., 0])
            elif kind == "shape_swarm":
                difference = (difference[:, :, :, SWARMS.index(spec["swarm"]), :]
                              - difference[:, :, :, 0, :])
            elif kind == "shape_eta":
                difference = difference[:, :, 1] - difference[:, :, 0]
            column = difference.mean(axis=tuple(range(2, difference.ndim)))
        columns.append(column)
    return np.stack(columns, axis=-1)


def diagnostics(raw: pd.DataFrame, source: Path, out: Path) -> dict:
    keys = KEYS[:5]
    rows = []
    for key, block in raw.groupby(keys, sort=True):
        pivot = block.pivot(index="run", columns="method", values="final_score")
        for baseline in ["pso", "pso_ea"]:
            differences = (pivot.pso_ocba - pivot[baseline]).to_numpy()
            low, high = np.quantile(differences, [0.25, 0.75])
            width = high - low
            outliers = int(np.count_nonzero((differences < low - 1.5 * width)
                                            | (differences > high + 1.5 * width)))
            influence = np.max(np.abs(differences - differences.mean())) / 29
            skewness = float(skew(differences, bias=False)) if np.std(differences) else 0.0
            rows.append({**dict(zip(keys, key)), "baseline": baseline,
                         "mean": float(differences.mean()), "skewness": skewness,
                         "tukey_outliers": outliers, "largest_leave_one_run_mean_shift": influence})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "paired_difference_diagnostics.csv", index=False)
    original = pd.read_csv(source / "synthetic_pairwise.csv")
    merged = original.merge(frame, on=keys + ["baseline"], validate="one_to_one")
    retained = merged.loc[merged.baseline.eq("pso_ea") & merged.practical_win]
    return {
        "contrasts": len(frame), "absolute_skewness_above_one": int(frame.skewness.abs().gt(1).sum()),
        "contrasts_with_tukey_outliers": int(frame.tukey_outliers.gt(0).sum()),
        "ea_threshold_cells": len(retained),
        "ea_cells_whose_mean_stays_above_threshold_after_any_one_run_removal":
            int((retained["mean"] - retained.largest_leave_one_run_mean_shift).ge(0.02).sum()),
        "interpretation": "descriptive diagnostics, no independent outlier removal or new primary decisions",
    }


def nn_threshold_analysis(source: Path, out: Path) -> dict:
    """Exploratory mean-threshold tests; retain the original NN family."""
    raw = pd.read_csv(source / "nn_main_runs.csv")
    records = []
    for key, group in raw.groupby(["dataset", "split_seed"]):
        pivot = group.pivot(index="run", columns="method", values="final_score")
        if len(pivot) != 456:
            raise ValueError("Unexpected confirmatory NN sample size")
        for baseline in ["pso", "pso_ea"]:
            values = (pivot.pso_ocba - pivot[baseline]).to_numpy()
            se = values.std(ddof=1) / np.sqrt(len(values))
            radius = t.ppf(1 - 0.05 / (2 * 12), len(values) - 1) * se
            records.append({"dataset": key[0], "split_seed": key[1], "baseline": baseline,
                            "mean": values.mean(), "family_ci_low": values.mean() - radius,
                            "family_ci_high": values.mean() + radius,
                            "p_threshold_raw": ttest_1samp(values, 0.01, alternative="greater").pvalue})
    frame = pd.DataFrame(records)
    frame["holm_p_threshold"] = holm_adjust(frame.p_threshold_raw)
    frame.to_csv(out / "nn_mean_thresholds.csv", index=False)
    return {"contrasts": len(frame), "threshold": 0.01,
            "threshold_rejections_holm": int(frame.holm_p_threshold.lt(.05).sum()),
            "simultaneous_upper_below_threshold": int(frame.family_ci_high.lt(.01).sum()),
            "ea_upper_below_threshold": int((frame.baseline.eq("pso_ea") & frame.family_ci_high.lt(.01)).sum()),
            "largest_upper_bound": float(frame.family_ci_high.max())}


def run(source: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    inputs = {name: sha256(source / name) for name in [
        "synthetic_runs.csv", "synthetic_pairwise.csv", "nn_main_runs.csv", "protocol.json"]}
    design = {"status": "post_hoc_existing_data", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_inputs": inputs, "source_directory": str(source), "contrasts": contrast_specs(),
              "bootstrap_repetitions": 99999, "bootstrap_seed": 10222026,
              "fixed_strata": list(FUNCTIONS), "seed_blocks_per_stratum": 30,
              "new_primary_claims": False, "source_sha256": {name: sha256(ROOT / name) for name in [
                  "analyze_ocba_interactions.py", "paired_mean_inference.py", "study_statistics.py"]}}
    path = out / "analysis_plan.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        for key in design:
            if key != "created_utc" and saved[key] != design[key]:
                raise ValueError("Post-hoc analysis plan changed; use a new output directory")
    else:
        path.write_text(json.dumps(design, indent=2), encoding="utf-8")
    raw = pd.read_csv(source / "synthetic_runs.csv")
    values = make_contrasts(paired_tensor(raw))
    inferred = simultaneous_mean_intervals(values)
    rows = []
    for index, spec in enumerate(contrast_specs()):
        row = dict(spec)
        row.update({key: float(inferred[key][index]) for key in [
            "mean", "se", "ci_low", "ci_high", "adjusted_p_zero"]})
        row["lower_above_002"] = row["ci_low"] > .02
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "mean_contrasts.csv", index=False)
    lofo = []
    for excluded, function in enumerate(FUNCTIONS):
        estimates = np.delete(values, excluded, axis=0).mean(axis=(0, 1))
        lofo.extend({"excluded_function": function, "id": spec["id"], "mean": float(estimates[i])}
                    for i, spec in enumerate(contrast_specs()))
    pd.DataFrame(lofo).to_csv(out / "leave_one_function_out.csv", index=False)
    report = {"status": "exploratory mean inference conditional on eight fixed functions",
              "simultaneous_family": 23, "bootstrap_repetitions": inferred["repetitions"],
              "critical_max_abs_t": inferred["critical_max_abs_t"], "contrasts": rows,
              "diagnostics": diagnostics(raw, source, out),
              "nn_thresholds": nn_threshold_analysis(source, out)}
    plot = frame.loc[frame.kind.eq("shape")]
    fig, ax = plt.subplots(figsize=(7.5, 3.3), constrained_layout=True)
    labels = ["Student t3 - Gaussian", "Lognormal - Gaussian", "Contaminated - Gaussian"]
    for index, row in enumerate(plot.itertuples()):
        ax.errorbar(row.mean, index, xerr=[[row.mean-row.ci_low], [row.ci_high-row.mean]],
                    fmt="o", color="black", capsize=4)
    ax.set_yticks(range(3), labels)
    ax.axvline(0, color="0.5", linestyle="--")
    ax.set_xlabel("Difference of mean (PSO-OCBA - PSO-EA) effects\nSimultaneous 95% max-t bootstrap intervals, family of 23")
    ax.invert_yaxis()
    fig.savefig(out / "noise_mean_interactions.png", dpi=220)
    plt.close(fig)
    (out / "analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"family_size": 23, "shape_contrasts": rows[:3],
                      "diagnostics": report["diagnostics"], "nn": report["nn_thresholds"]}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.source.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
