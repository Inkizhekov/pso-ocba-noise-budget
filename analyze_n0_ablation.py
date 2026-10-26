"""Audit and analyze focused n0 ablation without expanding the primary family."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from analyze_robust_ocba_study import audit_job
from paired_mean_inference import simultaneous_mean_intervals
from run_n0_ablation import OUT, ROOT, TASKS, INITIAL_REPEATS, METHODS, jobs, job_id
from run_robust_ocba_study import NOISES, sha256
from study_statistics import validate_grid


PAIRS = (("pso_ocba", "pso_ea"), ("pso_ocba_rs", "pso_ea"),
         ("pso_ocba", "pso_ocba_rs"))


def contrast_data(raw: pd.DataFrame, historical: pd.DataFrame) -> tuple[list[dict], np.ndarray]:
    keys = ["dataset", "run", "noise", "n0", "method"]
    validate_grid(raw, list(product(TASKS, range(30), NOISES, INITIAL_REPEATS, METHODS)), keys)
    grid = pd.MultiIndex.from_product([TASKS, range(30), NOISES, INITIAL_REPEATS, METHODS], names=keys)
    scores = raw.set_index(keys).final_score.reindex(grid).to_numpy().reshape(2, 30, 4, 3, 3)
    source = historical.loc[historical.dataset.isin(TASKS) & historical.eta.eq(2)
                            & historical.swarm.eq(12) & historical.budget.eq(8640)
                            & historical.method.isin(METHODS)]
    old_keys = ["dataset", "run", "noise", "method"]
    validate_grid(source, list(product(TASKS, range(30), NOISES, METHODS)), old_keys)
    old_grid = pd.MultiIndex.from_product([TASKS, range(30), NOISES, METHODS], names=old_keys)
    old_scores = source.set_index(old_keys).final_score.reindex(old_grid).to_numpy().reshape(2, 30, 4, 3)
    matched = raw.loc[raw.n0.eq(3)].merge(source, on=old_keys, suffixes=("_new", "_old"))
    if not np.array_equal(matched.seed_new, matched.seed_old):
        raise ValueError("Historical bridge does not preserve shared seed blocks")
    specs, columns = [], []
    for first, baseline in PAIRS:
        effect = scores[..., METHODS.index(first)] - scores[..., METHODS.index(baseline)]
        old_effect = old_scores[..., METHODS.index(first)] - old_scores[..., METHODS.index(baseline)]
        for noise_index, noise in enumerate(NOISES):
            for n_index, n0 in enumerate(INITIAL_REPEATS):
                specs.append({"kind": "mean", "noise": noise, "method": first,
                              "baseline": baseline, "n0": n0})
                columns.append(effect[:, :, noise_index, n_index])
            for n_index, n0 in enumerate(INITIAL_REPEATS[1:], 1):
                specs.append({"kind": "n0_change", "noise": noise, "method": first,
                              "baseline": baseline, "n0": n0})
                columns.append(effect[:, :, noise_index, n_index] - effect[:, :, noise_index, 0])
            specs.append({"kind": "generation_bridge", "noise": noise, "method": first,
                          "baseline": baseline, "n0": 3})
            columns.append(effect[:, :, noise_index, 0] - old_effect[:, :, noise_index])
    return specs, np.stack(columns, axis=-1)


def run(out: Path, workers: int) -> dict:
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if protocol["smoke"]:
        raise ValueError("Smoke cannot enter the ablation inference")
    for name, expected in protocol["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Ablation source changed: {name}")
    selected = jobs()
    audited = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(audit_job, str(out / "synthetic" / job_id(job))) for job in selected]
        for index, future in enumerate(as_completed(futures), 1):
            audited.append(future.result())
            print(f"n0 trace audit {index}/{len(selected)}", flush=True)
    if sum(row["calls"] for row in audited) != 18662400:
        raise ValueError("Ablation call-count mismatch")
    raw = pd.read_csv(out / "runs.csv")
    if not raw.calls.eq(8640).all() or not raw.generations.eq(60).all():
        raise ValueError("Ablation failed to hold budget and generation count fixed")
    source = ROOT / protocol["historical_source"]
    if sha256(source) != protocol["historical_source_sha256"]:
        raise ValueError("Historical bridge source changed")
    specs, values = contrast_data(raw, pd.read_csv(source))
    inferred = simultaneous_mean_intervals(values, seed=11222026)
    rows = [{**spec, **{key: float(inferred[key][i]) for key in [
        "mean", "se", "ci_low", "ci_high", "adjusted_p_zero"]}} for i, spec in enumerate(specs)]
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(out / "mean_contrasts.csv", index=False)
    summary = raw.groupby(["noise", "n0", "method"], sort=True).agg(
        mean_score=("final_score", "mean"), median_initial_variance_ratio=("initial_variance_ratio_median", "mean"),
        mean_initial_variance_p99=("initial_variance_ratio_p99", "mean"),
        mean_fraction_variance_above_ten=("initial_variance_ratio_above_ten", "mean"),
        mean_new_points=("evaluated_pools", "mean"), mean_generations=("generations", "mean"),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    changes = contrasts.loc[contrasts.kind.eq("n0_change")]
    report = {"status": "focused exploratory inference conditional on Ackley and Rosenbrock",
              "searches": len(raw), "calls": int(raw.calls.sum()), "fixed_generation_count": 60,
              "bootstrap_family": len(rows), "bootstrap_repetitions": inferred["repetitions"],
              "critical_max_abs_t": inferred["critical_max_abs_t"],
              "n0_changes_significant": int(changes.adjusted_p_zero.lt(.05).sum()),
              "n0_changes_positive": int((changes.adjusted_p_zero.lt(.05) & changes["mean"].gt(0)).sum()),
              "n0_changes_negative": int((changes.adjusted_p_zero.lt(.05) & changes["mean"].lt(0)).sum()),
              "summed_worker_elapsed_hours": sum(json.loads((Path(row["folder"]) / "complete.json").read_text())["seconds"]
                                                  for row in audited) / 3600,
              "contrasts": rows}
    (out / "trace_audit.json").write_text(json.dumps(audited, indent=2), encoding="utf-8")
    (out / "analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True, sharey=True)
    for ax, pair in zip(axes, [PAIRS[0], PAIRS[2]]):
        data = contrasts.loc[contrasts.kind.eq("mean") & contrasts.method.eq(pair[0]) & contrasts.baseline.eq(pair[1])]
        for offset, (n0, marker, color) in enumerate([(3, "o", "0"), (5, "s", "0.35"), (10, "^", "0.65")]):
            block = data.loc[data.n0.eq(n0)].set_index("noise").loc[list(NOISES)]
            ys = np.arange(4) + (offset - 1) * .2
            ax.errorbar(block["mean"], ys, xerr=[block["mean"]-block.ci_low, block.ci_high-block["mean"]],
                        fmt=marker, color=color, capsize=3, label=f"n0={n0}")
        ax.axvline(0, color="0.5", linestyle="--")
        ax.set_title("PSO-OCBA minus " + ("PSO-EA" if pair[1] == "pso_ea" else "OCBA-RS"))
        ax.set_yticks(range(4), ["Gaussian", "Student t3", "Lognormal", "Contaminated"])
        ax.set_xlabel("Mean score effect; simultaneous 95% intervals")
    axes[0].invert_yaxis()
    axes[0].legend(loc="best")
    fig.savefig(out / "n0_ablation.png", dpi=220)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "contrasts"}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--audit-workers", type=int, default=6)
    args = parser.parse_args()
    if args.audit_workers < 1:
        parser.error("audit-workers must be positive")
    run(args.out.resolve(), args.audit_workers)


if __name__ == "__main__":
    main()
