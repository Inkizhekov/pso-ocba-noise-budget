"""Post-hoc sensitivity of existing OCBA results; no optimization is run."""
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
from scipy.stats import norm

from analyze_ocba_interactions import contrast_specs, make_contrasts, paired_tensor
from ocba_benchmarks import FUNCTIONS
from paired_mean_inference import simultaneous_mean_intervals, stratified_mean_se
from run_ocba_instance_validation import CONTRASTS, INSTANCES, METHODS, NOISES
from run_robust_ocba_study import sha256
from study_statistics import validate_grid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/ocba_scope_20261023"
SOURCE = ROOT / "results/ocba_instances_20261022"


def instance_tensor(raw: pd.DataFrame) -> np.ndarray:
    """Return 8 families x 2 instances x 30 seed blocks x 5 contrasts."""
    keys = ["dataset", "instance", "run", "noise", "method"]
    grid = list(product(FUNCTIONS, INSTANCES, range(30), NOISES, METHODS))
    validate_grid(raw, grid, keys)
    index = pd.MultiIndex.from_tuples(grid, names=keys)
    data = raw.set_index(keys).final_score.reindex(index).to_numpy()
    scores = data.reshape(8, 2, 30, 2, 3)
    ea, rs = scores[..., 2] - scores[..., 0], scores[..., 2] - scores[..., 1]
    return np.stack([ea[..., 0], ea[..., 1], ea[..., 1] - ea[..., 0],
                     rs[..., 0], rs[..., 1]], axis=-1)


def bootstrap_draws(data: np.ndarray, repetitions: int, seed: int) -> dict:
    """Retain paired fixed-stratum draws for conditional sensitivity/MDE."""
    mean, se = stratified_mean_se(data)
    if repetitions < 999 or np.any(se <= 0):
        raise ValueError("Nondegenerate contrasts and >=999 draws required")
    tasks, runs, contrasts = data.shape
    rng = np.random.default_rng(seed)
    noise = np.empty((repetitions, contrasts))
    errors = np.empty_like(noise)
    for start in range(0, repetitions, 1000):
        size = min(1000, repetitions - start)
        weights = rng.multinomial(runs, np.full(runs, 1 / runs), (size, tasks))
        sums = np.einsum("bfr,frk->bfk", weights, data, optimize=True)
        squares = np.einsum("bfr,frk->bfk", weights, data * data, optimize=True)
        noise[start:start + size] = sums.sum(axis=1) / (tasks * runs) - mean
        variance = np.maximum(squares - sums * sums / runs, 0) / (runs - 1)
        errors[start:start + size] = np.sqrt(
            variance.sum(axis=1) / (tasks * tasks * runs))
    statistics = np.divide(np.abs(noise), errors, out=np.full_like(noise, np.inf),
                           where=errors > 0)
    maxima = statistics.max(axis=1)
    critical = float(np.quantile(maxima, .95, method="higher"))
    if not np.isfinite(critical):
        raise ValueError("Degenerate bootstrap family")
    return {"mean": mean, "se": se, "noise": noise, "boot_se": errors,
            "maxima": maxima, "critical": critical}


def conditional_mde(draws: dict, columns: list[int], specs: list[dict]) -> list[dict]:
    """Conditional location-shift sensitivity, not achieved or planned power.

    Recenter within fixed tasks; retain the paired empirical residual law.
    A hypothetical positive shift delta is detected when
    (bootstrap_noise + delta) / bootstrap_SE > fixed_family_critical.
    Delta90 is the 90th percentile of critical*SE - noise. The opposite
    direction uses +noise. The original 23-family critical stays fixed.
    """
    c = draws["critical"]
    rows = []
    for index in columns:
        positive = c * draws["boot_se"][:, index] - draws["noise"][:, index]
        negative = c * draws["boot_se"][:, index] + draws["noise"][:, index]
        rows.append({**specs[index], "se": float(draws["se"][index]),
                     "mean": float(draws["mean"][index]),
                     "mde90_positive": float(np.quantile(positive, .9, method="higher")),
                     "mde90_negative": float(np.quantile(negative, .9, method="higher")),
                     "normal_location_approximation": float((c + norm.ppf(.9)) * draws["se"][index])})
    return rows


def hierarchical_sensitivity(values: np.ndarray, repetitions: int, seed: int) -> dict:
    """Exploratory three-level bootstrap under exchangeable-family scenario.

    Resample families, instances within each selected family, and complete
    seed vectors within each selected instance. Studentize by the SD of
    the eight resampled family means / sqrt(8). These eight chosen families
    are not a probability sample: this does NOT establish population coverage.
    """
    if values.ndim != 4 or values.shape[:2] != (8, 2):
        raise ValueError("Expected eight families with two instances each")
    families, instances, runs, contrasts = values.shape
    mean = values.mean(axis=(0, 1, 2))
    se = values.mean(axis=(1, 2)).std(axis=0, ddof=1) / np.sqrt(families)
    if np.any(se <= 0):
        raise ValueError("Degenerate family variability")
    rng = np.random.default_rng(seed)
    maxima = np.empty(repetitions)
    for start in range(0, repetitions, 250):
        size = min(250, repetitions - start)
        selected_f = rng.integers(families, size=(size, families, 1, 1))
        selected_i = rng.integers(instances, size=(size, families, instances, 1))
        selected_r = rng.integers(runs, size=(size, families, instances, runs))
        sample = values[selected_f, selected_i, selected_r]
        family_means = sample.mean(axis=(2, 3))
        boot_mean = family_means.mean(axis=1)
        boot_se = family_means.std(axis=1, ddof=1) / np.sqrt(families)
        statistics = np.divide(np.abs(boot_mean - mean), boot_se,
                               out=np.full_like(boot_mean, np.inf), where=boot_se > 0)
        maxima[start:start + size] = statistics.max(axis=1)
    critical = float(np.quantile(maxima, .95, method="higher"))
    return {"mean": mean.tolist(), "se": se.tolist(),
            "ci_low": (mean - critical * se).tolist(),
            "ci_high": (mean + critical * se).tolist(),
            "critical": critical, "repetitions": repetitions, "seed": seed,
            "family_size": contrasts,
            "scope": "exploratory exchangeability scenario, eight selected families; not a probability sample or a population CI"}


def run(out: Path, repetitions: int = 99999) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    inputs = [SOURCE / "runs.csv", SOURCE / "protocol.json",
              ROOT / "results/ocba_robustness_20261022/synthetic_runs.csv",
              ROOT / "results/ocba_mean_analysis_20261022/mean_contrasts.csv"]
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "post_hoc_sensitivity_of_existing_observations_no_new_searches",
            "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs},
            "source_sha256": sha256(Path(__file__)), "repetitions": repetitions,
            "fixed_validation": "16 fixed instances, 30 independent paired seeds per instance",
            "lofo_family": "40 contrasts: 8 excluded whole function families x 5 original contrasts, jointly corrected",
            "hierarchical": "resample 8 families, 2 instances per draw, 30 full seed vectors; exchangeability sensitivity only",
            "mde": "conditional empirical location-shift simulation; fixed 23-family critical; target rejection probability .90; not prospective power or equivalence",
            "threshold": "compare existing simultaneous lower bounds of five mean contrasts to .02; cell counts remain descriptive",
            "new_primary_claims": False}
    path = out / "analysis_plan.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if {k: v for k, v in plan.items() if k != "created_utc"} != {
                k: v for k, v in saved.items() if k != "created_utc"}:
            raise ValueError("Analysis changed; choose a new output directory")
    else:
        path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    values = instance_tensor(pd.read_csv(SOURCE / "runs.csv"))
    per_instance = [{"dataset": f, "instance": inst, **{
        name: float(values[i, j, :, k].mean()) for k, name in enumerate(CONTRASTS)}}
        for i, f in enumerate(FUNCTIONS) for j, inst in enumerate(INSTANCES)]
    instance_frame = pd.DataFrame(per_instance)
    instance_frame.to_csv(out / "per_instance_effects.csv", index=False)
    instance_frame.groupby("dataset", sort=False)[list(CONTRASTS)].mean().to_csv(
        out / "per_family_effects.csv")
    # Normalize the 14 retained strata to equal 1/14, while retaining the
    # 16-stratum shared resampling representation and zeroing the excluded two.
    lofo = np.concatenate([values.reshape(16, 30, 5) * np.repeat(
        (np.arange(8) != f).astype(float), 2)[:, None, None] * (16 / 14)
        for f in range(8)], axis=-1)
    inferred = simultaneous_mean_intervals(lofo, repetitions=repetitions, seed=13232026)
    rows = [{"excluded": function, "id": name, **{
        key: float(inferred[key][f * 5 + k]) for key in ("mean", "se", "ci_low", "ci_high")}}
        for f, function in enumerate(FUNCTIONS) for k, name in enumerate(CONTRASTS)]
    pd.DataFrame(rows).to_csv(out / "leave_one_family_out.csv", index=False)
    hierarchy = hierarchical_sensitivity(values, repetitions, 14232026)
    pd.DataFrame([{"id": name, **{key: hierarchy[key][k] for key in
                  ("mean", "se", "ci_low", "ci_high")}} for k, name in enumerate(CONTRASTS)]).to_csv(
                      out / "hierarchical_sensitivity.csv", index=False)
    original = pd.read_csv(ROOT / "results/ocba_robustness_20261022/synthetic_runs.csv")
    original_contrasts = make_contrasts(paired_tensor(original))
    draws = bootstrap_draws(original_contrasts, repetitions, 10222026)
    archived = pd.read_csv(ROOT / "results/ocba_mean_analysis_20261022/mean_contrasts.csv")
    np.testing.assert_allclose(draws["mean"], archived["mean"], rtol=0, atol=2e-14)
    specs = contrast_specs()
    mde = conditional_mde(draws, [i for i, spec in enumerate(specs)
                                 if spec["kind"] == "shape_swarm"], specs)
    pd.DataFrame(mde).to_csv(out / "swarm_mde90.csv", index=False)
    sensitivity = []
    for n in (9999, repetitions):
        c = float(np.quantile(draws["maxima"][:n], .95, method="higher"))
        for i, spec in enumerate(specs):
            sensitivity.append({"id": spec["id"], "repetitions": n, "critical": c,
                                "mean": draws["mean"][i],
                                "ci_low": draws["mean"][i] - c * draws["se"][i],
                                "ci_high": draws["mean"][i] + c * draws["se"][i]})
    pd.DataFrame(sensitivity).to_csv(out / "bootstrap_repetitions.csv", index=False)
    threshold = archived.loc[archived.kind.eq("mean")].copy()
    threshold["simultaneous_lower_above_002"] = threshold.ci_low.gt(.02)
    threshold.to_csv(out / "mean_threshold_bounds.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for i, inst in enumerate(INSTANCES):
        part = instance_frame.loc[instance_frame.instance.eq(inst)].set_index("dataset").loc[list(FUNCTIONS)]
        ax.scatter(part.noise_interaction, np.arange(8) + (i - .5) * .22,
                   marker=("o", "x")[i], s=42, color=("#0072B2", "#D55E00")[i], label=f"Экземпляр {inst}")
    ax.axvline(0, color="black", lw=.8)
    ax.axvline(values[..., 2].mean(), color="black", ls="--", lw=1, label="Среднее 16 экземпляров")
    ax.set_yticks(np.arange(8), [str(f).capitalize() for f in FUNCTIONS])
    ax.set_xlabel("Разность эффектов OCBA - EA: логнормальный минус нормальный шум")
    ax.set_title("Описательные средние по 30 парным seed для каждого экземпляра", fontsize=10)
    ax.grid(axis="x", alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "instance_effects.png", dpi=240)
    plt.close(fig)
    interaction_lofo = [r for r in rows if r["id"] == "noise_interaction"]
    summary = {"status": plan["status"], "searches_added": 0,
               "per_instance_interaction_range": [float(values.mean(axis=2)[..., 2].min()),
                                                   float(values.mean(axis=2)[..., 2].max())],
               "lofo_interaction_mean_range": [min(r["mean"] for r in interaction_lofo), max(r["mean"] for r in interaction_lofo)],
               "lofo_interaction_lowest_joint_lower": min(r["ci_low"] for r in interaction_lofo),
               "lofo_joint_family_size": 40, "lofo_critical": inferred["critical_max_abs_t"],
               "hierarchical": hierarchy,
               "swarm_mde90": mde,
               "mean_lower_above_002": int(threshold.simultaneous_lower_above_002.sum()),
               "bootstrap_critical_9999": sensitivity[0]["critical"],
               "bootstrap_critical_99999": draws["critical"],
               "maximum_ci_endpoint_change": float(max(
                   abs(sensitivity[i]["ci_low"] - sensitivity[len(specs) + i]["ci_low"])
                   for i in range(len(specs)))),
               "scope": "conditional fixed-grid inference; hierarchical and MDE calculations are exploratory sensitivity only"}
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.out.resolve())


if __name__ == "__main__":
    main()
