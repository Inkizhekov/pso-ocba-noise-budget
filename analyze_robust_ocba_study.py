"""Audit complete robustness archives and analyze the frozen comparisons."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import binomtest, t, ttest_1samp, wilcoxon

from ocba_benchmarks import ControlledBenchmark
from run_robust_ocba_study import (
    BUDGETS, DEFAULT_OUT, NN_DATASETS, NN_METHODS, NN_SPLITS, NOISES,
    SYNTH_METHODS, SWARMS, RobustBenchmark, job_id, jobs, sha256, source_hashes,
)
from study_statistics import bootstrap_ci, holm_adjust, paired_effect, validate_grid


KEYS = ["dataset", "noise", "eta", "swarm", "budget"]
NN_KEYS = ["dataset", "split_seed"]


def benchmark_metadata_matches(saved: dict, expected: dict) -> bool:
    if saved.keys() != expected.keys():
        return False
    for name, value in expected.items():
        if name in {"rotation", "scale"}:
            # QR and calibration reductions can differ by rounding across BLAS CPUs.
            try:
                actual, reference = np.asarray(saved[name]), np.asarray(value)
                if (actual.shape != reference.shape
                        or not np.issubdtype(actual.dtype, np.number)
                        or not np.allclose(actual, reference, rtol=1e-12, atol=1e-14)):
                    return False
            except (ValueError, TypeError):
                return False
        elif saved[name] != value:
            return False
    return True


def check_benchmark_metadata(benchmark: RobustBenchmark, saved: dict) -> bool:
    if benchmark_metadata_matches(saved, benchmark.metadata()):
        return False
    # Initial archives inherited the Gaussian label even for other noise families.
    legacy = {**ControlledBenchmark.metadata(benchmark), "noise_family": benchmark.noise,
              "noise_mean": 0, "noise_variance_before_heteroskedastic_factor": 1}
    if benchmark_metadata_matches(saved, legacy):
        return True
    raise ValueError("Benchmark transformation or calibration changed")


def audit_job(folder_name: str) -> dict:
    folder = Path(folder_name)
    metadata = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    for name, expected in metadata["sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Archive changed: {folder / name}")
    frame = pd.read_csv(folder / "runs.csv")
    benchmark = None
    legacy_noise_description = False
    if metadata["job"]["phase"] == "synthetic":
        job = metadata["job"]
        benchmark = RobustBenchmark(job["dataset"], job["eta"], job["noise"])
        legacy_noise_description = check_benchmark_metadata(benchmark, metadata["task"])
    rows = {(row.method, int(row.run)): row for row in frame.itertuples()}
    pool_ids = {key: set() for key in rows}
    selected_coordinates = {}
    selected_truth = {}
    with gzip.open(folder / "pools.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key, identifier = (row["method"], int(row["run"])), int(row["pool_id"])
            if key not in rows or identifier in pool_ids[key]:
                raise ValueError("Unknown search or duplicate pool")
            pool_ids[key].add(identifier)
            if identifier == rows[key].selected_pool:
                selected_coordinates[key] = json.loads(row["coordinates"])
                selected_truth[key] = float(row["true_score"])
    counts = {key: 0 for key in rows}
    selected = {key: [0, 0.0] for key in rows}
    with gzip.open(folder / "evaluations.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            key = row[0], int(row[1])
            call, identifier = int(row[2]), int(row[4])
            if key not in rows or identifier not in pool_ids[key] or call != counts[key] + 1:
                raise ValueError("Broken evaluation sequence or pool link")
            counts[key] = call
            if identifier == rows[key].selected_pool:
                value = float(row[6])
                state = selected[key]
                state[0] += 1
                state[1] += (value - state[1]) / state[0]
    for key, row in rows.items():
        if counts[key] != row.budget or selected[key][0] == 0:
            raise ValueError("Incorrect evaluation budget or missing selected pool")
        if abs(selected[key][1] - row.selected_observed_score) > 1e-11:
            raise ValueError("Selected mean cannot be reconstructed")
        if not np.allclose(selected_coordinates[key], json.loads(row.selected_coordinates), rtol=0, atol=1e-14):
            raise ValueError("Selected coordinates disagree with pool archive")
        if metadata["job"]["phase"] == "synthetic":
            if abs(selected_truth[key] - row.final_score) > 1e-11:
                raise ValueError("Synthetic final score differs from pool truth")
            recomputed = benchmark.true_score(np.asarray(selected_coordinates[key]))
            if abs(recomputed-row.final_score) > 1e-11:
                raise ValueError("Synthetic score cannot be recomputed from coordinates")
        else:
            scores = np.asarray(json.loads(row.assessment_scores))
            correct = np.asarray(json.loads(row.assessment_correct_counts))
            seeds = json.loads(row.assessment_seeds)
            if (len(scores) != 3 or len(set(seeds)) != 3
                    or not np.allclose(scores * row.n_test, correct, atol=1e-9, rtol=0)
                    or abs(correct.sum()/row.assessment_denominator - row.final_score) > 1e-12):
                raise ValueError("Invalid final NN refits or accuracy lattice")
    return {"folder": folder.as_posix(), "searches": len(frame),
            "legacy_noise_description": legacy_noise_description,
            "calls": sum(counts.values()), "selected_means_reconstructed": len(frame),
            "completion_sha256": sha256(folder / "complete.json"),
            "sha256": metadata["sha256"]}


def audit_phase(phase: str, out: Path, workers: int) -> dict:
    phase_jobs = jobs(phase, out)
    cache_path = out / f"{phase}_trace_audit.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(cached["jobs"]) == len(phase_jobs):
            expected_jobs = {job_id(job): job for job in phase_jobs}
            for item in cached["jobs"]:
                folder = Path(item["folder"])
                completion = folder / "complete.json"
                record = json.loads(completion.read_text(encoding="utf-8"))
                if (record["job"] != expected_jobs[folder.name]
                        or record["sha256"] != item["sha256"]):
                    raise ValueError("Completion metadata disagree with frozen job or audit")
                digest = sha256(completion)
                if item.get("completion_sha256", digest) != digest:
                    raise ValueError("Previously audited completion metadata changed")
                item["completion_sha256"] = digest
                for name, expected in item["sha256"].items():
                    if sha256(folder / name) != expected:
                        raise ValueError("Previously audited archive changed")
            cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
            return cached
    folders = [out / phase / job_id(job) for job in phase_jobs]
    if not all((folder / "complete.json").exists() for folder in folders):
        raise ValueError(f"Incomplete phase: {phase}")
    audited = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(audit_job, str(folder)) for folder in folders]
        for future in as_completed(futures):
            audited.append(future.result())
            if len(audited) % 48 == 0 or len(audited) == len(folders):
                print(f"audit {phase}: {len(audited)}/{len(folders)}", flush=True)
    result = {"jobs": audited, "searches": sum(row["searches"] for row in audited),
              "calls": sum(row["calls"] for row in audited)}
    cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def synthetic_analysis(raw: pd.DataFrame, out: Path) -> dict:
    conditions = [("gaussian", 0)] + list(product(NOISES, [1, 2]))
    expected = [(job["dataset"], job["noise"], job["eta"], job["swarm"], job["budget"], m, r)
                for job in jobs("synthetic", out) for m, r in product(SYNTH_METHODS, range(30))]
    validate_grid(raw, expected, KEYS + ["method", "run"])
    if len(conditions) != 9 or len(raw) != 77760 or int(raw.calls.sum()) != 293932800:
        raise ValueError("Incorrect robustness design totals")
    rows, summaries = [], []
    for key, block in raw.groupby(KEYS, sort=True):
        pivot = block.pivot(index="run", columns="method", values="final_score")
        for baseline in ["pso", "pso_ea", "pso_ocba_rs"]:
            result = dict(zip(KEYS, key))
            result.update(baseline=baseline)
            result.update(paired_effect(pivot.pso_ocba.to_numpy(), pivot[baseline].to_numpy()))
            difference = np.round(pivot.pso_ocba.to_numpy()-pivot[baseline].to_numpy(), 12)
            nonzero = difference[difference != 0]
            positives = int(np.count_nonzero(nonzero > 0))
            result["sign_positive_pairs"] = positives
            result["sign_nonzero_pairs"] = len(nonzero)
            result["sign_p_raw"] = float(binomtest(positives, len(nonzero), 0.5).pvalue) if len(nonzero) else 1.0
            rows.append(result)
        for method, group in block.groupby("method"):
            summaries.append({**dict(zip(KEYS, key)), "method": method,
                              "mean_score": float(group.final_score.mean()),
                              "mean_regret": float(-group.final_score.mean()),
                              "success_fraction": float(group.final_score.ge(-0.1).mean()),
                              "mean_new_points": float(group.evaluated_pools.mean())})
    pairs = pd.DataFrame(rows)
    primary = pairs.baseline.isin(["pso", "pso_ea"])
    if int(primary.sum()) != 1296 or int((~primary).sum()) != 648:
        raise ValueError("Incorrect new comparison families")
    pairs.loc[primary, "holm_p"] = holm_adjust(pairs.loc[primary, "wilcoxon_p_raw"])
    pairs.loc[~primary, "holm_p"] = holm_adjust(pairs.loc[~primary, "wilcoxon_p_raw"])
    pairs.loc[primary, "holm_sign_sensitivity"] = holm_adjust(pairs.loc[primary, "sign_p_raw"])
    pairs.loc[~primary, "holm_sign_sensitivity"] = holm_adjust(pairs.loc[~primary, "sign_p_raw"])
    pairs["practical_win"] = pairs.mean_difference.ge(0.02) & pairs.holm_p.lt(0.05)
    pairs["significant_loss"] = pairs.mean_difference.lt(0) & pairs.holm_p.lt(0.05)
    pairs.to_csv(out / "synthetic_pairwise.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "synthetic_summary.csv", index=False)
    # Functions are fixed strata. Shared seeds stay paired within each stratum.
    suite = []
    for key, block in raw.groupby(["noise", "eta", "swarm", "budget"], sort=True):
        pivot = block.pivot(index=["dataset", "run"], columns="method", values="final_score")
        for baseline in ["pso", "pso_ea", "pso_ocba_rs"]:
            diff = (pivot.pso_ocba - pivot[baseline]).unstack("run").to_numpy()
            rng = np.random.default_rng(50319)
            estimates = np.zeros(5000)
            for values in diff:
                estimates += values[rng.integers(0, 30, (5000, 30))].mean(axis=1)/8
            lo, hi = np.quantile(estimates, [0.025, 0.975])
            mask = pairs.baseline.eq(baseline)
            for name, value in zip(["noise", "eta", "swarm", "budget"], key):
                mask &= pairs[name].eq(value)
            tests = pairs.loc[mask]
            suite.append({**dict(zip(["noise", "eta", "swarm", "budget"], key)),
                          "baseline": baseline, "mean_difference": float(diff.mean()),
                          "ci_low": float(lo), "ci_high": float(hi),
                          "practical_wins": int(tests.practical_win.sum()),
                          "significant_losses": int(tests.significant_loss.sum()),
                          "fixed_tasks": 8})
    suite_frame = pd.DataFrame(suite)
    suite_frame.to_csv(out / "suite_effects.csv", index=False)
    counts = pairs.groupby(["noise", "baseline"]).agg(
        comparisons=("baseline", "size"), wins=("practical_win", "sum"),
        losses=("significant_loss", "sum")).reset_index()
    counts.to_csv(out / "synthetic_counts.csv", index=False)
    eta_counts = pairs.groupby(["eta", "baseline"]).agg(
        comparisons=("baseline", "size"), wins=("practical_win", "sum"),
        losses=("significant_loss", "sum")).reset_index()
    eta_counts.to_csv(out / "synthetic_eta_counts.csv", index=False)
    noiseless = raw.loc[raw.eta.eq(0)].pivot(
        index=["dataset", "swarm", "budget", "run"], columns="method", values="final_score")
    if (not np.array_equal(noiseless.pso_ea.to_numpy(), noiseless.pso_ocba.to_numpy())
            or (noiseless.pso_ocba > noiseless.pso + 1e-12).any()):
        raise ValueError("Noiseless trajectory control failed")
    interactions = []
    for key, block in raw.groupby(["noise", "eta", "swarm"], sort=True):
        pivot = block.pivot(index=["dataset", "run"], columns=["method", "budget"], values="final_score")
        for baseline in ["pso", "pso_ea"]:
            contrast = (pivot[("pso_ocba", 8640)] - pivot[(baseline, 8640)]
                        - pivot[("pso_ocba", 540)] + pivot[(baseline, 540)])
            matrix = contrast.unstack("run").to_numpy()
            rng, estimates = np.random.default_rng(78131), np.zeros(5000)
            for values in matrix:
                estimates += values[rng.integers(0, 30, (5000, 30))].mean(axis=1)/8
            lo, hi = np.quantile(estimates, [0.025, 0.975])
            interactions.append({"noise": str(key[0]), "eta": int(key[1]), "swarm": int(key[2]),
                                 "baseline": baseline, "mean_difference": float(matrix.mean()),
                                 "ci_low": float(lo), "ci_high": float(hi),
                                 "status": "exploratory pointwise; paired across budgets"})
    pd.DataFrame(interactions).to_csv(out / "budget_interactions.csv", index=False)
    summary_frame = pd.DataFrame(summaries)
    targets = []
    for key, block in summary_frame.groupby(["dataset", "noise", "eta", "swarm", "method"]):
        hit = block.loc[block.success_fraction.ge(0.8), "budget"]
        targets.append({**dict(zip(["dataset", "noise", "eta", "swarm", "method"], key)),
                        "minimum_tested_budget": int(hit.min()) if len(hit) else None,
                        "right_censored_after": 8640 if not len(hit) else None})
    pd.DataFrame(targets).to_csv(out / "budget_targets.csv", index=False)
    return {"searches": len(raw), "calls": int(raw.calls.sum()),
            "primary_family": 1296, "secondary_family": 648,
            "counts": counts.to_dict("records"),
            "eta_counts": eta_counts.to_dict("records"),
            "sign_sensitivity_status": "additional exploratory exact sign tests; original primary decisions unchanged",
            "ea_wins_retained_by_sign_test": int((pairs.baseline.eq("pso_ea") & pairs.practical_win
                & pairs.holm_sign_sensitivity.lt(0.05)
                & pairs.sign_positive_pairs.gt(pairs.sign_nonzero_pairs/2)).sum()),
            "budget_interactions": interactions,
            "ea_wins": pairs.loc[pairs.baseline.eq("pso_ea") & pairs.practical_win].to_dict("records")}


def nn_analysis(raw: pd.DataFrame, out: Path, plan: dict) -> dict:
    n = plan["runs_per_cell"]
    expected = list(product(NN_DATASETS, NN_SPLITS, NN_METHODS, range(n)))
    validate_grid(raw, expected, NN_KEYS + ["method", "run"])
    if not raw.power_plan_sha256.eq(sha256(out / "power_plan.json")).all():
        raise ValueError("NN data refer to another power plan")
    if not raw.groupby(NN_KEYS + ["run"]).assessment_seeds.nunique().eq(1).all():
        raise ValueError("Final seeds are not paired across NN methods")
    if not raw.groupby(NN_KEYS + ["method"]).assessment_seeds.nunique().eq(n).all():
        raise ValueError("Final seed panels are not independent between searches")
    rows, summary = [], []
    for key, block in raw.groupby(NN_KEYS):
        pivot = block.pivot(index="run", columns="method", values="final_score")
        denominator = int(block.assessment_denominator.iloc[0])
        for baseline in ["pso", "pso_ea"]:
            differences = (np.rint(pivot.pso_ocba.to_numpy()*denominator)
                           - np.rint(pivot[baseline].to_numpy()*denominator))/denominator
            mean, std = float(differences.mean()), float(differences.std(ddof=1))
            se = std / np.sqrt(n)
            p = float(ttest_1samp(differences, 0).pvalue) if std else (1.0 if mean == 0 else 0.0)
            critical = float(t.ppf(0.975, n-1))
            bonferroni = float(t.ppf(1-0.05/(2*12), n-1))
            rank_p = 1.0 if np.all(differences == 0) else float(wilcoxon(differences).pvalue)
            rows.append({**dict(zip(NN_KEYS, key)), "baseline": baseline, "n_pairs": n,
                         "mean_difference": mean, "std_difference": std,
                         "ci_low": mean-critical*se, "ci_high": mean+critical*se,
                         "family_ci_low": mean-bonferroni*se,
                         "family_ci_high": mean+bonferroni*se,
                         "paired_t_p_raw": p, "wilcoxon_sensitivity_p_raw": rank_p})
        for method, group in block.groupby("method"):
            summary.append({**dict(zip(NN_KEYS, key)), "method": method,
                            "n_searches": n, "mean_accuracy": float(group.final_score.mean()),
                            "std_accuracy": float(group.final_score.std(ddof=1))})
    pairs = pd.DataFrame(rows)
    if len(pairs) != 12:
        raise ValueError("NN primary family must contain 12 comparisons")
    pairs["holm_p"] = holm_adjust(pairs.paired_t_p_raw)
    pairs["holm_wilcoxon_sensitivity"] = holm_adjust(pairs.wilcoxon_sensitivity_p_raw)
    pairs["significant"] = pairs.holm_p.lt(0.05)
    pairs["practical_win"] = pairs.significant & pairs.mean_difference.ge(0.01)
    pairs.to_csv(out / "nn_pairwise.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "nn_summary.csv", index=False)
    return {"searches": len(raw), "calls": int(raw.calls.sum()), "final_refits": 3*len(raw),
            "runs_per_cell": n, "primary_family": 12,
            "significant_comparisons": int(pairs.significant.sum()),
            "practical_wins": int(pairs.practical_win.sum()),
            "one_pp_gain_excluded_by_simultaneous_ci": int(pairs.family_ci_high.lt(0.01).sum()),
            "comparisons": pairs.to_dict("records")}


def plots(out: Path, include_nn: bool = True) -> None:
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    data = pd.read_csv(out / "suite_effects.csv")
    data = data.loc[data.eta.eq(2) & data.baseline.isin(["pso", "pso_ea"])]
    limit = max(0.02, float(data.mean_difference.abs().max()))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    plt.rcParams.update({"font.size": 12})
    noise_labels = {"gaussian": "Gaussian", "student_t3": "Student t3",
                    "lognormal": "Centered lognormal", "contaminated": "Contaminated"}
    fig, axes = plt.subplots(4, 2, figsize=(10, 11), constrained_layout=True)
    for col, baseline in enumerate(["pso", "pso_ea"]):
        for row, noise in enumerate(NOISES):
            ax = axes[row, col]
            block = data.loc[data.baseline.eq(baseline) & data.noise.eq(noise)]
            values = block.pivot(index="swarm", columns="budget", values="mean_difference").reindex(index=SWARMS, columns=BUDGETS).to_numpy()
            drawing = ax.imshow(values, cmap="cividis", norm=norm, aspect="auto")
            for i, swarm in enumerate(SWARMS):
                for j, budget in enumerate(BUDGETS):
                    item = block.loc[block.swarm.eq(swarm) & block.budget.eq(budget)].iloc[0]
                    value = values[i, j]
                    if abs(value) < 1e-12:
                        ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor="white",edgecolor="gray",hatch="xxx"))
                    elif abs(value) < .02:
                        ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor="gray",hatch="//"))
                    label = f"{value:+.1e}" if 1e-12 < abs(value) < .0005 else f"{value:+.3f}"
                    ax.text(j, i, f"{label}\n{int(item.practical_wins)}/8", ha="center", va="center", fontsize=11,
                            color="black" if value >= 0 or abs(value)<1e-12 else "white",
                            bbox={"facecolor":"white" if value>=0 else "#172a43", "alpha":.72, "edgecolor":"none", "pad":1})
            ax.set_xticks(range(3), BUDGETS)
            ax.set_yticks(range(3), SWARMS)
            ax.set_title(f"{noise_labels[noise]}; vs {'PSO' if baseline=='pso' else 'PSO-EA'}", fontsize=12)
            ax.set_xlabel("Evaluation budget")
            if col == 0:
                ax.set_ylabel("Swarm size")
    bar = fig.colorbar(drawing, ax=axes, shrink=.8)
    bar.set_label("PSO-OCBA minus control; normalized score")
    bar.ax.axhline(0, color="black", linewidth=1.5)
    fig.savefig(figures / "robustness_map.png", dpi=220)
    plt.close(fig)
    if not include_nn:
        return
    nn = pd.read_csv(out / "nn_pairwise.csv")
    fig, ax = plt.subplots(figsize=(9.8, 4.6), constrained_layout=True)
    labels = []
    for i, (key, block) in enumerate(nn.groupby(NN_KEYS, sort=True)):
        labels.append(f"{key[0].title()} / split {key[1]}")
        for baseline, offset, color, marker in [("pso", -.12, "#0072B2", "o"), ("pso_ea", .12, "#D55E00", "s")]:
            row = block.loc[block.baseline.eq(baseline)].iloc[0]
            x = 100*row.mean_difference
            errors = np.array([[x-100*row.ci_low], [100*row.ci_high-x]])
            ax.errorbar(x, i+offset, xerr=errors, fmt=marker, color=color,
                        markerfacecolor=color if row.holm_p<.05 else "white", capsize=3,
                        label=("vs PSO" if baseline=="pso" else "vs PSO-EA") if i==0 else None)
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(1, color="0.4", linestyle=":", linewidth=1,
               label="Planning effect: 1 percentage point")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Paired mean test accuracy difference, percentage points\nPointwise 95% confidence intervals")
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(figures / "nn_confirmation.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit-workers", type=int, default=6)
    parser.add_argument("--audit-phase", choices=["synthetic", "nn_pilot", "nn_main"])
    args = parser.parse_args()
    out = args.out.resolve()
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("smoke") or protocol["source_sha256"] != source_hashes():
        raise ValueError("Smoke results or changed sources cannot enter the main analysis")
    if args.audit_phase:
        audit_phase(args.audit_phase, out, args.audit_workers)
        return
    audits = {phase: audit_phase(phase, out, args.audit_workers)
              for phase in ["synthetic", "nn_pilot", "nn_main"]}
    plan = json.loads((out / "power_plan.json").read_text(encoding="utf-8"))
    if plan["protocol_sha256"] != sha256(out / "protocol.json") or plan["pilot_runs_sha256"] != sha256(out / "nn_pilot_runs.csv"):
        raise ValueError("Power-plan provenance mismatch")
    validation = json.loads((out / "power_validation.json").read_text(encoding="utf-8"))
    if validation["power_plan_sha256"] != sha256(out / "power_plan.json"):
        raise ValueError("Independent power validation refers to another plan")
    validator_path = Path(__file__).resolve().parent / "validate_robust_power.py"
    if (validation["validator_sha256"] != sha256(validator_path)
            or validation["pilot_sha256"] != sha256(out / "nn_pilot_runs.csv")
            or validation["runs_per_cell"] != plan["runs_per_cell"]):
        raise ValueError("Independent power-validation provenance mismatch")
    data_manifest = json.loads((Path(__file__).resolve().parent / "datasets/uci/manifest.json").read_text(encoding="utf-8"))
    data_inputs = {}
    for name in NN_DATASETS:
        path = Path(__file__).resolve().parent / f"datasets/uci/{name}.csv"
        actual = sha256(path)
        if actual != data_manifest[name]["sha256"]:
            raise ValueError(f"Dataset changed: {name}")
        data_inputs[str(path)] = actual
    (out / "dataset_inputs.json").write_text(json.dumps(data_inputs, indent=2), encoding="utf-8")
    canonical_tasks = []
    for job in jobs("synthetic", out):
        raw_metadata = json.loads((out / "synthetic" / job_id(job) / "complete.json").read_text(encoding="utf-8"))
        task = dict(raw_metadata["task"])
        # The parent benchmark retains its Gaussian template description.
        # Preserve that signed raw field, and publish the effective sampler
        # separately, using the actual frozen job and protocol definitions.
        task["inherited_gaussian_template"] = task.pop("noise")
        task["noise"] = "eta*(0.6+0.8*x[0])*Z; Z=" + protocol["synthetic"]["noise_distributions"][job["noise"]]
        canonical_tasks.append({"id": job_id(job), "task": task})
    (out / "canonical_task_metadata.json").write_text(json.dumps({
        "note": "Effective distributions are defined by noise_family, the frozen job and protocol. Raw inherited Gaussian template fields are retained unchanged in signed complete.json files.",
        "tasks": canonical_tasks,
    }, indent=2), encoding="utf-8")
    synthetic = synthetic_analysis(pd.read_csv(out / "synthetic_runs.csv"), out)
    nn = nn_analysis(pd.read_csv(out / "nn_main_runs.csv"), out, plan)
    plots(out)
    report = {"synthetic": synthetic, "nn": nn, "power_plan": plan,
              "power_validation": validation,
              "runtime_environment": json.loads((out / "runtime_environment.json").read_text(encoding="utf-8")),
              "pilot_searches": audits["nn_pilot"]["searches"],
              "pilot_calls": audits["nn_pilot"]["calls"],
              "all_searches": synthetic["searches"]+nn["searches"]+audits["nn_pilot"]["searches"],
              "all_calls": synthetic["calls"]+nn["calls"]+audits["nn_pilot"]["calls"],
              "fixed_legacy_primary_family_preserved": True}
    (out / "article_data.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    names = ["protocol.json", "power_plan.json", "power_validation.json", "runtime_environment.json", "dataset_inputs.json", "canonical_task_metadata.json",
             "synthetic_runs.csv", "nn_pilot_runs.csv", "nn_main_runs.csv",
             "synthetic_pairwise.csv", "synthetic_summary.csv", "synthetic_counts.csv", "synthetic_eta_counts.csv", "suite_effects.csv",
             "budget_interactions.csv", "budget_targets.csv",
             "nn_pairwise.csv", "nn_summary.csv", "article_data.json", "synthetic_trace_audit.json",
             "nn_pilot_trace_audit.json", "nn_main_trace_audit.json",
             "figures/robustness_map.png", "figures/nn_confirmation.png"]
    manifest = {name: sha256(out / name) for name in names}
    manifest[str(Path(__file__).resolve())] = sha256(Path(__file__))
    validator = Path(__file__).resolve().parent / "validate_robust_power.py"
    manifest[str(validator)] = sha256(validator)
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"synthetic_searches": synthetic["searches"], "nn_searches": nn["searches"],
                      "nn_significant": nn["significant_comparisons"], "all_calls": report["all_calls"]}, indent=2))


if __name__ == "__main__":
    main()
