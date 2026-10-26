"""Audit and analyze the fixed OCBA extension without changing its protocol."""
from __future__ import annotations

import argparse
import gzip
import hashlib
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from ocba_benchmarks import FUNCTIONS, VALIDATION_FUNCTIONS
from ocba_engine import REAL_METHODS, SYNTHETIC_METHODS
from run_ocba_study import (BUDGETS, DATASETS, DEFAULT_OUT, DIMENSIONS,
                            NOISE_LEVELS, SPLITS, cell_id, make_jobs,
                            source_hashes)
from study_statistics import bootstrap_ci, holm_adjust, paired_effect, validate_grid

KEY = ["dataset", "dimension", "eta", "budget"]
REAL_KEY = ["dataset", "split_seed", "budget"]
NAMES = {"random": "Random", "pso": "PSO", "pso_r3": "PSO-R3",
         "pso_ea": "PSO-EA", "pso_ocba_rs": "PSO-OCBA-RS",
         "pso_ocba": "PSO-OCBA"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_results(raw: pd.DataFrame, phase: str) -> None:
    if phase == "synthetic":
        columns = KEY + ["method", "run"]
        expected = list(product(FUNCTIONS, DIMENSIONS, NOISE_LEVELS, BUDGETS,
                                SYNTHETIC_METHODS, range(30)))
    else:
        columns = REAL_KEY + ["method", "run"]
        expected = list(product(DATASETS, SPLITS, [60, 180], REAL_METHODS, range(6)))
    validate_grid(raw, expected, columns)
    if phase == "real":
        if not raw.groupby(REAL_KEY + ["run"]).assessment_seeds.nunique().eq(1).all():
            raise ValueError("Final seed panels are not paired across methods")
        if not raw.groupby(REAL_KEY + ["method"]).assessment_seeds.nunique().eq(6).all():
            raise ValueError("Final seed panels were reused across HPO runs")
        for row in raw.itertuples():
            scores = np.asarray(json.loads(row.assessment_scores))
            counts = np.asarray(json.loads(row.assessment_correct_counts))
            if (len(scores) != 3 or len(set(json.loads(row.assessment_seeds))) != 3
                    or not np.allclose(scores * row.n_test, counts, atol=1e-9)
                    or abs(counts.sum() / row.assessment_denominator - row.final_score) > 1e-12):
                raise ValueError("Invalid independent final assessment")


def summaries(raw: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, block in raw.groupby(keys + ["method"], sort=True):
        values = block.final_score.to_numpy()
        lo, hi = bootstrap_ci(values)
        row = dict(zip(keys + ["method"], key))
        row.update(n=len(values), mean=float(values.mean()), ci_low=lo, ci_high=hi,
                   std=float(values.std(ddof=1)),
                   mean_pools=float(block.evaluated_pools.mean()),
                   mean_error_fraction=float(block.error_fraction.mean()),
                   success_rate=float(np.mean(values >= -0.1)))
        rows.append(row)
    return pd.DataFrame(rows)


def comparisons(raw: pd.DataFrame, phase: str) -> pd.DataFrame:
    keys = KEY if phase == "synthetic" else REAL_KEY
    methods = SYNTHETIC_METHODS if phase == "synthetic" else REAL_METHODS
    rows = []
    for key, block in raw.groupby(keys, sort=True):
        pivot = block.pivot(index="run", columns="method", values="final_score")
        pairs = [(method, "pso") for method in methods if method != "pso"]
        pairs.append(("pso_ocba", "pso_ea"))
        denom = None if phase == "synthetic" else int(block.assessment_denominator.iloc[0])
        local = []
        for contender, baseline in pairs:
            row = dict(zip(keys, key))
            row.update(method=contender, baseline=baseline)
            row.update(paired_effect(pivot[contender].to_numpy(),
                                     pivot[baseline].to_numpy(), denom))
            local.append(row)
        family = local[:-1] if phase == "synthetic" else local
        for row, adjusted in zip(family, holm_adjust([r["wilcoxon_p_raw"] for r in family])):
            row["holm_within_cell"] = float(adjusted)
        rows.extend(local)
    result = pd.DataFrame(rows)
    if phase == "synthetic":
        primary = (result.method.eq("pso_ocba") & result.baseline.isin(["pso", "pso_ea"]))
        if primary.sum() != 384:
            raise ValueError("Incorrect primary multiple-testing family")
        result.loc[primary, "holm_primary_global"] = holm_adjust(
            result.loc[primary, "wilcoxon_p_raw"].to_numpy())
        result["practical_win"] = (result.mean_difference.ge(0.02)
                                   & result.holm_primary_global.lt(0.05))
        result["significant_loss"] = (result.mean_difference.lt(0)
                                      & result.holm_primary_global.lt(0.05))
    return result


def fixed_suite_ci(block: pd.DataFrame, value: str) -> tuple[float, float, float]:
    """Resample run seeds inside fixed function/dimension strata.

    Conditions sharing a run seed stay together. No inference over a random
    population of functions or independent noise/budget conditions is implied.
    """
    rng = np.random.default_rng(702003)
    estimates = []
    means = []
    for _, stratum in block.groupby(["dataset", "dimension"], sort=True):
        values = stratum.groupby("run", sort=True)[value].mean().to_numpy()
        means.append(float(values.mean()))
        estimates.append(values[rng.integers(0, len(values), (4000, len(values)))].mean(axis=1))
    low, high = np.quantile(np.mean(estimates, axis=0), [0.025, 0.975])
    return float(np.mean(means)), float(low), float(high)


def condition_map(raw: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot = raw.pivot(index=KEY + ["run"], columns="method", values="final_score").reset_index()
    for (eta, budget), block in pivot.groupby(["eta", "budget"], sort=True):
        for baseline in ["pso", "pso_ea"]:
            block = block.copy()
            block["gain"] = block.pso_ocba - block[baseline]
            mean, lo, hi = fixed_suite_ci(block, "gain")
            relevant = pairwise[(pairwise.eta == eta) & (pairwise.budget == budget)
                                & (pairwise.method == "pso_ocba") & (pairwise.baseline == baseline)]
            rows.append({"eta": eta, "budget": budget, "baseline": baseline,
                         "mean_difference": mean, "ci_low": lo, "ci_high": hi,
                         "practical_wins": int(relevant.practical_win.sum()),
                         "significant_losses": int(relevant.significant_loss.sum()),
                         "tasks": len(relevant)})
    return pd.DataFrame(rows)


def payoff_targets(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, block in summary.groupby(["dataset", "dimension", "eta", "method"], sort=True):
        reached = block[block.success_rate >= 0.8]
        row = dict(zip(["dataset", "dimension", "eta", "method"], key))
        row.update(minimum_tested_budget=int(reached.budget.min()) if len(reached) else None,
                   censored=not bool(len(reached)), target_regret=0.1,
                   empirical_success_threshold=0.8)
        rows.append(row)
    return pd.DataFrame(rows)


def budget_savings(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot = targets.pivot(index=["dataset", "dimension", "eta"], columns="method",
                          values="minimum_tested_budget")
    for method in SYNTHETIC_METHODS:
        if method == "pso":
            continue
        a, b = pivot[method], pivot.pso
        both = a.notna() & b.notna()
        savings = 1 - a[both] / b[both]
        rows.append({"method": method, "both_reached": int(both.sum()),
                     "method_only": int((a.notna() & b.isna()).sum()),
                     "pso_only": int((a.isna() & b.notna()).sum()),
                     "neither": int((a.isna() & b.isna()).sum()),
                     "smaller_budget": int((a[both] < b[both]).sum()),
                     "equal_budget": int((a[both] == b[both]).sum()),
                     "larger_budget": int((a[both] > b[both]).sum()),
                     "conditional_mean_saving": float(savings.mean())})
    return pd.DataFrame(rows)


def noise_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, dimension, budget), block in raw.groupby(["dataset", "dimension", "budget"]):
        p = block.pivot(index="run", columns=["method", "eta"], values="final_score")
        for method in SYNTHETIC_METHODS:
            if method in {"random", "pso"}:
                continue
            values = p[(method, 2.0)] - p[("pso", 2.0)] - p[(method, 0.0)] + p[("pso", 0.0)]
            lo, hi = bootstrap_ci(values.to_numpy())
            rows.append({"dataset": dataset, "dimension": dimension, "budget": budget,
                         "method": method, "mean_interaction": float(values.mean()),
                         "ci_low": lo, "ci_high": hi, "ci_policy": "pointwise_exploratory"})
    return pd.DataFrame(rows)


def fit_router(raw: pd.DataFrame, pairwise: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cells = pairwise[(pairwise.method == "pso_ocba") & (pairwise.baseline == "pso")].copy()
    features = pd.DataFrame({"eta": cells.eta, "log_budget": np.log(cells.budget),
                             "dimension": cells.dimension}, index=cells.index)
    training = ~cells.dataset.isin(VALIDATION_FUNCTIONS)
    model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=12, random_state=20261022)
    model.fit(features.loc[training], cells.loc[training, "mean_difference"].ge(0.02))
    cells["choose_ocba"] = model.predict(features).astype(bool)
    cells["partition"] = np.where(training, "discovery", "validation")
    validation = raw[raw.dataset.isin(VALIDATION_FUNCTIONS)].pivot(
        index=KEY + ["run"], columns="method", values="final_score").reset_index()
    validation = validation.merge(cells[KEY + ["choose_ocba"]], on=KEY, validate="many_to_one")
    validation["policy_score"] = np.where(validation.choose_ocba, validation.pso_ocba, validation.pso)
    effects = []
    for baseline in ["pso", "pso_ocba"]:
        validation["gain"] = validation.policy_score - validation[baseline]
        mean, lo, hi = fixed_suite_ci(validation, "gain")
        effects.append({"dataset": "fixed_validation_suite", "dimension": 0,
                        "baseline": baseline, "mean_difference": mean, "ci_low": lo, "ci_high": hi})
        for (dataset, dimension), block in validation.groupby(["dataset", "dimension"]):
            values = block.groupby("run").gain.mean().to_numpy()
            lo, hi = bootstrap_ci(values)
            effects.append({"dataset": dataset, "dimension": dimension,
                            "baseline": baseline, "mean_difference": float(values.mean()),
                            "ci_low": lo, "ci_high": hi})
    info = {"rule": export_text(model, feature_names=list(features.columns)),
            "training_cells": int(training.sum()), "validation_cells": int((~training).sum()),
            "training_positive_labels": int(cells.loc[training, "mean_difference"].ge(0.02).sum()),
            "validation_choose_ocba": int(cells.loc[~training, "choose_ocba"].sum()),
            "validation_correct_labels": int((cells.loc[~training, "choose_ocba"] ==
                                                cells.loc[~training, "mean_difference"].ge(0.02)).sum()),
            "feature_names": list(features.columns),
            "tree": {"children_left": model.tree_.children_left.tolist(),
                     "children_right": model.tree_.children_right.tolist(),
                     "feature": model.tree_.feature.tolist(),
                     "threshold": model.tree_.threshold.tolist(),
                     "value": model.tree_.value.tolist()},
            "ci_policy": "pointwise stratified bootstrap of run blocks, fixed validation functions"}
    return info, cells, pd.DataFrame(effects)


def audit_archives(out: Path) -> dict:
    """Check all call counts and hashes, reconstruct selected sample means.

    pandas chunks retain every evaluation, including adaptive repeats. Pool
    links and final selected coordinates are checked in every search.
    """
    files = {}
    data_root = Path(__file__).resolve().parent / "datasets/uci"
    dataset_inputs = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    for name, metadata in dataset_inputs.items():
        if sha256(data_root / f"{name}.csv") != metadata["sha256"]:
            raise ValueError(f"UCI input changed: {name}")
    calls = {"synthetic": 0, "real": 0}
    for phase in calls:
        for job in make_jobs(phase):
            directory = out / phase / cell_id(job)
            info = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
            if info["job"] != job:
                raise ValueError("Archive job does not match the protocol")
            endpoints = pd.read_csv(directory / "runs.csv")
            pools = pd.read_csv(directory / "pools.csv.gz")
            events = pd.read_csv(directory / "evaluations.csv.gz")
            keys = ["method", "run"]
            counts = events.groupby(keys)["call"].agg(["size", "min", "max", "nunique"])
            if not (counts["size"].eq(job["budget"]).all()
                    and counts["min"].eq(1).all() and counts["max"].eq(job["budget"]).all()
                    and counts["nunique"].eq(job["budget"]).all()):
                raise ValueError(f"Broken call accounting in {directory}")
            if len(events) != info["calls"]:
                raise ValueError("Call count differs from completion marker")
            means = events.groupby(keys + ["pool_id"]).score.mean().rename("reconstructed_mean").reset_index()
            links = means.merge(pools, on=keys + ["pool_id"], how="left", validate="one_to_one", indicator=True)
            if not links._merge.eq("both").all():
                raise ValueError("Evaluation refers to a missing pool")
            selected = endpoints.merge(links, left_on=keys + ["selected_pool"],
                                       right_on=keys + ["pool_id"], validate="one_to_one")
            if len(selected) != len(endpoints) or not np.allclose(
                    selected.selected_observed_score, selected.reconstructed_mean, atol=1e-10, rtol=1e-10):
                raise ValueError("Selected endpoint differs from its recorded samples")
            for row in selected.itertuples():
                if not np.allclose(json.loads(row.selected_coordinates), json.loads(row.coordinates), atol=1e-14):
                    raise ValueError("Selected coordinates differ from pool log")
            if phase == "synthetic" and not np.allclose(selected.final_score, selected.true_score, atol=1e-10):
                raise ValueError("Final oracle score differs from selected pool")
            calls[phase] += len(events)
            for filename in ["runs.csv", "evaluations.csv.gz", "pools.csv.gz", "complete.json"]:
                path = directory / filename
                files[str(path.relative_to(out)).replace("\\", "/")] = sha256(path)
    return {"calls": calls, "files_sha256": files, "dataset_inputs": dataset_inputs,
            "all_selected_pools_reconstructed": True}


def figures(out: Path, condition: pd.DataFrame, summary: pd.DataFrame,
            pairwise: pd.DataFrame) -> None:
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    vmax = float(condition.mean_difference.abs().max())
    for ax, baseline in zip(axes, ["pso", "pso_ea"]):
        block = condition[condition.baseline == baseline]
        matrix = block.pivot(index="eta", columns="budget", values="mean_difference")
        image = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i, eta in enumerate(matrix.index):
            for j, budget in enumerate(matrix.columns):
                row = block[(block.eta == eta) & (block.budget == budget)].iloc[0]
                ax.text(j, i, f"{row.mean_difference:+.3f}\n{int(row.practical_wins)}/16 wins", ha="center", va="center",
                        color="white" if abs(row.mean_difference) > 0.6 * vmax else "black", fontsize=9)
        ax.set(xticks=range(3), xticklabels=BUDGETS, yticks=range(4), yticklabels=NOISE_LEVELS,
               xlabel="Evaluation budget", ylabel="Noise eta",
               title=f"PSO-OCBA minus {NAMES[baseline]}")
    fig.colorbar(image, ax=axes, shrink=0.8, label="Normalized score gain")
    fig.savefig(directory / "payoff_map.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), constrained_layout=True)
    for ax, dimension in zip(axes, DIMENSIONS):
        block = pairwise[(pairwise.dimension == dimension) & (pairwise.budget == 540)
                         & (pairwise.method == "pso_ocba") & (pairwise.baseline == "pso")]
        matrix = block.pivot(index="dataset", columns="eta", values="mean_difference").reindex(FUNCTIONS)
        limit = max(0.01, float(pairwise[pairwise.method.eq("pso_ocba") & pairwise.baseline.eq("pso")
                                             & pairwise.budget.eq(540)].mean_difference.abs().max()))
        im = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        for i in range(8):
            for j in range(4):
                val = matrix.iloc[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > .6 * limit else "black")
        ax.set(xticks=range(4), xticklabels=NOISE_LEVELS, yticks=range(8),
               yticklabels=[s.title() for s in FUNCTIONS], xlabel="Noise eta", title=f"d = {dimension}, B = 540")
    fig.colorbar(im, ax=axes, shrink=.85, label="PSO-OCBA minus PSO")
    fig.savefig(directory / "task_heterogeneity.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-audit", action="store_true", help="Reuse a completed immutable audit")
    args = parser.parse_args()
    out = args.out
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("smoke") or protocol["source_sha256"] != source_hashes():
        raise ValueError("Only the full study with unchanged sources can be analyzed")
    syn = pd.read_csv(out / "synthetic_runs.csv")
    real = pd.read_csv(out / "real_runs.csv")
    validate_results(syn, "synthetic")
    validate_results(real, "real")
    print("Validated complete endpoint grids", flush=True)
    audit_path = out / "audit.json"
    if args.skip_audit:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        audit = audit_archives(out)
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Audited all trace links and selected means", flush=True)
    syn_summary, real_summary = summaries(syn, KEY), summaries(real, REAL_KEY)
    syn_pairs, real_pairs = comparisons(syn, "synthetic"), comparisons(real, "real")
    condition = condition_map(syn, syn_pairs)
    targets = payoff_targets(syn_summary)
    savings = budget_savings(targets)
    interactions = noise_interactions(syn)
    router, router_cells, router_effects = fit_router(syn, syn_pairs)
    exports = {"synthetic_summary": syn_summary, "real_summary": real_summary,
               "synthetic_pairwise": syn_pairs, "real_pairwise": real_pairs,
               "condition_map": condition, "budget_targets": targets, "budget_savings": savings,
               "noise_interactions": interactions, "router_cells": router_cells,
               "router_effects": router_effects}
    for name, frame in exports.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "router.json").write_text(json.dumps(router, indent=2), encoding="utf-8")
    figures(out, condition, syn_summary, syn_pairs)
    real_overview = real.groupby(["dataset", "budget", "method"]).final_score.mean().reset_index()
    real_overview.to_csv(out / "real_overview.csv", index=False)
    payload = {"protocol": protocol, "n_synthetic": len(syn), "n_real": len(real),
               "search_calls": int(syn.calls.sum() + real.calls.sum()), "final_refits": 3 * len(real),
               "router": router, "real_overview": real_overview.to_dict("records"),
               "counts": {baseline: {"wins": int(block.practical_win.sum()),
                                      "losses": int(block.significant_loss.sum()),
                                      "positive_means": int(block.mean_difference.gt(0).sum())}
                          for baseline, block in syn_pairs[syn_pairs.method.eq("pso_ocba")].groupby("baseline")},
               **{name: frame.replace({np.nan: None}).to_dict("records") for name, frame in exports.items()}}
    (out / "article_data.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    manifest_names = ["protocol.json", "synthetic_runs.csv", "real_runs.csv", "audit.json",
                      "article_data.json", *[f"{name}.csv" for name in exports],
                      "real_overview.csv", "router.json", "figures/payoff_map.png", "figures/task_heterogeneity.png",
                      "../../datasets/uci/manifest.json", "../../datasets/uci/ionosphere.csv", "../../datasets/uci/sonar.csv"]
    manifest = {name: sha256(out / name) for name in manifest_names}
    manifest["../../analyze_ocba_study.py"] = sha256(Path(__file__))
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["n_synthetic", "n_real", "search_calls", "final_refits", "counts"]}, indent=2))
    print(condition.to_string(index=False))
    print(router["rule"])
    print(router_effects.to_string(index=False))


if __name__ == "__main__":
    main()
