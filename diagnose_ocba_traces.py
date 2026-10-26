"""Reconstruct record choices and convergence from archived calls only."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
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
from scipy.stats import spearmanr

from run_robust_ocba_study import sha256

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/ocba_instances_20261022"
OUT = ROOT / "results/ocba_scope_20261023"


def reconstruct_cell(folder_name: str) -> tuple[list[dict], list[dict]]:
    """Replay Welford means and strict record comparisons, without objective calls."""
    folder = Path(folder_name)
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if any(sha256(folder / name) != digest for name, digest in complete["sha256"].items()):
        raise ValueError(f"Changed trace: {folder.name}")
    raw = pd.read_csv(folder / "runs.csv")
    records = {(r.method, int(r.run)): r for r in raw.itertuples()}
    pools: dict[tuple[str, int], dict[int, dict]] = {key: {} for key in records}
    generations: dict[tuple[str, int], dict[int, list[int]]] = {key: {} for key in records}
    with gzip.open(folder / "pools.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key, pool, generation = (row["method"], int(row["run"])), int(row["pool_id"]), int(row["generation"])
            pools[key][pool] = {"mean": 0., "count": 0, "truth": float(row["true_score"]),
                                "particle": int(row["particle"])}
            generations[key].setdefault(generation, []).append(pool)
    curves, diagnoses = [], []
    key = None
    generation = -1
    personal: list[int | None] = []
    calls = comparisons = false_accept = false_reject = false_global = allocation = best_allocation = 0
    candidates: list[int] = []

    def finalize_generation() -> None:
        nonlocal comparisons, false_accept, false_reject, false_global
        if key is None:
            return
        state = pools[key]
        for identifier in generations[key][generation]:
            pool = state[identifier]
            particle = pool["particle"]
            previous_id = personal[particle]
            previous = state[previous_id] if previous_id is not None else None
            accept = previous is None or pool["mean"] > previous["mean"]
            if previous is not None:
                comparisons += 1
                false_accept += int(accept and pool["truth"] < previous["truth"] - 1e-12)
                false_reject += int(not accept and pool["truth"] > previous["truth"] + 1e-12)
            if accept:
                personal[particle] = identifier
        selected = max(personal, key=lambda p: state[p]["mean"])
        best_personal = max(state[p]["truth"] for p in personal)
        false_global += int(state[selected]["truth"] < best_personal - 1e-12)
        curves.append({"dataset": records[key].dataset, "instance": records[key].instance,
                       "noise": records[key].noise, "method": key[0], "run": key[1],
                       "generation": generation, "calls": calls,
                       "selected_regret": -state[selected]["truth"],
                       "gbest_oracle_gap": best_personal - state[selected]["truth"]})

    def finalize_search() -> None:
        if key is None:
            return
        row, state = records[key], pools[key]
        selected = max(personal, key=lambda p: state[p]["mean"])
        if (calls != row.calls or comparisons != row.comparisons or false_accept != row.false_accept
                or false_reject != row.false_reject or selected != row.selected_pool):
            raise ValueError(f"Record reconstruction mismatch: {folder.name}, {key}")
        np.testing.assert_allclose([state[selected]["truth"], state[selected]["mean"]],
                                   [row.final_score, row.selected_observed_score], rtol=0, atol=1e-11)
        diagnoses.append({"dataset": row.dataset, "instance": row.instance, "noise": row.noise,
                          "method": key[0], "run": key[1], "calls": calls,
                          "final_regret": -row.final_score, "comparisons": comparisons,
                          "false_accept": false_accept, "false_reject": false_reject,
                          "pbest_error_rate": (false_accept + false_reject) / comparisons,
                          "gbest_error_rate": false_global / row.generations,
                          "allocation_calls": allocation,
                          "oracle_best_candidate_allocation_share": best_allocation / allocation if allocation else 0.})

    with gzip.open(folder / "evaluations.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            incoming_key = (row[0], int(row[1]))
            incoming_generation = int(row[3])
            if incoming_key != key or incoming_generation != generation:
                finalize_generation()
                if incoming_key != key:
                    finalize_search()
                    key = incoming_key
                    personal = [None] * int(records[key].swarm)
                    calls = comparisons = false_accept = false_reject = false_global = allocation = best_allocation = 0
                generation = incoming_generation
                candidates = list(generations[key][generation])
                if any(p is not None for p in personal):
                    previous_global = max((p for p in personal if p is not None), key=lambda p: pools[key][p]["mean"])
                    candidates.append(previous_global)
            call, identifier = int(row[2]), int(row[4])
            if call != calls + 1 or identifier not in candidates:
                raise ValueError("Invalid call order or generation candidate")
            calls = call
            pool = pools[key][identifier]
            pool["count"] += 1
            pool["mean"] += (float(row[6]) - pool["mean"]) / pool["count"]
            if row[7] == "allocation":
                allocation += 1
                best_allocation += int(pool["truth"] >= max(pools[key][p]["truth"] for p in candidates) - 1e-12)
    finalize_generation()
    finalize_search()
    if len(diagnoses) != len(records):
        raise ValueError("Incomplete trace reconstruction")
    return diagnoses, curves


def run(source: Path, out: Path, workers: int) -> dict:
    folders = sorted((source / "cells").iterdir())
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(reconstruct_cell, map(str, folders)))
    diagnosis = pd.DataFrame([row for rows, _ in results for row in rows])
    curve = pd.DataFrame([row for _, rows in results for row in rows])
    if source.name == "ocba_instances_20261022" and (len(diagnosis) != 2880 or diagnosis.calls.sum() != 24883200):
        raise ValueError("Wrong validation trace count")
    out.mkdir(parents=True, exist_ok=True)
    diagnosis.to_csv(out / "record_diagnostics.csv", index=False)
    curve.to_csv(out / "record_trajectories.csv.gz", index=False, compression="gzip")
    metrics = ["pbest_error_rate", "gbest_error_rate", "oracle_best_candidate_allocation_share", "final_regret"]
    summary = diagnosis.groupby(["noise", "method"])[metrics].mean().reset_index()
    summary.to_csv(out / "record_diagnostic_means.csv", index=False)
    associations = []
    for key, group in diagnosis.groupby(["dataset", "instance", "noise", "method"]):
        for metric in metrics[:-1]:
            rho = (spearmanr(group[metric], group.final_regret).statistic
                   if group[metric].nunique() > 1 and group.final_regret.nunique() > 1
                   else float("nan"))
            associations.append({**dict(zip(["dataset", "instance", "noise", "method"], key)),
                                 "metric": metric, "rho": float(rho)})
    pd.DataFrame(associations).to_csv(out / "diagnostic_associations.csv", index=False)
    means = curve.groupby(["noise", "method", "calls"]).selected_regret.mean().reset_index()
    means.to_csv(out / "convergence_means.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, noise, label in zip(axes, ["gaussian", "lognormal"], ["Нормальный шум", "Логнормальный шум"]):
        for method, color, style, name in zip(
                ["pso_ea", "pso_ocba", "pso_ocba_rs"], ["#0072B2", "#D55E00", "#009E73"],
                ["-", "--", "-."], ["PSO-EA", "PSO-OCBA", "PSO-OCBA-RS"]):
            group = means.loc[means.noise.eq(noise) & means.method.eq(method)]
            ax.plot(group.calls, group.selected_regret, color=color, ls=style, label=name, lw=1.6)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Израсходованные оценки")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Средний regret выбранного рекорда")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "record_convergence.png", dpi=240)
    plt.close(fig)
    report = {"searches": len(diagnosis), "calls_replayed": int(diagnosis.calls.sum()),
              "new_objective_calls": 0, "generation_records": len(curve),
              "all_final_selections_and_pbest_counts_match": True,
              "source_sha256": sha256(Path(__file__)),
              "means": summary.to_dict(orient="records"),
              "scope": "descriptive diagnostics only; gbest error compares retained personal records, not all visited points; oracle labels never used by search; correlations are not causal"}
    (out / "trace_diagnostics_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    run(args.source.resolve(), args.out.resolve(), args.workers)


if __name__ == "__main__":
    main()
