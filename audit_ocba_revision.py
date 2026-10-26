"""Separate linear-weight and numerical audit of the revised analysis.

This is an internal computational cross-check, not an external replication.
The reference bootstrap below does not call the production inference routine.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_1samp, wilcoxon

from run_robust_ocba_study import sha256


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/ocba_audit_20261022"
FUNCTIONS = ("sphere", "ellipsoid", "rosenbrock", "ackley", "rastrigin", "griewank", "levy", "zakharov")
NOISES = ("gaussian", "student_t3", "lognormal", "contaminated")
SWARMS, BUDGETS = (6, 12, 24), (540, 2160, 8640)


def ledger_23() -> list[dict]:
    cells = list(product(NOISES, (1, 2), SWARMS, BUDGETS))
    records = []

    def add(name, kind, formula, function):
        weights = {"|".join(map(str, cell)): str(function(*cell)) for cell in cells if function(*cell)}
        records.append({"id": name, "kind": kind, "formula": formula,
                        "stratum_weight": "1/8", "seed_weight": "1/30", "weights": weights})

    for noise in NOISES[1:]:
        add(f"shape_{noise}", "shape", f"average(D[{noise}]-D[gaussian]) over eta,M,B",
            lambda z, e, m, b, n=noise: Fraction(int(z == n)-int(z == 'gaussian'), 18))
    for noise, budget in product(NOISES[1:], BUDGETS[1:]):
        add(f"shape_budget_{noise}_{budget}", "shape_budget", f"shape at B={budget} minus shape at B=540; average eta,M",
            lambda z, e, m, b, n=noise, target=budget:
                Fraction((int(z == n)-int(z == 'gaussian'))*(int(b == target)-int(b == 540)), 6))
    for noise, swarm in product(NOISES[1:], SWARMS[1:]):
        add(f"shape_swarm_{noise}_{swarm}", "shape_swarm", f"shape at M={swarm} minus shape at M=6; average eta,B",
            lambda z, e, m, b, n=noise, target=swarm:
                Fraction((int(z == n)-int(z == 'gaussian'))*(int(m == target)-int(m == 6)), 6))
    for noise in NOISES[1:]:
        add(f"shape_eta_{noise}", "shape_eta", "shape at eta=2 minus shape at eta=1; average M,B",
            lambda z, e, m, b, n=noise: Fraction((int(z == n)-int(z == 'gaussian'))*(1 if e == 2 else -1), 9))
    for noise in NOISES:
        add(f"mean_{noise}", "mean", f"average D[{noise}] over eta,M,B",
            lambda z, e, m, b, n=noise: Fraction(int(z == n), 18))
    add("mean_all", "mean", "(mean_gaussian+mean_student_t3+mean_lognormal+mean_contaminated)/4",
        lambda z, e, m, b: Fraction(1, 72))
    return records


def ledger_72() -> list[dict]:
    records = []
    for first, base in [("pso_ocba", "pso_ea"), ("pso_ocba_rs", "pso_ea"), ("pso_ocba", "pso_ocba_rs")]:
        for noise in NOISES:
            for kind, n0 in [("mean", 3), ("mean", 5), ("mean", 10), ("n0_change", 5), ("n0_change", 10), ("generation_bridge", 3)]:
                weights = {f"new|{noise}|{n0}|{first}": "1", f"new|{noise}|{n0}|{base}": "-1"}
                if kind != "mean":
                    reference = "new" if kind == "n0_change" else "old"
                    weights[f"{reference}|{noise}|3|{first}"] = "-1"
                    weights[f"{reference}|{noise}|3|{base}"] = "1"
                records.append({"id": f"{kind}_{noise}_{first}_{base}_{n0}", "kind": kind,
                                "noise": noise, "method": first, "baseline": base, "n0": n0,
                                "formula": "signed scores in weights; new uses 12M, old uses 6M",
                                "stratum_weight": "1/2", "seed_weight": "1/30", "weights": weights})
    return records


def weighted_values(raw: pd.DataFrame, tasks: list, run_keys: list[str], coordinate_keys: list[str],
                    cells: list[tuple], records: list[dict]) -> np.ndarray:
    index = pd.MultiIndex.from_tuples([(*task, run, *cell) for task in tasks for run in range(30) for cell in cells],
                                     names=run_keys + coordinate_keys)
    selected = raw.set_index(run_keys + coordinate_keys).reindex(index)
    if selected.final_score.isna().any() or selected.index.duplicated().any():
        raise ValueError("Incomplete independent audit grid")
    data = selected.final_score.to_numpy().reshape(len(tasks), 30, len(cells))
    mapping = {"|".join(map(str, cell)): j for j, cell in enumerate(cells)}
    matrix = np.zeros((len(cells), len(records)))
    for k, record in enumerate(records):
        for name, coefficient in record["weights"].items():
            matrix[mapping[name], k] = float(Fraction(coefficient))
    return data @ matrix


def reference_bootstrap(data: np.ndarray, seed: int, repetitions: int = 99999) -> dict:
    """Task-centered matrix products independently implement the same resamples."""
    tasks, runs, k = data.shape
    centers = data.mean(axis=1)
    centered = data - centers[:, None, :]
    mean = centers.mean(axis=0)
    se = np.sqrt(np.sum(np.sum(centered**2, axis=1)/(runs-1), axis=0)/(tasks*tasks*runs))
    rng = np.random.default_rng(seed)
    maxima = []
    for start in range(0, repetitions, 1000):
        size = min(1000, repetitions-start)
        weights = rng.multinomial(runs, np.full(runs, 1/runs), (size, tasks))
        sums, variances = np.zeros((size, k)), np.zeros((size, k))
        for f in range(tasks):
            total = weights[:, f] @ centered[f]
            sums += total
            variances += (weights[:, f] @ (centered[f]**2) - total**2/runs)/(runs-1)
        sampled_se = np.sqrt(np.maximum(variances, 0)/(tasks*tasks*runs))
        statistic = np.divide(np.abs(sums/(tasks*runs)), sampled_se,
                              out=np.full_like(sums, np.inf), where=sampled_se > 0)
        maxima.extend(statistic.max(axis=1))
    maxima = np.asarray(maxima)
    critical = float(np.quantile(maxima, .95, method="higher"))
    p = np.array([(1+np.count_nonzero(maxima >= value))/(repetitions+1) for value in np.abs(mean/se)])
    return {"mean": mean, "se": se, "ci_low": mean-critical*se, "ci_high": mean+critical*se,
            "adjusted_p_zero": p, "critical": critical}


def reference_holm(pvalues: np.ndarray) -> np.ndarray:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    result, largest = np.zeros(len(pvalues)), 0.
    for rank, index in enumerate(order):
        largest = max(largest, (len(order)-rank)*pvalues[index])
        result[index] = min(1., largest)
    return result


def audit_rank_and_nn(source: Path) -> dict:
    raw = pd.read_csv(source / "synthetic_runs.csv")
    archived = pd.read_csv(source / "synthetic_pairwise.csv")
    keys = ["dataset", "noise", "eta", "swarm", "budget"]
    computed = []
    for key, block in raw.groupby(keys):
        pivot = block.pivot(index="run", columns="method", values="final_score")
        for baseline in ["pso", "pso_ea", "pso_ocba_rs"]:
            d = np.round((pivot.pso_ocba-pivot[baseline]).to_numpy(), 12)
            p = 1. if np.all(d == 0) else float(wilcoxon(d, zero_method="wilcox", method="auto").pvalue)
            rng = np.random.default_rng(27117)
            sampled = d[rng.integers(0, 30, (4000, 30))].mean(axis=1)
            ci = np.quantile(sampled, [.025, .975])
            computed.append({**dict(zip(keys, key)), "baseline": baseline,
                             "mean_difference": d.mean(), "wilcoxon_p_raw": p,
                             "ci_low": ci[0], "ci_high": ci[1]})
    frame = pd.DataFrame(computed)
    for mask in [frame.baseline.ne("pso_ocba_rs"), frame.baseline.eq("pso_ocba_rs")]:
        frame.loc[mask, "holm_p"] = reference_holm(frame.loc[mask, "wilcoxon_p_raw"].to_numpy())
    columns = ["mean_difference", "wilcoxon_p_raw", "holm_p", "ci_low", "ci_high"]
    expected = archived.set_index(keys+["baseline"]).sort_index()
    actual = frame.set_index(keys+["baseline"]).sort_index()
    np.testing.assert_allclose(actual[columns], expected[columns], rtol=1e-10, atol=1e-12)
    nn = pd.read_csv(source / "nn_main_runs.csv")
    rows = []
    for (dataset, split), block in nn.groupby(["dataset", "split_seed"]):
        pivot = block.pivot(index="run", columns="method", values="final_score")
        for base in ["pso", "pso_ea"]:
            d = (pivot.pso_ocba-pivot[base]).to_numpy()
            se = d.std(ddof=1)/np.sqrt(len(d))
            radius = t.ppf(1-.05/24, len(d)-1)*se
            rows.append({"dataset": dataset, "split_seed": split, "baseline": base,
                         "mean_difference": d.mean(), "family_ci_low": d.mean()-radius,
                         "family_ci_high": d.mean()+radius, "paired_t_p_raw": ttest_1samp(d, 0).pvalue})
    nn_frame = pd.DataFrame(rows)
    nn_frame["holm_p"] = reference_holm(nn_frame.paired_t_p_raw.to_numpy())
    nn_expected = pd.read_csv(source / "nn_pairwise.csv").set_index(["dataset", "split_seed", "baseline"]).sort_index()
    nn_actual = nn_frame.set_index(["dataset", "split_seed", "baseline"]).sort_index()
    np.testing.assert_allclose(nn_actual, nn_expected[nn_actual.columns], rtol=1e-10, atol=1e-12)
    zeros = frame.loc[frame.wilcoxon_p_raw.eq(1) & frame.mean_difference.eq(0)]
    return {"analytical_rank_contrasts": len(frame), "nn_mean_contrasts": len(rows),
            "zero_contrasts_kept_with_p_one": len(zeros), "numeric_results_match": True}


def run(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "results/ocba_robustness_20261022"
    raw = pd.read_csv(source / "synthetic_runs.csv")
    records23, records72 = ledger_23(), ledger_72()
    cells = list(product(NOISES, (1, 2), SWARMS, BUDGETS))
    pivot = raw.loc[raw.eta.gt(0)].pivot(index=["dataset", "run", "noise", "eta", "swarm", "budget"], columns="method", values="final_score")
    effects = (pivot.pso_ocba-pivot.pso_ea).rename("final_score").reset_index()
    values23 = weighted_values(effects, [(f,) for f in FUNCTIONS], ["dataset", "run"],
                               ["noise", "eta", "swarm", "budget"], cells, records23)
    n0 = pd.read_csv(ROOT / "results/ocba_n0_ablation_20261022/runs.csv")
    n0["era"] = "new"
    old = raw.loc[raw.dataset.isin(["ackley", "rosenbrock"]) & raw.eta.eq(2) & raw.swarm.eq(12) & raw.budget.eq(8640)].copy()
    old["era"], old["n0"] = "old", 3
    methods = ("pso_ea", "pso_ocba_rs", "pso_ocba")
    cells72 = list(product(["new"], NOISES, (3, 5, 10), methods))+list(product(["old"], NOISES, [3], methods))
    values72 = weighted_values(pd.concat([n0, old.loc[old.method.isin(methods)]]),
                               [("ackley",), ("rosenbrock",)], ["dataset", "run"],
                               ["era", "noise", "n0", "method"], cells72, records72)
    report = {"status": "internal separate-implementation cross-check; no external replication claimed",
              "rank_and_nn": audit_rank_and_nn(source), "families": []}
    for size, records, values, seed, folder, keys in [
            (23, records23, values23, 10222026, "ocba_mean_analysis_20261022", ["id"]),
            (72, records72, values72, 11222026, "ocba_n0_ablation_20261022", ["kind", "noise", "method", "baseline", "n0"])]:
        inferred = reference_bootstrap(values, seed)
        actual = pd.DataFrame([{**record, **{k: float(inferred[k][i]) for k in [
            "mean", "se", "ci_low", "ci_high", "adjusted_p_zero"]}} for i, record in enumerate(records)])
        expected = pd.read_csv(ROOT / "results" / folder / "mean_contrasts.csv").set_index(keys).sort_index()
        checked = actual.set_index(keys).sort_index()
        columns = ["mean", "se", "ci_low", "ci_high", "adjusted_p_zero"]
        np.testing.assert_allclose(checked[columns], expected[columns], rtol=1e-9, atol=1e-11)
        pd.DataFrame([{**record, "weights": json.dumps(record["weights"], sort_keys=True)} for record in records]).to_csv(out / f"contrast_ledger_{size}.csv", index=False)
        (out / f"contrast_ledger_{size}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        report["families"].append({"size": size, "bootstrap_repetitions": 99999,
                                   "critical": inferred["critical"], "matches_archived_intervals_and_p": True,
                                   "maximum_absolute_error": float(np.max(np.abs(checked[columns].to_numpy()-expected[columns].to_numpy())))})
        print(f"Independent weight and bootstrap cross-check passed: {size}", flush=True)
    report["input_sha256"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in [
        source / "synthetic_runs.csv", source / "synthetic_pairwise.csv", source / "nn_main_runs.csv",
        ROOT / "results/ocba_n0_ablation_20261022/runs.csv", Path(__file__)]}
    (out / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(run(args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
