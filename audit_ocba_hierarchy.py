"""Diagnose the retained hierarchical bootstrap without objective evaluations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np
import pandas as pd

from analyze_ocba_scope import instance_tensor
from ocba_benchmarks import FUNCTIONS
from run_ocba_instance_validation import CONTRASTS
from run_robust_ocba_study import sha256

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/ocba_final_20261024"
SOURCE = ROOT / "results/ocba_instances_20261022/runs.csv"
QUANTILES = (0.001, 0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99, 0.999)


def resample_diagnostics(
    values: np.ndarray, repetitions: int, seed: int, level: str = "full"
) -> dict:
    """Use the retained index order; reduced levels share all random indices.

    Reduced levels are distribution diagnostics, not alternative primary tests.
    Family-only uses original family means. Family-instance additionally samples
    the two instance means. Full additionally samples 30 paired seed vectors.
    All three consume the same f/i/r draws to isolate added resampling levels.
    """
    if values.shape != (8, 2, 30, 5) or not np.isfinite(values).all():
        raise ValueError("Expected a finite 8 x 2 x 30 x 5 paired tensor")
    if repetitions < 999 or level not in {"family", "family_instance", "full"}:
        raise ValueError("Invalid repetitions or resampling level")
    family_means = values.mean(axis=(1, 2))
    instance_means = values.mean(axis=2)
    center = family_means.mean(axis=0)
    original_se = family_means.std(axis=0, ddof=1) / np.sqrt(8)
    if np.any(original_se <= 0):
        raise ValueError("Nonzero between-family variability is required")
    rng = np.random.default_rng(seed)
    means = np.empty((repetitions, 5))
    errors = np.empty_like(means)
    unique = np.empty(repetitions, dtype=np.int64)
    griewank = np.empty_like(unique)
    griewank_index = list(FUNCTIONS).index("griewank")
    for start in range(0, repetitions, 250):
        size = min(250, repetitions - start)
        f = rng.integers(8, size=(size, 8, 1, 1))
        i = rng.integers(2, size=(size, 8, 2, 1))
        r = rng.integers(30, size=(size, 8, 2, 30))
        if level == "full":
            groups = values[f, i, r].mean(axis=(2, 3))
        elif level == "family_instance":
            groups = instance_means[f[..., 0], i[..., 0]].mean(axis=2)
        else:
            groups = family_means[f[:, :, 0, 0]]
        stop = start + size
        means[start:stop] = groups.mean(axis=1)
        errors[start:stop] = groups.std(axis=1, ddof=1) / np.sqrt(8)
        labels = f[:, :, 0, 0]
        unique[start:stop] = (np.diff(np.sort(labels, axis=1), axis=1) != 0).sum(axis=1) + 1
        griewank[start:stop] = (labels == griewank_index).sum(axis=1)
    statistics = np.divide(
        np.abs(means - center), errors,
        out=np.full_like(means, np.inf), where=errors > 0,
    )
    maxima = statistics.max(axis=1)
    critical = float(np.quantile(maxima, 0.95, method="higher"))
    if not np.isfinite(critical):
        raise ValueError("The joint studentized critical is not finite")
    return {
        "mean": center, "se": original_se, "bootstrap_mean": means,
        "bootstrap_se": errors, "statistics": statistics, "maxima": maxima,
        "critical": critical, "ci_low": center - critical * original_se,
        "ci_high": center + critical * original_se, "unique_families": unique,
        "griewank_draws": griewank, "seed": seed, "level": level,
    }


def summarize(draws: dict) -> dict:
    """Summaries distinguish resampled means from studentized joint intervals."""
    quantiles = []
    for column, name in enumerate(CONTRASTS):
        for quantity, array in [
            ("bootstrap_mean", draws["bootstrap_mean"][:, column]),
            ("bootstrap_se", draws["bootstrap_se"][:, column]),
            ("absolute_studentized_error", draws["statistics"][:, column]),
        ]:
            quantiles.append({"contrast": name, "quantity": quantity, **{
                str(q): float(np.quantile(array, q)) for q in QUANTILES
            }})
    winners = np.argmax(draws["statistics"], axis=1)
    upper = draws["maxima"] >= draws["critical"]
    interaction = draws["statistics"][:, 2]
    interaction_upper = interaction >= np.quantile(interaction, 0.95, method="higher")
    return {
        "level": draws["level"], "seed": draws["seed"],
        "repetitions": len(draws["maxima"]), "critical": draws["critical"],
        "mean": draws["mean"].tolist(), "se": draws["se"].tolist(),
        "ci_low": draws["ci_low"].tolist(), "ci_high": draws["ci_high"].tolist(),
        "zero_bootstrap_se_counts": (draws["bootstrap_se"] == 0).sum(axis=0).tolist(),
        "interaction_resampled_mean_nonpositive": int((draws["bootstrap_mean"][:, 2] <= 0).sum()),
        "joint_maximum_winner_counts": {name: int((winners == j).sum()) for j, name in enumerate(CONTRASTS)},
        "upper_joint_tail_winner_counts": {name: int((upper & (winners == j)).sum()) for j, name in enumerate(CONTRASTS)},
        "joint_upper_tail_size": int(upper.sum()),
        "interaction_upper_t_tail": {
            "size": int(interaction_upper.sum()),
            "fraction_without_griewank": float((draws["griewank_draws"][interaction_upper] == 0).mean()),
            "median_bootstrap_se": float(np.median(draws["bootstrap_se"][interaction_upper, 2])),
            "median_bootstrap_mean": float(np.median(draws["bootstrap_mean"][interaction_upper, 2])),
        },
        "quantiles": quantiles,
        "quantile_interpretation": "descriptive distribution diagnostics; not replacement joint intervals or posterior probabilities",
    }


def run(out: Path, repetitions: int = 99999) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    original = ROOT / "results/ocba_scope_20261023/hierarchical_sensitivity.csv"
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post_hoc_numerical_diagnostic_no_new_primary_family",
        "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in (SOURCE, original)},
        "source_sha256": sha256(Path(__file__)), "repetitions": repetitions,
        "seeds": [14232026, 14242026],
        "levels": ["full", "family_instance", "family"],
        "purpose": "audit the retained interval, studentization, resampling levels and Monte Carlo sensitivity",
        "new_objective_calls": 0,
    }
    target = out / "hierarchy_diagnostic_plan.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "created_utc"} != {
            k: v for k, v in plan.items() if k != "created_utc"
        }:
            raise ValueError("Diagnostic code or inputs changed; use another output directory")
    else:
        target.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    values = instance_tensor(pd.read_csv(SOURCE))
    results = []
    for level, seed in [("full", 14232026), ("full", 14242026),
                        ("family_instance", 14232026), ("family", 14232026)]:
        draws = resample_diagnostics(values, repetitions, seed, level)
        result = summarize(draws)
        results.append(result)
        if level == "full" and seed == 14232026:
            saved = pd.read_csv(original)
            for key in ("mean", "se", "ci_low", "ci_high"):
                np.testing.assert_allclose(draws[key], saved[key], rtol=0, atol=1e-12)
            pd.DataFrame({
                "interaction_mean": draws["bootstrap_mean"][:, 2],
                "interaction_bootstrap_se": draws["bootstrap_se"][:, 2],
                "interaction_abs_t": draws["statistics"][:, 2],
                "joint_max_abs_t": draws["maxima"],
                "joint_winner": np.asarray(CONTRASTS)[np.argmax(draws["statistics"], axis=1)],
                "unique_families": draws["unique_families"],
                "griewank_draws": draws["griewank_draws"],
            }).to_csv(out / "hierarchy_draw_diagnostics.csv.gz", index=False, compression="gzip")
    report = {
        "status": plan["status"], "retained_intervals_reproduced": True,
        "results": results, "new_objective_calls": 0,
        "unchanged_estimand": "equal-weight average on the fixed chosen families; hierarchical scenario is exploratory",
        "audit_randomness": "Original and reference bootstrap use identical RNG algorithm, seed, batch dimensions and index order; independently coded arithmetic does not use independent resamples. The alternate seed checks Monte Carlo sensitivity separately.",
    }
    (out / "hierarchy_diagnostic_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results:
        print(json.dumps({k: r[k] for k in ("level", "seed", "critical", "interaction_resampled_mean_nonpositive", "upper_joint_tail_winner_counts")}))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.out.resolve())


if __name__ == "__main__":
    main()
