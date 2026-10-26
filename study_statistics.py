"""Statistical procedures shared by the experiment and manuscript analysis."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TEST_SIZES = {"digits": 360, "wine": 36, "breast_cancer": 114}


def bootstrap_ci(values: np.ndarray, seed: int = 27_117,
                 n_boot: int = 4000) -> tuple[float, float]:
    """Percentile interval for a mean of independent sampling units."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if not len(data):
        return float("nan"), float("nan")
    if len(data) == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    estimates = data[rng.integers(0, len(data), (n_boot, len(data)))].mean(axis=1)
    return tuple(float(v) for v in np.quantile(estimates, [0.025, 0.975]))


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(
        1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    )
    return adjusted


def paired_effect(contender: np.ndarray, baseline: np.ndarray,
                  denominator: int | None = None) -> dict[str, float | int | str]:
    """Use integer accuracy units for ranks, and score units for effect sizes.

    A denominator describes the mean of equally sized test assessments, e.g.
    three fits on 36 test objects have denominator 108. Continuous objectives
    use differences rounded to 12 decimal places to remove serialization noise.
    """
    a, b = np.asarray(contender, dtype=float), np.asarray(baseline, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired samples must be nonempty 1-D arrays of equal size")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paired samples must contain finite values")
    if denominator is not None:
        if denominator <= 0:
            raise ValueError("accuracy denominator must be positive")
        if (np.max(np.abs(a * denominator - np.rint(a * denominator))) > 1e-7
                or np.max(np.abs(b * denominator - np.rint(b * denominator))) > 1e-7):
            raise ValueError("accuracy values do not lie on the stated lattice")
        ranks = np.rint(a * denominator).astype(np.int64) - np.rint(b * denominator).astype(np.int64)
        difference = ranks / denominator
        policy = "integer_correct_counts"
    else:
        difference = np.round(a - b, decimals=12)
        ranks = difference
        policy = "difference_rounded_12_decimals"
    p = 1.0 if np.all(ranks == 0) else float(
        wilcoxon(ranks, zero_method="wilcox", method="auto").pvalue
    )
    lo, hi = bootstrap_ci(difference)
    return {"n_pairs": len(a), "mean_difference": float(np.mean(difference)),
            "ci_low": lo, "ci_high": hi, "wilcoxon_p_raw": p,
            "rank_policy": policy, "zero_pairs": int(np.sum(ranks == 0))}


def validate_grid(raw: pd.DataFrame, expected: Sequence[tuple],
                  columns: Sequence[str]) -> None:
    """Reject missing, extra and duplicated cells, rather than existing groups only."""
    if raw.empty or raw.duplicated(list(columns)).any():
        raise ValueError("empty results or duplicate experimental cells")
    normalize = lambda row: tuple(None if pd.isna(v) else v for v in row)
    actual = {normalize(row) for row in raw[list(columns)].itertuples(index=False, name=None)}
    wanted = {normalize(row) for row in expected}
    if actual != wanted:
        raise ValueError(f"incomplete experimental grid: missing={len(wanted-actual)}, extra={len(actual-wanted)}")
    if not raw.calls.eq(raw.budget).all() or not np.isfinite(raw.final_score).all():
        raise ValueError("invalid budget accounting or non-finite final scores")


def noise_interactions(raw: pd.DataFrame, methods: Sequence[str],
                       shared_seed_across_functions: bool) -> pd.DataFrame:
    """High-minus-zero effects; cluster shared seeds or keep functions separate.

    The functions are fixed benchmarks. No inference over a population of tasks
    is made, and no overlapping data splits are pooled as independent trials.
    """
    rows = []
    by_budget = raw.groupby("budget", sort=True)
    for budget, budget_data in by_budget:
        for method in methods:
            per_function = {}
            for dataset, block in budget_data.groupby("dataset", sort=True):
                pivot = block.pivot(index="run", columns=["method", "sigma"], values="final_score")
                per_function[dataset] = (
                    pivot[(method, 0.06)] - pivot[("pso", 0.06)]
                    - pivot[(method, 0.0)] + pivot[("pso", 0.0)]
                )
            if shared_seed_across_functions:
                matrix = pd.DataFrame(per_function)
                if matrix.isna().any().any():
                    raise ValueError("unmatched seed clusters across functions")
                groups = {"fixed_suite_seed_clusters": matrix.mean(axis=1)}
            else:
                groups = per_function
            for dataset, effects in groups.items():
                lo, hi = bootstrap_ci(effects.to_numpy(), seed=90210)
                rows.append({"budget": budget, "dataset": dataset, "method": method,
                             "n_units": len(effects), "mean_interaction": float(effects.mean()),
                             "ci_low": lo, "ci_high": hi,
                             "sampling_unit": "shared_run_seed" if shared_seed_across_functions else "run_within_function",
                             "ci_multiplicity": "pointwise_exploratory"})
    return pd.DataFrame(rows)
