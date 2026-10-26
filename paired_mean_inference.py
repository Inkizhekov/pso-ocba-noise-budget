"""Simultaneous inference for paired means conditional on fixed task strata."""
from __future__ import annotations

import numpy as np


def stratified_mean_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use independent seed blocks within each fixed, equally weighted task."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 3 or data.shape[1] < 3 or not np.isfinite(data).all():
        raise ValueError("Expected finite task x seed x contrast data with >=3 seeds")
    tasks, runs, _ = data.shape
    mean = data.mean(axis=(0, 1))
    se = np.sqrt(data.var(axis=1, ddof=1).sum(axis=0) / (tasks * tasks * runs))
    return mean, se


def simultaneous_mean_intervals(values: np.ndarray, *, repetitions: int = 99999,
                                seed: int = 10222026,
                                confidence: float = 0.95) -> dict:
    """Single-step, centered studentized max-|t| bootstrap over shared seeds.

    All coordinates of a seed block are resampled together, independently
    within each task. Task identities are never resampled. Coverage and FWER
    are bootstrap approximations, not finite-sample distribution-free bounds.
    """
    data = np.asarray(values, dtype=float)
    mean, se = stratified_mean_se(data)
    if repetitions < 999 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap size or confidence")
    if np.any(se <= 0):
        raise ValueError("Degenerate mean contrasts require separate treatment")
    tasks, runs, contrasts = data.shape
    rng = np.random.default_rng(seed)
    maxima = np.empty(repetitions)
    for start in range(0, repetitions, 1000):
        size = min(1000, repetitions - start)
        # Multinomial counts resample seed rows. Sharing each count vector
        # across all contrasts preserves their paired dependence.
        weights = rng.multinomial(runs, np.full(runs, 1 / runs), (size, tasks))
        sums = np.einsum("bfr,frk->bfk", weights, data, optimize=True)
        squares = np.einsum("bfr,frk->bfk", weights, data * data, optimize=True)
        bootstrap_mean = sums.sum(axis=1) / (tasks * runs)
        variance = np.maximum(squares - sums * sums / runs, 0) / (runs - 1)
        bootstrap_se = np.sqrt(variance.sum(axis=1) / (tasks * tasks * runs))
        # A zero resampled SE cannot yield a narrow interval by accident.
        statistic = np.divide(
            np.abs(bootstrap_mean - mean), bootstrap_se,
            out=np.full_like(bootstrap_mean, np.inf), where=bootstrap_se > 0,
        )
        maxima[start:start + size] = statistic.max(axis=1)
    critical = float(np.quantile(maxima, confidence, method="higher"))
    if not np.isfinite(critical):
        raise ValueError("Bootstrap family is too degenerate for studentization")
    observed = np.abs(mean / se)
    adjusted = np.array([(1 + np.count_nonzero(maxima >= value)) / (repetitions + 1)
                         for value in observed])
    return {
        "mean": mean, "se": se, "ci_low": mean - critical * se,
        "ci_high": mean + critical * se, "adjusted_p_zero": adjusted,
        "critical_max_abs_t": critical, "repetitions": repetitions,
        "seed": seed, "family_size": contrasts, "confidence": confidence,
    }
