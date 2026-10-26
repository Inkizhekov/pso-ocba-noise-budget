"""PSO-specific OCBA for the Gaussian max-min selection model.

Reference: Zhang et al., IEEE TEVC 21(2), 206-219 (2017),
doi:10.1109/TEVC.2016.2592185, selection models (4) and (6).
Historical personal-best estimates are fixed thresholds. The old global best
is a sampled alternative. Both unchanged-global and new-global cases reduce
to a star of Gaussian comparison rates plus personal threshold rates.

This solves the Gaussian model numerically rather than substituting ordinary
single-best OCBA or relying on the dominant-center approximation of the paper's
closed-form corollaries. Finite-sample regularization is explicit below.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _inputs(means, variances, center, thresholds=None):
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if (means.ndim != 1 or len(means) < 2 or means.shape != variances.shape
            or not np.isfinite(means).all() or not np.isfinite(variances).all()
            or (variances < 0).any() or not 0 <= center < len(means)):
        raise ValueError("invalid OCBA means, variances or center")
    limits = (np.full(len(means), np.nan) if thresholds is None
              else np.asarray(thresholds, dtype=float))
    if limits.shape != means.shape or np.isinf(limits).any():
        raise ValueError("invalid personal-best thresholds")
    variance_floor = max(1e-12, float(variances.max()) * 1e-8)
    variances = np.maximum(variances, variance_floor)
    gap_floor = 1e-6 * max(1.0, float(np.ptp(means)))
    return means, variances, limits, gap_floor


def gaussian_star_allocation(means, variances, center: int,
                             thresholds=None) -> np.ndarray:
    """Maximize the minimum personal/global Gaussian selection rate.

    The Gaussian global comparison rate is gap^2 / (2*(v_i/a_i+v_b/a_b)).
    The personal threshold rate is a_i*gap_personal^2/(2*v_i).
    Homogeneity permits fixing the minimum rate to one and minimizing total
    allocations. Conditional on the center weight, each leaf has two lower
    bounds. The remaining scalar objective is convex.
    """
    original_variances = np.asarray(variances, dtype=float)
    mu, var, limits, gap_floor = _inputs(means, variances, center, thresholds)
    if float(original_variances.max()) < 1e-12:
        return np.full(len(mu), 1 / len(mu))
    leaf = np.array([i for i in range(len(mu)) if i != center])
    gap2 = np.maximum(np.abs(mu[leaf] - mu[center]), gap_floor) ** 2
    h = 2 * var[center] / gap2
    # Normalize the largest center singularity to one; u must stay above it.
    base = float(h.max())
    h = h / base
    a = (2 * var[leaf] / gap2) / base
    personal = np.zeros(len(mu))
    finite = np.isfinite(limits)
    personal[finite] = (2 * var[finite]
                        / np.maximum(np.abs(mu[finite] - limits[finite]), gap_floor) ** 2
                        / base)
    lower = max(1.0 + 1e-9, float(personal[center]))

    def objective(u: float) -> float:
        # Each leaf must satisfy both bounds; the larger one is binding.
        leaf_weights = np.maximum(personal[leaf], a * u / (u - h))
        return float(u + leaf_weights.sum())

    upper = max(4.0, 2 * lower)
    for _ in range(80):
        if objective(upper) >= objective(max(lower, upper / 2)):
            break
        upper *= 2
    else:
        raise RuntimeError("Could not bracket the OCBA scalar optimum")
    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded",
                             options={"xatol": 1e-9, "maxiter": 200})
    if not result.success:
        raise RuntimeError(f"OCBA allocation solver failed: {result.message}")
    u = min((lower, float(result.x), upper), key=objective)
    weights = np.empty(len(mu))
    weights[center] = u
    weights[leaf] = np.maximum(personal[leaf], a * u / (u - h))
    weights /= weights.sum()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise RuntimeError("Invalid OCBA allocation")
    return weights


def classic_ocba_allocation(means, variances, center: int) -> np.ndarray:
    """Ordinary best-selection OCBA, equations (7)-(8) in Zhang et al."""
    original_variances = np.asarray(variances, dtype=float)
    mu, var, _, gap_floor = _inputs(means, variances, center)
    if float(original_variances.max()) < 1e-12:
        return np.full(len(mu), 1 / len(mu))
    leaf = np.arange(len(mu)) != center
    gap = np.maximum(np.abs(mu[leaf] - mu[center]), gap_floor)
    weights = np.zeros(len(mu))
    weights[leaf] = var[leaf] / gap ** 2
    weights[center] = np.sqrt(var[center] * np.sum(weights[leaf] ** 2 / var[leaf]))
    return weights / weights.sum()


def integer_top_up(weights: np.ndarray, counts: np.ndarray,
                   additional: int) -> np.ndarray:
    """Project target proportions onto cumulative nondecreasing integer counts."""
    weights, counts = np.asarray(weights, float), np.asarray(counts, int)
    if (weights.shape != counts.shape or weights.ndim != 1
            or not np.isfinite(weights).all() or (weights <= 0).any()
            or (counts < 0).any() or additional < 0):
        raise ValueError("invalid integer allocation inputs")
    if additional == 0:
        return np.zeros(len(counts), dtype=int)
    weights = weights / weights.sum()
    total = int(counts.sum()) + additional
    # Past samples cannot be removed. Clamp over-allocated candidates, then
    # redistribute the remaining budget only among the free candidates.
    fixed = np.zeros(len(counts), dtype=bool)
    for _ in range(len(counts) + 1):
        multiplier = (total - counts[fixed].sum()) / weights[~fixed].sum()
        newly_fixed = (~fixed) & (multiplier * weights < counts)
        if not newly_fixed.any():
            break
        fixed |= newly_fixed
    target = np.maximum(counts, multiplier * weights)
    rounded = np.maximum(counts, np.floor(target + 1e-10).astype(int))
    missing = total - int(rounded.sum())
    if missing < 0 or missing > len(counts):
        raise RuntimeError("Integer projection lost the budget")
    order = np.argsort(-(target - rounded), kind="stable")
    rounded[order[:missing]] += 1
    additions = rounded - counts
    if int(additions.sum()) != additional or (additions < 0).any():
        raise RuntimeError("Integer projection violated the budget")
    return additions


def gaussian_rates(alpha, means, variances, center, thresholds=None):
    """Expose model rates for independent optimizer validation and diagnostics."""
    mu, var, limits, floor = _inputs(means, variances, center, thresholds)
    alpha = np.asarray(alpha, float)
    leaf = np.arange(len(mu)) != center
    rates = (np.maximum(np.abs(mu[leaf] - mu[center]), floor) ** 2
             / (2 * (var[leaf] / alpha[leaf] + var[center] / alpha[center])))
    finite = np.isfinite(limits)
    personal = (alpha[finite] * np.maximum(np.abs(mu[finite] - limits[finite]), floor) ** 2
                / (2 * var[finite]))
    return np.concatenate([rates, personal])
