import unittest

import numpy as np
from scipy.optimize import minimize

from ocba import (
    classic_ocba_allocation, gaussian_rates, gaussian_star_allocation,
    integer_top_up,
)
from ocba_benchmarks import ControlledBenchmark, FUNCTIONS
from ocba_engine import SYNTHETIC_METHODS, run_search


class OCBATests(unittest.TestCase):
    def test_two_alternative_gaussian_optimum(self):
        alpha = gaussian_star_allocation([1.0, 0.0], [4.0, 1.0], 0)
        np.testing.assert_allclose(alpha, [2 / 3, 1 / 3], atol=1e-6)

    def test_personal_threshold_changes_allocation(self):
        mean, variance = [0.8, 0.7, 1.0], [0.01, 0.01, 0.01]
        ordinary = classic_ocba_allocation(mean, variance, 2)
        specific = gaussian_star_allocation(mean, variance, 2, [0.799, 0.1, np.nan])
        self.assertGreater(specific[0], 0.95)
        self.assertGreater(specific[0], ordinary[0])

    def test_zero_variance_and_ties_are_finite(self):
        for variance in ([0, 0, 0], [0.01, 0, 1]):
            alpha = gaussian_star_allocation([1, 1, 1], variance, 0, [1, 1, np.nan])
            self.assertTrue(np.isfinite(alpha).all())
            self.assertAlmostEqual(alpha.sum(), 1)
            self.assertTrue((alpha > 0).all())

    def test_integer_projection_never_reclaims_old_samples(self):
        rng = np.random.default_rng(91)
        for _ in range(100):
            weights = rng.random(7) + 0.001
            counts = rng.integers(0, 100, 7)
            extra = int(rng.integers(0, 40))
            additions = integer_top_up(weights, counts, extra)
            self.assertEqual(int(additions.sum()), extra)
            self.assertTrue((additions >= 0).all())

    def test_scalar_solver_against_independent_slsqp(self):
        rng = np.random.default_rng(410)
        for _ in range(16):
            n = 5
            means = rng.normal(size=n)
            variances = rng.uniform(0.3, 2.0, n)
            center = int(np.argmax(means))
            thresholds = means + rng.choice([-1, 1], n) * rng.uniform(0.2, 2, n)
            thresholds[center] = np.nan
            alpha = gaussian_star_allocation(means, variances, center, thresholds)
            uniform = np.full(n, 1 / n)
            scale = min(gaussian_rates(uniform, means, variances, center, thresholds))
            result = minimize(
                lambda z: -z[-1], np.append(uniform, 0.9), method="SLSQP",
                bounds=[(1e-9, 1)] * n + [(0, None)],
                constraints=[
                    {"type": "eq", "fun": lambda z: z[:-1].sum() - 1},
                    {"type": "ineq", "fun": lambda z: gaussian_rates(
                        z[:-1], means, variances, center, thresholds) / scale - z[-1]},
                ], options={"maxiter": 500, "ftol": 1e-10},
            )
            self.assertTrue(result.success, result.message)
            scalar_rate = min(gaussian_rates(alpha, means, variances, center, thresholds))
            self.assertAlmostEqual(scalar_rate / scale, result.x[-1], delta=2e-4)

    def test_benchmark_optima_and_reproducibility(self):
        for name in FUNCTIONS:
            task = ControlledBenchmark(name, 3, 1.0)
            self.assertAlmostEqual(task.true_score(task.optimum), 0, places=10)
            self.assertEqual(task.observe(task.optimum, 5), task.observe(task.optimum, 5))

    def test_all_methods_budget_and_reproducibility(self):
        task = ControlledBenchmark("sphere", 3, 1.0)
        for method in SYNTHETIC_METHODS:
            events, pools = [], []
            first = run_search(task, method, 99, 61, events.append, pools.append)
            second = run_search(task, method, 99, 61)
            self.assertEqual(first["calls"], 61)
            self.assertEqual(len(events), 61)
            self.assertEqual(first["final_score"], second["final_score"])
            self.assertEqual([r[0] for r in events], list(range(1, 62)))
            self.assertTrue(all(0 <= i[2] < len(pools) for i in events))


if __name__ == "__main__":
    unittest.main()
