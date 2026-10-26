import unittest
import numpy as np
from experiment import (
    METHODS, SyntheticObjective, TabularNNObjective, bootstrap_ci, run_once,
)


class ScientificConsistencyTests(unittest.TestCase):
    def test_optimum_of_synthetic_benchmarks(self):
        midpoint = np.full(3, 0.5)
        for name in ("ackley", "rastrigin"):
            task = SyntheticObjective(name, sigma=0.1)
            self.assertAlmostEqual(task.true_score(midpoint), 1.0, places=12)
            self.assertLess(task.true_score(np.zeros(3)), 1.0)

    def test_noise_is_reproducible(self):
        task = SyntheticObjective("ackley", sigma=0.04)
        x = np.array([0.3, 0.5, 0.7])
        self.assertEqual(task.observe(x, seed=123), task.observe(x, seed=123))

    def test_exact_budget_and_repeated_results(self):
        task = SyntheticObjective("rastrigin", sigma=0.04)
        for method in METHODS:
            first, events = run_once(task, method, run_id=1, budget=19,
                                     master_seed=22222)
            second, _ = run_once(task, method, run_id=1, budget=19,
                                 master_seed=22222)
            self.assertEqual(first["calls"], 19)
            self.assertEqual(len(events), 19)
            self.assertAlmostEqual(first["final_score"], second["final_score"])
            self.assertTrue(0 <= first["x0"] <= 1)

    def test_model_splits_and_ranges(self):
        nn = TabularNNObjective("wine", epochs=3)
        self.assertEqual(sum(nn.sizes), 178)
        decoded = nn.decode(np.array([0.0, 1.0, 0.5]))
        self.assertEqual(decoded["hidden"], 16)
        self.assertTrue(0 < decoded["learning_rate_init"] < 1)
        self.assertTrue(0 < nn.observe(np.full(3, 0.5), seed=42) <= 1)

    def test_bootstrap_ci_for_constant(self):
        lo, hi = bootstrap_ci(np.ones(5))
        self.assertAlmostEqual(lo, 1)
        self.assertAlmostEqual(hi, 1)


if __name__ == "__main__":
    unittest.main()
