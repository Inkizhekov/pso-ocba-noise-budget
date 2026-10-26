"""Compatibility and budget checks for the parameterized robustness engine."""
import unittest

import numpy as np

from ocba_benchmarks import ControlledBenchmark
from ocba_engine import run_search as frozen_search
from robust_ocba_engine import run_search
from run_robust_ocba_study import RobustBenchmark, jobs, DEFAULT_OUT


class RobustEngineTests(unittest.TestCase):
    def test_frozen_default_trajectory_is_preserved(self):
        for method in ["pso", "pso_ea", "pso_ocba_rs", "pso_ocba"]:
            task = ControlledBenchmark("ackley", 8, 2)
            old_events, new_events = [], []
            before = frozen_search(task, method, 12341, 180, old_events.append)
            after = run_search(task, method, 12341, 180, new_events.append)
            self.assertEqual(old_events, new_events)
            for key in ["final_score", "selected_observed_score", "selected_coordinates",
                        "selected_pool", "calls", "generations", "false_accept", "false_reject"]:
                self.assertEqual(before[key], after[key])

    def test_swarm_and_incomplete_tail_budgets(self):
        for swarm in [6, 12, 24]:
            for budget in [swarm + 1, 6 * swarm + 5, 540]:
                for method in ["pso", "pso_ea", "pso_ocba_rs", "pso_ocba"]:
                    task = RobustBenchmark("levy", 2, "student_t3")
                    events = []
                    result = run_search(task, method, 1247, budget, events.append,
                                        swarm_size=swarm)
                    self.assertEqual(len(events), budget)
                    self.assertEqual(result["calls"], budget)
                    self.assertTrue(np.isfinite(result["final_score"]))

    def test_normal_transform_matches_frozen_noise(self):
        before = ControlledBenchmark("sphere", 8, 2)
        after = RobustBenchmark("sphere", 2, "gaussian")
        for seed in range(10):
            x = np.random.default_rng(seed + 10).random(8)
            self.assertEqual(before.observe(x, seed), after.observe(x, seed))

    def test_zero_noise_is_identical_for_all_distributions(self):
        x = np.full(8, 0.5)
        for noise in ["gaussian", "student_t3", "lognormal", "contaminated"]:
            task = RobustBenchmark("sphere", 0, noise)
            self.assertEqual(task.true_score(x), task.observe(x, 1))

    def test_frozen_grid_size(self):
        grid = jobs("synthetic", DEFAULT_OUT)
        self.assertEqual(len(grid), 648)
        self.assertEqual(sum(job["runs"] * 4 for job in grid), 77760)
        self.assertEqual(sum(job["runs"] * 4 * job["budget"] for job in grid), 293932800)


if __name__ == "__main__":
    unittest.main()
