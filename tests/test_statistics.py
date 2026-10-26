import json
import unittest

import numpy as np
import pandas as pd

from experiment import SyntheticObjective, TabularNNObjective, run_once, summarize
from study_statistics import (
    holm_adjust, noise_interactions, paired_effect, validate_grid,
)


class StatisticalRegressionTests(unittest.TestCase):
    def test_accuracy_ranks_ignore_serialization_roundoff(self):
        baseline = np.array([100, 101, 102, 103, 104, 105]) / 108
        candidate = baseline + np.array([1, 1, -1, 0, 2, -2]) / 108
        exact = paired_effect(candidate, baseline, 108)
        noisy = paired_effect(candidate + 1e-16, baseline - 1e-16, 108)
        self.assertEqual(exact, noisy)
        self.assertEqual(exact["zero_pairs"], 1)

    def test_accuracy_lattice_is_validated(self):
        with self.assertRaises(ValueError):
            paired_effect(np.array([0.123]), np.array([0.5]), 108)

    def test_holm_matches_known_example(self):
        np.testing.assert_allclose(
            holm_adjust([0.04, 0.001, 0.03, 0.2]), [0.09, 0.004, 0.09, 0.2]
        )

    def test_missing_whole_method_is_rejected(self):
        raw = pd.DataFrame([{"method": "pso", "run": 0, "calls": 8,
                             "budget": 8, "final_score": 0.5}])
        with self.assertRaises(ValueError):
            validate_grid(raw, [("pso", 0), ("random", 0)], ["method", "run"])

    def test_cluster_interval_counts_shared_seeds_once(self):
        rows = []
        for dataset in ("a", "b"):
            for run, value in enumerate([0.01, 0.02, 0.03]):
                for method in ("pso", "pso_r3"):
                    for sigma in (0.0, 0.06):
                        score = 0.8 + (value if method == "pso_r3" and sigma else 0)
                        rows.append({"dataset": dataset, "run": run,
                                     "method": method, "sigma": sigma,
                                     "budget": 36, "final_score": score})
        result = noise_interactions(pd.DataFrame(rows), ["pso_r3"], True)
        self.assertEqual(result.iloc[0].n_units, 3)
        self.assertAlmostEqual(result.iloc[0].mean_interaction, 0.02)

    def test_new_final_panels_and_explicit_legacy_mode(self):
        task = SyntheticObjective("ackley", 0.02)
        def seeds(run, method="random", policy="per_run"):
            row, _ = run_once(task, method, run, 8, 123,
                              assessment_seed_policy=policy)
            return json.loads(row["assessment_seeds"])
        self.assertNotEqual(seeds(0), seeds(1))
        self.assertEqual(seeds(1, "random"), seeds(1, "pso"))
        self.assertEqual(seeds(0, policy="legacy_shared"),
                         seeds(1, policy="legacy_shared"))

    def test_pairing_preserves_budget_and_split_strata(self):
        rows = []
        for budget in (36, 90):
            for split in (3101, 3102):
                for run in (0, 1):
                    for method in ("pso", "random"):
                        rows.append({"dataset": "ackley", "sigma": 0.02,
                                     "method": method, "budget": budget,
                                     "split_seed": split, "run": run,
                                     "final_score": 0.8 + 0.01 * (method == "random"),
                                     "search_seconds": 1.0, "error_fraction": 0.0})
        _, pairs = summarize(pd.DataFrame(rows))
        self.assertEqual(len(pairs), 4)
        self.assertTrue(pairs.n_pairs.eq(2).all())

    def test_unmatched_pairs_are_rejected(self):
        rows = [{"dataset": "ackley", "sigma": 0.02, "method": method,
                 "run": run, "final_score": 0.8, "search_seconds": 1.0,
                 "error_fraction": 0.0} for method, run in [("pso", 0), ("random", 1)]]
        with self.assertRaises(ValueError):
            summarize(pd.DataFrame(rows))

    def test_final_assessments_are_recorded_and_reconstruct_mean(self):
        task = TabularNNObjective("wine", epochs=2)
        row, events = run_once(task, "random", 0, 8, 321, assessment_repeats=2)
        counts = json.loads(row["assessment_correct_counts"])
        self.assertEqual(len(counts), 2)
        self.assertEqual(row["assessment_denominator"], 72)
        self.assertAlmostEqual(sum(counts) / 72, row["final_score"])
        self.assertTrue(all("x0" in event and "x2" in event for event in events))


if __name__ == "__main__":
    unittest.main()
