"""Meaningful guards for holdout separation and censored budget comparisons."""
import unittest
from itertools import product

import numpy as np
import pandas as pd

from analyze_ocba_study import budget_savings, fit_router, fixed_suite_ci
from ocba_benchmarks import FUNCTIONS, VALIDATION_FUNCTIONS
from ocba_engine import SYNTHETIC_METHODS


class AnalysisTests(unittest.TestCase):
    def test_validation_labels_cannot_change_trained_rule(self):
        cells, runs = [], []
        for name, dimension, eta, budget in product(FUNCTIONS, [3, 8], [0, .25, 1, 2], [60, 180, 540]):
            gain = .1 if eta >= 1 and budget >= 180 else -.1
            key = dict(dataset=name, dimension=dimension, eta=eta, budget=budget)
            cells.append({**key, "method": "pso_ocba", "baseline": "pso", "mean_difference": gain})
            if name in VALIDATION_FUNCTIONS:
                for run, method in product(range(4), SYNTHETIC_METHODS):
                    runs.append({**key, "run": run, "method": method,
                                 "final_score": gain if method == "pso_ocba" else 0})
        pairs, raw = pd.DataFrame(cells), pd.DataFrame(runs)
        first, _, _ = fit_router(raw, pairs)
        pairs.loc[pairs.dataset.isin(VALIDATION_FUNCTIONS), "mean_difference"] *= -1
        second, _, _ = fit_router(raw, pairs)
        self.assertEqual(first["tree"], second["tree"])
        self.assertEqual(first["training_cells"], 144)
        self.assertEqual(first["validation_cells"], 48)

    def test_suite_resampling_preserves_shared_condition_blocks(self):
        raw = pd.DataFrame({"dataset": ["a"] * 4, "dimension": [3] * 4,
                            "run": [0, 0, 1, 1], "gain": [1, -1, 2, -2]})
        self.assertEqual(fixed_suite_ci(raw, "gain"), (0.0, 0.0, 0.0))

    def test_censored_targets_never_imply_finite_savings(self):
        rows = []
        for task, method in product(["a", "b", "c"], SYNTHETIC_METHODS):
            budget = 180 if method == "pso" else 60
            if task == "b" and method == "pso":
                budget = np.nan
            if task == "c" and method != "pso":
                budget = np.nan
            rows.append(dict(dataset=task, dimension=3, eta=1,
                             method=method, minimum_tested_budget=budget))
        result = budget_savings(pd.DataFrame(rows))
        self.assertTrue(result.both_reached.eq(1).all())
        self.assertTrue(result.method_only.eq(1).all())
        self.assertTrue(result.pso_only.eq(1).all())
        np.testing.assert_allclose(result.conditional_mean_saving, 2 / 3)


if __name__ == "__main__":
    unittest.main()
