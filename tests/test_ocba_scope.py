"""Check sensitivity resampling, complete pairing, and trace reconstruction."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_ocba_scope import bootstrap_draws, conditional_mde, instance_tensor
from diagnose_ocba_traces import reconstruct_cell
from paired_mean_inference import simultaneous_mean_intervals


class ScopeTests(unittest.TestCase):
    def test_retained_draws_reproduce_existing_bootstrap(self):
        data = np.random.default_rng(16).normal(size=(3, 30, 5))
        data += np.arange(3)[:, None, None] * 20
        draws = bootstrap_draws(data, 1999, 42)
        expected = simultaneous_mean_intervals(data, repetitions=1999, seed=42)
        np.testing.assert_allclose(draws["critical"], expected["critical_max_abs_t"], atol=1e-12)
        np.testing.assert_allclose(draws["mean"] - draws["critical"] * draws["se"],
                                   expected["ci_low"], atol=1e-12)

    def test_mde_quantile_corresponds_to_ninety_percent_rejection(self):
        noise = np.linspace(-1, 1, 10001)[:, None]
        draws = {"critical": 3., "noise": noise, "boot_se": np.ones_like(noise),
                 "mean": [0.], "se": [1.]}
        result = conditional_mde(draws, [0], [{"id": "shift"}])[0]
        self.assertAlmostEqual(result["mde90_positive"], 3.8)
        rejection = (noise[:, 0] + result["mde90_positive"]) > 3
        self.assertAlmostEqual(rejection.mean(), .9, places=3)

    def test_instance_tensor_rejects_missing_pair(self):
        raw = pd.read_csv("results/ocba_instances_20261022/runs.csv")
        values = instance_tensor(raw)
        self.assertEqual(values.shape, (8, 2, 30, 5))
        with self.assertRaises(ValueError):
            instance_tensor(raw.iloc[1:])

    def test_trace_replay_matches_every_final_record_and_error_counter(self):
        root = Path("results/ocba_instances_smoke_20261022/cells/sphere_i101_gaussian")
        diagnoses, curves = reconstruct_cell(str(root))
        self.assertEqual(len(diagnoses), 3)
        self.assertEqual(sum(r["calls"] for r in diagnoses), 25920)
        self.assertEqual(len(curves), 360)
        for record in diagnoses:
            self.assertEqual(record["comparisons"], 12 * 119)
            self.assertTrue(0 <= record["gbest_error_rate"] <= 1)


if __name__ == "__main__":
    unittest.main()
