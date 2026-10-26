"""Check mean inference, shared-seed contrasts and feasible n0 ablations."""
import unittest

import numpy as np

from analyze_ocba_interactions import contrast_specs, make_contrasts
from paired_mean_inference import stratified_mean_se, simultaneous_mean_intervals
from robust_ocba_engine import run_search
from run_robust_ocba_study import RobustBenchmark


class PairedMeanTests(unittest.TestCase):
    def test_stratified_se_respects_fixed_tasks(self):
        data = np.array([[[1.0], [2.0], [3.0]], [[100.0], [102.0], [104.0]]])
        mean, se = stratified_mean_se(data)
        self.assertAlmostEqual(mean[0], 52.0)
        self.assertAlmostEqual(se[0], np.sqrt((1 + 4) / 12))

    def test_perfectly_paired_contrasts_do_not_receive_independent_noise(self):
        data = np.random.default_rng(43).normal(size=(3, 30, 1))
        single = simultaneous_mean_intervals(data, repetitions=1999, seed=17)
        joint = simultaneous_mean_intervals(np.repeat(data, 2, axis=-1), repetitions=1999, seed=17)
        np.testing.assert_allclose(joint["ci_low"], single["ci_low"][0])
        np.testing.assert_allclose(joint["ci_high"], single["ci_high"][0])
        self.assertEqual(joint["critical_max_abs_t"], single["critical_max_abs_t"])

    def test_shift_changes_mean_but_not_uncertainty(self):
        data = np.random.default_rng(54).normal(size=(3, 30, 2))
        before = simultaneous_mean_intervals(data, repetitions=1999, seed=8)
        after = simultaneous_mean_intervals(data + .75, repetitions=1999, seed=8)
        np.testing.assert_allclose(after["ci_low"], before["ci_low"] + .75)
        np.testing.assert_allclose(after["ci_high"], before["ci_high"] + .75)

    def test_degenerate_data_are_not_called_significant(self):
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            simultaneous_mean_intervals(np.ones((2, 30, 2)), repetitions=1999)

    def test_interactions_recover_known_factorial_effects(self):
        base = np.arange(8 * 30).reshape(8, 30, 1, 1, 1, 1) / 100
        z = np.arange(4).reshape(1, 1, 4, 1, 1, 1)
        eta = np.arange(2).reshape(1, 1, 1, 2, 1, 1)
        swarm = np.arange(3).reshape(1, 1, 1, 1, 3, 1)
        budget = np.arange(3).reshape(1, 1, 1, 1, 1, 3)
        values = base + z * (2 + 3 * eta + 5 * swarm + 7 * budget)
        contrasts = make_contrasts(values)
        specs = contrast_specs()
        self.assertEqual(len(specs), 23)
        np.testing.assert_allclose(contrasts[:, :, 0], 15.5)
        np.testing.assert_allclose(contrasts[:, :, 1], 31)
        np.testing.assert_allclose(contrasts[:, :, 2], 46.5)
        expected = {
            "shape_budget_student_t3_2160": 7,
            "shape_budget_lognormal_8640": 28,
            "shape_swarm_contaminated_24": 30,
            "shape_eta_student_t3": 3,
        }
        for index, spec in enumerate(specs):
            if spec["id"] in expected:
                np.testing.assert_allclose(contrasts[:, :, index], expected[spec["id"]])

    def test_n0_requires_enough_generation_budget(self):
        task = RobustBenchmark("sphere", 2, "student_t3")
        with self.assertRaises(ValueError):
            run_search(task, "pso_ocba", 2, 144, initial_repeats=10)
        for n0 in [3, 5, 10]:
            for method in ["pso_ea", "pso_ocba_rs", "pso_ocba"]:
                events = []
                result = run_search(task, method, 1234, 144, events.append,
                                    swarm_size=6, initial_repeats=n0,
                                    generation_multiplier=12)
                self.assertEqual(result["calls"], 144)
                self.assertEqual(result["generations"], 2)
                initial = [row for row in events if row[-1] in {"new", "initial_repeat"}]
                self.assertEqual(len(initial), 2 * 6 * n0)


if __name__ == "__main__":
    unittest.main()
