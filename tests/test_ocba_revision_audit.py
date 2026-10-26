"""Check validation independence and explicit contrast definitions."""
import unittest
from fractions import Fraction

import numpy as np

from audit_ocba_revision import ledger_23, ledger_72, reference_bootstrap
from paired_mean_inference import simultaneous_mean_intervals
from run_ocba_instance_validation import NewInstance
from run_robust_ocba_study import RobustBenchmark


class RevisionAuditTests(unittest.TestCase):
    def test_transforms_are_new_reproducible_and_shared_across_noises(self):
        old = RobustBenchmark("ackley", 2, "gaussian")
        first = NewInstance("ackley", 101, "gaussian")
        repeated = NewInstance("ackley", 101, "lognormal")
        second = NewInstance("ackley", 102, "gaussian")
        self.assertNotEqual(old.design_seed, first.design_seed)
        self.assertFalse(np.array_equal(old.rotation, first.rotation))
        self.assertFalse(np.array_equal(first.rotation, second.rotation))
        np.testing.assert_array_equal(first.rotation, repeated.rotation)
        self.assertAlmostEqual(first.true_score(first.optimum), 0., places=12)

    def test_all_23_weights_include_fifth_grand_mean(self):
        records = ledger_23()
        self.assertEqual(len(records), 23)
        self.assertEqual(len({r["id"] for r in records}), 23)
        self.assertEqual(len(records[-1]["weights"]), 72)
        for row in records:
            total = sum(Fraction(x) for x in row["weights"].values())
            self.assertEqual(total, 1 if row["kind"] == "mean" else 0)
        average = {}
        for row in records[-5:-1]:
            for key, value in row["weights"].items():
                average[key] = Fraction(value)/4
        self.assertEqual(average, {k: Fraction(v) for k, v in records[-1]["weights"].items()})

    def test_n0_ledger_has_complete_pairs_and_correct_bridges(self):
        records = ledger_72()
        self.assertEqual(len(records), 72)
        self.assertEqual(len({r["id"] for r in records}), 72)
        self.assertEqual(sum(r["kind"] == "mean" for r in records), 36)
        self.assertEqual(sum(r["kind"] == "n0_change" for r in records), 24)
        self.assertEqual(sum(r["kind"] == "generation_bridge" for r in records), 12)
        for row in records:
            self.assertEqual(sum(Fraction(x) for x in row["weights"].values()), 0)
            self.assertEqual(any(k.startswith("old|") for k in row["weights"]),
                             row["kind"] == "generation_bridge")

    def test_separate_bootstrap_implementation_matches_with_task_offsets(self):
        rng = np.random.default_rng(42)
        data = rng.normal(size=(3, 30, 5)) + np.arange(3)[:, None, None]*10
        reference = reference_bootstrap(data, 157, repetitions=1999)
        production = simultaneous_mean_intervals(data, seed=157, repetitions=1999)
        for key in ["mean", "se", "ci_low", "ci_high", "adjusted_p_zero"]:
            np.testing.assert_allclose(reference[key], production[key], rtol=1e-9, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
