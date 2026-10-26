"""Regression checks for the diagnostic hierarchy calculation."""
from __future__ import annotations

import unittest

import numpy as np

from analyze_ocba_scope import hierarchical_sensitivity
from audit_ocba_hierarchy import resample_diagnostics


class HierarchyDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(116)
        self.values = rng.normal(size=(8, 2, 30, 5))
        self.values += np.arange(8)[:, None, None, None] * 0.2

    def test_full_diagnostic_reproduces_retained_hierarchy(self) -> None:
        actual = resample_diagnostics(self.values, 1999, 251)
        expected = hierarchical_sensitivity(self.values, 1999, 251)
        for key in ("mean", "se", "ci_low", "ci_high", "critical"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)

    def test_resampling_levels_collapse_without_within_family_variation(self) -> None:
        family = self.values.mean(axis=(1, 2))
        values = np.broadcast_to(family[:, None, None, :], (8, 2, 30, 5)).copy()
        levels = [resample_diagnostics(values, 1999, 251, level)
                  for level in ("family", "family_instance", "full")]
        for actual in levels[1:]:
            np.testing.assert_allclose(actual["bootstrap_mean"], levels[0]["bootstrap_mean"], atol=1e-12)
            np.testing.assert_allclose(actual["bootstrap_se"], levels[0]["bootstrap_se"], atol=1e-12)
            np.testing.assert_array_equal(actual["unique_families"], levels[0]["unique_families"])

    def test_location_shift_changes_centers_not_studentization(self) -> None:
        shift = np.array([0.25, -0.5, 1.0, 0.1, -0.1])
        original = resample_diagnostics(self.values, 1999, 251)
        shifted = resample_diagnostics(self.values + shift, 1999, 251)
        np.testing.assert_allclose(shifted["bootstrap_mean"], original["bootstrap_mean"] + shift, atol=1e-12)
        np.testing.assert_allclose(shifted["statistics"], original["statistics"], rtol=1e-11, atol=1e-12)

    def test_nonfinite_inputs_are_rejected(self) -> None:
        values = self.values.copy()
        values[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            resample_diagnostics(values, 999, 251)


if __name__ == "__main__":
    unittest.main()
