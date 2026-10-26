"""Check that archive audits reject inconsistencies beyond file hashes."""
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd
import numpy as np

from analyze_robust_ocba_study import audit_job, check_benchmark_metadata
from ocba_benchmarks import ControlledBenchmark
from run_robust_ocba_study import RobustBenchmark, NOISES, sha256


ROOT = Path(__file__).resolve().parent.parent
SMOKE = ROOT / "results/ocba_robustness_nn_smoke_20261022/nn_pilot/sonar_s7103_r0023"


class ArchiveAuditTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "results/robust_audit_test_temp"
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.folder = Path(self.temporary.name) / "cell"
        shutil.copytree(SMOKE, self.folder)

    def tearDown(self):
        parent = (ROOT / "results/robust_audit_test_temp").resolve()
        if not Path(self.temporary.name).resolve().is_relative_to(parent):
            raise ValueError("Temporary audit folder escaped the workspace")
        self.temporary.cleanup()

    def update_recorded_hash(self, name):
        path = self.folder / "complete.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["sha256"][name] = sha256(self.folder / name)
        path.write_text(json.dumps(record), encoding="utf-8")

    def test_recorded_search_is_reconstructed(self):
        result = audit_job(str(self.folder))
        self.assertEqual(result["searches"], 3)
        self.assertEqual(result["calls"], 1620)

    def test_changed_selected_mean_is_rejected(self):
        path = self.folder / "runs.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "selected_observed_score"] += .01
        frame.to_csv(path, index=False)
        self.update_recorded_hash("runs.csv")
        with self.assertRaisesRegex(ValueError, "Selected mean"):
            audit_job(str(self.folder))

    def test_missing_archived_observation_is_rejected(self):
        path = self.folder / "evaluations.csv.gz"
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            lines = stream.readlines()
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.writelines(lines[:10] + lines[11:])
        self.update_recorded_hash("evaluations.csv.gz")
        with self.assertRaisesRegex(ValueError, "sequence"):
            audit_job(str(self.folder))

    def test_changed_final_assessments_are_rejected(self):
        path = self.folder / "runs.csv"
        frame = pd.read_csv(path)
        counts = json.loads(frame.loc[0, "assessment_correct_counts"])
        counts[0] += 1
        frame.loc[0, "assessment_correct_counts"] = json.dumps(counts)
        frame.to_csv(path, index=False)
        self.update_recorded_hash("runs.csv")
        with self.assertRaisesRegex(ValueError, "final NN"):
            audit_job(str(self.folder))


class BenchmarkMetadataTests(unittest.TestCase):
    def test_cpu_rounding_is_accepted_but_parameter_changes_are_rejected(self):
        task = RobustBenchmark("sphere", 2, "gaussian")
        metadata = task.metadata()
        metadata["rotation"][0][0] = float(np.nextafter(metadata["rotation"][0][0], np.inf))
        metadata["scale"] *= 1 + 5e-15
        self.assertFalse(check_benchmark_metadata(task, metadata))
        for field, value in (("scale", task.scale * (1 + 1e-7)),
                             ("design_seed", task.design_seed + 1)):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "transformation or calibration"):
                    check_benchmark_metadata(task, {**metadata, field: value})
        metadata["rotation"][0][0] += 1e-7
        with self.assertRaisesRegex(ValueError, "transformation or calibration"):
            check_benchmark_metadata(task, metadata)

    def test_exact_current_and_legacy_metadata_are_supported(self):
        for noise in NOISES:
            with self.subTest(noise=noise):
                task = RobustBenchmark("sphere", 2, noise)
                self.assertFalse(check_benchmark_metadata(task, task.metadata()))
                legacy = {**ControlledBenchmark.metadata(task), "noise_family": noise,
                          "noise_mean": 0, "noise_variance_before_heteroskedastic_factor": 1}
                self.assertTrue(check_benchmark_metadata(task, legacy))
                for field, replacement in (("noise_family", "wrong"), ("noise", "wrong"),
                                           ("noise_mean", 1), ("eta", 100)):
                    changed = {**legacy, field: replacement}
                    with self.assertRaisesRegex(ValueError, "transformation or calibration"):
                        check_benchmark_metadata(task, changed)


if __name__ == "__main__":
    unittest.main()
