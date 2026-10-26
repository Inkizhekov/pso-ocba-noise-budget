"""Check that archive audits reject inconsistencies beyond file hashes."""
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from analyze_robust_ocba_study import audit_job
from run_robust_ocba_study import sha256


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


if __name__ == "__main__":
    unittest.main()
