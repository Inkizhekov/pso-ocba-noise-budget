"""Exercise the historical CLI through the public entry point in subprocesses."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent.parent


class HistoricalEntrypointTests(unittest.TestCase):
    def invoke(self, arguments: list[str], cwd: Path) -> subprocess.CompletedProcess:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        return subprocess.run(
            [sys.executable, str(ROOT / "experiment.py"), *arguments],
            cwd=cwd, env=environment, text=True, capture_output=True, timeout=120,
        )

    def test_explicit_legacy_help_from_repository(self) -> None:
        result = self.invoke(["legacy", "--help"], ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--assessment-seed-policy", result.stdout)

    def test_old_mode_alias_from_another_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for mode in (["--mode", "smoke"], ["--mode=smoke"]):
                with self.subTest(mode=mode):
                    result = self.invoke([*mode, "--help"], Path(temporary))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("Historical --mode detected", result.stderr)
                    self.assertIn("--master-seed", result.stdout)

    def test_relative_output_stays_relative_to_the_caller(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            caller = Path(temporary)
            result = self.invoke([
                "legacy", "--mode", "smoke", "--runs", "2", "--budget", "19",
                "--skip-real", "--out", "pilot_output",
            ], caller)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (caller / "pilot_output" / "runs.csv").open(newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(int(row["calls"]) == 19 for row in rows))


if __name__ == "__main__":
    unittest.main()
