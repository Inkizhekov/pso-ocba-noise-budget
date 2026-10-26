"""Check safe restoration and corruption rejection for primary release data."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from research_integrity import digest
from verify_primary_traces import verify_part


class PrimaryTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def asset(self, name: str = "results/cell/runs.csv") -> tuple[Path, dict]:
        data = b"run,score\n1,0.25\n"
        path = self.root / "data.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr(name, data)
        part = {"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
                "files": [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]}
        return path, part

    def test_restore_is_idempotent_and_preserves_changed_files(self) -> None:
        path, part = self.asset()
        destination = self.root / "restored"
        self.assertEqual(verify_part(path, part, destination)["files_restored"], 1)
        self.assertEqual(verify_part(path, part, destination)["files_restored"], 0)
        target = destination / part["files"][0]["path"]
        target.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "not overwritten"):
            verify_part(path, part, destination)
        self.assertEqual(target.read_bytes(), b"changed")

    def test_corrupt_file_is_rejected_before_restoration(self) -> None:
        path, part = self.asset()
        part["files"][0]["sha256"] = "0" * 64
        destination = self.root / "restored"
        with self.assertRaisesRegex(ValueError, "Primary file checksum"):
            verify_part(path, part, destination)
        self.assertFalse((destination / part["files"][0]["path"]).exists())

    def test_path_traversal_is_rejected(self) -> None:
        path, part = self.asset("results/../../escape.csv")
        with self.assertRaisesRegex(ValueError, "Unsafe payload path"):
            verify_part(path, part, self.root / "restored")
        self.assertFalse((self.root / "escape.csv").exists())


if __name__ == "__main__":
    unittest.main()
