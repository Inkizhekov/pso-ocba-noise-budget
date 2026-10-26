"""Regression checks for editable documentation and frozen scientific bytes."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import prepare_data
from refresh_release_manifest import MANIFESTS, refresh
from verify_reviewer_package import verify_directory


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ReleaseIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        subprocess.run(["git", "init", "--quiet", str(self.root)], check=True)
        raw = b"run,score\n1,0.25\n"
        source = self.root / "results/example/runs.csv.gz"
        source.parent.mkdir(parents=True)
        source.write_bytes(gzip.compress(raw, mtime=0))
        (self.root / "README.md").write_bytes(b"Original documentation\n")
        item = {"compressed_path": "results/example/runs.csv.gz",
                "restored_path": "results/example/runs.csv", "restored_bytes": len(raw),
                "restored_sha256": checksum(raw)}
        self.transport = {"format": "ocba-github-web-upload-v1", "compressed_files": [item],
                          "sha256": {item["compressed_path"]: checksum(source.read_bytes()),
                                     "README.md": checksum((self.root / "README.md").read_bytes())}}
        self.logical = {"format": "ocba-minimal-reviewer-v1", "full_primary_traces_included": False,
                        "trace_examples": [], "sha256": {item["restored_path"]: checksum(raw),
                        "README.md": self.transport["sha256"]["README.md"]}}
        for name, value in zip(MANIFESTS, (self.transport, self.logical)):
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")
        self.patch_root = patch.object(prepare_data, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

    def test_document_edits_do_not_block_preparation_or_data_checks(self) -> None:
        (self.root / "README.md").write_text("Edited documentation", encoding="utf-8")
        prepare_data.verify_transport(self.transport)
        prepare_data.restore_one(self.transport["compressed_files"][0])
        with patch("verify_reviewer_package.check_ledger", return_value={"checked": True}):
            self.assertEqual(verify_directory(self.root)["files_verified"], 1)
            with self.assertRaisesRegex(ValueError, "README.md"):
                verify_directory(self.root, strict_release=True)
        with self.assertRaisesRegex(ValueError, "README.md"):
            prepare_data.verify_transport(self.transport, strict_release=True)

    def test_changed_transport_is_rejected(self) -> None:
        (self.root / "results/example/runs.csv.gz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "runs.csv.gz"):
            prepare_data.verify_transport(self.transport)
        with self.assertRaisesRegex(ValueError, "refresh refused"):
            refresh(self.root, write=True)

    def test_restore_is_exact_idempotent_and_preserves_changed_target(self) -> None:
        item = self.transport["compressed_files"][0]
        prepare_data.restore_one(item)
        prepare_data.restore_one(item)
        target = self.root / item["restored_path"]
        self.assertEqual(checksum(target.read_bytes()), item["restored_sha256"])
        target.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "not overwritten"):
            prepare_data.restore_one(item)
        self.assertEqual(target.read_bytes(), b"changed")

    def test_refresh_cannot_bless_changed_restored_results(self) -> None:
        prepare_data.restore_one(self.transport["compressed_files"][0])
        (self.root / "results/example/runs.csv").write_bytes(b"changed")
        before = (self.root / MANIFESTS[1]).read_bytes()
        with self.assertRaisesRegex(ValueError, "refresh refused"):
            refresh(self.root, write=True)
        self.assertEqual((self.root / MANIFESTS[1]).read_bytes(), before)

    def test_refresh_updates_docs_without_restoring_csv(self) -> None:
        (self.root / "README.md").write_text("Edited documentation", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            refresh(self.root, write=False)
        refresh(self.root, write=True)
        refresh(self.root, write=False)
        manifest = json.loads((self.root / MANIFESTS[0]).read_text(encoding="utf-8"))
        prepare_data.verify_transport(manifest, strict_release=True)
        self.assertFalse((self.root / "results/example/runs.csv").exists())

    def test_new_research_requires_explicit_addition(self) -> None:
        (self.root / "article.pdf").write_bytes(b"new scientific artifact")
        with self.assertRaisesRegex(ValueError, "--add-research required"):
            refresh(self.root, write=True)
        refresh(self.root, write=True, additions={"article.pdf"})
        (self.root / "article.pdf").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "refresh refused"):
            refresh(self.root, write=True, additions={"article.pdf"})


if __name__ == "__main__":
    unittest.main()
