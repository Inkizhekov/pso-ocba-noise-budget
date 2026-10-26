"""Select frozen scientific artifacts separately from editable release files."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath


def is_research_artifact(name: str) -> bool:
    path = PurePosixPath(name)
    return name.startswith(("data/", "datasets/", "results/")) or path.suffix.lower() in {
        ".docx", ".pdf"
    }


def selected_hashes(manifest: dict, *, strict_release: bool = False) -> dict[str, str]:
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("The manifest has no file checksums")
    selected = {name: value for name, value in hashes.items()
                if strict_release or is_research_artifact(name)}
    if not selected:
        raise ValueError("The manifest has no scientific artifacts")
    return selected


def checked_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
        raise ValueError(f"Unsafe payload path: {name}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Payload path leaves the repository: {name}")
    return path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
