# Data availability

Scientific release: v2026.10.27.

The public repository includes the individual results of all 126720 searches,
input datasets, frozen protocols, power plans, statistical tables and three
examples of complete primary observations. It supports recomputation of the
reported statistics and reconstruction of these examples. It does not include
complete primary observations for all 351216000 objective evaluations.

## Complete primary observations

[Release v2026.10.27](https://github.com/Inkizhekov/pso-ocba-noise-budget/releases/tag/v2026.10.27)
provides 15 independently extractable `primary_traces_*.zip` assets. They
contain the 3806 official job folders, with 15470 files covering all seven
search series. Each included byte was checked against the preserved full
archive's manifest; each job completion was also checked against
`data/job_inventory.csv`. Old manuscripts, build scripts, development history
and unrelated technical runs are excluded.

The additional `scientific_sources_20261024.zip` asset preserves the original
bytes of 26 scientific modules needed to identify the implementation behind
the archived protocols. Its checksum is recorded in the primary-data manifest
and `SHA256SUMS.txt`. It contains no earlier manuscripts or editorial builders.

The source repository remains the compact review package. Downloading its
source ZIP does not automatically download the primary-data release assets.
No DOI (Digital Object Identifier) has been assigned to this release.

`data/primary_traces_manifest.json` records every asset's size and SHA-256,
the scientific phase, and every enclosed file's size and SHA-256. The release
also supplies `SHA256SUMS.txt`. Each asset is smaller than 2 GiB, the
[GitHub release asset limit](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases#storage-and-bandwidth-quotas).

After downloading the 15 ZIP files to one directory, verify them with Python's
standard library:

```powershell
py -3.13 verify_primary_traces.py --assets D:/Downloads/ocba_primary
```

To verify and restore one phase into a separate review directory:

```powershell
py -3.13 verify_primary_traces.py --assets D:/Downloads/ocba_primary --phase nn_confirmatory --extract-to reviewer_checks/full_primary
```

The available phases are `original_synthetic`, `original_nn`,
`expanded_synthetic`, `nn_pilot`, `nn_confirmatory`, `n0_ablation`, and
`new_instances`. The script requires only assets belonging to the selected
phase. Restored job folders appear below the destination's `results/`.
An existing differing file is refused and is not overwritten. Omitting
`--extract-to` checks assets without restoring their contents.

These commands check transport and file integrity. They do not rerun searches
or claim an independent statistical replication. Existing scientific auditors
can inspect the restored job folders; the compact repository still contains
the frozen protocols, aggregate tables and analysis scripts.

## Preserved original archive

The original archive below remains local and unchanged. It includes manuscript
and development history, so it is not the distribution uploaded to GitHub:

| Item | Value |
|---|---|
| Filename | `ocba_reviewer_full_20261024.zip` |
| Size in bytes | `18438980862` |
| SHA-256 (Secure Hash Algorithm, 256-bit) | `2470084cb1dc7c73d3d44ade36992ee5fcb2b57817d1f9638ffa1f12194d70c4` |

The filename and checksum identify the preserved original archive. The release
assets above provide its official primary data without requiring this older
history-containing ZIP. Questions can be discussed through
[repository issues](https://github.com/Inkizhekov/pso-ocba-noise-budget/issues).

The two repository manifests record transport and restored-file checksums.
The Git commit identifies the accompanying code, documentation and manifests.
See [verification and maintenance](GITHUB_UPLOAD_RU.md) for default scientific
checks and optional strict release checks. A GitHub-generated ZIP has no
separately supplied `.zip.sha256` file in this repository.

Code and technical documentation use MIT; manuscripts, generated research
outputs and third-party datasets have the separate scope described in
[reuse information](../THIRD_PARTY_NOTICES.md).
