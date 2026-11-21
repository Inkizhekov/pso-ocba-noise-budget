# Data availability

The [current English preprint by A. A. Inkizhekov and L. A. Valuiskaya](https://github.com/Inkizhekov/pso-ocba-noise-budget/releases/tag/v2026.11.22) uses the November analysis. The [21 November manuscript and calculations](https://github.com/Inkizhekov/pso-ocba-noise-budget/releases/tag/v2026.11.21) include the exploratory decomposition of 2,880 saved validation searches. The [published per-search losses and eight contrasts](../results/ocba_visit_selection_20261121) reproduce Table 11. See the [analysis instructions](VISIT_SELECTION.md). The primary-data release is v2026.10.27.

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

For protocol-bound source verification, use the corrected
[scientific_sources_20261027.zip](https://github.com/Inkizhekov/pso-ocba-noise-budget/releases/download/v2026.10.27/scientific_sources_20261027.zip)
and its [SHA-256 file](https://github.com/Inkizhekov/pso-ocba-noise-budget/releases/download/v2026.10.27/scientific_sources_20261027.sha256).
The archive is 143176 bytes; its SHA-256 is
`da8e41e5a98eabc8580506479397647acdf181c7d77e14e281a42a35b0def2fd`.
It includes the exact 20261022 modules for original and expanded protocols,
and the 20261024 modules for the ablation and new-instance protocols. Three
historical modules missing from the earlier source asset are restored with
their original protocol-matching hashes. No source-hash guard is bypassed.
The earlier `scientific_sources_20261024.zip`, primary-data manifest and
`SHA256SUMS.txt` remain unchanged; the new archive has its own checksum and
internal `SOURCE_RECOVERY_MANIFEST.json`.

Extract the corrected source archive into a separate directory. From there,
use the pinned environment to run the included verification commands, replacing
the example paths with your downloaded repository and assets:

```powershell
python verify_sources.py --repository D:/Review/pso-ocba-noise-budget-2026.10.27
python audit_saved_phase.py --repository D:/Review/pso-ocba-noise-budget-2026.10.27 --assets D:/Review/assets --out D:/Review/pilot_audit --workers 2
```

The second command requires `primary_traces_12_nn_pilot.zip` and a new output
directory. It runs the original analyzer with source verification enabled on
144 saved jobs, 432 searches and 233280 evaluations; it does not retrain models.
The archive's README describes the other phase choices.

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

## Contrast definitions and October manuscripts

The [contrast ledgers](CONTRAST_LEDGERS.md) export exact weights for the
unchanged families of 23 and 72 mean contrasts. The additional exports and
the revised article texts in `manuscripts/revision_20261027/` are identified
by the accompanying repository manifests. They do not replace the scientific
protocols, primary observations, saved estimates or release tag. Older
manuscript files remain available for byte-level reference.
