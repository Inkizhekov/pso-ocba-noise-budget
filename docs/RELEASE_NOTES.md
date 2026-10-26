# Repository corrections, 26 October 2026

## Scientific release v2026.10.27

- Fixed the historical command-line entry point, including invocation from
  another directory and caller-relative output paths. Added subprocess
  regression checks for the public `legacy` and historical `--mode` interfaces.
- Documented Git as a prerequisite for the full test suite and maintainer tools.
- Added versioned software and unpublished-manuscript citation metadata.
- Limited cross-CPU rounding tolerance to the recomputed QR rotation and
  calibration scale in the trace auditor (`rtol=1e-12`, `atol=1e-14`). All
  other benchmark metadata and archived-file checksums remain strict; the
  separate reconstruction of selected true scores is still checked.
- Prepared separate primary-data assets from the inventoried official jobs,
  excluding manuscript revisions, build history and unrelated technical runs.

The changes do not alter optimization rules or the recorded research results.

## Initial repository corrections

The preserved scientific CSV files, datasets, protocols, manuscript DOCX files
and earlier release ZIP files have not been changed.

- New synthetic job metadata now names the actual standardized noise family.
  Previously, `noise_family` was correct but an inherited `noise` description
  always said Gaussian. Noise generation, random-number draws and search
  rules are unchanged. Existing archived metadata is preserved as evidence.
  The trace auditor recognizes the exact original format and the corrected
  format, and continues to reject other parameter or calibration changes.
- The MIT License now covers the author's code and technical documentation;
  research outputs, manuscripts and third-party datasets have separate terms.
- Default preparation and verification check scientific artifacts. Strict
  release checks also validate code and documentation. A maintainer refresh
  refuses to replace checksums of changed frozen scientific artifacts.
- New result paths beginning with `results/new_` are ignored, matching the
  Russian README examples. Automated checks run on Windows and Python 3.13.3.
- PDF reading copies accompany the existing editable manuscripts. The full
  primary archive is identified accurately as not publicly hosted.

`data/source_equivalence.json` is the preserved report for the initial public
snapshot, commit `672bc7779bb9a92248ce1be72ad22c0141134bfe`. It is not a new
claim of source identity after these corrections. Fresh jobs record current
source hashes; run them in new output directories. Do not change archived
protocol hashes to resume an old run with edited code.
