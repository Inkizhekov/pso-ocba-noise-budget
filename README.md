# PSO-OCBA under noise and a finite evaluation budget

[English](README.md) | [Русский](README.ru.md)

[Scientific release v2026.10.27](https://github.com/Inkizhekov/pso-ocba-noise-budget/releases/tag/v2026.10.27)
provides a fixed reference for this repository. Use [CITATION.cff](CITATION.cff)
to cite the software or accompanying unpublished manuscript. No journal
publication or DOI (Digital Object Identifier) is claimed.

GitHub browser-upload edition, 26 October 2026.
The manuscripts include concise dataset descriptions and primary-source UCI citations. Numerical results are unchanged. Start with the verification commands below.
All seven scientific series are represented by individual search results. Complete
primary observations are included only for three audit examples; this is not a full-trace archive.

This project studies when repeated evaluations improve Particle Swarm
Optimization (PSO) under stochastic noise and a fixed number of objective
calls. Optimal Computing Budget Allocation (OCBA) distributes additional
evaluations among candidate solutions instead of allocating them uniformly.

The project implements a controlled empirical comparison of existing methods.
It does not introduce a new optimizer or establish a universal applicability
rule. The current manuscript is
[article_vak_swarm_noise_ocba_datasets_20261025.docx](article_vak_swarm_noise_ocba_datasets_20261025.docx),
with [supplementary appendices](article_vak_swarm_noise_ocba_supplement_datasets_20261025.docx)
(appendices А-З in the Russian manuscript).

For a detailed walkthrough of the experiment, code, data formats, statistics,
and troubleshooting, see the [Russian experiment guide](docs/EXPERIMENT_GUIDE_RU.md).

## Manuscripts in both languages

- [English article](article_vak_swarm_noise_ocba_english_20261026.docx) and [appendices A-H](article_vak_swarm_noise_ocba_supplement_english_20261026.docx).
- [Russian article](article_vak_swarm_noise_ocba_datasets_20261025.docx) and [appendices](article_vak_swarm_noise_ocba_supplement_datasets_20261025.docx).
- PDF reading copies: [English article](manuscripts/article_en.pdf), [English appendices](manuscripts/supplement_en.pdf), [Russian article](manuscripts/article_ru.pdf), [Russian appendices](manuscripts/supplement_ru.pdf).
- [MIT License for code and technical documentation](LICENSE); see the separate scope for research materials below.
- [Input attributions and reuse information](THIRD_PARTY_NOTICES.md).
- [Bibliography](docs/REFERENCES.md).

The English files translate the current Russian version. Results, limitations and editable equations are preserved. The two language versions are current deliverables, not revision history.

## Prepare the included results

After downloading the complete repository, restore the table compressed for browser upload before running any saved-result verification:

```powershell
py -3.13 prepare_data.py
py -3.13 verify_reviewer_package.py
```

The standard-library preparation script checks frozen scientific artifacts and
restores the original CSV byte for byte. It requires no network access. An
existing changed CSV is rejected, not overwritten. The compressed copy remains
available. Running preparation again is safe. Edits to code or documentation
do not block these data checks. For an exact release-byte check, including
code and documentation, use `prepare_data.py --verify-only --strict-release`
and `verify_reviewer_package.py --strict-release`. Maintainers can check or
refresh release checksums using [the maintenance instructions](docs/GITHUB_UPLOAD_RU.md).

For publication steps, see [the upload instructions](docs/GITHUB_UPLOAD_RU.md).

## What is compared

| Method | Code identifier | Evaluation policy |
|---|---|---|
| PSO | `pso` | One evaluation per new particle position |
| PSO-EA, Equal Allocation | `pso_ea` | Initial repeats followed by uniform target allocation |
| PSO-OCBA-RS, Ranking and Selection | `pso_ocba_rs` | Ordinary OCBA for selecting the best candidate |
| PSO-OCBA | `pso_ocba` | PSO-specific allocation accounting for personal and global record decisions |

All four methods use the same particle movement rules. The comparisons
separate the benefit of repeated measurements, adaptive allocation, and
the additional personal-record model.

For repeat-based methods, a full generation uses `6*M` evaluations, with
`M` particles, three initial observations per new position, and allocation
increments of `M`. The previous global record remains a sampled candidate
with its accumulated observations. Uniform allocation therefore targets
cumulative counts; it is not simply six fresh evaluations of every particle.

An equal evaluation budget does not imply equal elapsed time or equal
arithmetic cost. All methods maximize quality. For synthetic tasks, the
true optimum is zero and negative scores closer to zero are better.

## Environment

The recorded scientific environment uses Windows and Python 3.13.3.
Git must also be installed to run the complete test suite and maintain release
manifests. Check it with `git --version`. Data preparation and saved-result
verification use Python and do not require Git.
[requirements.txt](requirements.txt) pins the numerical dependencies:

```text
numpy==2.3.5
scipy==1.17.0
scikit-learn==1.8.0
optuna==4.8.0
pandas==2.2.3
matplotlib==3.10.8
```

The current study runs on the CPU (Central Processing Unit); no GPU
(Graphics Processing Unit) is required. Optuna supports the historical
Tree-structured Parzen Estimator (TPE) baseline.

Run these PowerShell commands from the directory containing `experiment.py`.
Create the environment only if it does not already exist:

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

`py -3.13` selects an installed Python 3.13 interpreter. Check its exact
patch version when reproducing the recorded environment. Explicit interpreter
paths avoid the need to activate the environment. In the author's working
copy, `.review_venv` is an existing environment with the pinned dependencies;
replace `.venv` in examples if using it. Environments are not included
in the reviewer archives.

The neural-network study reads `datasets/uci/ionosphere.csv`,
`datasets/uci/sonar.csv`, and their `manifest.json`. These inputs are included
in the complete package and checked against their recorded SHA-256
(Secure Hash Algorithm, 256-bit) digests. A fresh checkout without the inputs
can obtain them using `fetch_ocba_data.py`; an intact reviewer package
does not need to download them again.

## Start with a dry run

```powershell
.\.venv\Scripts\python.exe experiment.py --help
.\.venv\Scripts\python.exe experiment.py --dry-run --out results/my_study
```

`--dry-run` prints the protocol and expected workload without writing result
files or running searches. The main synthetic grid contains 77760 searches
and 293932800 objective evaluations. The independent neural-network pilot
contains 432 searches. The confirmatory sample size is determined from
the pilot; it is not hard-coded to the manuscript's observed value of 456.

A synthetic smoke test checks execution on one technical condition:

```powershell
.\.venv\Scripts\python.exe experiment.py --smoke --dry-run --out results/my_smoke
.\.venv\Scripts\python.exe experiment.py --smoke --workers 1 --out results/my_smoke
```

This runs one search for each of four methods at a budget of 8640,
for 34560 calls in total. Smoke output directory names must contain
`smoke`. A smoke run is not a scientific sample and cannot be promoted
to a full study by reusing the same directory.

## Run the main study

The following command starts the full main pipeline, including the neural
networks. It is a substantial computation:

```powershell
.\.venv\Scripts\python.exe experiment.py --out results/my_study --workers 8 --audit-workers 6
```

Without `--out`, the destination is `results/ocba_current`.
The pipeline executes these stages in order:

| Stage | Purpose |
|---|---|
| `freeze` | Save the design, source digests, and actual runtime environment |
| `synthetic` | Run the expanded synthetic grid |
| `nn_pilot` | Collect independent neural-network pilot observations |
| `plan` | Freeze a pilot-derived confirmatory sample size |
| `validate` | Independently simulate and check the power plan before main observations |
| `nn_main` | Run the confirmatory neural-network study using separate search seeds |
| `analyze` | Audit raw traces and calculate the main tables and figures |

Stages can also be run individually:

```powershell
.\.venv\Scripts\python.exe experiment.py --phase freeze --out results/my_study
.\.venv\Scripts\python.exe experiment.py --phase synthetic --workers 8 --out results/my_study
.\.venv\Scripts\python.exe experiment.py --phase nn_pilot --workers 8 --out results/my_study
.\.venv\Scripts\python.exe experiment.py --phase plan --out results/my_study
.\.venv\Scripts\python.exe experiment.py --phase validate --out results/my_study
.\.venv\Scripts\python.exe experiment.py --phase nn_main --workers 8 --out results/my_study
.\.venv\Scripts\python.exe experiment.py --phase analyze --audit-workers 6 --out results/my_study
```

The full `analyze` stage needs all main-study components, including the
neural-network results and validated power plan. There are no current
`--budget`, `--runs`, or `--epochs` options: the main design is fixed
in code. Those options belong to the historical pilot interface.

## Resume and source integrity

The original pilot remains available through the public entry point:

```powershell
.\.venv\Scripts\python.exe experiment.py legacy --help
```

The historical `--mode` interface is also supported. Relative output paths
remain relative to the caller's working directory.

After interruption, repeat the same command with the same output directory.
Completed jobs are recognized by `complete.json`; an incomplete job can be
recomputed in full. There is no checkpoint inside an individual search.
Raw-file integrity is checked during analysis. Do not run two controlling
processes against the same output directory.

Protocols record byte-level source digests. Even a comment changes a source
digest, so the current commented sources require a fresh output directory
when an older directory was frozen against different bytes. Keep recorded
digests intact. Recorded protocols are preserved. `data/source_equivalence.json`
is the report for the initial public commit, not a claim of source identity
after the metadata correction. See [repository corrections](docs/RELEASE_NOTES.md).

An existing power plan is reused only after checking its links to the pilot,
protocol, and planning code. Independent power validation must precede
`nn_main`; the main results are not used to replan the sample size.

Worker counts can change on resume. Numerical library thread counts are
limited to one per worker. Appended progress logs can repeat completed jobs
after resume, so count unique jobs rather than summing every progress line.

## Experimental design

### Expanded synthetic series

| Parameter | Values |
|---|---|
| Functions | Sphere, Ellipsoid, Rosenbrock, Ackley, Rastrigin, Griewank, Levy, Zakharov |
| Dimension | 8 |
| Swarm sizes | 6, 12, 24 |
| Evaluation budgets | 540, 2160, 8640 |
| Nonzero noise levels | 1, 2 |
| Noise families | Gaussian, standardized Student t3, centered standardized lognormal, contaminated Gaussian mixture |
| Zero-noise control | Gaussian label only, noise level 0 |
| Searches | 30 per method and condition |

All noise families have zero mean and unit variance before multiplication
by `eta*(0.6 + 0.8*x[0])`. Moments are matched at a fixed position; methods
can visit different positions. Nonnormal noise stresses the misspecified
Gaussian allocation model without changing the model itself.

There are `8*(1 + 4*2)*3*3 = 648` conditions. Four methods and 30 searches
per condition produce 77760 searches. True quality is the negative
analytical loss divided by a fixed calibration standard deviation.
The synthetic practical threshold 0.02 is in normalized score units,
not percentage points of accuracy.

### Neural-network series

The study uses Ionosphere and Sonar with three fixed stratified data splits
each, an approximately 60/20/20 train/validation/test partition, and a small
Multilayer Perceptron (MLP). Scaling is fitted only on training data during
search. After selection, the model and scaler are fitted on train plus
validation data, and performance is assessed on the held-out test set.

The search selects hidden-layer width in multiples of 16 from 16 to 128,
learning rate from `1e-4` to about `3.16e-2`, and L2 regularization from
`1e-6` to `1e-2`. Training uses 20 epochs, Adam, and batch size 64.
PSO, EA, and OCBA are compared at `M=12`, `B=540`.

An independent pilot has 24 searches per method and condition. Planning
targets 90% marginal model-based power for a one-percentage-point accuracy
effect, using conservative per-test alpha `0.05/12`, noncentral t calculations,
and simulation checks. The recorded plan selected 456 searches per method
and condition, yielding 8208 confirmatory searches.

Each selected configuration receives three new final training runs, averaged
into one search outcome. These fits are counted outside the search budget;
they are not three independent searches. More search seeds do not increase
the number of independent test objects or make overlapping splits
independent datasets.

### Other series

The default entry point does not rerun every historical or supplementary
series. Separate scripts cover the focused initial-repeat ablation,
new function instances, and post-hoc mean analyses:

```powershell
.\.venv\Scripts\python.exe run_n0_ablation.py --out results/my_n0_study --workers 8
.\.venv\Scripts\python.exe analyze_n0_ablation.py --out results/my_n0_study --audit-workers 6
.\.venv\Scripts\python.exe run_ocba_instance_validation.py --out results/my_instances --workers 8
```

The n0 ablation compares 3, 5, and 10 initial observations on two functions,
holding the generation budget at `12*M` in all arms. It also requires the
recorded historical bridge table in
`results/ocba_robustness_20261022/synthetic_runs.csv`.

The instance study checks two new transformations of each original function,
16 instances in total, at `M=12`, `B=8640`, `eta=2`, with Gaussian and
lognormal noise. These are new instances of the same eight families,
not eight newly sampled function families.

Post-hoc analysis scripts are tied to the article's recorded inputs and
expected grids. An output-path option does not make them general-purpose
analyzers for arbitrary data. See the detailed guide before adapting them.

## Code map

| Module | Responsibility |
|---|---|
| [experiment.py](experiment.py) | Main pipeline, provenance checks, and historical dispatch |
| [run_robust_ocba_study.py](run_robust_ocba_study.py) | Frozen design, noise generators, parallel jobs, and raw output |
| [robust_ocba_engine.py](robust_ocba_engine.py) | One complete search with configurable swarm and initial repeats |
| [ocba_engine.py](ocba_engine.py) | Observation pools, sampling, budget accounting, and the earlier fixed engine |
| [ocba.py](ocba.py) | Gaussian PSO-specific allocation, ordinary OCBA, and integer top-ups |
| [ocba_benchmarks.py](ocba_benchmarks.py) | Analytical functions, transformations, calibration, and true scores |
| [run_ocba_study.py](run_ocba_study.py) | Earlier OCBA grid and tabular dataset loading |
| [plan_robust_nn_power.py](plan_robust_nn_power.py) | Pilot-only sample-size planning |
| [validate_robust_power.py](validate_robust_power.py) | Independent simulation validation of the plan |
| [analyze_robust_ocba_study.py](analyze_robust_ocba_study.py) | Raw-trace audit and primary analyses |
| [study_statistics.py](study_statistics.py) | Grid validation, paired ranks, Holm adjustment, pointwise intervals |
| [paired_mean_inference.py](paired_mean_inference.py) | Joint inference for paired mean contrasts on fixed tasks |
| [analyze_ocba_interactions.py](analyze_ocba_interactions.py) | Exploratory noise-shape and interaction contrasts |
| [analyze_ocba_scope.py](analyze_ocba_scope.py) | Per-family effects, leave-one-family-out and hierarchical sensitivity |
| [diagnose_ocba_traces.py](diagnose_ocba_traces.py) | Reconstruction and diagnosis of personal and global records |
| [audit_ocba_revision.py](audit_ocba_revision.py) | Separately coded arithmetic audit of recorded comparisons |
| [Tabular objective implementation](historical/ocba_20261022/experiment.py) | Required runtime import; only three dependency files are retained |

## Outputs and recorded results

Each completed job contains:

- `runs.csv`: one row per full search, including `final_score`, selected
  coordinates, observed selected mean, calls, generations, and final assessments.
- `evaluations.csv.gz`: one row per search-time objective observation.
- `pools.csv.gz`: created positions, particle indices, and synthetic true scores.
- `complete.json`: job metadata, totals, elapsed duration, and file digests.

Compare methods using `final_score`, not the noisy
`selected_observed_score` used to select the winner. Pair results by condition
and `run`; `pool_id` is unique only within an individual search.

| Recorded directory | Contents |
|---|---|
| `results/ocba_extension_20261022` | Earlier small-budget synthetic and neural-network series |
| `results/ocba_robustness_20261022` | Expanded grid, independent pilot, confirmatory neural-network series |
| `results/ocba_mean_analysis_20261022` | Exploratory mean contrasts and neural-network threshold checks |
| `results/ocba_n0_ablation_20261022` | Initial-repeat ablation |
| `results/ocba_instances_20261022` | New-instance searches and five mean contrasts |
| `results/ocba_scope_20261023` | Generalization sensitivity and record diagnostics |

Across all seven official search series, the recorded totals are 3806 jobs,
126720 searches, 351216000 search-time objective evaluations, and 28080
final neural-network fits. Technical smoke runs are excluded.
Summed worker elapsed times are not CPU time or whole-run wall time.

## Statistical interpretation

Primary synthetic comparisons use a family of 1296 two-sided Wilcoxon tests
with Holm adjustment. The 648 OCBA-versus-RS comparisons form a separate
secondary family. All-zero comparisons remain in the family with p=1.
Wilcoxon is supplemented by a sign-test sensitivity analysis; neither
should be treated as a general test of the mean under arbitrary asymmetry.

A cell satisfying adjusted p < 0.05 and an estimated mean gain >= 0.02
is counted descriptively. This conjunction does not test whether the true
mean exceeds 0.02. Direct mean inference is reported separately.

Post-hoc families use a single-step centered studentized max-t bootstrap,
preserving paired seed vectors within fixed tasks. The 23-contrast family
is exploratory; correction does not undo selection of questions after
seeing the original results. Bootstrap coverage is approximate, and this
implementation is not the step-down Romano-Wolf procedure.

On the 16 fixed new instances, the lognormal-minus-Gaussian difference
in the OCBA-versus-EA effect is about `1.1668 [0.9621, 1.3714]`.
A later hierarchical sensitivity scenario gives `[-3.0175, 5.3511]`.
These assess different sources of variation. Transfer to an arbitrary
new function family is not established. The ordinary Gaussian mean effect
against EA is also positive on the original fixed grid, so the results
do not imply that OCBA helps only under nonnormal noise.

The personal-record allocation model did not establish superiority over
ordinary OCBA-RS on the tested designs. The confirmatory neural-network
series did not establish a positive one-percentage-point accuracy benefit
on the two tabular datasets and small MLP. These findings do not prove
universal dominance, exact equivalence, or absence of every smaller effect.

## Verify the prepared repository

This is the minimal review package. It includes the current manuscript and
appendices, the English README and detailed Russian guide, current commented
scientific code, pinned dependencies, the data underlying all 126720 searches,
plans, statistical tables, record diagnostics, and three raw-trace examples.
Older manuscripts, Word build history, revision logs, redundant archives,
environments, and caches are excluded.

The `historical` directory contains only three files required by existing
Python imports, including the tabular neural-network objective. It is not
the full development-history snapshot.

Check frozen scientific files, search totals, and final-fit counts without scientific dependencies:

```powershell
py -3.13 prepare_data.py
py -3.13 verify_reviewer_package.py
```

After installing requirements, recompute the recorded statistics:

```powershell
.\.venv\Scripts\python.exe verify_reviewer_package.py --recompute
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The recomputation checks original-series comparisons, 1944 expanded rank
contrasts, the neural comparisons, joint mean families 23 and 72, five
new-instance contrasts, leave-one-family-out and hierarchical sensitivity,
MDE and bootstrap sensitivity, convergence endpoints, and the trace examples.
It does not train all networks again or replay all 351216000 objective calls.
The included tests execute small technical checks, not the full study.

`reviewer_package_manifest.json` records restored-file SHA-256 checksums;
`github_payload_manifest.json` records the uploaded representation. The two
manifests exclude themselves and each other to avoid circular checksums;
their bytes are identified by the Git commit. A GitHub-generated ZIP has no
separately supplied `.zip.sha256` file in this repository.
Download the whole repository with GitHub's Code / Download ZIP, extract it, then run prepare_data.py before verification.

Verification writes new reports under `reviewer_checks`. Complete primary
observations for the official jobs are distributed separately as 15 ZIP assets
in [release v2026.10.27](https://github.com/Inkizhekov/pso-ocba-noise-budget/releases/tag/v2026.10.27).
See [data availability](docs/DATA_AVAILABILITY.md) for checksums, phase selection
and safe restoration with `verify_primary_traces.py`. The original full archive,
which also contains development history, remains preserved locally.
Nothing in this compact package claims external
independent replication. Source-byte verification, saved-data recomputation,
and a new execution of the study are distinct checks.



## Included tests and documents

Run the scientific tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The selected existing tests cover allocation, exact budgets, pairing, incomplete
generations, raw-trace reconstruction, and statistical calculations. Tests for
the discarded publication-history workflow are not included. The final Word
documents are supplied as verified artifacts; their historical build chain
is not needed for scientific review and is excluded.
