# Exact weights for the existing contrast families

These files document the already reported exploratory families. They do not
introduce new hypotheses, modify results or change the resampling procedure.

| Family | CSV (comma-separated values) | JSON (JavaScript Object Notation) | Weight rows |
|---|---|---|---|
| 23 means and noise-shape interactions | [CSV](../data/contrast_ledger_23.csv) | [JSON](../data/contrast_ledger_23.json) | 10368 |
| 72 initial-sample ablation contrasts | [CSV](../data/contrast_ledger_72.csv) | [JSON](../data/contrast_ledger_72.json) | 432 |

Each row identifies a scientific series, function, distribution, noise level,
swarm, budget, initial sample size, generation multiplier and method. The
coefficient is the exact rational number `numerator / denominator`. It multiplies
the cell's mean `final_score` over the 30 shared run indices 0 through 29.
Summing these terms gives the reported contrast estimate. Positive quality
differences favor the first method. Units are normalized quality, not percentages.

The 23-family contains 3 shape contrasts, 6 budget interactions, 6 swarm-size
interactions, 3 noise-level interactions and 5 means. `mean_all` averages the
four distribution-specific means uniformly. Its 576 nonzero-noise conditions
retain equal weights. The 72-family contains 36 method means, 24 changes from
initial sample size 3 to 5 or 10, and 12 contrasts against the original
generation-budget setting. The two ablation functions have equal weight.

Coefficients describe the estimator; they are not independent sampling units.
For bootstrap inference, keep the entire vector of shared seeds across
conditions together within each fixed function, exactly as the scientific
analyzers do. Do not resample individual ledger rows independently.

After installing the pinned requirements and running `prepare_data.py`:

```powershell
.\.venv\Scripts\python.exe export_contrast_ledgers.py --check
```

The exporter independently constructs rational weights, evaluates them against
the existing aggregate observations, and compares the results with
`analyze_ocba_interactions.py` and `analyze_n0_ablation.py` at absolute and
relative tolerance `1e-12`. The 72 definitions are also checked in their
original order. The saved [validation record](../data/contrast_ledger_validation.json)
identifies the input and output digests. JSON files additionally record the
scientific implementation hashes. Checking writes nothing and runs no searches.
Generation into another directory uses `--out PATH`; differing existing files
are refused, never silently overwritten.
