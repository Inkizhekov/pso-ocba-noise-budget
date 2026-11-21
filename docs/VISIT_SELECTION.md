# Where the final quality difference arose

Table 11 in the November English manuscript uses the completed validation searches. Each of the 16 fixed instances has 30 seed blocks. Each block contains three methods under both Gaussian and centered lognormal noise. This exploratory family was selected after the validation results were available.

The true optimum quality is zero. For one search, let q* be the highest noise-free quality among evaluated points and q be the noise-free quality of the returned point. Visit loss equals -q*. Final selection loss equals q* - q. Their sum equals the search's final regret (-q). The analyzer computes each paired difference as control loss minus PSO-OCBA loss. A positive difference favors PSO-OCBA. These estimates describe trajectories the algorithms actually visited.

For each of the two controls and each noise distribution, the analyzer estimates both components. It resamples the 30 complete seed vectors within each fixed instance and weights all 16 instances equally. The eight simultaneous 95% intervals use the maximum absolute centered studentized statistic across the eight contrasts, 99,999 draws and random seed 21112026. The resulting critical value is 2.81805. Interval endpoints from different coordinates cannot be added.

## Reproduce Table 11

Install the pinned Python 3.13 environment using the repository README. Run from the directory containing the analyzer. Choose a fresh output path each time.

```text
python analyze_visit_selection.py --components results/ocba_visit_selection_20261121/search_components.csv --out review/table11
```

Compare the eight rows in `review/table11/component_contrasts.csv` with [the published table](../results/ocba_visit_selection_20261121/component_contrasts.csv). This route checks 2,880 search outcomes and selected point identifiers against the archived per-search results. The `raw_traces_checked` flag in the output is false on this route.

## Rebuild the compact input from the primary observations

Download [the new-instance primary observations](https://github.com/Inkizhekov/pso-ocba-noise-budget/releases/download/v2026.10.27/primary_traces_15_new_instances.zip) into a directory called `downloads`. From the complete repository or the accompanying reproducibility archive run:

```text
python verify_primary_traces.py --assets downloads --phase new_instances --extract-to review/primary
python analyze_visit_selection.py --cells review/primary/results/ocba_instances_20261022/cells --out review/from_primary
```

The analyzer checks completion and file digests for all 32 jobs, joins the selected point back to each search, and reads the true qualities of 4,147,200 visited pools. The compact `search_components.csv` contains a row for every official validation search. No additional objective evaluations enter either analysis. The accompanying `analysis.json` identifies which route produced the table.

Under lognormal noise, equal allocation had 0.6500 units more visit loss and 0.6476 units more final selection loss than PSO-OCBA. Against ordinary OCBA the corresponding differences were -0.0723 and -0.1472. These are normalized quality units on the fixed tasks. Record choices influence subsequent visits, so the decomposition alone cannot establish a causal mechanism.
