"""Verify the minimal reviewer package without manuscript revision history."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from typing import Callable
from zipfile import ZipFile

from research_integrity import selected_hashes


ROOT = Path(__file__).resolve().parent
MANIFEST = "reviewer_package_manifest.json"
CHUNK = 8 * 1024 * 1024
SERIES = (
    ("original_synthetic", "ocba_extension_20261022/synthetic_runs.csv", 192, 34560, 8985600, 0),
    ("original_nn", "ocba_extension_20261022/real_runs.csv", 30, 720, 86400, 2160),
    ("expanded_synthetic", "ocba_robustness_20261022/synthetic_runs.csv", 648, 77760, 293932800, 0),
    ("nn_pilot", "ocba_robustness_20261022/nn_pilot_runs.csv", 144, 432, 233280, 1296),
    ("nn_confirmatory", "ocba_robustness_20261022/nn_main_runs.csv", 2736, 8208, 4432320, 24624),
    ("n0_ablation", "ocba_n0_ablation_20261022/runs.csv", 24, 2160, 18662400, 0),
    ("new_instances", "ocba_instances_20261022/runs.csv", 32, 2880, 24883200, 0),
)


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
        raise ValueError(f"Unsafe package path: {name}")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_ledger(read: Callable[[str], bytes], manifest: dict) -> dict:
    inventory = list(csv.DictReader(io.StringIO(read("data/job_inventory.csv").decode("utf-8"))))
    keys = {(row["phase"], row["job_id"]) for row in inventory}
    if len(keys) != len(inventory):
        raise ValueError("Repeated job in the inventory")
    totals = Counter()
    report = []
    for label, relative, jobs, searches, calls, fits in SERIES:
        rows = list(csv.DictReader(io.StringIO(read("results/" + relative).decode("utf-8-sig"))))
        indexed = []
        actual_calls = actual_fits = 0
        for row in rows:
            key = tuple(row.get(k, "") for k in (
                "dataset", "dimension", "noise", "eta", "swarm", "budget", "split_seed", "instance", "n0", "method", "run"
            ))
            indexed.append(key)
            if int(row["calls"]) != int(row["budget"]) or not math.isfinite(float(row["final_score"])):
                raise ValueError(f"Invalid budget or score in {label}")
            actual_calls += int(row["calls"])
            scores = json.loads(row.get("assessment_scores") or "[]")
            if fits:
                seeds = json.loads(row["assessment_seeds"])
                if len(scores) != 3 or len(set(seeds)) != 3:
                    raise ValueError(f"Invalid final assessment in {label}")
                if not math.isclose(math.fsum(scores) / 3, float(row["final_score"]), abs_tol=1e-12):
                    raise ValueError(f"Final assessment mean differs in {label}")
            elif scores:
                raise ValueError(f"Unexpected final neural fit in {label}")
            actual_fits += len(scores)
        if len(set(indexed)) != len(rows) or (len(rows), actual_calls, actual_fits) != (searches, calls, fits):
            raise ValueError(f"Search-table totals differ for {label}")
        selected = [row for row in inventory if row["phase"] == label]
        observed = (len(selected), sum(int(r["searches"]) for r in selected),
                    sum(int(r["calls"]) for r in selected), sum(int(r["final_nn_fits"]) for r in selected))
        if observed != (jobs, searches, calls, fits):
            raise ValueError(f"Job inventory differs for {label}: {observed}")
        item = dict(phase=label, jobs=jobs, searches=searches, calls=calls, final_nn_fits=fits)
        report.append(item)
        totals.update({k: item[k] for k in ("jobs", "searches", "calls", "final_nn_fits")})
    if dict(totals) != manifest["official_totals"]:
        raise ValueError("Manifest totals differ from the scientific ledger")
    return {"series": report, "totals": dict(totals), "all_raw_observations_replayed": False}


def load_manifest(data: bytes) -> dict:
    manifest = json.loads(data)
    if manifest.get("format") != "ocba-minimal-reviewer-v1" or manifest.get("full_primary_traces_included") is not False:
        raise ValueError("Not a minimal reviewer package of the expected format")
    for name in manifest["sha256"]:
        safe_name(name)
    return manifest


def verify_archive(path: Path, destination: Path | None) -> dict:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(n.casefold() for n in names)):
            raise ValueError("Duplicate or case-colliding archive entries")
        for item in archive.infolist():
            safe_name(item.filename)
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive contains a symbolic link")
        manifest = load_manifest(archive.read(MANIFEST))
        if set(names) != set(manifest["sha256"]) | {MANIFEST}:
            raise ValueError("Manifest does not cover the complete archive")
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=False)
        for name, expected in manifest["sha256"].items():
            hasher = hashlib.sha256()
            target = None
            try:
                if destination is not None:
                    output = destination / name
                    if not output.resolve().is_relative_to(destination.resolve()):
                        raise ValueError("Extraction leaves its destination")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    target = output.open("xb")
                with archive.open(name) as stream:
                    for chunk in iter(lambda: stream.read(CHUNK), b""):
                        hasher.update(chunk)
                        if target is not None:
                            target.write(chunk)
            finally:
                if target is not None:
                    target.close()
            if hasher.hexdigest() != expected:
                raise ValueError(f"File checksum mismatch: {name}")
        ledger = check_ledger(archive.read, manifest)
        if destination is not None:
            (destination / MANIFEST).write_bytes(archive.read(MANIFEST))
    return {"files_verified": len(manifest["sha256"]), "ledger": ledger,
            "full_primary_traces_included": False, "trace_examples": manifest["trace_examples"]}


def verify_directory(root: Path, *, strict_release: bool = False) -> dict:
    manifest = load_manifest((root / MANIFEST).read_bytes())
    hashes = selected_hashes(manifest, strict_release=strict_release)
    for name, expected in hashes.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"File checksum mismatch: {name}")
    return {"files_verified": len(hashes), "strict_release": strict_release,
            "ledger": check_ledger(lambda name: (root / name).read_bytes(), manifest),
            "full_primary_traces_included": False, "trace_examples": manifest["trace_examples"]}


def compare_frame(actual, expected, keys: list[str], columns: list[str] | None = None) -> dict:
    import numpy as np
    import pandas as pd

    a = actual.set_index(keys).sort_index()
    e = expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, e.index, check_names=False)
    columns = columns or list(e.columns)
    largest = 0.0
    for name in columns:
        if pd.api.types.is_numeric_dtype(a[name]) and pd.api.types.is_numeric_dtype(e[name]):
            av, ev = a[name].to_numpy(dtype=float), e[name].to_numpy(dtype=float)
            np.testing.assert_allclose(av, ev, rtol=1e-8, atol=1e-11, equal_nan=True, err_msg=name)
            finite = np.isfinite(av) & np.isfinite(ev)
            if finite.any():
                largest = max(largest, float(np.abs(av[finite] - ev[finite]).max()))
        else:
            pd.testing.assert_series_equal(a[name].fillna("<missing>").astype(str),
                                           e[name].fillna("<missing>").astype(str), check_names=False)
    return {"rows": len(a), "columns": len(columns), "maximum_absolute_difference": largest}


def recompute(out: Path) -> dict:
    import numpy as np
    import pandas as pd

    from analyze_n0_ablation import contrast_data
    from analyze_ocba_interactions import nn_threshold_analysis
    from analyze_ocba_scope import instance_tensor, run as scope_run
    from analyze_ocba_study import comparisons, validate_results
    from analyze_robust_ocba_study import audit_job, synthetic_analysis
    from audit_ocba_hierarchy import run as hierarchy_run
    from audit_ocba_revision import run as independent_run
    from diagnose_ocba_traces import reconstruct_cell
    from experiment import verify_datasets, verify_power_plan, verify_power_validation
    from paired_mean_inference import simultaneous_mean_intervals
    from run_ocba_instance_validation import CONTRASTS

    out.mkdir(parents=True, exist_ok=False)
    robust = ROOT / "results/ocba_robustness_20261022"
    report = {"scope": "saved-search recomputation and retained trace examples, not full independent replication"}
    verify_datasets()
    plan, validation = verify_power_plan(robust), verify_power_validation(robust)
    report["power_provenance"] = {"runs_per_cell": plan["runs_per_cell"],
                                  "minimum_power_lower_bound": validation["minimum_power_lower_bound"]}
    print("Checking original-series comparisons", flush=True)
    original = ROOT / "results/ocba_extension_20261022"
    report["original"] = {}
    for phase, filename, keys in (
        ("synthetic", "synthetic", ["dataset", "dimension", "eta", "budget", "method", "baseline"]),
        ("real", "real", ["dataset", "split_seed", "budget", "method", "baseline"]),
    ):
        raw = pd.read_csv(original / f"{filename}_runs.csv")
        validate_results(raw, phase)
        report["original"][phase] = compare_frame(comparisons(raw, phase),
            pd.read_csv(original / f"{filename}_pairwise.csv"), keys)
    print("Checking expanded comparisons and sign-test sensitivity", flush=True)
    primary = out / "primary"
    primary.mkdir()
    expanded = pd.read_csv(robust / "synthetic_runs.csv")
    summary = synthetic_analysis(expanded, primary)
    report["expanded"] = compare_frame(pd.read_csv(primary / "synthetic_pairwise.csv"),
        pd.read_csv(robust / "synthetic_pairwise.csv"), ["dataset", "noise", "eta", "swarm", "budget", "baseline"])
    report["sign_sensitivity_retained_ea_cells"] = summary["ea_wins_retained_by_sign_test"]
    print("Checking 1944 ranks, 12 neural contrasts, and joint families 23 and 72", flush=True)
    report["independent_arithmetic"] = independent_run(out / "independent")
    threshold = out / "nn_threshold"
    threshold.mkdir()
    report["nn_threshold"] = nn_threshold_analysis(robust, threshold)
    compare_frame(pd.read_csv(threshold / "nn_mean_thresholds.csv"),
        pd.read_csv(ROOT / "results/ocba_mean_analysis_20261022/nn_mean_thresholds.csv"), ["dataset", "split_seed", "baseline"])
    print("Checking new instances and five joint contrasts", flush=True)
    instances = pd.read_csv(ROOT / "results/ocba_instances_20261022/runs.csv")
    tensor = instance_tensor(instances)
    inferred = simultaneous_mean_intervals(tensor.reshape(16, 30, 5), seed=12222026)
    frame = pd.DataFrame([{"id": name, **{key: inferred[key][i] for key in (
        "mean", "se", "ci_low", "ci_high", "adjusted_p_zero")}} for i, name in enumerate(CONTRASTS)])
    report["instances"] = compare_frame(frame,
        pd.read_csv(ROOT / "results/ocba_instances_20261022/mean_contrasts.csv"), ["id"])
    print("Checking scope, MDE, and bootstrap sensitivity", flush=True)
    scope = out / "scope"
    report["scope"] = scope_run(scope)
    recorded = ROOT / "results/ocba_scope_20261023"
    for filename, keys in (
        ("per_instance_effects.csv", ["dataset", "instance"]),
        ("per_family_effects.csv", ["dataset"]),
        ("leave_one_family_out.csv", ["excluded", "id"]),
        ("hierarchical_sensitivity.csv", ["id"]),
        ("swarm_mde90.csv", ["id"]),
        ("bootstrap_repetitions.csv", ["repetitions", "id"]),
        ("mean_threshold_bounds.csv", ["id"]),
    ):
        compare_frame(pd.read_csv(scope / filename), pd.read_csv(recorded / filename), keys)
    print("Checking hierarchical studentization diagnostics", flush=True)
    diagnostics = hierarchy_run(out / "hierarchy")
    archived = json.loads((ROOT / "results/ocba_final_20261024/hierarchy_diagnostic_summary.json").read_text())
    for actual, expected in zip(diagnostics["results"], archived["results"], strict=True):
        for key in ("mean", "se", "ci_low", "ci_high", "critical"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-9, atol=1e-11)
    report["hierarchy_diagnostic_cases"] = len(diagnostics["results"])
    print("Checking saved record diagnostics and convergence endpoints", flush=True)
    keys = ["dataset", "instance", "noise", "method", "run"]
    rates = pd.read_csv(recorded / "record_diagnostics.csv").set_index(keys).sort_index()
    raw = instances.set_index(keys).sort_index()
    np.testing.assert_array_equal(rates.false_accept, raw.false_accept)
    np.testing.assert_array_equal(rates.false_reject, raw.false_reject)
    np.testing.assert_allclose(rates.pbest_error_rate, (raw.false_accept + raw.false_reject) / raw.comparisons, atol=1e-12)
    curves = pd.read_csv(recorded / "record_trajectories.csv.gz")
    endpoints = curves.sort_values("calls").groupby(keys).tail(1).set_index(keys).sort_index()
    np.testing.assert_allclose(endpoints.selected_regret, -raw.final_score, atol=1e-11)
    global_errors = curves.assign(error=curves.gbest_oracle_gap.gt(1e-12)).groupby(keys).error.mean().sort_index()
    np.testing.assert_allclose(global_errors, rates.gbest_error_rate, atol=1e-12)
    if len(curves) != 345600 or len(rates) != 2880:
        raise ValueError("Incorrect record diagnostic grid")
    means = rates.reset_index().groupby(["noise", "method"], sort=True)[[
        "pbest_error_rate", "gbest_error_rate", "oracle_best_candidate_allocation_share", "final_regret"
    ]].mean().reset_index()
    compare_frame(means, pd.read_csv(recorded / "record_diagnostic_means.csv"), ["noise", "method"])
    report["retained_record_trajectories"] = len(curves)
    report["trace_examples"] = []
    manifest = load_manifest((ROOT / MANIFEST).read_bytes())
    for example in manifest["trace_examples"]:
        folder = ROOT / example["path"]
        if example["kind"] == "instance_records":
            searches, generations = reconstruct_cell(str(folder))
            result = {"searches": len(searches), "generations": len(generations)}
        else:
            result = audit_job(str(folder))
        report["trace_examples"].append({"path": example["path"], "result": result})
    report["all_primary_traces_replayed"] = False
    report["new_searches_for_statistical_recomputation"] = 0
    (out / "recomputation_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--strict-release", action="store_true", help="Also check release code and documentation bytes")
    parser.add_argument("--out", type=Path, help="New verification report directory")
    args = parser.parse_args()
    if args.extract_to is not None and args.archive is None:
        parser.error("--extract-to requires --archive")
    if args.archive and args.recompute and args.extract_to is None:
        parser.error("--recompute with --archive requires --extract-to")
    started = time.monotonic()
    if args.archive:
        target = args.extract_to.resolve() if args.extract_to else None
        report = verify_archive(args.archive.resolve(), target)
        if args.recompute:
            command = [sys.executable, str(target / Path(__file__).name), "--recompute"]
            if args.out:
                command.extend(["--out", str(args.out.resolve())])
            subprocess.run(command, cwd=target, check=True)
        print(json.dumps({**report, "elapsed_seconds": time.monotonic() - started}, indent=2))
        return
    report = verify_directory(ROOT, strict_release=args.strict_release)
    output = args.out.resolve() if args.out else ROOT / "reviewer_checks" / time.strftime("check_%Y%m%d_%H%M%S")
    if output.exists():
        raise FileExistsError("Use a new verification output directory")
    if args.recompute:
        report["recomputation"] = recompute(output)
    else:
        output.mkdir(parents=True)
    report["elapsed_seconds"] = time.monotonic() - started
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"files_verified": report["files_verified"], "totals": report["ledger"]["totals"],
                      "statistics_recomputed": args.recompute, "report": str(output / "verification.json")}, indent=2))


if __name__ == "__main__":
    main()
