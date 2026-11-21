"""Export exact cell-mean weights for the existing families of 23 and 72 contrasts."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import io
from itertools import product
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIELDS = (
    "contrast_id", "kind", "series", "dataset", "noise", "eta", "swarm", "budget",
    "n0", "generation_multiplier", "method", "numerator", "denominator",
)
INPUTS = {
    "expanded_synthetic": "results/ocba_robustness_20261022/synthetic_runs.csv",
    "n0_ablation": "results/ocba_n0_ablation_20261022/runs.csv",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def primitive(spec: dict, dataset: str, noise: str, method: str,
              coefficient: Fraction, *, series: str = "expanded_synthetic",
              eta: int = 2, swarm: int = 12, budget: int = 8640,
              n0: int = 3, generation_multiplier: int = 6) -> dict:
    return {
        "contrast_id": spec["id"], "kind": spec["kind"], "series": series,
        "dataset": dataset, "noise": noise, "eta": eta, "swarm": swarm,
        "budget": budget, "n0": n0,
        "generation_multiplier": generation_multiplier,
        "method": method, "numerator": coefficient.numerator,
        "denominator": coefficient.denominator,
    }


def weights23() -> tuple[list[dict], list[dict]]:
    from analyze_ocba_interactions import contrast_specs
    from run_robust_ocba_study import BUDGETS, FUNCTIONS, NOISES, SWARMS

    specs, rows = contrast_specs(), []
    for spec in specs:
        kind, noise = spec["kind"], spec["noise"]
        terms = []
        if kind == "mean":
            noises = NOISES if noise == "all" else (noise,)
            terms = [(z, eta, m, b, Fraction(1, len(noises) * 18))
                     for z, eta, m, b in product(noises, (1, 2), SWARMS, BUDGETS)]
        elif kind == "shape":
            terms = [(z, eta, m, b, Fraction(sign, 18))
                     for (z, sign), eta, m, b in product(
                         ((noise, 1), ("gaussian", -1)), (1, 2),
                         SWARMS, BUDGETS)]
        elif kind == "shape_budget":
            terms = [(z, eta, m, b, Fraction(sign * budget_sign, 6))
                     for (z, sign), eta, m, (b, budget_sign) in product(
                         ((noise, 1), ("gaussian", -1)), (1, 2), SWARMS,
                         ((spec["budget"], 1), (540, -1)))]
        elif kind == "shape_swarm":
            terms = [(z, eta, m, b, Fraction(sign * swarm_sign, 6))
                     for (z, sign), eta, (m, swarm_sign), b in product(
                         ((noise, 1), ("gaussian", -1)), (1, 2),
                         ((spec["swarm"], 1), (6, -1)), BUDGETS)]
        elif kind == "shape_eta":
            terms = [(z, eta, m, b, Fraction(sign * eta_sign, 9))
                     for (z, sign), (eta, eta_sign), m, b in product(
                         ((noise, 1), ("gaussian", -1)), ((2, 1), (1, -1)),
                         SWARMS, BUDGETS)]
        else:
            raise ValueError(f"Unknown contrast kind: {kind}")
        for dataset, (z, eta, swarm, budget, weight), (method, sign) in product(
                FUNCTIONS, terms, (("pso_ocba", 1), ("pso_ea", -1))):
            rows.append(primitive(
                spec, dataset, z, method, sign * weight / len(FUNCTIONS),
                eta=eta, swarm=swarm, budget=budget,
            ))
    return specs, rows


def weights72() -> tuple[list[dict], list[dict]]:
    from analyze_n0_ablation import PAIRS
    from run_n0_ablation import TASKS
    from run_robust_ocba_study import NOISES

    specs, rows = [], []
    for (method, baseline), noise in product(PAIRS, NOISES):
        definitions = [("mean", n0) for n0 in (3, 5, 10)]
        definitions += [("n0_change", n0) for n0 in (5, 10)]
        definitions += [("generation_bridge", 3)]
        for kind, n0 in definitions:
            spec = {
                "id": f"{kind}_{noise}_{method}_minus_{baseline}_n{n0}",
                "kind": kind, "noise": noise, "method": method,
                "baseline": baseline, "n0": n0,
            }
            specs.append(spec)
            terms = [("n0_ablation", n0, 12, 1)]
            if kind == "n0_change":
                terms.append(("n0_ablation", 3, 12, -1))
            elif kind == "generation_bridge":
                terms.append(("expanded_synthetic", 3, 6, -1))
            combinations = product(
                TASKS, terms, ((method, 1), (baseline, -1)),
            )
            for dataset, term, (name, sign) in combinations:
                series, repeats, generation, coefficient = term
                rows.append(primitive(
                    spec, dataset, noise, name,
                    Fraction(sign * coefficient, len(TASKS)),
                    series=series, n0=repeats,
                    generation_multiplier=generation,
                ))
    return specs, rows


def cell_key(row: dict) -> tuple:
    return (
        row["series"], row["dataset"], row["noise"], int(row["eta"]),
        int(row["swarm"]), int(row["budget"]), int(row["n0"]),
        int(row["generation_multiplier"]), row["method"],
    )


def verify_estimators(families: list[tuple[list[dict], list[dict]]]) -> None:
    import numpy as np
    import pandas as pd
    from analyze_n0_ablation import contrast_data
    from analyze_ocba_interactions import make_contrasts, paired_tensor

    scores = defaultdict(list)
    frames = {}
    for series, relative in INPUTS.items():
        frames[series] = pd.read_csv(ROOT / relative)
        for row in frames[series].to_dict("records"):
            record = {**row, "series": series}
            if series == "expanded_synthetic":
                record.update(n0=3, generation_multiplier=6)
            scores[cell_key(record)].append(float(row["final_score"]))
    means = {
        key: math.fsum(values) / len(values) for key, values in scores.items()
    }
    target23 = make_contrasts(
        paired_tensor(frames["expanded_synthetic"]),
    ).mean(axis=(0, 1))
    target_specs72, values72 = contrast_data(
        frames["n0_ablation"], frames["expanded_synthetic"],
    )
    target72 = values72.mean(axis=(0, 1))
    for (specs, rows), target in zip(families, (target23, target72), strict=True):
        assert len(specs) == len(target)
        groups = defaultdict(list)
        for row in rows:
            key = cell_key(row)
            if len(scores[key]) != 30:
                raise ValueError(f"Expected 30 paired searches per cell: {key}")
            groups[row["contrast_id"]].append(row)
        estimates = []
        for spec in specs:
            terms = groups[spec["id"]]
            coefficients = [
                Fraction(r["numerator"], r["denominator"]) for r in terms
            ]
            if sum(coefficients, Fraction()) != 0:
                raise ValueError("Contrast coefficients do not sum to zero")
            estimates.append(math.fsum(
                float(coefficient) * means[cell_key(row)]
                for row, coefficient in zip(terms, coefficients, strict=True)
            ))
        np.testing.assert_allclose(estimates, target, rtol=1e-12, atol=1e-12)
    for ours, expected in zip(families[1][0], target_specs72, strict=True):
        assert {k: ours[k] for k in expected} == expected


def rendered_files() -> dict[str, bytes]:
    families = [weights23(), weights72()]
    verify_estimators(families)
    inputs = {path: digest(ROOT / path) for path in INPUTS.values()}
    sources = {name: digest(ROOT / name) for name in (
        "analyze_ocba_interactions.py", "analyze_n0_ablation.py",
        "paired_mean_inference.py",
    )}
    files = {}
    for size, (specs, rows) in zip((23, 72), families, strict=True):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        files[f"contrast_ledger_{size}.csv"] = stream.getvalue().encode("utf-8")
        grouped = [
            {**spec, "weights": [r for r in rows
                                if r["contrast_id"] == spec["id"]]}
            for spec in specs
        ]
        metadata = {
            "format": "ocba-cell-mean-contrast-ledger-v1", "family_size": size,
            "status": (
                "exploratory; unchanged existing contrasts, not new hypotheses"
            ),
            "estimator": (
                "sum((numerator/denominator) * mean(final_score over the 30 "
                "paired run indices 0..29) for each cell)"
            ),
            "pairing": (
                "For resampling, retain the whole shared-seed vector across "
                "cells within each fixed function; these ledgers define "
                "weights, not independent sampling units."
            ),
            "units": "normalized quality; larger final_score is better",
            "inputs_sha256": inputs, "implementation_sha256": sources,
            "contrasts": grouped,
        }
        files[f"contrast_ledger_{size}.json"] = (
            json.dumps(metadata, indent=2) + "\n"
        ).encode("utf-8")
    validation = {
        "families": [23, 72],
        "all_estimators_match_existing_implementations": True,
        "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12,
        "weights": (
            "exact rational coefficients on cell means; 30 shared seeds per cell"
        ),
        "rows": {"23": len(families[0][1]), "72": len(families[1][1])},
        "input_sha256": inputs,
        "files_sha256": {
            name: hashlib.sha256(value).hexdigest()
            for name, value in files.items()
        },
        "new_searches": 0, "existing_results_modified": False,
    }
    files["contrast_ledger_validation.json"] = (
        json.dumps(validation, indent=2) + "\n"
    ).encode("utf-8")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify files without writing",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    try:
        files = rendered_files()
        if not args.check:
            args.out.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            path = args.out / name
            if path.exists() and path.read_bytes() != content:
                raise ValueError(f"Existing ledger differs; preserved: {path}")
            if args.check and not path.exists():
                raise ValueError(f"Missing ledger: {path}")
            if not args.check:
                path.write_bytes(content)
        print(json.dumps({
            "families_verified": [23, 72], "files": len(files),
            "check_only": args.check, "new_searches": 0,
        }, indent=2))
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        parser.exit(1, f"Contrast-ledger verification failed: {exc}\n")


if __name__ == "__main__":
    main()
