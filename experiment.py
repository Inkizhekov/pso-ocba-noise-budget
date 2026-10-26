#!/usr/bin/env python3
"""Run the finite-budget PSO-OCBA study supporting the current manuscript.

The default runs the full frozen robustness and powered NN protocol. Use
--dry-run to inspect it, --phase to run one stage, or the legacy subcommand
to reproduce the original pilot. Historical Python imports remain available.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
HISTORICAL_ROOT = ROOT / "historical" / "ocba_20261022"
DEFAULT_OUT = ROOT / "results" / "ocba_current"
PHASES = (
    "freeze", "synthetic", "nn_pilot", "plan", "validate", "nn_main", "analyze",
)


def __getattr__(name: str) -> Any:
    """Preserve the original scientific API used by older scripts and tests."""
    legacy = importlib.import_module("historical.ocba_20261022.experiment")
    try:
        return getattr(legacy, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def verify_datasets() -> None:
    """Fail before a long run if the manuscript's local inputs have changed."""
    from run_robust_ocba_study import NN_DATASETS, sha256

    folder = ROOT / "datasets" / "uci"
    manifest = read_json(folder / "manifest.json")
    for name in NN_DATASETS:
        if sha256(folder / f"{name}.csv") != manifest[name]["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")


def verify_power_plan(out: Path) -> dict[str, Any]:
    """Resume only a plan tied to this protocol, pilot and planning code."""
    from run_robust_ocba_study import sha256

    plan = read_json(out / "power_plan.json")
    expected = {
        "protocol_sha256": sha256(out / "protocol.json"),
        "pilot_runs_sha256": sha256(out / "nn_pilot_runs.csv"),
        "planner_sha256": sha256(ROOT / "plan_robust_nn_power.py"),
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen power plan does not match protocol, pilot or code")
    nn = read_json(out / "protocol.json")["nn"]
    n = plan.get("runs_per_cell")
    if not isinstance(n, int) or n < 48 or n > 1536 or n % 24:
        raise ValueError("Invalid pilot-derived NN sample size")
    if (plan.get("main_data_used_in_planning") is not False
            or plan.get("family_size") != nn["global_primary_family_size"]
            or plan.get("delta_accuracy") != nn["practical_delta_accuracy"]
            or plan.get("target_marginal_power") != nn["target_power"]
            or plan.get("conservative_per_test_alpha") != 0.05 / 12):
        raise ValueError("Power-plan assumptions differ from the frozen protocol")
    return plan


def verify_power_validation(out: Path) -> dict[str, Any]:
    """Require independent validation before confirmatory observations."""
    from run_robust_ocba_study import sha256

    plan = verify_power_plan(out)
    validation = read_json(out / "power_validation.json")
    expected = {
        "power_plan_sha256": sha256(out / "power_plan.json"),
        "pilot_sha256": sha256(out / "nn_pilot_runs.csv"),
        "validator_sha256": sha256(ROOT / "validate_robust_power.py"),
        "runs_per_cell": plan["runs_per_cell"],
    }
    if any(validation.get(key) != value for key, value in expected.items()):
        raise ValueError("Independent power validation has stale provenance")
    if (validation.get("main_results_used") is not False
            or validation.get("sample_size_changed") is not False
            or validation.get("minimum_power_lower_bound", 0)
            < plan["target_marginal_power"]):
        raise ValueError("Independent validation does not meet the power criterion")
    return validation


def invoke(script: str, *arguments: str) -> None:
    command = [sys.executable, str(ROOT / script), *arguments]
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def run_stage(phase: str, out: Path, workers: int, audit_workers: int,
              smoke: bool) -> None:
    """Run a stage without replacing an already frozen power calculation."""
    if phase == "freeze":
        return
    if phase == "nn_pilot" and (out / "power_plan.json").exists():
        verify_power_plan(out)
        print("Reusing the pilot bound to the frozen power plan.", flush=True)
    elif phase in {"synthetic", "nn_pilot", "nn_main"}:
        if phase == "nn_main":
            verify_power_validation(out)
        arguments = ["--phase", phase, "--out", str(out), "--workers", str(workers)]
        if smoke:
            arguments.append("--smoke")
        invoke("run_robust_ocba_study.py", *arguments)
    elif phase == "plan":
        if not (out / "power_plan.json").exists():
            invoke("plan_robust_nn_power.py", "--out", str(out))
        verify_power_plan(out)
        print("Frozen power plan verified.", flush=True)
    elif phase == "validate":
        verify_power_plan(out)
        if not (out / "power_validation.json").exists():
            if (out / "nn_main").exists():
                raise ValueError("Independent validation must precede NN main data")
            invoke("validate_robust_power.py", "--out", str(out))
        verify_power_validation(out)
        print("Independent power validation verified.", flush=True)
    elif phase == "analyze":
        verify_power_validation(out)
        invoke("analyze_robust_ocba_study.py", "--out", str(out),
               "--audit-workers", str(audit_workers))
    else:
        raise ValueError(f"Unknown phase: {phase}")


def describe(out: Path, phases: list[str], smoke: bool) -> dict[str, Any]:
    from run_robust_ocba_study import jobs, protocol

    grid = jobs("synthetic", out)
    selected = [{**grid[-1], "runs": 1}] if smoke else grid
    n = None
    if (out / "power_plan.json").exists():
        n = verify_power_plan(out)["runs_per_cell"]
    return {
        "output": str(out), "phases": phases, "smoke": smoke,
        "synthetic_searches": (sum(job["runs"] * 4 for job in selected)
                               if "synthetic" in phases else 0),
        "synthetic_calls": (sum(job["runs"] * 4 * job["budget"] for job in selected)
                            if "synthetic" in phases else 0),
        "nn_pilot_searches": (3 if smoke else 432) if "nn_pilot" in phases else 0,
        "nn_main_runs_per_condition": n or "determined from independent pilot",
        "nn_main_searches": (18 * n if n else None) if "nn_main" in phases else 0,
        "protocol": protocol(smoke=smoke),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        epilog="Historical pilot: experiment.py legacy --help. "
               "The full study is expensive; inspect it with --dry-run first.",
    )
    result.add_argument("--phase", choices=["all", *PHASES], default="all",
                        help="default: all stages, including independent power planning")
    result.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="fresh result directory (default: results/ocba_current)")
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--audit-workers", type=int, default=6)
    result.add_argument("--smoke", action="store_true",
                        help="one technical cell; only synthetic or nn_pilot; separate smoke folder")
    result.add_argument("--dry-run", action="store_true",
                        help="print the protocol and workload without writing files or running searches")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    explicit_legacy = bool(arguments and arguments[0] == "legacy")
    old_mode = any(arg == "--mode" or arg.startswith("--mode=")
                   for arg in arguments)
    if explicit_legacy or old_mode:
        if not explicit_legacy:
            print("Historical --mode detected; use 'experiment.py legacy' explicitly.",
                  file=sys.stderr)
        command = [sys.executable, "-m", "historical.ocba_20261022.experiment"]
        command.extend(arguments[1:] if explicit_legacy else arguments)
        environment = os.environ.copy()
        # Keep caller-relative output paths while making repository imports visible.
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), *filter(None, [environment.get("PYTHONPATH", "")])]
        )
        try:
            return subprocess.run(command, check=False, env=environment).returncode
        except KeyboardInterrupt:
            return 130

    cli = parser()
    args = cli.parse_args(arguments)
    if args.workers < 1 or args.audit_workers < 1:
        cli.error("worker counts must be positive")
    out = args.out.resolve()
    phase = "synthetic" if args.smoke and args.phase == "all" else args.phase
    if args.smoke and phase not in {"freeze", "synthetic", "nn_pilot"}:
        cli.error("--smoke supports only freeze, synthetic or nn_pilot")
    if args.smoke and "smoke" not in out.name:
        cli.error("--smoke requires a separate --out directory containing 'smoke'")
    phases = list(PHASES) if phase == "all" else [phase]
    try:
        if args.dry_run:
            print(json.dumps(describe(out, phases, args.smoke), indent=2))
            return 0
        if any(p in phases for p in ("nn_pilot", "nn_main", "plan", "validate")):
            verify_datasets()
        from run_robust_ocba_study import ensure_protocol

        ensure_protocol(out, smoke=args.smoke)
        print(f"Protocol verified: {out / 'protocol.json'}", flush=True)
        for step in phases:
            run_stage(step, out, args.workers, args.audit_workers, args.smoke)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed cells can be resumed.", file=sys.stderr)
        return 130
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        return error.returncode if isinstance(error, subprocess.CalledProcessError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
