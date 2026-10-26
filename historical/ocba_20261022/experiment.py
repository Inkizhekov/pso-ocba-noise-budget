#!/usr/bin/env python3
"""Reproducible study of PSO hyperparameter optimization under noisy fitness.

All methods are charged for EVERY observed fitness call, including reevaluations.
The selected configuration is assessed with fresh, held-out evaluation seeds.
Synthetic tasks also expose an exact expected fitness (never visible to search).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Pin numerical threading before importing numpy/scikit-learn in shell or entrypoint.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy
import sklearn
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from study_statistics import TEST_SIZES, bootstrap_ci, holm_adjust, paired_effect

LOGGER = logging.getLogger("swarm_research")
METHODS = ("random", "tpe", "pso", "pso_r3", "pso_ar")
PSO_METHODS = {"pso", "pso_r3", "pso_ar"}
DIM = 3


class Objective(Protocol):
    name: str
    synthetic: bool

    def observe(self, x: np.ndarray, seed: int) -> float: ...
    def true_score(self, x: np.ndarray) -> float: ...
    def final_score(self, x: np.ndarray, seeds: list[int]) -> float: ...


@dataclass(frozen=True)
class SyntheticObjective:
    name: str
    sigma: float
    synthetic: bool = True

    def true_score(self, x: np.ndarray) -> float:
        z = (np.asarray(x, dtype=float) - 0.5) * 4
        if self.name == "rastrigin":
            raw = 10 * DIM + np.sum(z * z - 10 * np.cos(2 * np.pi * z))
            return float(1.0 - raw / 400.0)
        if self.name == "ackley":
            raw = (-20 * np.exp(-0.2 * np.sqrt(np.mean(z * z)))
                   - np.exp(np.mean(np.cos(2 * np.pi * z))) + 20 + np.e)
            return float(1.0 - raw / 50.0)
        raise ValueError(f"Unknown benchmark: {self.name}")

    def observe(self, x: np.ndarray, seed: int) -> float:
        actual_sigma = self.sigma * (0.6 + 0.8 * float(x[0]))
        return self.true_score(x) + np.random.default_rng(seed).normal(0, actual_sigma)

    def final_score(self, x: np.ndarray, seeds: list[int]) -> float:
        del seeds
        return self.true_score(x)


class TabularNNObjective:
    synthetic = False

    def __init__(self, name: str, epochs: int = 20, split_seed: int = 2026,
                 data: tuple[np.ndarray, np.ndarray] | None = None):
        loaders = {
            "digits": load_digits,
            "wine": load_wine,
            "breast_cancer": load_breast_cancer,
        }
        if name not in loaders and data is None:
            raise ValueError(f"Unknown data set {name}; options: {list(loaders)}")
        x, y = data if data is not None else loaders[name](return_X_y=True)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.name = name
        self.epochs = epochs
        self.split_seed = split_seed
        x_dev, self.x_test, y_dev, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=split_seed, stratify=y
        )
        x_train, x_valid, self.y_train, self.y_valid = train_test_split(
            x_dev, y_dev, test_size=0.25, random_state=split_seed + 1,
            stratify=y_dev
        )
        scaler = StandardScaler().fit(x_train)
        self.x_train = scaler.transform(x_train)
        self.x_valid = scaler.transform(x_valid)
        self.y_dev = y_dev
        full_scaler = StandardScaler().fit(x_dev)
        self.x_dev = full_scaler.transform(x_dev)
        self.x_test = full_scaler.transform(self.x_test)
        self.sizes = (len(x_train), len(x_valid), len(self.y_test))

    def true_score(self, x: np.ndarray) -> float:
        del x
        return float("nan")

    @staticmethod
    def decode(x: np.ndarray) -> dict[str, float | int]:
        values = np.clip(np.asarray(x, dtype=float), 0, 1)
        hidden = int(np.clip(16 * np.rint(1 + 7 * values[0]), 16, 128))
        return {
            "hidden": hidden,
            "learning_rate_init": float(10 ** (-4.0 + 2.5 * values[1])),
            "alpha": float(10 ** (-6.0 + 4 * values[2])),
        }

    def _fit(self, x: np.ndarray, seed: int, final: bool) -> float:
        p = self.decode(x)
        model = MLPClassifier(
            hidden_layer_sizes=(int(p["hidden"]),),
            alpha=float(p["alpha"]),
            learning_rate_init=float(p["learning_rate_init"]),
            batch_size=64,
            activation="relu",
            solver="adam",
            shuffle=True,
            max_iter=self.epochs,
            tol=0.0,
            n_iter_no_change=self.epochs + 1,
            random_state=int(seed),
            early_stopping=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            if final:
                model.fit(self.x_dev, self.y_dev)
                return float(model.score(self.x_test, self.y_test))
            model.fit(self.x_train, self.y_train)
            return float(model.score(self.x_valid, self.y_valid))

    def observe(self, x: np.ndarray, seed: int) -> float:
        return self._fit(x, seed=seed, final=False)

    def final_score(self, x: np.ndarray, seeds: list[int]) -> float:
        return float(np.mean([self._fit(x, seed=s, final=True) for s in seeds]))


@dataclass
class Incumbent:
    x: np.ndarray
    scores: list[float]
    truth: float

    @property
    def estimate(self) -> float:
        return float(np.mean(self.scores))


class Evaluator:
    def __init__(self, task: Objective, run_seed: int, budget: int,
                 method: str, run_id: int):
        self.task = task
        self.rng = np.random.default_rng(run_seed + 4_000_003)
        self.budget = budget
        self.method = method
        self.run_id = run_id
        self.calls = 0
        self.events: list[dict] = []
        self.start = time.perf_counter()

    @property
    def remaining(self) -> int:
        return self.budget - self.calls

    def sample(self, x: np.ndarray, reason: str, selected: Incumbent | None) -> float:
        if self.remaining <= 0:
            raise RuntimeError("Evaluation budget exceeded")
        seed = int(self.rng.integers(1, 2_147_483_000))
        value = self.task.observe(x, seed)
        self.calls += 1
        self.events.append({
            "dataset": self.task.name,
            "sigma": getattr(self.task, "sigma", np.nan),
            "method": self.method,
            "run": self.run_id,
            "call": self.calls,
            "reason": reason,
            "training_seed": seed,
            "x0": float(x[0]),
            "x1": float(x[1]),
            "x2": float(x[2]),
            "observed_fitness": value,
            "evaluation_true_fitness": self.task.true_score(x),
            "current_best_estimate": (selected.estimate if selected else np.nan),
            "elapsed_seconds": time.perf_counter() - self.start,
        })
        return value


def solve_random(task: Objective, ev: Evaluator, seed: int) -> tuple[Incumbent, dict]:
    rng = np.random.default_rng(seed)
    best: Incumbent | None = None
    while ev.remaining:
        x = rng.random(DIM)
        score = ev.sample(x, "new", best)
        if best is None or score > best.estimate:
            best = Incumbent(x, [score], task.true_score(x))
    if best is None:
        raise RuntimeError("Empty random search")
    return best, {"false_accept": 0, "false_reject": 0, "comparisons": 0}


def solve_tpe(task: Objective, ev: Evaluator, seed: int) -> tuple[Incumbent, dict]:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8),
    )
    best: Incumbent | None = None
    while ev.remaining:
        trial = study.ask()
        x = np.array([trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(DIM)])
        score = ev.sample(x, "new", best)
        study.tell(trial, score)
        if best is None or score > best.estimate:
            best = Incumbent(x, [score], task.true_score(x))
    if best is None:
        raise RuntimeError("Empty TPE search")
    return best, {"false_accept": 0, "false_reject": 0, "comparisons": 0}


def solve_pso(task: Objective, ev: Evaluator, seed: int, mode: str,
              swarm_size: int = 6, adaptive_gap: float = 0.025
              ) -> tuple[Incumbent, dict]:
    if mode not in PSO_METHODS:
        raise ValueError(mode)
    rng = np.random.default_rng(seed)
    positions = rng.random((swarm_size, DIM))
    velocities = rng.uniform(-0.1, 0.1, size=(swarm_size, DIM))
    personal: list[Incumbent | None] = [None] * swarm_size
    counters = {"false_accept": 0, "false_reject": 0, "comparisons": 0}
    turn = 0

    while ev.remaining:
        i = turn % swarm_size
        if turn >= swarm_size:
            leaders = [q for q in personal if q is not None]
            if not leaders:
                raise RuntimeError("PSO has no leader")
            leader = max(leaders, key=lambda incumbent: incumbent.estimate)
            local_best = personal[i]
            target = local_best.x if local_best is not None else positions[i]
            r1, r2 = rng.random(DIM), rng.random(DIM)
            velocities[i] = (0.729 * velocities[i]
                             + 1.49445 * r1 * (target - positions[i])
                             + 1.49445 * r2 * (leader.x - positions[i]))
            velocities[i] = np.clip(velocities[i], -0.2, 0.2)
            positions[i] = np.clip(positions[i] + velocities[i], 0, 1)
        x = positions[i].copy()
        previous = personal[i]
        incumbent = max((p for p in personal if p is not None),
                        key=lambda p: p.estimate, default=None)
        sample = ev.sample(x, "new", incumbent)
        values = [sample]

        if mode == "pso_r3":
            while len(values) < 3 and ev.remaining:
                values.append(ev.sample(x, "fixed_resample", incumbent))

        if (mode == "pso_ar" and previous is not None
                and abs(sample - previous.estimate) <= adaptive_gap):
            # Equal additional measurements of the challenger and incumbent.
            # This is a transparent heuristic baseline, NOT the OCBA algorithm.
            for _ in range(2):
                if ev.remaining >= 2:
                    values.append(ev.sample(x, "adaptive_challenger", incumbent))
                    previous.scores.append(
                        ev.sample(previous.x, "adaptive_incumbent", incumbent)
                    )

        current_estimate = float(np.mean(values))
        if previous is not None:
            counters["comparisons"] += 1
            accept = current_estimate > previous.estimate
            if task.synthetic:
                current_true = task.true_score(x)
                if accept and current_true < previous.truth - 1e-12:
                    counters["false_accept"] += 1
                if not accept and current_true > previous.truth + 1e-12:
                    counters["false_reject"] += 1
        else:
            accept = True

        if accept:
            personal[i] = Incumbent(x, values, task.true_score(x))
        turn += 1
    best = max((p for p in personal if p is not None), key=lambda p: p.estimate)
    return best, counters


def run_once(task: Objective, method: str, run_id: int, budget: int,
             master_seed: int, assessment_repeats: int = 3,
             adaptive_gap: float = 0.025,
             assessment_seed_policy: str = "per_run") -> tuple[dict, list[dict]]:
    if budget < 8:
        raise ValueError("budget must be >= 8")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if assessment_repeats < 1 or run_id < 0:
        raise ValueError("assessment_repeats must be positive and run_id nonnegative")
    if assessment_seed_policy not in {"per_run", "legacy_shared"}:
        raise ValueError("unknown assessment_seed_policy")
    # Same outer seed for all methods; separate noise RNG vs proposal RNG.
    seed = int(master_seed + 10_003 * run_id)
    ev = Evaluator(task, seed, budget, method, run_id)
    if method == "random":
        best, counts = solve_random(task, ev, seed)
    elif method == "tpe":
        best, counts = solve_tpe(task, ev, seed)
    else:
        best, counts = solve_pso(task, ev, seed, method, adaptive_gap=adaptive_gap)
    search_seconds = time.perf_counter() - ev.start
    seed_offset = 104_729 * run_id if assessment_seed_policy == "per_run" else 0
    assessment_seeds = [int(master_seed + 7_000_001 + seed_offset + k * 37) for k in range(
        assessment_repeats
    )]
    if task.synthetic:
        final = task.final_score(best.x, assessment_seeds)
        assessment_scores = []
    else:
        assessment_scores = [task.final_score(best.x, [s]) for s in assessment_seeds]
        final = float(np.mean(assessment_scores))
    n_test = int(task.sizes[2]) if hasattr(task, "sizes") and not task.synthetic else None
    correct_counts = ([int(round(s * n_test)) for s in assessment_scores]
                      if n_test is not None else [])
    row = {
        "dataset": task.name,
        "sigma": getattr(task, "sigma", np.nan),
        "synthetic": task.synthetic,
        "method": method,
        "run": run_id,
        "budget": budget,
        "calls": ev.calls,
        "assessment_seed_policy": assessment_seed_policy,
        "assessment_seeds": json.dumps(assessment_seeds),
        "assessment_scores": json.dumps(assessment_scores),
        "assessment_correct_counts": json.dumps(correct_counts),
        "assessment_repeats": assessment_repeats,
        "n_test": n_test,
        "assessment_denominator": n_test * assessment_repeats if n_test else None,
        "search_seconds": search_seconds,
        "selected_observed_score": best.estimate,
        "final_score": final,
        "false_accept": counts["false_accept"],
        "false_reject": counts["false_reject"],
        "comparisons": counts["comparisons"],
        "error_fraction": ((counts["false_accept"] + counts["false_reject"])
                           / counts["comparisons"]
                           if task.synthetic and counts["comparisons"] else np.nan),
        "x0": float(best.x[0]),
        "x1": float(best.x[1]),
        "x2": float(best.x[2]),
    }
    return row, ev.events


def pilot_noise(task: TabularNNObjective, master_seed: int,
                n_configs: int = 12, n_seeds: int = 8) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(master_seed + 808)
    scores = []
    positions = rng.random((n_configs, DIM))
    for k, x in enumerate(positions):
        for j in range(n_seeds):
            seed = int(master_seed + 70_001 + 100 * k + j)
            value = task.observe(x, seed)
            scores.append({"dataset": task.name, "config": k, "seed": seed,
                           "accuracy": value, "x0": float(x[0]),
                           "x1": float(x[1]), "x2": float(x[2])})
    data = pd.DataFrame(scores)
    grouped = data.groupby("config").accuracy
    within_var = grouped.var(ddof=1).mean()
    between_var = grouped.mean().var(ddof=1)
    estimate = {
        "dataset": task.name,
        "n_configs": n_configs,
        "n_seeds": n_seeds,
        "avg_within_var": float(within_var),
        "between_means_var": float(between_var),
        # Correct the variance of noisy group means for finite seed repeats.
        "between_corrected_var": float(max(0.0, between_var - within_var / n_seeds)),
        "rho_raw": float(within_var / between_var) if between_var > 0 else float("nan"),
        "rho_corrected": (float(within_var / max(1e-12, between_var-within_var/n_seeds))
                          if between_var - within_var/n_seeds > 0 else float("nan")),
    }
    return scores, estimate


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    pairs = []
    strata = [c for c in ("budget", "split_seed") if c in raw.columns]
    by = ["dataset", "sigma", *strata, "method"]
    for keys, group in raw.groupby(by, dropna=False):
        arr = group.final_score.to_numpy(float)
        lo, hi = bootstrap_ci(arr)
        summary_rows.append({
            **dict(zip(by, keys)),
            "n": len(arr), "mean": np.mean(arr),
            "sd": np.std(arr, ddof=1) if len(arr) > 1 else float("nan"),
            "median": np.median(arr), "ci_low": lo, "ci_high": hi,
            "mean_search_seconds": group.search_seconds.mean(),
            "mean_error_fraction": (
                np.nan if "synthetic" in group and not group.synthetic.all()
                else group.error_fraction.mean()
            ),
        })
    family_columns = ["dataset", "sigma", *strata]
    for family_key, block in raw.groupby(family_columns, dropna=False):
        family = dict(zip(family_columns, family_key))
        dataset = family["dataset"]
        if block.duplicated(["method", "run"]).any():
            raise ValueError("duplicate run ids within a statistical family")
        base = block.loc[block.method == "pso", ["run", "final_score"]].set_index("run")
        if base.empty:
            continue
        for method in METHODS:
            if method == "pso":
                continue
            contender = block.loc[block.method == method,
                                  ["run", "final_score"]].set_index("run")
            if contender.empty:
                continue
            if base.empty or not contender.index.equals(base.index):
                if set(contender.index) != set(base.index):
                    raise ValueError("unmatched run ids in a paired comparison")
            combined = contender.join(base, lsuffix="_method", rsuffix="_pso")
            denominator = None
            if "assessment_denominator" in block and block.assessment_denominator.notna().any():
                unique = block.assessment_denominator.dropna().unique()
                if len(unique) != 1:
                    raise ValueError("inconsistent test accuracy denominators")
                denominator = int(unique[0])
            elif dataset in TEST_SIZES:
                denominator = 3 * TEST_SIZES[dataset]
            effect = paired_effect(combined.final_score_method.to_numpy(),
                                   combined.final_score_pso.to_numpy(), denominator)
            pairs.append({**family,
                          "method": method, "baseline": "pso",
                          **effect})
    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df["holm_p"] = np.nan
        # Correction within each pre-defined dataset/noise family.
        for _, indices in pairs_df.groupby(family_columns, dropna=False).groups.items():
            positions = list(indices)
            p = pairs_df.loc[positions, "wilcoxon_p_raw"].to_numpy(float)
            pairs_df.loc[positions, "holm_p"] = holm_adjust(p)
    return pd.DataFrame(summary_rows), pairs_df


def make_figures(raw: pd.DataFrame, summary: pd.DataFrame,
                 pilot: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    synthetic = summary[summary.sigma.notna()]
    if not synthetic.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for method in METHODS:
            group = synthetic[synthetic.method == method]
            levels = sorted(group.sigma.unique())
            values = [group[group.sigma == level]["mean"].mean() for level in levels]
            ax.plot(levels, values, marker="o", label=method)
        ax.set(xlabel="Noise level (base standard deviation)",
               ylabel="Mean oracle fitness of selected candidate",
               title="Controlled synthetic benchmarks")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=9, ncol=2)
        fig.tight_layout()
        fig.savefig(output / "fig1_synthetic_noise.png", dpi=190)
        plt.close(fig)

        errors = synthetic[synthetic.method.isin(PSO_METHODS)]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for method in ["pso", "pso_r3", "pso_ar"]:
            group = errors[errors.method == method]
            levels = sorted(group.sigma.unique())
            values = [group[group.sigma == level].mean_error_fraction.mean()
                      for level in levels]
            ax.plot(levels, values, marker="s", label=method)
        ax.set(xlabel="Noise level (base standard deviation)",
               ylabel="Fraction of incorrect pbest decisions",
               title="Incorrect personal-best comparisons")
        ax.grid(alpha=0.22)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "fig2_ranking_errors.png", dpi=190)
        plt.close(fig)

    real = summary[summary.sigma.isna()]
    if not real.empty:
        datasets = sorted(real.dataset.unique())
        fig, axes = plt.subplots(1, len(datasets), figsize=(max(8, 3.3*len(datasets)), 4.5),
                                 sharey=False, squeeze=False)
        for ax, dataset in zip(axes[0], datasets):
            sub = real[real.dataset == dataset].set_index("method").reindex(METHODS)
            ax.errorbar(range(len(METHODS)), sub["mean"],
                        yerr=[sub["mean"]-sub["ci_low"], sub["ci_high"]-sub["mean"]],
                        fmt="o", capsize=3)
            ax.set_xticks(range(len(METHODS)), METHODS, rotation=55, ha="right")
            ax.set_title(dataset)
            ax.set_ylabel("Held-out test accuracy")
            ax.grid(alpha=0.18, axis="y")
        fig.suptitle("Neural network HPO: mean accuracy and 95% bootstrap CI")
        fig.tight_layout()
        fig.savefig(output / "fig3_real_data.png", dpi=190)
        plt.close(fig)

    if not pilot.empty:
        fig, axes = plt.subplots(1, len(pilot.dataset.unique()),
                                 figsize=(max(8, 3.4 * len(pilot.dataset.unique())), 4.2),
                                 squeeze=False)
        for ax, ds in zip(axes[0], sorted(pilot.dataset.unique())):
            grp = pilot[pilot.dataset == ds]
            positions = sorted(grp.config.unique())
            values = [grp.loc[grp.config == pos, "accuracy"].values for pos in positions]
            ax.boxplot(values, showfliers=False)
            ax.set_title(ds)
            ax.set_xlabel("Hyperparameter configuration")
            ax.set_ylabel("Validation accuracy")
            ax.tick_params(axis="x", labelsize=7)
        fig.suptitle("Observed seed-dependent validation score variation")
        fig.tight_layout()
        fig.savefig(output / "fig4_pilot_variance.png", dpi=190)
        plt.close(fig)


def metadata(args: argparse.Namespace) -> dict:
    return {
        "python": sys.version.split()[0],
        "system": platform.platform(),
        "cpu": platform.processor(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "optuna": optuna.__version__,
        "pandas": pd.__version__,
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "thread_settings": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "data_split": "stratified 60% train / 20% validation / 20% test",
        "optimization_objective": "validation accuracy, higher is better",
        "search_budget_unit": "individual objective training runs, including repeated evaluations",
        "final_model_policy": "refit train+validation, average independent test seeds",
        "note": "Python/scikit-learn CPU pilot, not MNIST or CIFAR-10; no external datasets.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--master-seed", type=int, default=20261022)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--assessment-seed-policy", choices=["per_run", "legacy_shared"],
                        default="per_run")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--data", nargs="+", default=["digits", "wine", "breast_cancer"],
                        choices=["digits", "wine", "breast_cancer"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    if args.skip_synthetic and args.skip_real:
        parser.error("At least one experiment must be selected")
    config = {"smoke": (2, 12), "pilot": (12, 36), "full": (30, 90)}
    runs, budget = config[args.mode]
    runs = args.runs if args.runs is not None else runs
    budget = args.budget if args.budget is not None else budget
    if runs < 2 or budget < 8:
        parser.error("--runs must be >= 2 and --budget must be >= 8")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.runs, args.budget = runs, budget
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metadata.json").write_text(
        json.dumps(metadata(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows: list[dict] = []
    events: list[dict] = []
    pilot_rows: list[dict] = []
    pilot_stats: list[dict] = []
    tasks: list[Objective] = []
    if not args.skip_synthetic:
        sigma_values = (0.0, 0.02, 0.06)
        tasks += [SyntheticObjective(name, sigma)
                  for name in ("ackley", "rastrigin") for sigma in sigma_values]
    if not args.skip_real:
        nn_tasks = [TabularNNObjective(name, epochs=args.epochs) for name in args.data]
        for nn in nn_tasks:
            LOGGER.info("Measuring pilot noise: %s split=%s", nn.name, nn.sizes)
            measured, stats = pilot_noise(
                nn, args.master_seed,
                n_configs=4 if args.mode == "smoke" else 12,
                n_seeds=3 if args.mode == "smoke" else 8,
            )
            pilot_rows.extend(measured)
            pilot_stats.append(stats)
        tasks.extend(nn_tasks)
    total = len(tasks) * len(METHODS) * runs
    done = 0
    try:
        for task in tasks:
            for method in METHODS:
                for run_id in range(runs):
                    row, trace = run_once(
                        task, method, run_id, budget, args.master_seed,
                        assessment_repeats=2 if args.mode == "smoke" else 3,
                        assessment_seed_policy=args.assessment_seed_policy,
                    )
                    rows.append(row)
                    events.extend(trace)
                    done += 1
                LOGGER.info("Completed %-13s %-8s %3d/%3d jobs", task.name, method, done, total)
            # Save checkpoints after every task.
            pd.DataFrame(rows).to_csv(args.out / "runs.csv", index=False)
            pd.DataFrame(events).to_csv(args.out / "evaluations.csv", index=False)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted: saving completed trial results")
        if rows:
            pd.DataFrame(rows).to_csv(args.out / "runs.csv", index=False)
        if events:
            pd.DataFrame(events).to_csv(args.out / "evaluations.csv", index=False)
        return 130

    raw = pd.DataFrame(rows)
    summary, comparisons = summarize(raw)
    pilot_data = pd.DataFrame(pilot_rows)
    raw.to_csv(args.out / "runs.csv", index=False)
    pd.DataFrame(events).to_csv(args.out / "evaluations.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    comparisons.to_csv(args.out / "pairwise.csv", index=False)
    pilot_data.to_csv(args.out / "pilot_scores.csv", index=False)
    pd.DataFrame(pilot_stats).to_csv(args.out / "pilot_noise.csv", index=False)
    make_figures(raw, summary, pilot_data, args.out / "figures")
    LOGGER.info("Finished %d optimization runs (%d objective calls); results: %s",
                len(rows), int(raw.calls.sum()), args.out.resolve())
    LOGGER.info("Summary:\n%s", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
