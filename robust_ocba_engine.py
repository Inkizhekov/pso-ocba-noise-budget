"""Configurable extension of the frozen matched PSO engine.

The numerical search loop is retained from ocba_engine.py, with swarm size
and initial repeat count made explicit. The default generation budget is
6*M; a focused ablation can choose a larger fixed multiple to fit n0=10.
Allocation increments are M. Frozen historical sources are never modified.
"""
from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np

from ocba import classic_ocba_allocation, gaussian_star_allocation, integer_top_up
from ocba_engine import Pool, Sampler, SYNTHETIC_METHODS


def run_search(task, method: str, seed: int, budget: int,
               event_writer: Callable | None = None,
               pool_writer: Callable | None = None, *,
               swarm_size: int = 6, initial_repeats: int = 3,
               generation_multiplier: int = 6) -> dict:
    if (method not in SYNTHETIC_METHODS or swarm_size < 2
            or initial_repeats < 2 or generation_multiplier < initial_repeats
            or budget < swarm_size):
        raise ValueError("invalid search method or budget")
    started = time.perf_counter()
    sampler = Sampler(task, seed, budget, event_writer, pool_writer)
    rng = np.random.default_rng(seed)
    dimension = getattr(task, "dimension", 3)
    comparisons = false_accept = false_reject = 0
    allocation_seconds = 0.0
    if method == "random":
        best = None
        while sampler.remaining:
            pool = sampler.new_pool(rng.random(dimension), -1)
            sampler.sample(pool, "new")
            if best is None or pool.mean > best.mean:
                best = pool
        generations = 0
    else:
        positions = rng.random((swarm_size, dimension))
        velocities = rng.uniform(-0.1, 0.1, (swarm_size, dimension))
        personal: list[Pool | None] = [None] * swarm_size
        generation = 0
        while sampler.remaining:
            sampler.generation = generation
            previous_global = max((p for p in personal if p is not None),
                                  key=lambda p: p.mean, default=None)
            if generation:
                local_x = np.array([p.x for p in personal])
                r1 = rng.random((swarm_size, dimension))
                r2 = rng.random((swarm_size, dimension))
                velocities = (0.729 * velocities + 1.49445 * r1 * (local_x - positions)
                              + 1.49445 * r2 * (previous_global.x - positions))
                velocities = np.clip(velocities, -0.2, 0.2)
                positions = np.clip(positions + velocities, 0, 1)
            current = [sampler.new_pool(x, i) for i, x in enumerate(positions)]
            # Freeze personal thresholds before the old global is remeasured.
            thresholds = np.array([p.mean if p else np.nan for p in personal])
            n0 = 1 if method == "pso" else initial_repeats
            per_generation = (swarm_size * n0 if method in {"pso", "pso_r3"}
                              else (generation_multiplier * swarm_size))
            available = min(per_generation, sampler.remaining)
            beginning = sampler.calls
            for repeat in range(n0):
                for pool in current:
                    if sampler.calls - beginning < available:
                        sampler.sample(pool, "new" if repeat == 0 else "initial_repeat")
            # Reuse the record's observations instead of creating a fresh pool.
            pools = current + ([previous_global] if previous_global else [])
            limits = np.append(thresholds, np.nan) if previous_global else thresholds
            while sampler.calls - beginning < available:
                counts = np.array([p.count for p in pools])
                # A truncated final generation may lack variance estimates.
                if any(p.count < 2 for p in current):
                    break
                means = np.array([p.mean for p in pools])
                variances = np.array([p.variance for p in pools])
                center = int(np.argmax(means))
                t0 = time.perf_counter()
                if method == "pso_ocba":
                    weights = gaussian_star_allocation(means, variances, center, limits)
                elif method == "pso_ocba_rs":
                    weights = classic_ocba_allocation(means, variances, center)
                else:
                    weights = np.full(len(pools), 1 / len(pools))
                extra = min(swarm_size, available - (sampler.calls - beginning))
                additions = integer_top_up(weights, counts, extra)
                allocation_seconds += time.perf_counter() - t0
                for pool, count in zip(pools, additions):
                    for _ in range(int(count)):
                        sampler.sample(pool, "allocation")
            for i, pool in enumerate(current):
                if not pool.count:
                    continue
                previous = personal[i]
                accept = previous is None or pool.mean > previous.mean
                if previous is not None:
                    comparisons += 1
                    if task.synthetic:
                        false_accept += int(accept and pool.truth < previous.truth - 1e-12)
                        false_reject += int(not accept and pool.truth > previous.truth + 1e-12)
                if accept:
                    personal[i] = pool
            generation += 1
        best = max((p for p in personal if p is not None), key=lambda p: p.mean)
        generations = generation
    search_seconds = time.perf_counter() - started
    assessment_seeds = [seed + 7000001 + 37 * k for k in range(3)]
    if task.synthetic:
        final_score = task.true_score(best.x)
        assessment_scores, correct_counts = [], []
        n_test = None
    else:
        # These test-set refits are outside the exhausted search budget.
        assessment_scores = [task.final_score(best.x, [s]) for s in assessment_seeds]
        final_score = float(np.mean(assessment_scores))
        n_test = task.sizes[2]
        correct_counts = [int(round(s * n_test)) for s in assessment_scores]
    return {
        "method": method, "seed": seed, "budget": budget, "calls": sampler.calls,
        "final_score": final_score, "selected_observed_score": best.mean,
        "selected_coordinates": json.dumps(best.x.tolist()),
        "selected_pool": best.identifier,
        "search_seconds": search_seconds, "allocation_seconds": allocation_seconds,
        "generations": generations, "evaluated_pools": sum(p.count > 0 for p in sampler.pools),
        "comparisons": comparisons, "false_accept": false_accept,
        "false_reject": false_reject,
        "error_fraction": ((false_accept + false_reject) / comparisons
                           if task.synthetic and comparisons else float("nan")),
        "assessment_seeds": json.dumps(assessment_seeds),
        "assessment_scores": json.dumps(assessment_scores),
        "assessment_correct_counts": json.dumps(correct_counts),
        "n_test": n_test, "assessment_denominator": 3 * n_test if n_test else None,
    }
