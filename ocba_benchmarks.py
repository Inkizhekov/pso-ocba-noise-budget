"""Shifted and rotated analytical benchmarks with a common noise scale."""
from __future__ import annotations

import numpy as np

FUNCTIONS = ("sphere", "ellipsoid", "rosenbrock", "ackley", "rastrigin",
             "griewank", "levy", "zakharov")
DISCOVERY_FUNCTIONS = FUNCTIONS[:6]
VALIDATION_FUNCTIONS = FUNCTIONS[6:]
SPANS = dict(zip(FUNCTIONS, [8.0, 10.0, 4.0, 10.0, 10.0, 30.0, 10.0, 10.0]))


def raw_loss(name: str, z: np.ndarray) -> np.ndarray:
    """Vectorized conventional zero-minimum functions, last axis is dimension."""
    d = z.shape[-1]
    if name == "sphere":
        return np.sum(z * z, axis=-1)
    if name == "ellipsoid":
        return np.sum(10 ** np.linspace(0, 6, d) * z * z, axis=-1)
    if name == "rosenbrock":
        q = 1 + z
        return np.sum(100 * (q[..., 1:] - q[..., :-1] ** 2) ** 2
                      + (q[..., :-1] - 1) ** 2, axis=-1)
    if name == "ackley":
        return (-20 * np.exp(-0.2 * np.sqrt(np.mean(z * z, axis=-1)))
                - np.exp(np.mean(np.cos(2 * np.pi * z), axis=-1)) + 20 + np.e)
    if name == "rastrigin":
        return 10 * d + np.sum(z * z - 10 * np.cos(2 * np.pi * z), axis=-1)
    if name == "griewank":
        return (np.sum(z * z, axis=-1) / 4000
                - np.prod(np.cos(z / np.sqrt(np.arange(1, d + 1))), axis=-1) + 1)
    if name == "levy":
        w = 1 + z / 4
        return (np.sin(np.pi * w[..., 0]) ** 2
                + np.sum((w[..., :-1] - 1) ** 2
                         * (1 + 10 * np.sin(np.pi * w[..., :-1] + 1) ** 2), axis=-1)
                + (w[..., -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[..., -1]) ** 2))
    if name == "zakharov":
        linear = np.sum(0.5 * np.arange(1, d + 1) * z, axis=-1)
        return np.sum(z * z, axis=-1) + linear ** 2 + linear ** 4
    raise ValueError(f"Unknown benchmark {name}")


class ControlledBenchmark:
    synthetic = True

    def __init__(self, name: str, dimension: int, eta: float,
                 calibration_size: int = 4096):
        if name not in FUNCTIONS or dimension < 2 or eta < 0:
            raise ValueError("invalid benchmark parameters")
        self.name, self.dimension, self.eta = name, dimension, eta
        self.sigma = eta
        index = FUNCTIONS.index(name)
        self.design_seed = 521003 + index * 10007 + dimension * 97
        rng = np.random.default_rng(self.design_seed)
        self.optimum = rng.uniform(0.25, 0.75, dimension)
        matrix, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        self.rotation = matrix
        calibration = np.random.default_rng(self.design_seed + 900001).random(
            (calibration_size, dimension))
        z = (calibration - self.optimum) @ self.rotation * SPANS[name]
        losses = raw_loss(name, z)
        self.scale = float(np.std(losses, ddof=1))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("invalid benchmark calibration")
        self.calibration_size = calibration_size

    def true_score(self, x: np.ndarray) -> float:
        z = (np.asarray(x) - self.optimum) @ self.rotation * SPANS[self.name]
        return -max(0.0, float(raw_loss(self.name, z))) / self.scale

    def observe(self, x: np.ndarray, seed: int) -> float:
        return self.draw_from_truth(x, self.true_score(x), seed)

    def draw_from_truth(self, x: np.ndarray, truth: float, seed: int) -> float:
        std = self.eta * (0.6 + 0.8 * float(x[0]))
        return truth + float(np.random.default_rng(seed).normal(0, std))

    def final_score(self, x: np.ndarray, seeds: list[int]) -> float:
        return self.true_score(x)

    def metadata(self) -> dict:
        return {"name": self.name, "dimension": self.dimension,
                "design_seed": self.design_seed, "span": SPANS[self.name],
                "optimum": self.optimum.tolist(), "rotation": self.rotation.tolist(),
                "scale": self.scale, "calibration_size": self.calibration_size,
                "score": "-raw_loss/SD_uniform_calibration; optimum=0",
                "noise": "eta*(0.6+0.8*x0)*N(0,1)"}
