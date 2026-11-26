"""Sampled-pair MaxEnt-PCA derived from He et al. (2010), Eqs. (15–23).

At fixed bandwidth, minimizing the unnormalized Gaussian kernel mean has
the same optima as maximizing estimated entropy, but a different gradient
scale. Bandwidth is updated between steps, not differentiated through.
"""

import math
import pickle
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models.base import InductiveAnomalyModel
from mark_ad_benchmark.optimizers import create_optimizer


class MaxEntPCAModel(InductiveAnomalyModel):
    """Learn an orthonormal subspace with sampled pairs and score L2 residuals."""

    def __init__(
        self,
        *,
        logger: ComponentLogger,
        n_components: int,
        standardize_features: bool,
        bandwidth_scale: float = 2.0,
        pair_batch_size: int = 1024,
        max_iterations: int = 1000,
        random_state: int = 42,
        optimizer: Mapping[str, object] | None = None,
    ) -> None:
        self.logger = logger
        self.n_components = n_components
        self.standardize_features = standardize_features
        self.bandwidth_scale = bandwidth_scale
        self.pair_batch_size = pair_batch_size
        self.max_iterations = max_iterations
        self.random_state = random_state

        self.validate_parameters()
        optimizer_settings = (
            {"name": "sgd", "learning_rate": 0.01} if optimizer is None else optimizer
        )
        self.optimizer = create_optimizer(
            optimizer_settings, logger.for_component("mark_ad_benchmark.optimizers")
        )
        self.optimizer_settings = dict(optimizer_settings)
        self.components: np.ndarray | None = None
        self.scaler: StandardScaler | None = None
        self._diagnostics: dict[str, object] = {}

    def validate_parameters(self) -> None:
        for name, value, minimum in (
            ("n_components", self.n_components, 1),
            ("pair_batch_size", self.pair_batch_size, 1),
            ("max_iterations", self.max_iterations, 1),
            ("random_state", self.random_state, 0),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                self._fail(f"MaxEnt-PCA {name} must be an integer >= {minimum}")
        if not isinstance(self.standardize_features, bool):
            self._fail("MaxEnt-PCA standardize_features must be a boolean")
        if (
            not isinstance(self.bandwidth_scale, (int, float))
            or isinstance(self.bandwidth_scale, bool)
            or not math.isfinite(self.bandwidth_scale)
            or self.bandwidth_scale <= 0
        ):
            self._fail("MaxEnt-PCA bandwidth_scale must be finite and positive")

    @property
    def diagnostics(self) -> Mapping[str, object]:
        """Report the update budget and numerical state, not claimed convergence."""
        return self._diagnostics

    def fit(self, features: Sequence[Sequence[float]]) -> None:
        """Fit on all training rows using fresh uniform ordered pairs each step."""
        # create numpy matrix
        matrix = self._feature_matrix(features)
        if self.n_components >= matrix.shape[1]:
            self._fail("MaxEnt-PCA components must be fewer than the feature width")
        if len(matrix) <= self.n_components:
            self._fail("MaxEnt-PCA training rows must exceed n_components")
        self.logger.info(
            f"Fitting MaxEnt-PCA on {len(matrix)} rows and {matrix.shape[1]} features with {self.pair_batch_size} pairs per update"
        )
        # 1. center and optionally standardize training features (documentation excerpt; keep synchronized)
        scaler = StandardScaler(with_std=self.standardize_features)
        matrix = scaler.fit_transform(matrix)
        if not np.isfinite(matrix).all():
            self._fail("MaxEnt-PCA preprocessing produced nonfinite features")
        # 2. initialize pca directions using thin svd (documentation excerpt; keep synchronized)
        _, _, right = np.linalg.svd(matrix, full_matrices=False)
        components = right[: self.n_components].T.copy()
        # initialize optimizer
        optimizer = create_optimizer(
            self.optimizer_settings,
            self.logger.for_component("mark_ad_benchmark.optimizers"),
        )
        # seed random generator for pair sampling
        rng = np.random.default_rng(self.random_state)
        # 3. compute initial squared bandwidth (documentation excerpt; keep synchronized)
        bandwidth_squared = self._bandwidth(matrix @ components)

        # 4. optimize for the configured update budget (documentation excerpt; keep synchronized)
        for _ in tqdm(
            range(1, self.max_iterations + 1),
            desc="MaxEnt-PCA",
            unit="update",
            disable=not sys.stderr.isatty(),
        ):
            # 4.1. sample ordered pairs (documentation excerpt; keep synchronized)
            pairs = rng.integers(len(matrix), size=(self.pair_batch_size, 2))
            # 4.2. compute feature differences (documentation excerpt; keep synchronized)
            differences = matrix[pairs[:, 0]] - matrix[pairs[:, 1]]
            # 4.3. estimate the kernel-mean gradient (documentation excerpt; keep synchronized)
            gradient = _pair_gradient(differences, components, bandwidth_squared)
            # 4.4. update the component matrix (documentation excerpt; keep synchronized)
            candidate = optimizer.step(components, gradient)
            # 4.5. compute thin svd of the candidate (documentation excerpt; keep synchronized)
            left, _, right = np.linalg.svd(candidate, full_matrices=False)
            # 4.6. set orthonormal components (documentation excerpt; keep synchronized)
            components = left @ right
            # 4.7. recompute squared bandwidth (documentation excerpt; keep synchronized)
            bandwidth_squared = self._bandwidth(matrix @ components)

        self.components = components
        self.scaler = scaler
        self.optimizer = optimizer
        self._diagnostics = {
            "iterations": self.max_iterations,
            "pairs_per_iteration": self.pair_batch_size,
            "final_bandwidth_squared": bandwidth_squared,
            "termination_reason": "iteration_budget",
        }
        self.logger.info(
            f"Completed MaxEnt-PCA fitting: updates={self.max_iterations}; no convergence claim"
        )

    def score(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        """Return baseline-compatible residuals using unchanged train preprocessing."""
        if self.components is None:
            self._fail("MaxEnt-PCA must be fitted before scoring")
        matrix = self._feature_matrix(features)
        if matrix.shape[1] != self.components.shape[0]:
            self._fail("MaxEnt-PCA scoring features have an incompatible width")
        # 5. score reconstruction residuals using training preprocessing (documentation excerpt; keep synchronized)
        matrix = self.scaler.transform(matrix)
        projected = matrix @ self.components
        residuals = np.linalg.norm(matrix - projected @ self.components.T, axis=1)
        if not np.isfinite(residuals).all():
            self._fail("MaxEnt-PCA produced nonfinite reconstruction residuals")
        self.logger.info(f"Scored {len(matrix)} rows with MaxEnt-PCA")
        return tuple(float(value) for value in residuals)

    def save(self, path: Path) -> None:
        """Save inference state, excluding the logger, optimizer, and training rows."""
        if self.components is None:
            self._fail("MaxEnt-PCA must be fitted before saving")
        with path.open("wb") as file:
            pickle.dump(
                {
                    "parameters": {
                        "n_components": self.n_components,
                        "standardize_features": self.standardize_features,
                        "bandwidth_scale": self.bandwidth_scale,
                        "pair_batch_size": self.pair_batch_size,
                        "max_iterations": self.max_iterations,
                        "random_state": self.random_state,
                        "optimizer": self.optimizer_settings,
                    },
                    "components": self.components,
                    "scaler": self.scaler,
                    "diagnostics": self._diagnostics,
                },
                file,
            )
        self.logger.info(f"Saved fitted MaxEnt-PCA model: {path}")

    @classmethod
    def load(cls, path: Path, logger: ComponentLogger) -> "MaxEntPCAModel":
        """Restore a trusted local artifact without fitting or sampling."""
        with path.open("rb") as file:
            payload = pickle.load(file)
        model = cls(logger=logger, **payload["parameters"])
        model.components = payload["components"]
        model.scaler = payload["scaler"]
        model._diagnostics = payload["diagnostics"]
        logger.info(f"Loaded fitted MaxEnt-PCA model: {path}")
        return model

    def _bandwidth(self, projected: np.ndarray) -> float:
        bandwidth_squared = self._bandwidth_squared(projected, self.bandwidth_scale)
        if not math.isfinite(bandwidth_squared) or bandwidth_squared <= 0:
            self._fail("MaxEnt-PCA requires finite, positive projected bandwidth")
        return bandwidth_squared

    @staticmethod
    def _bandwidth_squared(projected: np.ndarray, scale: float) -> float:
        centered = projected - projected.mean(axis=0)
        return float(2.0 * np.mean(np.sum(centered**2, axis=1)) / scale)

    def _feature_matrix(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        try:
            matrix = np.asarray(features, dtype=float)
        except (TypeError, ValueError):
            self._fail("MaxEnt-PCA requires a rectangular numeric feature matrix")
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
            self._fail("MaxEnt-PCA requires a nonempty two-dimensional feature matrix")
        if not np.isfinite(matrix).all():
            self._fail("MaxEnt-PCA requires finite feature values")
        return matrix

    def _fail(self, message: str) -> NoReturn:
        self.logger.error(message)
        raise ValueError(message)


def _pair_gradient(
    differences: np.ndarray, components: np.ndarray, bandwidth_squared: float
) -> np.ndarray:
    projected = differences @ components
    weights = np.exp(-np.sum(projected**2, axis=1) / (2.0 * bandwidth_squared))
    return -(differences.T @ (weights[:, None] * projected)) / (
        len(differences) * bandwidth_squared
    )
