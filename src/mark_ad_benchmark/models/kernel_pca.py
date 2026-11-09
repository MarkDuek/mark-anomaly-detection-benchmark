import math
import pickle
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import NoReturn

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models.base import InductiveAnomalyModel


class KernelType(StrEnum):
    LINEAR = "linear"
    POLY = "poly"
    RBF = "rbf"


class KernelPCAModel(InductiveAnomalyModel):
    """Train-centered Kernel PCA with feature-space reconstruction residuals."""

    def __init__(
        self, parameters: Mapping[str, object], logger: ComponentLogger
    ) -> None:
        self.logger = logger
        self.n_components = parameters.get("n_components")
        self.standardize_features = parameters.get("standardize_features")
        self.kernel = parameters.get("kernel")
        self.kernel_parameters = parameters.get("kernel_parameters", {})
        self.validate_parameters()
        self.pca = KernelPCA(
            n_components=self.n_components,
            kernel="precomputed",
            eigen_solver="arpack",
            random_state=0,
        )
        self.scaler = StandardScaler() if self.standardize_features else None
        self.train_features: np.ndarray | None = None
        self.kernel_mean = 0.0

    def validate_parameters(self) -> None:
        if (
            not isinstance(self.n_components, int)
            or isinstance(self.n_components, bool)
            or self.n_components < 1
        ):
            self._fail("Kernel PCA n_components must be a positive integer")
        try:
            self.kernel = KernelType(self.kernel)
        except (TypeError, ValueError):
            self._fail("Kernel PCA kernel must be linear, poly, or rbf")
        if not isinstance(self.standardize_features, bool):
            self._fail("Kernel PCA standardize_features must be a boolean")
        self.kernel_parameters = self._validate_kernel_parameters(
            self.kernel_parameters
        )

    def fit(self, features: Sequence[Sequence[float]]) -> None:
        """Fit scaling, kernel centering, and principal directions on train only."""
        matrix = self._feature_matrix(features)
        if self.n_components >= len(matrix):
            self._fail("Kernel PCA components must be fewer than the training rows")
        if self.scaler is not None:
            matrix = self.scaler.fit_transform(matrix)
        self.logger.info(
            f"Fitting Kernel PCA ({self.kernel}) with {self.n_components} components on {len(matrix)} rows"
        )
        kernel = self._kernel_matrix(matrix, matrix)
        self.pca.fit(kernel)
        self.kernel_mean = float(kernel.mean())
        self.train_features = matrix

    def score(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        """Return distances to the learned subspace in centered kernel space."""
        if self.train_features is None:
            self._fail("Kernel PCA must be fitted before scoring")
        matrix = self._feature_matrix(features)
        if matrix.shape[1] != self.train_features.shape[1]:
            self._fail("Kernel PCA scoring features have an incompatible width")
        if self.scaler is not None:
            matrix = self.scaler.transform(matrix)
        kernel = self._kernel_matrix(matrix, self.train_features)
        projected = self.pca.transform(kernel)
        squared_norm = np.einsum("ij,ij->i", matrix, matrix)
        if self.kernel is KernelType.POLY:
            squared_norm = (
                self.kernel_parameters["gamma"] * squared_norm
                + self.kernel_parameters["coef0"]
            ) ** self.kernel_parameters["degree"]
        elif self.kernel is KernelType.RBF:
            squared_norm = np.ones(len(matrix))
        centered_norm = squared_norm - 2.0 * kernel.mean(axis=1) + self.kernel_mean
        projected_norm = np.einsum("ij,ij->i", projected, projected)
        squared_residual = centered_norm - projected_norm
        tolerance = (
            100.0
            * np.finfo(float).eps
            * np.maximum(1.0, np.maximum(np.abs(squared_norm), np.abs(projected_norm)))
        )
        if not np.isfinite(squared_residual).all() or np.any(
            squared_residual < -tolerance
        ):
            self._fail("Kernel PCA produced invalid feature-space residuals")
        scores = tuple(
            float(value) for value in np.sqrt(np.maximum(squared_residual, 0))
        )
        self.logger.info(f"Scored {len(scores)} rows with Kernel PCA ({self.kernel})")
        return scores

    def save(self, path: Path) -> None:
        """Persist fitted kernel and scaling state without its logger."""
        if self.train_features is None:
            self._fail("Kernel PCA must be fitted before saving")
        with path.open("wb") as file:
            pickle.dump(
                {
                    "parameters": {
                        "n_components": self.n_components,
                        "kernel": self.kernel.value,
                        "standardize_features": self.standardize_features,
                        "kernel_parameters": self.kernel_parameters,
                    },
                    "pca": self.pca,
                    "scaler": self.scaler,
                    "train_features": self.train_features,
                    "kernel_mean": self.kernel_mean,
                },
                file,
            )
        self.logger.info(f"Saved fitted Kernel PCA model: {path}")

    @classmethod
    def load(cls, path: Path, logger: ComponentLogger) -> "KernelPCAModel":
        """Restore fitted Kernel PCA state from a trusted local artifact."""
        with path.open("rb") as file:
            payload = pickle.load(file)
        model = cls(payload["parameters"], logger)
        model.pca = payload["pca"]
        model.scaler = payload["scaler"]
        model.train_features = payload["train_features"]
        model.kernel_mean = payload["kernel_mean"]
        logger.info(f"Loaded fitted Kernel PCA model: {path}")
        return model

    def _validate_kernel_parameters(self, parameters: object) -> dict[str, object]:
        defaults = {
            KernelType.LINEAR: {},
            KernelType.POLY: {"gamma": 1.0, "degree": 2, "coef0": 1.0},
            KernelType.RBF: {"gamma": 1.0},
        }[self.kernel]
        if not isinstance(parameters, Mapping):
            self._fail("Kernel PCA kernel_parameters must be a mapping")
        for name in parameters:
            if name not in defaults:
                self._fail(
                    f"Kernel PCA {self.kernel} does not support kernel parameter {name}"
                )
        effective = {**defaults, **parameters}
        for name, value in effective.items():
            if name == "degree":
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    self._fail("Kernel PCA degree must be a positive integer")
            elif (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or (value <= 0 if name == "gamma" else value < 0)
            ):
                self._fail(
                    f"Kernel PCA {name} must be finite and "
                    + ("positive" if name == "gamma" else "nonnegative")
                )
        return effective

    def _feature_matrix(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        try:
            matrix = np.asarray(features, dtype=float)
        except (TypeError, ValueError):
            self._fail("Kernel PCA requires a rectangular numeric feature matrix")
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
            self._fail("Kernel PCA requires a nonempty two-dimensional feature matrix")
        if not np.isfinite(matrix).all():
            self._fail("Kernel PCA requires finite feature values")
        return matrix

    def _kernel_matrix(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        kernel = pairwise_kernels(
            left,
            right,
            metric=self.kernel.value,
            **self.kernel_parameters,
        )
        if not np.isfinite(kernel).all():
            self._fail("Kernel PCA requires finite kernel values")
        return kernel

    def _fail(self, message: str) -> NoReturn:
        self.logger.error(message)
        raise ValueError(message)
