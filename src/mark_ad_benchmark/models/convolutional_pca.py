"""Finite-filter batch ConvPCA following lixilinx/ConvPCA's MATLAB solver."""

import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

import numpy as np
from sklearn.preprocessing import StandardScaler

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models.base import InductiveAnomalyModel


class ConvolutionalPCAModel(InductiveAnomalyModel):
    """Learn finite convolutional filters and score timestamp reconstruction errors."""

    def __init__(
        self, parameters: Mapping[str, object], logger: ComponentLogger
    ) -> None:
        self.logger = logger
        self.n_components = parameters.get("n_components")
        self.filter_order = parameters.get("filter_order")
        self.standardize_features = parameters.get("standardize_features")
        self.step_size = parameters.get("step_size", 0.2)
        self.max_iterations = parameters.get("max_iterations", 200)
        for name, value, minimum in (
            ("n_components", self.n_components, 1),
            ("filter_order", self.filter_order, 0),
            ("max_iterations", self.max_iterations, 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                self._fail(f"Convolutional PCA {name} must be an integer >= {minimum}")
        if not isinstance(self.standardize_features, bool):
            self._fail("Convolutional PCA standardize_features must be a boolean")
        if (
            not isinstance(self.step_size, (int, float))
            or isinstance(self.step_size, bool)
            or not math.isfinite(self.step_size)
            or self.step_size <= 0
        ):
            self._fail("Convolutional PCA step_size must be finite and positive")
        self.scaler = StandardScaler(with_std=self.standardize_features)
        self.filters: np.ndarray | None = None
        self._diagnostics: dict[str, object] = {}

    @property
    def diagnostics(self) -> Mapping[str, object]:
        """Expose solver iterations, convergence, and full zero-padded fitting MSE."""
        return self._diagnostics

    def fit(self, features: Sequence[Sequence[float]]) -> None:
        """Fit centering/scaling and optimize finite filters using train data only."""
        matrix = self._feature_matrix(features)
        width = matrix.shape[1]
        if self.n_components >= width:
            self._fail(
                "Convolutional PCA components must be fewer than the feature width"
            )
        if len(matrix) <= 2 * self.filter_order:
            self._fail(
                "Convolutional PCA training rows must exceed twice the filter order"
            )
        matrix = self.scaler.fit_transform(matrix)
        self.logger.info(
            f"Fitting convolutional PCA with {self.n_components} components, order {self.filter_order}, and {len(matrix)} rows"
        )
        correlations = _lag_correlations(matrix, self.filter_order)
        frequencies = np.arange(self.n_components)
        basis = np.cos(
            np.pi * (np.arange(width)[:, None] + 0.5) * frequencies[None, :] / width
        ) * np.sqrt(2.0 / width)
        basis[:, 0] /= np.sqrt(2.0)
        filters = np.zeros((self.filter_order + 1, width, self.n_components))
        filters[self.filter_order // 2] = basis
        power = float(np.trace(correlations[2 * self.filter_order]))
        iterations, converged = 0, power == 0.0
        if power > 0.0:
            scale_squared = width / power
            momentum = np.zeros_like(filters)
            for iterations in range(1, self.max_iterations + 1):
                _, gradient, _ = _objective_gradient(correlations, filters)
                momentum = 0.9 * momentum + 0.1 * gradient
                filters = filters - self.step_size * scale_squared * momentum
                if not np.isfinite(filters).all():
                    self._fail(
                        "Convolutional PCA optimization produced nonfinite filters"
                    )
                if scale_squared * np.max(np.abs(gradient)) < 1e-6:
                    converged = True
                    break
        error, _, component_covariance = _objective_gradient(correlations, filters)
        if not math.isfinite(error):
            self._fail(
                "Convolutional PCA optimization produced nonfinite fitting error"
            )
        _, rotation = np.linalg.eigh(component_covariance + component_covariance.T)
        self.filters = filters @ rotation
        self._diagnostics = {
            "iterations": iterations,
            "converged": converged,
            "fitting_mse": max(0.0, error),
        }
        if not converged:
            self.logger.warning(
                f"Convolutional PCA reached {iterations} iterations before gradient convergence"
            )
        self.logger.info(
            f"Completed convolutional PCA fitting: iterations={iterations}, fitting_mse={error}"
        )

    def score(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        """Score a retrospective split independently, with zero-padded boundaries."""
        if self.filters is None:
            self._fail("Convolutional PCA must be fitted before scoring")
        matrix = self._feature_matrix(features)
        if matrix.shape[1] != self.filters.shape[1]:
            self._fail("Convolutional PCA scoring features have an incompatible width")
        matrix = self.scaler.transform(matrix)
        components = _apply_filter(_adjoint(self.filters), matrix)
        reconstructed = _apply_filter(self.filters, components)
        reconstructed = reconstructed[
            self.filter_order : self.filter_order + len(matrix)
        ]
        residuals = np.linalg.norm(matrix - reconstructed, axis=1)
        if not np.isfinite(residuals).all():
            self._fail("Convolutional PCA produced nonfinite reconstruction residuals")
        self.logger.info(f"Scored {len(matrix)} rows with convolutional PCA")
        return tuple(float(value) for value in residuals)

    def save(self, path: Path) -> None:
        """Persist learned filters and train-fitted scaling without the logger."""
        if self.filters is None:
            self._fail("Convolutional PCA must be fitted before saving")
        with path.open("wb") as file:
            pickle.dump(
                {
                    "parameters": {
                        "n_components": self.n_components,
                        "filter_order": self.filter_order,
                        "standardize_features": self.standardize_features,
                        "step_size": self.step_size,
                        "max_iterations": self.max_iterations,
                    },
                    "scaler": self.scaler,
                    "filters": self.filters,
                    "diagnostics": self._diagnostics,
                },
                file,
            )
        self.logger.info(f"Saved fitted convolutional PCA model: {path}")

    @classmethod
    def load(cls, path: Path, logger: ComponentLogger) -> "ConvolutionalPCAModel":
        """Restore a trusted local filter artifact with the current logger."""
        with path.open("rb") as file:
            payload = pickle.load(file)
        model = cls(payload["parameters"], logger)
        model.scaler = payload["scaler"]
        model.filters = payload["filters"]
        model._diagnostics = payload["diagnostics"]
        logger.info(f"Loaded fitted convolutional PCA model: {path}")
        return model

    def _feature_matrix(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        try:
            matrix = np.asarray(features, dtype=float)
        except (TypeError, ValueError):
            self._fail(
                "Convolutional PCA requires a rectangular numeric feature matrix"
            )
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
            self._fail(
                "Convolutional PCA requires a nonempty two-dimensional feature matrix"
            )
        if not np.isfinite(matrix).all():
            self._fail("Convolutional PCA requires finite feature values")
        return matrix

    def _fail(self, message: str) -> NoReturn:
        self.logger.error(message)
        raise ValueError(message)


def _polynomial_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    product = np.zeros((len(left) + len(right) - 1, left.shape[1], right.shape[2]))
    for i, coefficient in enumerate(left):
        for j, other in enumerate(right):
            product[i + j] += coefficient @ other
    return product


def _adjoint(filters: np.ndarray) -> np.ndarray:
    return filters[::-1].transpose(0, 2, 1)


def _lag_correlations(matrix: np.ndarray, order: int) -> np.ndarray:
    width = matrix.shape[1]
    correlations = np.zeros((4 * order + 1, width, width))
    correlations[2 * order] = matrix.T @ matrix / len(matrix)
    for lag in range(1, 2 * order + 1):
        covariance = matrix[lag:].T @ matrix[:-lag] / len(matrix)
        correlations[2 * order + lag] = covariance
        correlations[2 * order - lag] = covariance.T
    return correlations


def _objective_gradient(
    correlations: np.ndarray, filters: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    order = len(filters) - 1
    rx_w = _polynomial_product(correlations, filters)
    wh_rx_w = _polynomial_product(_adjoint(filters), rx_w)
    wh_w = _polynomial_product(_adjoint(filters), filters)
    component_covariance = wh_rx_w[3 * order]
    error = (
        np.trace(correlations[2 * order])
        - 2 * np.trace(component_covariance)
        + np.sum(wh_w * wh_rx_w[2 * order : 4 * order + 1])
    )
    gradient = -2 * rx_w[2 * order : 3 * order + 1]
    term1 = _polynomial_product(filters, wh_rx_w[2 * order : 4 * order + 1])
    gradient += term1[order : 2 * order + 1]
    term2 = _polynomial_product(rx_w[order : 4 * order + 1], wh_w)
    gradient += term2[2 * order : 3 * order + 1]
    return float(error), gradient, component_covariance


def _apply_filter(filters: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    result = np.zeros((len(matrix) + len(filters) - 1, filters.shape[1]))
    for lag, coefficient in enumerate(filters):
        result[lag : lag + len(matrix)] += matrix @ coefficient.T
    return result
