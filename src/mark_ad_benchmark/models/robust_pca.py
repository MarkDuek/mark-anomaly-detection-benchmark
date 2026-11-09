import sys
from collections.abc import Mapping, Sequence

import numpy as np

# import standardscaler dependency (documentation excerpt; keep synchronized)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models.base import TransductiveAnomalyModel


class RobustPCAModel(TransductiveAnomalyModel):
    """Principal Component Pursuit with row-wise sparse L1 anomaly scores."""

    def __init__(
        self, parameters: Mapping[str, object], logger: ComponentLogger
    ) -> None:
        # configure robust pca solver parameters (documentation excerpt; keep synchronized)
        self.lambda_value = parameters.get("lambda")
        self.tolerance = parameters.get("tolerance")
        self.max_iterations = parameters.get("max_iterations")
        self.logger = logger
        self.validate_parameters()
        self.low_rank_component: tuple[tuple[float, ...], ...] | None = None
        self.sparse_component: tuple[tuple[float, ...], ...] | None = None
        self._diagnostics: dict[str, object] = {}

    def validate_parameters(self) -> None:
        parameters = {
            "lambda": self.lambda_value,
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
        }
        self.lambda_value = positive_float(parameters, "lambda", self.logger)
        self.tolerance = positive_float(parameters, "tolerance", self.logger)
        self.max_iterations = positive_int(parameters, "max_iterations", self.logger)

    @property
    def diagnostics(self) -> Mapping[str, object]:
        """Expose convergence metadata for the most recent persisted artifact."""
        return self._diagnostics

    def fit_score(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        """Decompose one feature batch and return sparse row L1 magnitudes."""
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
            message = "Robust PCA requires a nonempty two-dimensional feature matrix"
            self.logger.error(message)
            raise ValueError(message)
        if not np.isfinite(matrix).all():
            message = "Robust PCA requires finite feature values"
            self.logger.error(message)
            raise ValueError(message)
        # 1. standardize the complete transductive feature batch (documentation excerpt; keep synchronized)
        matrix = StandardScaler().fit_transform(matrix)
        self.logger.info(
            f"Standardizing and decomposing {matrix.shape[0]} rows and {matrix.shape[1]} features with Robust PCA"
        )
        matrix_norm = float(np.linalg.norm(matrix, ord="fro"))
        if matrix_norm == 0.0:
            zeros = np.zeros_like(matrix)
            return self.finish(zeros, zeros, 0, 0.0, 0.0)
        # 2.1. initialize augmented-lagrangian solver state (documentation excerpt; keep synchronized)
        mu = matrix.size / (4.0 * float(np.abs(matrix).sum()))
        low_rank = np.zeros_like(matrix)
        sparse = np.zeros_like(matrix)
        dual = np.zeros_like(matrix)
        convergence_threshold = self.tolerance * matrix_norm
        # 2.2. alternate low-rank and sparse updates until convergence (documentation excerpt; keep synchronized)
        with tqdm(
            total=self.max_iterations,
            desc="Robust PCA",
            unit="iteration",
            leave=False,
            disable=not sys.stderr.isatty(),
        ) as progress:
            for iteration in range(1, self.max_iterations + 1):
                low_rank = singular_value_threshold(
                    matrix - sparse + dual / mu, 1.0 / mu
                )
                sparse = soft_threshold(
                    matrix - low_rank + dual / mu, self.lambda_value / mu
                )
                residual_matrix = matrix - low_rank - sparse
                residual = float(np.linalg.norm(residual_matrix, ord="fro"))
                dual += mu * residual_matrix
                progress.update()
                if residual <= convergence_threshold:
                    return self.finish(
                        low_rank, sparse, iteration, residual, convergence_threshold
                    )
            message = (
                "Robust PCA did not converge within "
                f"{self.max_iterations} iterations; residual={residual}"
            )
            self.logger.error(message)
            raise RuntimeError(message)

    def finish(
        self,
        low_rank: np.ndarray,
        sparse: np.ndarray,
        iterations: int,
        residual: float,
        convergence_threshold: float,
    ) -> tuple[float, ...]:
        """Store one decomposition and return its timestamp anomaly scores."""

        # 3. retain decomposition diagnostics and score sparse row magnitudes (documentation excerpt; keep synchronized)
        self.low_rank_component = tuple(tuple(row) for row in low_rank.tolist())
        self.sparse_component = tuple(tuple(row) for row in sparse.tolist())
        self._diagnostics = {
            "iterations": iterations,
            "residual": residual,
            "convergence_threshold": convergence_threshold,
        }
        scores = tuple(float(np.abs(row).sum()) for row in sparse)
        self.logger.info(
            f"Robust PCA converged in {iterations} iterations with residual={residual}"
        )
        return scores


def positive_float(
    parameters: Mapping[str, object], name: str, logger: ComponentLogger
) -> float:
    """Read one required strictly positive floating solver parameter."""
    value = parameters.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        message = f"Robust PCA {name} must be a positive number"
        logger.error(message)
        raise ValueError(message)
    return float(value)


def positive_int(
    parameters: Mapping[str, object], name: str, logger: ComponentLogger
) -> int:
    """Read one required strictly positive integer solver parameter."""
    value = parameters.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        message = f"Robust PCA {name} must be a positive integer"
        logger.error(message)
        raise ValueError(message)
    return value


def soft_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Apply element-wise soft thresholding."""
    return np.sign(matrix) * np.maximum(np.abs(matrix) - threshold, 0.0)


def singular_value_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Apply singular-value thresholding."""
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    return (left * np.maximum(singular_values - threshold, 0.0)) @ right
