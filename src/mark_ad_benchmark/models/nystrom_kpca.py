"""Train-only Nyström kernel PCA without a full training Gram matrix."""

import math
import pickle
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import NoReturn

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models.base import InductiveAnomalyModel


class NystromKernelType(StrEnum):
    LINEAR = "linear"
    POLY = "poly"
    RBF = "rbf"


def _uniform_landmarks(matrix: np.ndarray, count: int, seed: int | None) -> np.ndarray:
    # 1. sample distinct landmark indices (documentation excerpt; keep synchronized)
    indices = np.random.default_rng(seed).choice(len(matrix), count, replace=False)
    return matrix[indices]


# configure landmark samplers (documentation excerpt; keep synchronized)
_LANDMARK_SAMPLERS = {"uniform": _uniform_landmarks}


class NystromKPCA:
    """Project onto train-centered principal directions of Nyström kernel features."""

    def __init__(
        self,
        *,
        n_components: int | None = None,
        n_landmarks: int = 100,
        kernel: str = "rbf",
        kernel_params: Mapping[str, object] | None = None,
        sampling: str = "uniform",
        nystrom_solver: str = "eigh",
        pca_solver: str = "eigh",
        eigenvalue_strategy: str = "truncate",
        eigenvalue_tol: float = 1e-12,
        eigenvalue_tol_mode: str = "absolute",
        covariance_normalization: str = "population",
        random_state: int | None = None,
        standardize_features: bool = False,
        logger: ComponentLogger | None = None,
    ) -> None:
        self.logger = logger
        self.n_components = n_components
        self.n_landmarks = n_landmarks
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.sampling = sampling
        self.nystrom_solver = nystrom_solver
        self.pca_solver = pca_solver
        self.eigenvalue_strategy = eigenvalue_strategy
        self.eigenvalue_tol = eigenvalue_tol
        self.eigenvalue_tol_mode = eigenvalue_tol_mode
        self.covariance_normalization = covariance_normalization
        self.random_state = random_state
        self.standardize_features = standardize_features
        self.validate_parameters()
        self.eigenvalue_tol = float(self.eigenvalue_tol)
        self.landmarks_: np.ndarray | None = None
        self.normalizer_: np.ndarray | None = None
        self.feature_mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_width_: int | None = None
        self.feature_explained_variance_: np.ndarray | None = None
        self.feature_explained_variance_ratio_: np.ndarray | None = None
        self.feature_total_variance_: float | None = None

    def validate_parameters(self) -> None:
        if self.n_components is not None and (
            not isinstance(self.n_components, int)
            or isinstance(self.n_components, bool)
            or self.n_components < 1
        ):
            self._fail("Nyström KPCA n_components must be a positive integer or None")
        if (
            not isinstance(self.n_landmarks, int)
            or isinstance(self.n_landmarks, bool)
            or self.n_landmarks < 1
        ):
            self._fail("Nyström KPCA n_landmarks must be a positive integer")
        try:
            self.kernel = NystromKernelType(self.kernel)
        except (ValueError, TypeError):
            self._fail("Nyström KPCA kernel must be linear, poly, or rbf")
        self.kernel_params = self._kernel_parameters(self.kernel_params)
        if self.sampling not in _LANDMARK_SAMPLERS:
            self._fail("Nyström KPCA sampling must be uniform")
        if self.nystrom_solver not in ("eigh", "svd") or self.pca_solver not in (
            "eigh",
            "svd",
        ):
            self._fail("Nyström KPCA solvers must be eigh or svd")
        if self.eigenvalue_strategy != "truncate":
            self._fail("Nyström KPCA eigenvalue_strategy must be truncate")
        if (
            not isinstance(self.eigenvalue_tol, (int, float))
            or isinstance(self.eigenvalue_tol, bool)
            or not math.isfinite(self.eigenvalue_tol)
            or self.eigenvalue_tol <= 0
        ):
            self._fail("Nyström KPCA eigenvalue_tol must be finite and positive")
        if self.eigenvalue_tol_mode not in ("absolute", "relative"):
            self._fail("Nyström KPCA eigenvalue_tol_mode must be absolute or relative")
        if self.covariance_normalization not in ("population", "sample"):
            self._fail(
                "Nyström KPCA covariance_normalization must be population or sample"
            )
        if self.random_state is not None and (
            not isinstance(self.random_state, int)
            or isinstance(self.random_state, bool)
            or self.random_state < 0
        ):
            self._fail(
                "Nyström KPCA random_state must be a nonnegative integer or None"
            )
        if not isinstance(self.standardize_features, bool):
            self._fail("Nyström KPCA standardize_features must be a boolean")

    def fit(self, features: Sequence[Sequence[float]]) -> "NystromKPCA":
        """Learn landmarks, normalization, centering, and PCA from training rows."""
        self._fit_projection(features)
        return self

    def fit_transform(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        """Fit and return training PCA scores without a second kernel evaluation."""
        return self._fit_projection(features)

    def _fit_projection(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = self._feature_matrix(features)
        if self.n_landmarks >= len(matrix):
            self._fail("Nyström KPCA n_landmarks must be fewer than training rows")
        scaler = StandardScaler() if self.standardize_features else None
        if scaler is not None:
            matrix = scaler.fit_transform(matrix)
        if self.logger is not None:
            self.logger.info(
                f"Fitting Nyström KPCA on {len(matrix)} rows with {self.n_landmarks} landmarks"
            )

        # 1. select landmarks (documentation excerpt; keep synchronized)
        landmarks = _LANDMARK_SAMPLERS[self.sampling](
            matrix, self.n_landmarks, self.random_state
        )
        # 3. evaluate observation-landmark kernels (documentation excerpt; keep synchronized)
        landmark_kernel = self._kernel_matrix(landmarks, landmarks)
        cross_kernel = self._kernel_matrix(matrix, landmarks)

        # 4. construct orthonormal nystrom coordinates (documentation excerpt; keep synchronized)
        normalizer = self._normalizer(landmark_kernel)
        z = cross_kernel @ normalizer
        del cross_kernel, landmark_kernel

        # 5. center coordinates using training rows (documentation excerpt; keep synchronized)
        feature_mean = z.mean(axis=0)
        z -= feature_mean

        # 6. solve covariance directions or thin svd (documentation excerpt; keep synchronized)
        denominator = (
            len(matrix)
            if self.covariance_normalization == "population"
            else len(matrix) - 1
        )
        if self.pca_solver == "eigh":
            covariance = z.T @ z / denominator
            eigenvalues, directions = np.linalg.eigh(covariance)
            eigenvalues = eigenvalues[::-1]
            directions = directions[:, ::-1]
        else:
            _, singular_values, right = np.linalg.svd(z, full_matrices=False)
            eigenvalues = singular_values**2 / denominator
            directions = right.T
        # 7. select the retained component count (documentation excerpt; keep synchronized)
        total_variance = float(np.einsum("ij,ij->", z, z) / denominator)
        rank_tol = np.finfo(float).eps * max(z.shape) * max(0.0, float(eigenvalues[0]))
        if not math.isfinite(total_variance) or total_variance <= rank_tol:
            self._fail("Nyström KPCA requires positive centered feature-space variance")
        if np.any(eigenvalues < -rank_tol):
            self._fail("Nyström KPCA produced negative covariance eigenvalues")
        positive_rank = int(np.count_nonzero(eigenvalues > rank_tol))
        retained = z.shape[1] if self.n_components is None else self.n_components
        if retained > z.shape[1]:
            self._fail("Nyström KPCA n_components exceeds landmark effective rank")
        if self.n_components is not None and retained > positive_rank:
            self._fail("Nyström KPCA n_components exceeds positive covariance rank")

        self.landmarks_ = landmarks
        self.normalizer_ = normalizer
        self.feature_mean_ = feature_mean
        # 8. retain principal directions (documentation excerpt; keep synchronized)
        self.components_ = directions[:, :retained].copy()
        self.scaler_ = scaler
        self.feature_width_ = matrix.shape[1]
        # 7. record explained variance (documentation excerpt; keep synchronized)
        self.feature_total_variance_ = total_variance
        self.feature_explained_variance_ = np.maximum(eigenvalues[:retained], 0)
        self.feature_explained_variance_ratio_ = (
            self.feature_explained_variance_ / total_variance
        )

        # 8. project training observations (documentation excerpt; keep synchronized)
        projection = z @ self.components_
        if self.logger is not None:
            self.logger.info(
                f"Fitted Nyström KPCA: effective rank={z.shape[1]}, components={retained}"
            )
        return projection

    def transform(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        """Project new observations using unchanged training landmarks and mean."""
        # 9. project new rows (documentation excerpt; keep synchronized)
        centered = self._centered_features(features)
        projection = centered @ self.components_
        if not np.isfinite(projection).all():
            self._fail("Nyström KPCA produced nonfinite component scores")
        if self.logger is not None:
            self.logger.info(f"Projected {len(projection)} rows with Nyström KPCA")
        return projection

    def _centered_features(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        if (
            self.landmarks_ is None
            or self.normalizer_ is None
            or self.feature_mean_ is None
            or self.components_ is None
        ):
            self._fail("Nyström KPCA must be fitted before transforming")
        matrix = self._feature_matrix(features, width=self.feature_width_)
        if self.scaler_ is not None and len(matrix):
            matrix = self.scaler_.transform(matrix)
        if not len(matrix):
            return np.empty((0, self.normalizer_.shape[1]))
        # 9. map and center new rows (documentation excerpt; keep synchronized)
        cross_kernel = self._kernel_matrix(matrix, self.landmarks_)
        centered = cross_kernel @ self.normalizer_ - self.feature_mean_
        if not np.isfinite(centered).all():
            self._fail("Nyström KPCA produced nonfinite Nyström features")
        return centered

    def _normalizer(self, landmark_kernel: np.ndarray) -> np.ndarray:
        landmark_kernel = (landmark_kernel + landmark_kernel.T) / 2
        # 4. solve and truncate the landmark kernel (documentation excerpt; keep synchronized)
        if self.nystrom_solver == "eigh":
            values, vectors = np.linalg.eigh(landmark_kernel)
            indices = np.argsort(values)[::-1]
            values, vectors = values[indices], vectors[:, indices]
            signed_values = values
        else:
            signed_values = np.linalg.eigvalsh(landmark_kernel)
            vectors, values, _ = np.linalg.svd(landmark_kernel, full_matrices=False)
        cutoff = self.eigenvalue_tol
        if self.eigenvalue_tol_mode == "relative":
            cutoff *= max(0.0, float(np.max(signed_values)))
        if np.min(signed_values) < -cutoff:
            self._fail("Nyström KPCA landmark kernel has negative eigenvalues")
        keep = values > cutoff
        if not np.any(keep):
            self._fail("Nyström KPCA landmark kernel has no eigenvalues above cutoff")
        return vectors[:, keep] / np.sqrt(values[keep])

    def _kernel_parameters(
        self, parameters: Mapping[str, object] | None
    ) -> dict[str, object]:
        defaults = {
            NystromKernelType.LINEAR: {},
            NystromKernelType.POLY: {"gamma": 1.0, "degree": 2, "coef0": 1.0},
            NystromKernelType.RBF: {"gamma": 1.0},
        }[self.kernel]
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, Mapping):
            self._fail("Nyström KPCA kernel_params must be a mapping")
        for name in parameters:
            if name not in defaults:
                self._fail(
                    f"Nyström KPCA {self.kernel} does not support kernel parameter {name}"
                )
        effective = {**defaults, **parameters}
        for name, value in effective.items():
            if name == "degree":
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    self._fail("Nyström KPCA degree must be a positive integer")
            elif (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or (value <= 0 if name == "gamma" else value < 0)
            ):
                self._fail(
                    f"Nyström KPCA {name} must be finite and "
                    + ("positive" if name == "gamma" else "nonnegative")
                )
        return effective

    def _kernel_matrix(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        kernel = pairwise_kernels(
            left, right, metric=self.kernel.value, **self.kernel_params
        )
        if not np.isfinite(kernel).all():
            self._fail("Nyström KPCA requires finite kernel values")
        return kernel

    def _feature_matrix(
        self, features: Sequence[Sequence[float]], width: int | None = None
    ) -> np.ndarray:
        try:
            matrix = np.asarray(features, dtype=float)
        except (TypeError, ValueError):
            self._fail("Nyström KPCA requires a rectangular numeric feature matrix")
        if width is not None and matrix.shape == (0,):
            matrix = matrix.reshape(0, width)
        if (
            matrix.ndim != 2
            or (width is None and not len(matrix))
            or not matrix.shape[1]
        ):
            self._fail("Nyström KPCA requires a two-dimensional feature matrix")
        if width is not None and matrix.shape[1] != width:
            self._fail("Nyström KPCA features have an incompatible width")
        if not np.isfinite(matrix).all():
            self._fail("Nyström KPCA requires finite feature values")
        return matrix

    def _fail(self, message: str) -> NoReturn:
        if self.logger is not None:
            self.logger.error(message)
        raise ValueError(message)


class NystromKPCAModel(InductiveAnomalyModel):
    """Benchmark adapter scoring centered Nyström reconstruction residuals."""

    def __init__(
        self,
        *,
        logger: ComponentLogger,
        n_components: int | None = None,
        n_landmarks: int = 100,
        kernel: str = "rbf",
        kernel_params: Mapping[str, object] | None = None,
        sampling: str = "uniform",
        nystrom_solver: str = "eigh",
        pca_solver: str = "eigh",
        eigenvalue_strategy: str = "truncate",
        eigenvalue_tol: float = 1e-12,
        eigenvalue_tol_mode: str = "absolute",
        covariance_normalization: str = "population",
        random_state: int | None = None,
        standardize_features: bool = False,
    ) -> None:
        self.logger = logger
        self.estimator = NystromKPCA(
            logger=logger,
            n_components=n_components,
            n_landmarks=n_landmarks,
            kernel=kernel,
            kernel_params=kernel_params,
            sampling=sampling,
            nystrom_solver=nystrom_solver,
            pca_solver=pca_solver,
            eigenvalue_strategy=eigenvalue_strategy,
            eigenvalue_tol=eigenvalue_tol,
            eigenvalue_tol_mode=eigenvalue_tol_mode,
            covariance_normalization=covariance_normalization,
            random_state=random_state,
            standardize_features=standardize_features,
        )
        self.validate_parameters()

    def validate_parameters(self) -> None:
        self.estimator.validate_parameters()

    def fit(self, features: Sequence[Sequence[float]]) -> None:
        self.estimator.fit(features)
        if self.estimator.components_.shape[1] == self.estimator.normalizer_.shape[1]:
            self.estimator.components_ = None
            message = "Nyström KPCA anomaly scoring requires fewer components than effective rank"
            self.logger.error(message)
            raise ValueError(message)

    def score(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        """Return Euclidean reconstruction residuals in centered Nyström space."""
        # 10. score centered reconstruction residuals (documentation excerpt; keep synchronized)
        centered = self.estimator._centered_features(features)
        projected = centered @ self.estimator.components_
        residuals = np.linalg.norm(
            centered - projected @ self.estimator.components_.T, axis=1
        )
        if not np.isfinite(residuals).all():
            message = "Nyström KPCA produced nonfinite reconstruction residuals"
            self.logger.error(message)
            raise ValueError(message)
        self.logger.info(f"Scored {len(residuals)} rows with Nyström KPCA")
        return tuple(float(value) for value in residuals)

    def save(self, path: Path) -> None:
        """Persist fitted state without the process-specific logger or training data."""
        if self.estimator.components_ is None:
            message = "Nyström KPCA must be fitted before saving"
            self.logger.error(message)
            raise ValueError(message)
        state = {
            name: getattr(self.estimator, name)
            for name in (
                "landmarks_",
                "normalizer_",
                "feature_mean_",
                "components_",
                "scaler_",
                "feature_width_",
                "feature_explained_variance_",
                "feature_explained_variance_ratio_",
                "feature_total_variance_",
            )
        }
        parameters = {
            name: getattr(self.estimator, name)
            for name in (
                "n_components",
                "n_landmarks",
                "kernel",
                "kernel_params",
                "sampling",
                "nystrom_solver",
                "pca_solver",
                "eigenvalue_strategy",
                "eigenvalue_tol",
                "eigenvalue_tol_mode",
                "covariance_normalization",
                "random_state",
                "standardize_features",
            )
        }
        parameters["kernel"] = self.estimator.kernel.value
        with path.open("wb") as file:
            pickle.dump({"parameters": parameters, "state": state}, file)
        self.logger.info(f"Saved fitted Nyström KPCA model: {path}")

    @classmethod
    def load(cls, path: Path, logger: ComponentLogger) -> "NystromKPCAModel":
        """Restore a trusted local fitted artifact with the current logger."""
        with path.open("rb") as file:
            payload = pickle.load(file)
        model = cls(logger=logger, **payload["parameters"])
        for name, value in payload["state"].items():
            setattr(model.estimator, name, value)
        logger.info(f"Loaded fitted Nyström KPCA model: {path}")
        return model
