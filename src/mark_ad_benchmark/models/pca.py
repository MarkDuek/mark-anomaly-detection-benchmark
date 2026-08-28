import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models.base import InductiveAnomalyModel


class PCAModel(InductiveAnomalyModel):
    """Configurable train-scaled PCA with Euclidean reconstruction-residual scores."""

    def __init__(
        self, parameters: Mapping[str, object], logger: ComponentLogger
    ) -> None:
        n_components = parameters.get("n_components")
        if not isinstance(n_components, int):
            message = "PCA n_components must be an integer"
            logger.error(message)
            raise TypeError(message)
        if n_components < 1:
            message = "PCA requires at least one component"
            logger.error(message)
            raise ValueError(message)
        standardize_features = parameters.get("standardize_features")
        if not isinstance(standardize_features, bool):
            message = "PCA standardize_features must be a boolean"
            logger.error(message)
            raise TypeError(message)

        # 1. configure PCA and optional train-fitted standardization (documentation excerpt; keep synchronized)
        self.n_components = n_components
        self.standardize_features = standardize_features
        self.logger = logger
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler() if standardize_features else None
        self.feature_width: int | None = None

    def fit(self, features: Sequence[Sequence[float]]) -> None:
        """Fit PCA on training feature rows with fewer components than features."""
        feature_width = len(features[0])
        if self.n_components >= feature_width:
            message = "PCA components must be fewer than the feature width"
            self.logger.error(message)
            raise ValueError(message)
        self.logger.info(
            f"Fitting PCA with {self.n_components} components on {len(features)} rows"
        )

        # 2. fit PCA on train-only scaled features (documentation excerpt; keep synchronized)
        self.pca.fit(self.fit_scale(features))
        self.feature_width = feature_width

    def score(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        """Return Euclidean reconstruction residuals for feature rows."""
        if self.feature_width is None:
            message = "PCA must be fitted before scoring"
            self.logger.error(message)
            raise ValueError(message)
        if any(len(feature_row) != self.feature_width for feature_row in features):
            message = "PCA scoring features have an incompatible width"
            self.logger.error(message)
            raise ValueError(message)

        # 3. scale, reconstruct, and score each feature row (documentation excerpt; keep synchronized)
        scaled = self.scale(features)
        reconstructed = self.pca.inverse_transform(self.pca.transform(scaled))
        scores = tuple(
            math.dist(feature_row, reconstructed_row)
            for feature_row, reconstructed_row in zip(
                scaled, reconstructed, strict=True
            )
        )
        self.logger.info(f"Scored {len(scores)} rows with PCA")
        return scores

    def save(self, path: Path) -> None:
        """Persist fitted PCA state without its process-specific logger."""
        if self.feature_width is None:
            message = "PCA must be fitted before saving"
            self.logger.error(message)
            raise ValueError(message)

        # 4. persist fitted PCA state without the process-specific logger (documentation excerpt; keep synchronized)
        with path.open("wb") as file:
            pickle.dump(
                {
                    "n_components": self.n_components,
                    "standardize_features": self.standardize_features,
                    "feature_width": self.feature_width,
                    "pca": self.pca,
                    "scaler": self.scaler,
                },
                file,
            )
        self.logger.info(f"Saved fitted PCA model: {path}")

    @classmethod
    def load(cls, path: Path, logger: ComponentLogger) -> "PCAModel":
        """Restore fitted PCA state from a trusted training artifact."""
        with path.open("rb") as file:
            payload = pickle.load(file)

        # 5. restore fitted PCA state with the current process logger (documentation excerpt; keep synchronized)
        model = cls(
            {
                "n_components": payload["n_components"],
                "standardize_features": payload["standardize_features"],
            },
            logger,
        )
        model.feature_width = payload["feature_width"]
        model.pca = payload["pca"]
        model.scaler = payload["scaler"]
        logger.info(f"Loaded fitted PCA model: {path}")
        return model

    def fit_scale(
        self, features: Sequence[Sequence[float]]
    ) -> Sequence[Sequence[float]]:
        """Fit the configured train-only feature scaling transformation."""
        if self.scaler is None:
            return features
        return self.scaler.fit_transform(features)

    def scale(self, features: Sequence[Sequence[float]]) -> Sequence[Sequence[float]]:
        """Apply the configured feature scaling transformation."""
        if self.scaler is None:
            return features
        return self.scaler.transform(features)
