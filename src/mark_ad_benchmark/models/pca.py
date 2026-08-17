import math
import pickle
from collections.abc import Sequence
from pathlib import Path

from sklearn.decomposition import PCA

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models.base import AnomalyModel


class PCAModel(AnomalyModel):
    """Mean-centered PCA with Euclidean reconstruction-residual scores."""

    def __init__(self, n_components: int, logger: ComponentLogger) -> None:
        if n_components < 1:
            message = "PCA requires at least one component"
            logger.error(message)
            raise ValueError(message)
        self.n_components = n_components
        self.logger = logger
        self.pca = PCA(n_components=n_components)
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
        self.pca.fit(features)
        self.feature_width = feature_width

    def score(self, features: Sequence[Sequence[float]]) -> Sequence[float]:
        """Return Euclidean reconstruction residuals for feature rows."""
        if self.feature_width is None:
            message = "PCA must be fitted before scoring"
            self.logger.error(message)
            raise ValueError(message)
        if any(len(feature_row) != self.feature_width for feature_row in features):
            message = "PCA scoring features have an incompatible width"
            self.logger.error(message)
            raise ValueError(message)
        reconstructed = self.pca.inverse_transform(self.pca.transform(features))
        scores = tuple(
            math.dist(feature_row, reconstructed_row)
            for feature_row, reconstructed_row in zip(
                features, reconstructed, strict=True
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
        with path.open("wb") as file:
            pickle.dump(
                {
                    "n_components": self.n_components,
                    "feature_width": self.feature_width,
                    "pca": self.pca,
                },
                file,
            )
        self.logger.info(f"Saved fitted PCA model: {path}")

    @classmethod
    def load(cls, path: Path, logger: ComponentLogger) -> "PCAModel":
        """Restore fitted PCA state from a trusted training artifact."""
        with path.open("rb") as file:
            payload = pickle.load(file)
        model = cls(payload["n_components"], logger)
        model.feature_width = payload["feature_width"]
        model.pca = payload["pca"]
        logger.info(f"Loaded fitted PCA model: {path}")
        return model
