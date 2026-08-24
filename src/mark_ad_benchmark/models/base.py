from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path

from mark_ad_benchmark.logging import ComponentLogger


class AnomalyModel(ABC):
    """Base capability marker for anomaly models."""


class InductiveAnomalyModel(AnomalyModel):
    """Model that learns from one split and scores later splits."""

    @abstractmethod
    def fit(self, features: Sequence[Sequence[float]]) -> None:
        """Learn normal structure from protocol-provided feature rows."""

    @abstractmethod
    def score(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        """Return immutable higher-is-more-anomalous scores per feature row."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist a fitted model for later scoring."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path, logger: ComponentLogger) -> "InductiveAnomalyModel":
        """Restore a fitted model using the current process logger."""


class TransductiveAnomalyModel(AnomalyModel):
    """Model that fits and scores one batch as a single operation."""

    @abstractmethod
    def fit_score(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        """Fit on and return immutable anomaly scores for the same feature rows."""

    @property
    @abstractmethod
    def diagnostics(self) -> Mapping[str, object]:
        """Return metadata from the most recent batch decomposition."""
