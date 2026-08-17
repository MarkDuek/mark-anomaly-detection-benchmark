from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from mark_ad_benchmark.logging import ComponentLogger


class AnomalyModel(ABC):
    """Common fit-and-score contract for anomaly models."""

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
    def load(cls, path: Path, logger: ComponentLogger) -> "AnomalyModel":
        """Restore a fitted model using the current process logger."""
