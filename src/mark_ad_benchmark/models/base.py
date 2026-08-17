from abc import ABC, abstractmethod
from collections.abc import Sequence


class AnomalyModel(ABC):
    """Common fit-and-score contract for anomaly models."""

    @abstractmethod
    def fit(self, features: Sequence[Sequence[float]]) -> None:
        """Learn normal structure from protocol-provided feature rows."""

    @abstractmethod
    def score(self, features: Sequence[Sequence[float]]) -> Sequence[float]:
        """Return one higher-is-more-anomalous score per feature row."""
