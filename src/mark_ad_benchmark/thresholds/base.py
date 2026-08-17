from abc import ABC, abstractmethod
from collections.abc import Sequence


class ThresholdSelector(ABC):
    """Choose a binary anomaly threshold from labeled validation scores."""

    @abstractmethod
    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> float:
        """Return a threshold that the caller applies to future scores."""
