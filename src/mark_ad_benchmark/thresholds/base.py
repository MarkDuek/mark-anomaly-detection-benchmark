from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence


class ThresholdSelector(ABC):
    """Calibrate a binary anomaly threshold from labeled validation scores."""

    @abstractmethod
    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> float:
        """Fit calibration state and return the validation threshold."""

    @abstractmethod
    def threshold_for(self, scores: Sequence[float]) -> float:
        """Return the effective threshold for an unlabeled score sequence."""

    @abstractmethod
    def state(self) -> dict[str, object]:
        """Return JSON-serializable fitted calibration state."""

    @abstractmethod
    def load_state(self, state: Mapping[str, object]) -> None:
        """Restore fitted calibration state from a validated artifact."""
