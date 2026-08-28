import math
from collections.abc import Mapping, Sequence

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.thresholds.base import ThresholdSelector
from mark_ad_benchmark.thresholds.best_f1 import BestF1ThresholdSelector


class ValidationF1QuantileThresholdSelector(ThresholdSelector):
    """Transfer the validation F1-optimal prediction rate across score scales."""

    def __init__(self, logger: ComponentLogger) -> None:
        self.logger = logger
        self.tail_fraction: float | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> float:
        """Fit the F1-optimal validation tail fraction."""
        threshold = BestF1ThresholdSelector(self.logger).fit(scores, labels)
        self.tail_fraction = sum(score >= threshold for score in scores) / len(scores)
        self.logger.info(
            f"Selected validation F1 quantile: top {self.tail_fraction:.4%} of scores"
        )
        return threshold

    def threshold_for(self, scores: Sequence[float]) -> float:
        """Select the fitted upper-tail fraction from unlabeled scores."""
        if self.tail_fraction is None:
            message = "Validation F1 quantile must be fitted before use"
            self.logger.error(message)
            raise ValueError(message)
        if self.tail_fraction == 0.0:
            return math.inf
        count = math.ceil(self.tail_fraction * len(scores))
        threshold = sorted(scores, reverse=True)[count - 1]
        self.logger.info(
            f"Applied validation F1 quantile to scores: threshold={threshold}"
        )
        return threshold

    def state(self) -> dict[str, object]:
        """Persist the selected upper-tail fraction."""
        if self.tail_fraction is None:
            message = "Validation F1 quantile must be fitted before saving"
            self.logger.error(message)
            raise ValueError(message)
        return {"tail_fraction": self.tail_fraction}

    def load_state(self, state: Mapping[str, object]) -> None:
        """Restore the selected upper-tail fraction."""
        tail_fraction = state.get("tail_fraction")
        if (
            not isinstance(tail_fraction, (int, float))
            or isinstance(tail_fraction, bool)
            or not 0.0 <= tail_fraction <= 1.0
        ):
            message = (
                "Validation F1 quantile state must contain a fraction from zero to one"
            )
            self.logger.error(message)
            raise ValueError(message)
        self.tail_fraction = float(tail_fraction)
