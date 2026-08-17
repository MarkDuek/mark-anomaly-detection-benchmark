from abc import ABC, abstractmethod

from mark_ad_benchmark.models import AnomalyModel


class Protocol(ABC):
    """Benchmark execution contract for a defined fitting and scoring policy."""

    @abstractmethod
    def fit(self, model: AnomalyModel) -> None:
        """Fit a model according to protocol data-access policy."""

    @abstractmethod
    def score_validation(self, model: AnomalyModel) -> tuple[float, ...]:
        """Score validation data without fitting the model."""
