from abc import ABC, abstractmethod

from mark_ad_benchmark.models import AnomalyModel


class Protocol(ABC):
    """Benchmark execution contract for a defined fitting and scoring policy."""

    @abstractmethod
    def fit_and_score_validation(self, model: AnomalyModel) -> tuple[float, ...]:
        """Fit a model and score validation data according to protocol policy."""
