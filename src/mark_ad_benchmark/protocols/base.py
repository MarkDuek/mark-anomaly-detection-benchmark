from abc import ABC, abstractmethod

from mark_ad_benchmark.models import AnomalyModel


class Protocol(ABC):
    """Benchmark execution contract for a defined fitting and scoring policy."""

    @abstractmethod
    def run(self, model: AnomalyModel) -> dict[str, tuple[float, ...]]:
        """Fit and score a model according to this protocol's data-access rules."""
