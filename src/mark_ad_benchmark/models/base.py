from abc import ABC, abstractmethod
from collections.abc import Sequence


class AnomalyModel(ABC):
    @abstractmethod
    def fit(self, features: Sequence[Sequence[float]]) -> None:
        pass

    @abstractmethod
    def score(self, features: Sequence[Sequence[float]]) -> Sequence[float]:
        pass
