from collections.abc import Sequence
from pathlib import Path

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models import AnomalyModel


class ExampleModel(AnomalyModel):
    def fit(self, features: Sequence[Sequence[float]]) -> None:
        self.offset = features[0][0]

    def score(self, features: Sequence[Sequence[float]]) -> Sequence[float]:
        return [row[0] - self.offset for row in features]

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, path: Path, logger: ComponentLogger) -> "ExampleModel":
        return cls()


def test_anomaly_model_implements_persistence_contract() -> None:
    model = ExampleModel()
    model.fit([[2.0], [3.0]])

    assert model.score([[2.0], [5.0]]) == [0.0, 3.0]
