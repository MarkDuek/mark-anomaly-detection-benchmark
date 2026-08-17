from collections.abc import Sequence

import pytest

from mark_ad_benchmark.datasets.loader import DatasetSplit
from mark_ad_benchmark.logging import Logger, LogOutput
from mark_ad_benchmark.models import AnomalyModel
from mark_ad_benchmark.protocols import InductiveProtocol, Protocol


class CapturingOutput(LogOutput):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


class TrackingModel(AnomalyModel):
    def __init__(self) -> None:
        self.fit_features: Sequence[Sequence[float]] | None = None
        self.scored_features: list[Sequence[Sequence[float]]] = []

    def fit(self, features: Sequence[Sequence[float]]) -> None:
        self.fit_features = features

    def score(self, features: Sequence[Sequence[float]]) -> Sequence[float]:
        self.scored_features.append(features)
        return [row[0] for row in features]


def test_inductive_protocol_fits_train_and_scores_held_out_splits() -> None:
    output = CapturingOutput()
    logger = Logger(output).for_component("mark_ad_benchmark.protocols.inductive")
    train = split(((1.0,),))
    validation = split(((2.0,),))
    test = split(((3.0,),))
    protocol = InductiveProtocol.from_splits(
        {"train": train, "validation": validation, "test": test}, logger
    )
    model = TrackingModel()

    scores = protocol.run(model)

    assert model.fit_features == train.features
    assert isinstance(protocol, Protocol)
    assert model.scored_features == [validation.features, test.features]
    assert scores == {"validation": (2.0,), "test": (3.0,)}
    assert output.messages == [
        "[INFO] Prepared inductive train, validation, and test splits",
        "[INFO] Fitting model on train split",
        "[INFO] Scoring validation split",
        "[INFO] Scoring test split",
    ]


def test_inductive_protocol_rejects_missing_split() -> None:
    with pytest.raises(ValueError, match="Missing required splits"):
        InductiveProtocol.from_splits(
            {"train": split(((1.0,),))},
            Logger().for_component("mark_ad_benchmark.protocols.inductive"),
        )


def split(features: tuple[tuple[float, ...], ...]) -> DatasetSplit:
    return DatasetSplit(
        timestamps=tuple("2025-01-01" for _ in features),
        features=features,
        labels=tuple(0 for _ in features),
    )
