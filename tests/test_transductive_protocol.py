from collections.abc import Mapping, Sequence

import pytest

from mark_ad_benchmark.datasets.loader import DatasetSplit
from mark_ad_benchmark.logging import Logger, LogOutput
from mark_ad_benchmark.models import TransductiveAnomalyModel
from mark_ad_benchmark.protocols import Protocol, TransductiveProtocol


class CapturingOutput(LogOutput):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


class TrackingModel(TransductiveAnomalyModel):
    def __init__(self) -> None:
        self.batches: list[Sequence[Sequence[float]]] = []

    def validate_parameters(self) -> None:
        pass

    @property
    def diagnostics(self) -> Mapping[str, object]:
        return {}

    def fit_score(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        self.batches.append(features)
        return tuple(row[0] for row in features)


def test_transductive_protocol_fits_and_scores_each_split_independently() -> None:
    output = CapturingOutput()
    logger = Logger(output).for_component("mark_ad_benchmark.protocols.transductive")
    validation = split(((2.0,),))
    test = split(((3.0,),))
    protocol = TransductiveProtocol.from_splits({"validation": validation}, logger)
    validation_model = TrackingModel()
    test_model = TrackingModel()

    validation_scores = protocol.fit_score_validation(validation_model)
    test_scores = TransductiveProtocol.fit_score_test(test_model, test, logger)

    assert isinstance(protocol, Protocol)
    assert validation_model.batches == [validation.features]
    assert test_model.batches == [test.features]
    assert validation_scores == (2.0,)
    assert test_scores == (3.0,)
    assert output.messages == [
        "[INFO] Prepared transductive validation split",
        "[INFO] Fitting and scoring validation split",
        "[INFO] Fitting and scoring test split",
    ]


def test_transductive_protocol_rejects_missing_validation_split() -> None:
    with pytest.raises(ValueError, match="Missing required splits"):
        TransductiveProtocol.from_splits(
            {}, Logger().for_component("mark_ad_benchmark.protocols.transductive")
        )


def split(features: tuple[tuple[float, ...], ...]) -> DatasetSplit:
    return DatasetSplit(
        timestamps=tuple("2025-01-01" for _ in features),
        features=features,
        labels=tuple(0 for _ in features),
    )
