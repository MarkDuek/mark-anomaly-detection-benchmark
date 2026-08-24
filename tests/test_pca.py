import math
from pathlib import Path

import pytest

from mark_ad_benchmark.datasets.loader import DatasetSplit
from mark_ad_benchmark.logging import Logger, LogOutput
from mark_ad_benchmark.models import PCAModel
from mark_ad_benchmark.protocols import InductiveProtocol


class CapturingOutput(LogOutput):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def test_pca_scores_reconstruction_residuals_through_inductive_protocol() -> None:
    output = CapturingOutput()
    logger = Logger(output)
    protocol = InductiveProtocol.from_splits(
        {
            "train": split(((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))),
            "validation": split(((3.0, 3.0), (1.0, 3.0))),
        },
        logger.for_component("mark_ad_benchmark.protocols.inductive"),
    )
    model = PCAModel(
        {"n_components": 1, "feature_scaling": "standard"},
        logger.for_component("mark_ad_benchmark.models.pca"),
    )

    protocol.fit(model)
    validation_scores = protocol.score_validation(model)
    test_scores = InductiveProtocol.score_test(
        model, split(((2.0, 2.0), (0.0, 2.0))), protocol.logger
    )

    assert validation_scores == pytest.approx((0.0, math.sqrt(3)))
    assert test_scores == pytest.approx((0.0, math.sqrt(3)))
    assert output.messages == [
        "[INFO] Prepared inductive train and validation splits",
        "[INFO] Fitting model on train split",
        "[INFO] Fitting PCA with 1 components on 3 rows",
        "[INFO] Scoring validation split",
        "[INFO] Scored 2 rows with PCA",
        "[INFO] Scoring test split",
        "[INFO] Scored 2 rows with PCA",
    ]


def test_pca_rejects_full_feature_reconstruction() -> None:
    model = PCAModel(
        {"n_components": 2, "feature_scaling": "standard"},
        Logger().for_component("mark_ad_benchmark.models.pca"),
    )

    with pytest.raises(ValueError, match="components must be fewer"):
        model.fit(((0.0, 0.0), (1.0, 1.0)))


def test_pca_uses_configured_feature_scaling() -> None:
    logger = Logger().for_component("mark_ad_benchmark.models.pca")
    model = PCAModel({"n_components": 1, "feature_scaling": "none"}, logger)

    model.fit(((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)))

    assert model.score(((1.0, 3.0),)) == pytest.approx((math.sqrt(2),))


def test_pca_save_and_load_preserves_scores(tmp_path: Path) -> None:
    logger = Logger().for_component("mark_ad_benchmark.models.pca")
    model = PCAModel({"n_components": 1, "feature_scaling": "standard"}, logger)
    model.fit(((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)))
    path = tmp_path / "model.pkl"
    model.save(path)

    loaded = PCAModel.load(path, logger)

    assert loaded.score(((1.0, 3.0),)) == pytest.approx(model.score(((1.0, 3.0),)))


def split(features: tuple[tuple[float, ...], ...]) -> DatasetSplit:
    return DatasetSplit(
        timestamps=tuple("2025-01-01" for _ in features),
        features=features,
        labels=tuple(0 for _ in features),
    )
