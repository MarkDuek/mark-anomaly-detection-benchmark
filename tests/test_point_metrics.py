import math

import pytest

from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.metrics import point_metrics
from mark_ad_benchmark.metrics.ranking import ranking_curves
from mark_ad_benchmark.thresholds import (
    BestF1ThresholdSelector,
    ValidationF1QuantileThresholdSelector,
)


def test_validation_f1_threshold_applies_to_point_metrics() -> None:
    labels = (0, 1, 1, 0)
    scores = (0.1, 0.8, 0.6, 0.4)
    logger = Logger().for_component("mark_ad_benchmark.metrics.point")

    threshold = BestF1ThresholdSelector(logger).fit(scores, labels)
    predictions = tuple(int(score >= threshold) for score in scores)
    metrics = point_metrics(labels, predictions, logger)

    assert threshold == 0.6
    assert (metrics.precision, metrics.recall, metrics.f1) == pytest.approx(
        (1.0, 1.0, 1.0)
    )


def test_validation_f1_threshold_selects_no_predictions_without_positives() -> None:
    threshold = BestF1ThresholdSelector(
        Logger().for_component("mark_ad_benchmark.thresholds.best_f1")
    ).fit((0.2, 0.8), (0, 0))

    assert threshold == math.inf


def test_validation_f1_quantile_preserves_prediction_rate_across_scores() -> None:
    selector = ValidationF1QuantileThresholdSelector(
        Logger().for_component("mark_ad_benchmark.thresholds.validation_f1_quantile")
    )

    validation_threshold = selector.fit((0.1, 0.8, 0.6, 0.4), (0, 1, 1, 0))
    selector.load_state(selector.state())
    test_threshold = selector.threshold_for((0.2, 0.9, 0.7, 0.5, 0.3, 0.1))

    assert validation_threshold == 0.6
    assert test_threshold == 0.5
    assert ranking_curves((0, 1, 1, 0), (0.1, 0.8, 0.6, 0.4), validation_threshold)[
        "operating_point"
    ] == {"fpr": 0.0, "tpr": 1.0, "recall": 1.0, "precision": 1.0}
    assert ranking_curves(
        (0, 1, 0, 1, 0, 0), (0.2, 0.9, 0.7, 0.5, 0.3, 0.1), test_threshold
    )["operating_point"] == {"fpr": 0.25, "tpr": 1.0, "recall": 1.0, "precision": 2 / 3}
