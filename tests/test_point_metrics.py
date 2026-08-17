import pytest

from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.metrics import point_metrics
from mark_ad_benchmark.thresholds import BestF1ThresholdSelector


def test_validation_f1_threshold_applies_to_point_metrics() -> None:
    labels = (0, 1, 1, 0)
    scores = (0.1, 0.8, 0.6, 0.4)
    logger = Logger().for_component("mark_ad_benchmark.metrics.point")

    threshold = BestF1ThresholdSelector(logger).fit(scores, labels)
    metrics = point_metrics(labels, scores, threshold, logger)

    assert threshold == 0.6
    assert (metrics.precision, metrics.recall, metrics.f1) == pytest.approx(
        (1.0, 1.0, 1.0)
    )
