import pytest

from mark_ad_benchmark.metrics.events import (
    event_metrics,
    point_adjust_predictions,
    ranges,
)


def test_event_metrics_use_frozen_predictions_and_adjust_true_event_hits() -> None:
    labels = (0, 1, 1, 0)
    predictions = (0, 1, 0, 0)

    metrics = event_metrics(labels, predictions, recall_existence_weight=0.2)

    assert point_adjust_predictions(labels, predictions) == (0, 1, 1, 0)
    assert metrics.point_adjusted_f1 == pytest.approx(1.0)
    assert metrics.event_f1 == pytest.approx(1.0)
    assert metrics.range_f1 == pytest.approx(0.75)


def test_ranges_closes_a_trailing_event_at_the_final_index() -> None:
    assert ranges((0, 0, 1, 1)) == ((2, 3),)
