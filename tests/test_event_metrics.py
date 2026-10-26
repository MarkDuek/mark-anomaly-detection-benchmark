import pytest

from mark_ad_benchmark.metrics.events import event_metrics
from mark_ad_benchmark.metrics.intervals import extract_intervals


def test_event_metrics_use_frozen_predictions() -> None:
    labels = (0, 1, 1, 0)
    predictions = (0, 1, 0, 0)

    metrics = event_metrics(labels, predictions)

    assert (metrics.precision, metrics.recall, metrics.f1) == pytest.approx(
        (1.0, 1.0, 1.0)
    )


def test_event_precision_counts_predicted_events_not_predicted_points() -> None:
    metrics = event_metrics(
        (0, 1, 1, 1, 0, 0, 0),
        (0, 1, 1, 0, 0, 1, 0),
    )

    assert (metrics.precision, metrics.recall, metrics.f1) == pytest.approx(
        (0.5, 1.0, 2 / 3)
    )


def test_extract_intervals_closes_a_trailing_event_at_the_final_index() -> None:
    assert extract_intervals((0, 0, 1, 1)) == ((2, 3),)


@pytest.mark.parametrize(
    ("labels", "predictions"),
    [((0, 0), (0, 0)), ((0, 1), (0, 0)), ((0, 0), (0, 1))],
)
def test_event_metrics_without_an_event_score_zero(labels, predictions) -> None:
    metrics = event_metrics(labels, predictions)

    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)


def test_one_prediction_can_hit_two_true_events() -> None:
    metrics = event_metrics((0, 1, 0, 1, 0), (0, 1, 1, 1, 0))

    assert (metrics.precision, metrics.recall, metrics.f1) == (1.0, 1.0, 1.0)
