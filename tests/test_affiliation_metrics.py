import pytest

from mark_ad_benchmark.metrics.affiliation import affiliation_metrics


def test_affiliation_metrics_are_perfect_for_matching_events() -> None:
    metrics = affiliation_metrics((0, 1, 1, 0, 0, 1, 0), (0, 1, 1, 0, 0, 1, 0))

    assert (metrics.precision, metrics.recall, metrics.f1) == pytest.approx(
        (1.0, 1.0, 1.0)
    )


def test_affiliation_metrics_are_zero_without_predictions() -> None:
    metrics = affiliation_metrics((0, 1, 1, 0), (0, 0, 0, 0))

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_affiliation_metrics_reward_nearby_predictions() -> None:
    labels = (0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0)
    near = affiliation_metrics(labels, (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0))
    far = affiliation_metrics(labels, (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    assert near.precision > far.precision
    assert near.recall > far.recall
