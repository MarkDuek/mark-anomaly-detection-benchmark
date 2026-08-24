import pytest

from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.metrics.evaluation import MetricSettings, evaluate_metrics
from mark_ad_benchmark.metrics.ranking import ranking_metrics
from mark_ad_benchmark.metrics.vus import vus_prediction_metrics, vus_ranking_metrics


def test_ranking_metrics_are_perfect_for_separated_scores() -> None:
    metrics = ranking_metrics((0, 1, 0, 1), (0.1, 0.9, 0.2, 0.8))

    assert metrics.auc_roc == pytest.approx(1.0)
    assert metrics.auc_pr == pytest.approx(1.0)


def test_ranking_metrics_are_undefined_for_one_class_labels() -> None:
    assert ranking_metrics((0, 0), (0.1, 0.2)).auc_roc is None


def test_evaluation_uses_the_supplied_frozen_threshold() -> None:
    metrics = evaluate_metrics(
        (0, 1),
        (0.9, 0.1),
        threshold=0.5,
        settings=MetricSettings(0, 2, 0.2),
        logger=Logger().for_component("mark_ad_benchmark.metrics.evaluation"),
    )

    assert metrics.point.f1 == 0.0
    assert metrics.event.event_f1 == 0.0


def test_vus_at_zero_window_matches_perfect_point_predictions() -> None:
    labels = (0, 1, 1, 0)
    scores = (0.1, 0.9, 0.8, 0.2)
    predictions = (0, 1, 1, 0)

    ranking = vus_ranking_metrics(labels, scores, max_window=0, samples=4)
    prediction = vus_prediction_metrics(labels, predictions, max_window=0)

    assert ranking.auc_roc == pytest.approx(1.0)
    assert ranking.auc_pr == pytest.approx(1.0)
    assert prediction.precision == pytest.approx(1.0)
    assert prediction.recall == pytest.approx(1.0)
    assert prediction.f1 == pytest.approx(1.0)
