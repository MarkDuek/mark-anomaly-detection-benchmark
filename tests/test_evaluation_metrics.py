from dataclasses import asdict

import pytest
from sklearn.metrics import auc

from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.metrics.evaluation import MetricSettings, evaluate_metrics
from mark_ad_benchmark.metrics.point import point_metrics
from mark_ad_benchmark.metrics.ranking import ranking_curves, ranking_metrics
from mark_ad_benchmark.metrics.vus import vus_prediction_metrics, vus_ranking_metrics


def test_ranking_metrics_are_perfect_for_separated_scores() -> None:
    metrics = ranking_metrics((0, 1, 0, 1), (0.1, 0.9, 0.2, 0.8))

    assert metrics.auc_roc == pytest.approx(1.0)
    assert metrics.auc_pr == pytest.approx(1.0)


def test_ranking_metrics_are_undefined_for_one_class_labels() -> None:
    assert ranking_metrics((0, 0), (0.1, 0.2)).auc_roc is None


def test_curves_match_auc_and_stepwise_ap_at_applied_threshold() -> None:
    labels = (1, 0, 1, 0)
    scores = (0.9, 0.8, 0.3, 0.2)
    metrics = ranking_metrics(labels, scores)
    curves = ranking_curves(labels, scores, 0.5)

    assert auc(curves["roc"]["fpr"], curves["roc"]["tpr"]) == pytest.approx(
        metrics.auc_roc
    )
    recall = curves["pr"]["recall"]
    precision = curves["pr"]["precision"]
    assert sum(
        (recall[index] - recall[index + 1]) * precision[index]
        for index in range(len(recall) - 1)
    ) == pytest.approx(metrics.auc_pr)
    assert curves["operating_point"] == {
        "fpr": 0.5,
        "tpr": 0.5,
        "recall": 0.5,
        "precision": 0.5,
    }
    assert ranking_curves(labels, scores, float("inf"))["operating_point"] == {
        "fpr": 0.0,
        "tpr": 0.0,
        "recall": 0.0,
        "precision": 0.0,
    }
    assert curves["pr"]["precision"][-1] == 1.0
    assert ranking_curves((0, 0), (0.2, 0.4), 0.3) == {
        "roc": None,
        "pr": None,
        "operating_point": None,
    }


def test_evaluation_uses_the_supplied_frozen_threshold() -> None:
    metrics = evaluate_metrics(
        (0, 1),
        (0.9, 0.1),
        threshold=0.5,
        settings=MetricSettings(0, 2, 0.2),
        logger=Logger().for_component("mark_ad_benchmark.metrics.evaluation"),
    )

    assert metrics.point.f1 == 0.0
    assert metrics.event.f1 == 0.0
    assert metrics.range.f1 == 0.0
    assert set(asdict(metrics)) == {
        "point",
        "ranking",
        "event",
        "range",
        "affiliation",
        "vus_ranking",
        "vus_prediction",
    }
    assert set(asdict(metrics.event)) == {"precision", "recall", "f1"}
    assert set(asdict(metrics.range)) == {"precision", "recall", "f1"}
    assert set(asdict(metrics.point)) == {"precision", "recall", "f1"}
    assert set(asdict(metrics.affiliation)) == {"precision", "recall", "f1"}


def test_point_and_event_metrics_share_thresholded_predictions() -> None:
    labels = (0, 1, 1, 0, 0, 1, 0)
    scores = (0.1, 0.5, 0.7, 0.1, 0.1, 0.5, 0.1)
    threshold = 0.7
    metrics = evaluate_metrics(
        labels,
        scores,
        threshold,
        MetricSettings(0, 2, 0.2),
        Logger().for_component("mark_ad_benchmark.metrics.evaluation"),
    )

    assert metrics.point == point_metrics(
        labels,
        (0, 0, 1, 0, 0, 0, 0),
        Logger().for_component("mark_ad_benchmark.metrics.point"),
    )
    assert (
        metrics.point.precision,
        metrics.point.recall,
        metrics.point.f1,
    ) == pytest.approx((1.0, 1 / 3, 0.5))
    assert (
        metrics.event.precision,
        metrics.event.recall,
        metrics.event.f1,
    ) == pytest.approx((1.0, 0.5, 2 / 3))


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ((0, 1), (0.1,), "equal lengths"),
        ((0, 2), (0.1, 0.9), "binary labels"),
        ((0, 1), (0.1, float("nan")), "finite scores"),
    ],
)
def test_evaluation_validates_inputs_before_thresholding(
    labels, scores, message
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_metrics(
            labels,
            scores,
            0.5,
            MetricSettings(0, 2, 0.2),
            Logger().for_component("mark_ad_benchmark.metrics.evaluation"),
        )


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
