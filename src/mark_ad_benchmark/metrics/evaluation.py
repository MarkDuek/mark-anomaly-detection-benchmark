from collections.abc import Sequence
from dataclasses import dataclass

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.metrics.events import EventMetrics, event_metrics
from mark_ad_benchmark.metrics.point import PointMetrics, calculate_point_metrics
from mark_ad_benchmark.metrics.ranking import RankingMetrics, ranking_metrics
from mark_ad_benchmark.metrics.vus import (
    VusPredictionMetrics,
    VusRankingMetrics,
    vus_prediction_metrics,
    vus_ranking_metrics,
)


@dataclass(frozen=True)
class MetricSettings:
    vus_max_window: int
    vus_threshold_samples: int
    range_recall_existence_weight: float


@dataclass(frozen=True)
class EvaluationMetrics:
    point: PointMetrics
    ranking: RankingMetrics
    event: EventMetrics
    vus_ranking: VusRankingMetrics
    vus_prediction: VusPredictionMetrics

    @property
    def precision(self) -> float:
        """Expose point precision for existing result consumers."""
        return self.point.precision

    @property
    def recall(self) -> float:
        """Expose point recall for existing result consumers."""
        return self.point.recall

    @property
    def f1(self) -> float:
        """Expose point F1 for existing selection behavior."""
        return self.point.f1


def evaluate_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    settings: MetricSettings,
    logger: ComponentLogger,
) -> EvaluationMetrics:
    """Evaluate ranking and frozen-prediction metric families together."""
    predictions = tuple(int(score >= threshold) for score in scores)
    metrics = EvaluationMetrics(
        point=calculate_point_metrics(labels, scores, threshold),
        ranking=ranking_metrics(labels, scores),
        event=event_metrics(
            labels, predictions, settings.range_recall_existence_weight
        ),
        vus_ranking=vus_ranking_metrics(
            labels, scores, settings.vus_max_window, settings.vus_threshold_samples
        ),
        vus_prediction=vus_prediction_metrics(
            labels, predictions, settings.vus_max_window
        ),
    )
    logger.info(f"Evaluated ranking, event, and VUS metrics at threshold {threshold}")
    return metrics
