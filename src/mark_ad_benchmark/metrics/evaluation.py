import sys
from collections.abc import Sequence
from dataclasses import dataclass

from tqdm import tqdm

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
    with tqdm(
        total=1,
        desc="Point metrics",
        unit="metric",
        leave=False,
        disable=not sys.stderr.isatty(),
    ) as progress:
        point = calculate_point_metrics(labels, scores, threshold)
        progress.update()
    with tqdm(
        total=1,
        desc="Ranking metrics",
        unit="metric",
        leave=False,
        disable=not sys.stderr.isatty(),
    ) as progress:
        ranking = ranking_metrics(labels, scores)
        progress.update()
    with tqdm(
        total=1,
        desc="Event metrics",
        unit="metric",
        leave=False,
        disable=not sys.stderr.isatty(),
    ) as progress:
        event = event_metrics(
            labels, predictions, settings.range_recall_existence_weight
        )
        progress.update()
    with tqdm(
        total=1,
        desc="VUS ranking",
        unit="metric",
        leave=False,
        disable=not sys.stderr.isatty(),
    ) as progress:
        vus_ranking = vus_ranking_metrics(
            labels, scores, settings.vus_max_window, settings.vus_threshold_samples
        )
        progress.update()
    with tqdm(
        total=settings.vus_max_window + 1,
        desc="VUS prediction",
        unit="window",
        leave=False,
        disable=not sys.stderr.isatty(),
    ) as progress:
        vus_prediction = vus_prediction_metrics(
            labels, predictions, settings.vus_max_window, progress.update
        )
    metrics = EvaluationMetrics(
        point=point,
        ranking=ranking,
        event=event,
        vus_ranking=vus_ranking,
        vus_prediction=vus_prediction,
    )
    logger.info(f"Evaluated ranking, event, and VUS metrics at threshold {threshold}")
    return metrics
