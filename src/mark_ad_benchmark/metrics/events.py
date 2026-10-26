from collections.abc import Sequence
from dataclasses import dataclass

from mark_ad_benchmark.metrics.intervals import Interval, extract_intervals

Event = Interval


@dataclass(frozen=True)
class EventMetrics:
    precision: float
    recall: float
    f1: float


def event_metrics(labels: Sequence[int], predictions: Sequence[int]) -> EventMetrics:
    """Calculate event precision, recall, and F1 from binary vectors."""

    # 1. extract label and predicted intervals (documentation excerpt; keep synchronized)
    label_intervals = extract_intervals(labels)
    predicted_intervals = extract_intervals(predictions)

    precision = event_precision(label_intervals, predicted_intervals)
    recall = event_recall(label_intervals, predicted_intervals)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return EventMetrics(precision=precision, recall=recall, f1=f1)


def _events_overlap(first: Event, second: Event) -> bool:
    # 2. check whether two events share a timestamp (documentation excerpt; keep synchronized)
    return first[0] <= second[1] and second[0] <= first[1]


def event_precision(
    label_intervals: Sequence[Event], predicted_intervals: Sequence[Event]
) -> float:
    """Return the fraction of predicted events overlapping a true event."""
    # 3. average hits across predicted events (documentation excerpt; keep synchronized)
    return (
        sum(
            any(_events_overlap(predicted, label) for label in label_intervals)
            for predicted in predicted_intervals
        )
        / len(predicted_intervals)
        if predicted_intervals
        else 0.0
    )


def event_recall(
    label_intervals: Sequence[Event], predicted_intervals: Sequence[Event]
) -> float:
    """Return the fraction of true events overlapping a predicted event."""
    # 4. average hits across true events (documentation excerpt; keep synchronized)
    return (
        sum(
            any(_events_overlap(label, predicted) for predicted in predicted_intervals)
            for label in label_intervals
        )
        / len(label_intervals)
        if label_intervals
        else 0.0
    )
