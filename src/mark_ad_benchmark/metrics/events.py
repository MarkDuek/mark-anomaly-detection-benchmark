from collections.abc import Sequence
from dataclasses import dataclass

Event = tuple[int, int]


@dataclass(frozen=True)
class EventMetrics:
    precision: float
    recall: float
    f1: float


def extract_events(values: Sequence[int]) -> tuple[Event, ...]:
    """Return contiguous positive events with inclusive endpoints."""

    # 1. extract maximal contiguous events (documentation excerpt; keep synchronized)
    result = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        elif not value and start is not None:
            result.append((start, index - 1))
            start = None
    if start is not None:
        result.append((start, len(values) - 1))
    return tuple(result)


def _events_overlap(first: Event, second: Event) -> bool:
    # 2. check whether two events share a timestamp (documentation excerpt; keep synchronized)
    return first[0] <= second[1] and second[0] <= first[1]


def event_precision(
    true_events: Sequence[Event], predicted_events: Sequence[Event]
) -> float:
    """Return the fraction of predicted events overlapping a true event."""
    # 3. average hits across predicted events (documentation excerpt; keep synchronized)
    return (
        sum(
            any(_events_overlap(predicted, true) for true in true_events)
            for predicted in predicted_events
        )
        / len(predicted_events)
        if predicted_events
        else 0.0
    )


def event_recall(
    true_events: Sequence[Event], predicted_events: Sequence[Event]
) -> float:
    """Return the fraction of true events overlapping a predicted event."""
    # 4. average hits across true events (documentation excerpt; keep synchronized)
    return (
        sum(
            any(_events_overlap(true, predicted) for predicted in predicted_events)
            for true in true_events
        )
        / len(true_events)
        if true_events
        else 0.0
    )


def event_metrics(labels: Sequence[int], predictions: Sequence[int]) -> EventMetrics:
    """Calculate event precision, recall, and F1 from binary vectors."""
    # 5. assemble study-defined event metrics (documentation excerpt; keep synchronized)

    true_events = extract_events(labels)
    predicted_events = extract_events(predictions)

    precision = event_precision(true_events, predicted_events)
    recall = event_recall(true_events, predicted_events)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return EventMetrics(precision, recall, f1)
