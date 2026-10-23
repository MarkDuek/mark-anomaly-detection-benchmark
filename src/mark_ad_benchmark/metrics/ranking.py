from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class RankingMetrics:
    auc_roc: float | None
    auc_pr: float | None


def ranking_metrics(labels: Sequence[int], scores: Sequence[float]) -> RankingMetrics:
    """Calculate ranking metrics, returning None when one class is absent."""
    if len(set(labels)) != 2:
        return RankingMetrics(auc_roc=None, auc_pr=None)
    return RankingMetrics(
        auc_roc=float(roc_auc_score(labels, scores)),
        auc_pr=float(average_precision_score(labels, scores)),
    )


def ranking_curves(
    labels: Sequence[int], scores: Sequence[float], threshold: float
) -> dict[str, object]:
    """Return full ROC/PR coordinates and the applied-threshold operating point."""
    if len(set(labels)) != 2:
        return {"roc": None, "pr": None, "operating_point": None}

    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    precision, recall, _ = precision_recall_curve(labels, scores)
    positives = sum(labels)
    negatives = len(labels) - positives
    true_positives = sum(
        label == 1 and score >= threshold
        for label, score in zip(labels, scores, strict=True)
    )
    false_positives = sum(
        label == 0 and score >= threshold
        for label, score in zip(labels, scores, strict=True)
    )
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr": {"recall": recall.tolist(), "precision": precision.tolist()},
        "operating_point": {
            "fpr": false_positives / negatives,
            "tpr": true_positives / positives,
            "recall": true_positives / positives,
            "precision": (
                true_positives / (true_positives + false_positives)
                if true_positives + false_positives
                else 0.0
            ),
        },
    }
