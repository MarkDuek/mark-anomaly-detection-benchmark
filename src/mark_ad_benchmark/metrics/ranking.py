from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.metrics import average_precision_score, roc_auc_score


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
