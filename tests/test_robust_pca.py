import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.models import RobustPCAModel


def test_robust_pca_recovers_sparse_component_and_scores_its_row_l1_norm() -> None:
    low_rank = np.array([[value, 2 * value, -value] for value in range(1, 13)])
    sparse = np.zeros_like(low_rank)
    sparse[2, 1] = 12.0
    sparse[8, 2] = -10.0
    features = tuple(tuple(row) for row in low_rank + sparse)
    model = RobustPCAModel(
        lambda_value=1 / np.sqrt(12),
        tolerance=1e-7,
        max_iterations=1000,
        logger=Logger().for_component("mark_ad_benchmark.models.robust_pca"),
    )

    scores = model.fit_score(features)

    recovered_low_rank = np.array(model.low_rank_component)
    recovered_sparse = np.array(model.sparse_component)
    standardized = StandardScaler().fit_transform(features)
    assert np.allclose(recovered_low_rank + recovered_sparse, standardized, atol=1e-5)
    assert scores == pytest.approx(tuple(np.abs(recovered_sparse).sum(axis=1)))
    assert model.diagnostics["residual"] <= model.diagnostics["convergence_threshold"]


def test_robust_pca_requires_explicit_solver_parameters() -> None:
    with pytest.raises(TypeError, match="lambda_value"):
        RobustPCAModel(
            tolerance=1e-7,
            max_iterations=1000,
            logger=Logger().for_component("mark_ad_benchmark.models.robust_pca"),
        )
