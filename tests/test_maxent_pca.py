import pickle
from pathlib import Path

import numpy as np
import pytest

from mark_ad_benchmark.datasets.loader import DatasetSplit
from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.models import MaxEntPCAModel
from mark_ad_benchmark.models.maxent_pca import _pair_gradient
from mark_ad_benchmark.models.pca import PCAModel
from mark_ad_benchmark.optimizers import SGD
from mark_ad_benchmark.protocols import InductiveProtocol


def model(**overrides) -> MaxEntPCAModel:
    return MaxEntPCAModel(
        {
            "n_components": 1,
            "standardize_features": False,
            "max_iterations": 5,
            "pair_batch_size": 16,
            **overrides,
        },
        Logger().for_component("mark_ad_benchmark.models.maxent_pca"),
    )


def test_bandwidth_identity_matches_all_ordered_pairs() -> None:
    projected = np.array([[1.0, 2.0], [-3.0, 4.0], [2.0, -1.0]])
    differences = projected[:, None, :] - projected[None, :, :]
    expected = np.sum(differences**2) / (3.0 * len(projected) ** 2)

    fitted = model(bandwidth_scale=3.0)
    assert fitted._bandwidth(projected) == pytest.approx(expected)
    assert fitted._bandwidth(projected + 100.0) == pytest.approx(expected)


def test_pair_gradient_matches_finite_differences_and_paper_laplacian() -> None:
    matrix = np.array([[-2.0, 0.5, 1.0], [1.0, 2.0, -1.0], [3.0, -0.2, 0.4]])
    components, _ = np.linalg.qr(np.array([[1.0, 0.2], [-0.5, 1.0], [0.3, -0.2]]))
    bandwidth_squared = 1.7
    differences = matrix[:, None, :] - matrix[None, :, :]
    gradient = _pair_gradient(
        differences.reshape(-1, matrix.shape[1]), components, bandwidth_squared
    )

    def potential(basis):
        projected = matrix @ basis
        distances = np.sum((projected[:, None, :] - projected[None, :, :]) ** 2, axis=2)
        return np.exp(-distances / (2.0 * bandwidth_squared)).mean()

    numerical = np.empty_like(components)
    for index in np.ndindex(components.shape):
        perturbation = np.zeros_like(components)
        perturbation[index] = 1e-6
        numerical[index] = (
            potential(components + perturbation) - potential(components - perturbation)
        ) / 2e-6
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-9)

    kernel = np.exp(
        -np.sum((differences @ components) ** 2, axis=2) / (2.0 * bandwidth_squared)
    )
    weights = kernel / (bandwidth_squared * kernel.sum())
    laplacian = np.diag(weights.sum(axis=1)) - weights
    entropy_gradient = 2.0 * matrix.T @ laplacian @ matrix @ components
    np.testing.assert_allclose(gradient, -kernel.mean() * entropy_gradient, atol=1e-14)

    optimizer = SGD(0.01, Logger().for_component("mark_ad_benchmark.optimizers.sgd"))
    candidate = optimizer.step(components, gradient)
    beta = 2.0 * optimizer.learning_rate * kernel.mean()
    paper_candidate = components + beta * matrix.T @ laplacian @ matrix @ components
    np.testing.assert_allclose(candidate, paper_candidate, atol=1e-14)
    left, _, right = np.linalg.svd(candidate, full_matrices=False)
    np.testing.assert_allclose((left @ right).T @ (left @ right), np.eye(2), atol=1e-14)


@pytest.mark.parametrize("standardize", [False, True])
def test_protocol_scoring_matches_baseline_convention_and_preserves_state(
    standardize: bool, tmp_path: Path
) -> None:
    def split(rows):
        return DatasetSplit(
            tuple("2026-01-01" for _ in rows), rows, tuple(0 for _ in rows)
        )

    protocol = InductiveProtocol.from_splits(
        {
            "train": split(((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))),
            "validation": split(((3.0, 3.0), (1.0, 3.0))),
        },
        Logger().for_component("mark_ad_benchmark.protocols.inductive"),
    )
    fitted = model(standardize_features=standardize)
    protocol.fit(fitted)
    components = fitted.components.copy()
    center = fitted.scaler.mean_.copy()
    diagnostics = dict(fitted.diagnostics)
    scores = protocol.score_validation(fitted)
    baseline = PCAModel(
        {"n_components": 1, "standardize_features": standardize}, fitted.logger
    )
    baseline.fit(protocol.train.features)

    assert scores == pytest.approx(
        baseline.score(protocol.validation.features), abs=1e-12
    )
    np.testing.assert_array_equal(fitted.components, components)
    np.testing.assert_array_equal(fitted.scaler.mean_, center)
    assert fitted.diagnostics == diagnostics
    assert diagnostics["termination_reason"] == "iteration_budget"
    assert "converged" not in diagnostics
    assert "last_subspace_change" not in diagnostics

    path = tmp_path / "model.pkl"
    fitted.save(path)
    with path.open("rb") as file:
        assert set(pickle.load(file)) == {
            "parameters",
            "components",
            "scaler",
            "diagnostics",
        }
    restored = MaxEntPCAModel.load(path, fitted.logger)
    assert protocol.score_validation(restored) == scores
    assert restored.diagnostics == diagnostics
    assert restored.parameters == fitted.parameters


def test_sampling_is_seeded_and_every_fit_resets_sampling() -> None:
    matrix = np.random.default_rng(5).normal(size=(40, 3)) * [4.0, 2.0, 0.3]
    fitted = model()
    fitted.fit(matrix)
    components = fitted.components.copy()
    fitted.fit(matrix)
    np.testing.assert_array_equal(fitted.components, components)
    np.testing.assert_allclose(
        fitted.components.T @ fitted.components, np.eye(1), atol=1e-14
    )

    other = model(random_state=43)
    other.fit(matrix)
    assert not np.array_equal(other.components, components)


def test_fit_uses_one_sampled_pair_batch_per_iteration(monkeypatch) -> None:
    from mark_ad_benchmark.models import maxent_pca

    original = maxent_pca._pair_gradient
    batches = []

    def capture(differences, components, bandwidth_squared):
        batches.append(differences.copy())
        return original(differences, components, bandwidth_squared)

    monkeypatch.setattr(maxent_pca, "_pair_gradient", capture)
    fitted = model(pair_batch_size=7)
    matrix = np.random.default_rng(1).normal(size=(30, 3))
    fitted.fit(matrix)

    assert len(batches) == fitted.max_iterations
    assert all(batch.shape == (7, 3) for batch in batches)
    rng = np.random.default_rng(fitted.random_state)
    scaled = fitted.scaler.transform(matrix)
    for batch in batches:
        pairs = rng.integers(len(matrix), size=(7, 2))
        np.testing.assert_array_equal(batch, scaled[pairs[:, 0]] - scaled[pairs[:, 1]])
    assert not hasattr(fitted, "training_history")
    assert "monitor_interval" not in fitted.parameters
    assert fitted.scaler.n_samples_seen_ == len(matrix)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_components": True},
        {"pair_batch_size": 0},
        {"max_iterations": 0},
        {"random_state": -1},
        {"standardize_features": None},
        {"bandwidth_scale": 0},
        {"bandwidth_scale": float("nan")},
        {"optimizer": {"name": "adam"}},
    ],
)
def test_rejects_invalid_configuration(overrides) -> None:
    with pytest.raises(ValueError):
        model(**overrides)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        ([], "nonempty"),
        ([1.0, 2.0], "two-dimensional"),
        ([[1.0, 2.0], [3.0]], "rectangular"),
        ([[1.0, np.nan], [2.0, 3.0]], "finite feature"),
        ([[1.0], [2.0]], "fewer than"),
        ([[1.0, 2.0]], "rows must exceed"),
        ([[1.0, 2.0], [1.0, 2.0]], "positive projected bandwidth"),
    ],
)
def test_rejects_invalid_training_data(features, message) -> None:
    with pytest.raises(ValueError, match=message):
        model().fit(features)


def test_rejects_unfitted_operations_and_incompatible_scoring(tmp_path: Path) -> None:
    fitted = model()
    with pytest.raises(ValueError, match="fitted before scoring"):
        fitted.score([[1.0, 2.0]])
    with pytest.raises(ValueError, match="fitted before saving"):
        fitted.save(tmp_path / "unfitted.pkl")
    fitted.fit([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(ValueError, match="incompatible width"):
        fitted.score([[1.0]])
    with pytest.raises(ValueError, match="finite feature"):
        fitted.score([[np.inf, 0.0]])
