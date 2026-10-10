from pathlib import Path

import numpy as np
import pytest

from mark_ad_benchmark.experiment import Experiment
from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.models import ConvolutionalPCAModel, PCAModel
from mark_ad_benchmark.models.convolutional_pca import (
    _lag_correlations,
    _objective_gradient,
)


def model(**parameters: object) -> ConvolutionalPCAModel:
    return ConvolutionalPCAModel(
        {
            "n_components": 1,
            "filter_order": 2,
            "standardize_features": False,
            **parameters,
        },
        Logger().for_component("mark_ad_benchmark.models.convolutional_pca"),
    )


def reference_reconstruction(filters: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    order, width, components = len(filters) - 1, filters.shape[1], filters.shape[2]
    encoded = np.zeros((len(matrix) + order, components))
    for component in range(components):
        for feature in range(width):
            encoded[:, component] += np.convolve(
                matrix[:, feature], filters[::-1, feature, component]
            )
    reconstructed = np.zeros((len(matrix) + 2 * order, width))
    for feature in range(width):
        for component in range(components):
            reconstructed[:, feature] += np.convolve(
                encoded[:, component], filters[:, feature, component]
            )
    return reconstructed


def reference_cost(filters: np.ndarray, matrix: np.ndarray) -> float:
    order = len(filters) - 1
    padded = np.pad(matrix, ((order, order), (0, 0)))
    residual = padded - reference_reconstruction(filters, matrix)
    return float(np.sum(residual**2) / len(matrix))


def reference_gradient(filters: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    gradient = np.zeros_like(filters)
    delta = 1e-6
    for index in np.ndindex(filters.shape):
        plus, minus = filters.copy(), filters.copy()
        plus[index] += delta
        minus[index] -= delta
        gradient[index] = (
            reference_cost(plus, matrix) - reference_cost(minus, matrix)
        ) / (2 * delta)
    return gradient


@pytest.mark.parametrize("order", [0, 1, 2])
def test_matlab_objective_and_truncated_gradient_match_time_domain(order: int) -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(20, 3))
    filters = rng.normal(scale=0.2, size=(order + 1, 3, 2))
    correlations = _lag_correlations(matrix, order)

    error, gradient, _ = _objective_gradient(correlations, filters)

    assert error == pytest.approx(reference_cost(filters, matrix))
    # ConvPCA.m uses half the real-valued MSE derivative.
    np.testing.assert_allclose(
        2 * gradient, reference_gradient(filters, matrix), atol=1e-8
    )


def test_matlab_momentum_and_power_normalized_updates() -> None:
    matrix = np.random.default_rng(3).normal(size=(20, 2))
    centered = matrix - matrix.mean(axis=0)
    expected = np.zeros((3, 2, 1))
    expected[1, :, 0] = np.sqrt(0.5)
    momentum = np.zeros_like(expected)
    scale_squared = 2 / (np.sum(centered**2) / len(centered))
    for _ in range(2):
        gradient = reference_gradient(expected, centered) / 2
        momentum = 0.9 * momentum + 0.1 * gradient
        expected -= 0.2 * scale_squared * momentum

    convolutional = model(max_iterations=2)
    convolutional.fit(matrix)

    np.testing.assert_allclose(convolutional.filters, expected, atol=1e-9)
    assert convolutional.diagnostics["iterations"] == 2
    assert convolutional.diagnostics["fitting_mse"] == pytest.approx(
        reference_cost(expected, centered)
    )


@pytest.mark.parametrize("standardize_features", [False, True])
def test_zero_order_reduces_to_pca(standardize_features: bool) -> None:
    train = ((1.0, 2.0), (2.0, 1.0), (3.0, 4.0), (4.0, 3.0), (5.0, 7.0))
    query = ((2.0, 3.0), (10.0, -3.0))
    convolutional = model(
        filter_order=0, max_iterations=1000, standardize_features=standardize_features
    )
    pca = PCAModel(
        {"n_components": 1, "standardize_features": standardize_features},
        Logger().for_component("mark_ad_benchmark.models.pca"),
    )
    convolutional.fit(train)
    pca.fit(train)

    assert convolutional.score(query) == pytest.approx(pca.score(query), rel=1e-5)


def test_scores_align_zero_padded_filtering_without_refitting() -> None:
    train = np.random.default_rng(7).normal(size=(30, 3))
    query = np.random.default_rng(8).normal(size=(7, 3)) + 10
    convolutional = model(n_components=2, standardize_features=True)
    convolutional.fit(train)
    mean = convolutional.scaler.mean_.copy()
    filters = convolutional.filters.copy()
    scaled = convolutional.scaler.transform(query)
    reconstructed = reference_reconstruction(filters, scaled)[2:9]

    scores = convolutional.score(query)

    assert isinstance(scores, tuple)
    assert len(scores) == len(query)
    assert scores == pytest.approx(np.linalg.norm(scaled - reconstructed, axis=1))
    np.testing.assert_array_equal(convolutional.scaler.mean_, mean)
    np.testing.assert_array_equal(convolutional.filters, filters)
    component_covariance = _objective_gradient(
        _lag_correlations(convolutional.scaler.transform(train), 2), filters
    )[2]
    np.testing.assert_allclose(
        component_covariance, np.diag(np.diag(component_covariance)), atol=1e-12
    )


def test_temporal_filters_improve_on_static_pca_for_delayed_signals() -> None:
    rng = np.random.default_rng(12)
    source = rng.normal(size=502)
    for index in range(1, len(source)):
        source[index] += 0.8 * source[index - 1]
    features = np.column_stack((source[1:], source[:-1]))
    features += rng.normal(scale=0.05, size=features.shape)
    convolutional = model(filter_order=4, max_iterations=500)
    convolutional.fit(features)
    pca = PCAModel(
        {"n_components": 1, "standardize_features": False},
        Logger().for_component("mark_ad_benchmark.models.pca"),
    )
    pca.fit(features)

    convolutional_mse = np.mean(np.square(convolutional.score(features)[4:-4]))
    pca_mse = np.mean(np.square(pca.score(features)[4:-4]))

    assert convolutional_mse < 0.5 * pca_mse


def test_save_and_load_preserves_scores_and_diagnostics(tmp_path: Path) -> None:
    features = np.random.default_rng(4).normal(size=(20, 2))
    convolutional = model(standardize_features=True)
    convolutional.fit(features)
    path = tmp_path / "model.pkl"
    convolutional.save(path)
    logger = Logger().for_component("mark_ad_benchmark.models.convolutional_pca")

    loaded = ConvolutionalPCAModel.load(path, logger)

    assert loaded.score(features) == pytest.approx(convolutional.score(features))
    assert loaded.diagnostics == convolutional.diagnostics
    assert loaded.logger is logger


def test_checked_in_experiment_resolves_filter_order_candidates() -> None:
    experiment = Experiment.from_path(
        Path(
            "experiments/creditcard-convolutional-pca-inductive-validation-f1-quantile.yaml"
        ),
        Logger().for_component("mark_ad_benchmark.experiment"),
    )
    models = tuple(
        experiment.create_model(parameters)
        for parameters in experiment.model_parameter_candidates()
    )

    assert tuple(candidate.filter_order for candidate in models) == (0, 1, 3, 10)
    assert all(
        candidate.n_components == 13
        and candidate.standardize_features is False
        and candidate.max_iterations == 200
        and candidate.step_size == 0.2
        for candidate in models
    )


def test_constant_training_data_has_zero_reconstruction_error() -> None:
    convolutional = model()
    features = ((10.0, 20.0),) * 8
    convolutional.fit(features)

    assert convolutional.score(features) == (0.0,) * 8
    assert convolutional.diagnostics["converged"] is True
    assert convolutional.diagnostics["iterations"] == 0


def test_rejects_invalid_data_and_incompatible_dimensions(tmp_path: Path) -> None:
    convolutional = model()
    with pytest.raises(ValueError, match="fitted before scoring"):
        convolutional.score(((1.0, 2.0),))
    with pytest.raises(ValueError, match="fitted before saving"):
        convolutional.save(tmp_path / "model.pkl")
    with pytest.raises(ValueError, match="finite feature values"):
        convolutional.fit(((float("nan"), 2.0),) * 8)
    with pytest.raises(ValueError, match="twice the filter order"):
        convolutional.fit(((1.0, 2.0),) * 4)
    with pytest.raises(ValueError, match="fewer than the feature width"):
        model(n_components=2).fit(((1.0, 2.0),) * 8)
    convolutional.fit(np.random.default_rng(2).normal(size=(20, 2)))
    with pytest.raises(ValueError, match="incompatible width"):
        convolutional.score(((1.0,),))


@pytest.mark.parametrize(
    "parameters",
    [
        {"filter_order": -1},
        {"n_components": 0},
        {"max_iterations": 0},
        {"step_size": float("inf")},
        {"standardize_features": "true"},
    ],
)
def test_rejects_invalid_solver_parameters(parameters: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Convolutional PCA"):
        model(**parameters)
