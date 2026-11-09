import numpy as np
import pytest

from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.optimizers import SGD, Optimizer, create_optimizer


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 2)])
def test_sgd_updates_arbitrary_arrays_without_mutating_inputs(shape: tuple) -> None:
    parameters = np.arange(np.prod(shape), dtype=float).reshape(shape)
    gradient = np.full(shape, 2.0)
    original = parameters.copy()
    optimizer = create_optimizer(
        {"name": "sgd", "learning_rate": 0.25},
        Logger().for_component("mark_ad_benchmark.optimizers"),
    )

    updated = optimizer.step(parameters, gradient)

    assert isinstance(optimizer, Optimizer)
    np.testing.assert_array_equal(updated, original - 0.5)
    np.testing.assert_array_equal(parameters, original)
    np.testing.assert_array_equal(gradient, np.full(shape, 2.0))
    assert not np.shares_memory(parameters, updated)
    np.testing.assert_array_equal(optimizer.step(parameters, gradient), updated)


@pytest.mark.parametrize(
    "learning_rate", [0, -1, True, float("nan"), float("inf"), None]
)
def test_sgd_rejects_invalid_learning_rates(learning_rate: object) -> None:
    with pytest.raises(ValueError, match="learning_rate must be finite and positive"):
        SGD(learning_rate, Logger().for_component("mark_ad_benchmark.optimizers.sgd"))


@pytest.mark.parametrize(
    ("parameters", "gradient", "message"),
    [
        (np.zeros((2, 1)), np.zeros((2,)), "equal shape"),
        (np.array([]), np.array([]), "nonempty"),
        (np.array([np.nan]), np.zeros(1), "requires finite"),
        (np.zeros(1), np.array([np.inf]), "requires finite"),
        (np.array([1e308]), np.array([-1e308]), "produced nonfinite"),
    ],
)
def test_sgd_rejects_invalid_updates(parameters, gradient, message) -> None:
    optimizer = SGD(2.0, Logger().for_component("mark_ad_benchmark.optimizers.sgd"))
    with pytest.raises(ValueError, match=message):
        optimizer.step(parameters, gradient)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (None, "must be a mapping"),
        ({"name": "adam"}, "Unsupported optimizer"),
        ({"name": "sgd"}, "learning_rate"),
        ({"name": "sgd", "learning_rate": 0.1, "momentum": 0.9}, "supports only"),
    ],
)
def test_optimizer_selection_rejects_unsupported_settings(settings, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        create_optimizer(
            settings, Logger().for_component("mark_ad_benchmark.optimizers")
        )
