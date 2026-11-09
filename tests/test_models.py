from collections.abc import Sequence
from pathlib import Path

import pytest

from mark_ad_benchmark.logging import ComponentLogger, Logger
from mark_ad_benchmark.models import (
    AnomalyModel,
    ConvolutionalPCAModel,
    KernelPCAModel,
    MaxEntPCAModel,
    NystromKPCAModel,
    PCAModel,
    RobustPCAModel,
)


class ExampleModel(AnomalyModel):
    def validate_parameters(self) -> None:
        pass

    def fit(self, features: Sequence[Sequence[float]]) -> None:
        self.offset = features[0][0]

    def score(self, features: Sequence[Sequence[float]]) -> tuple[float, ...]:
        return tuple(row[0] - self.offset for row in features)

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, path: Path, logger: ComponentLogger) -> "ExampleModel":
        return cls()


def test_anomaly_model_implements_persistence_contract() -> None:
    model = ExampleModel()
    model.fit([[2.0], [3.0]])

    assert model.score([[2.0], [5.0]]) == (0.0, 3.0)


def test_anomaly_model_requires_parameter_validation() -> None:
    class MissingValidation(AnomalyModel):
        pass

    with pytest.raises(TypeError, match="validate_parameters"):
        MissingValidation()


@pytest.mark.parametrize(
    ("model_class", "parameters", "attribute"),
    [
        (PCAModel, {"n_components": 1, "standardize_features": False}, "n_components"),
        (
            KernelPCAModel,
            {"n_components": 1, "standardize_features": False, "kernel": "rbf"},
            "n_components",
        ),
        (
            ConvolutionalPCAModel,
            {"n_components": 1, "standardize_features": False, "filter_order": 0},
            "n_components",
        ),
        (
            MaxEntPCAModel,
            {"n_components": 1, "standardize_features": False},
            "n_components",
        ),
        (NystromKPCAModel, {"n_components": 1}, "n_components"),
        (
            RobustPCAModel,
            {"lambda": 0.1, "tolerance": 1e-6, "max_iterations": 10},
            "max_iterations",
        ),
    ],
)
def test_models_validate_parameters_without_feature_data(
    model_class, parameters, attribute
) -> None:
    model = model_class(parameters, Logger().for_component(model_class.__module__))
    assert model.validate_parameters() is None

    target = getattr(model, "estimator", model)
    setattr(target, attribute, 0)
    with pytest.raises(
        ValueError, match=attribute if attribute != "n_components" else "component"
    ):
        model.validate_parameters()
