import pickle
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel

from mark_ad_benchmark.experiment import Experiment
from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.models import KernelPCAModel, KernelType, PCAModel

TRAIN = ((1.0, 2.0), (2.0, 1.0), (3.0, 4.0), (4.0, 3.0), (5.0, 7.0))
QUERY = ((2.0, 3.0), (10.0, -3.0))


def model(kernel: str = "linear", **parameters: object) -> KernelPCAModel:
    return KernelPCAModel(
        {
            "n_components": 1,
            "kernel": kernel,
            "standardize_features": False,
            **parameters,
        },
        Logger().for_component("mark_ad_benchmark.models.kernel_pca"),
    )


@pytest.mark.parametrize("standardize_features", [False, True])
def test_linear_kernel_matches_pca_residuals(standardize_features: bool) -> None:
    kernel_pca = model(standardize_features=standardize_features)
    pca = PCAModel(
        {"n_components": 1, "standardize_features": standardize_features},
        Logger().for_component("mark_ad_benchmark.models.pca"),
    )
    kernel_pca.fit(TRAIN)
    pca.fit(TRAIN)

    assert kernel_pca.score(QUERY) == pytest.approx(pca.score(QUERY))


@pytest.mark.parametrize(
    ("kernel", "kernel_parameters"),
    [
        (KernelType.LINEAR, {}),
        (KernelType.POLY, {"gamma": 1.0, "degree": 3, "coef0": 1.0}),
        (KernelType.RBF, {"gamma": 0.5}),
    ],
)
def test_valid_kernel_configuration(
    kernel: KernelType, kernel_parameters: dict[str, object]
) -> None:
    kernel_pca = model(kernel.value, kernel_parameters=kernel_parameters)

    assert kernel_pca.kernel is kernel
    assert kernel_pca.kernel_parameters == kernel_parameters
    assert kernel_pca.kernel_parameters is not kernel_parameters
    assert not any(hasattr(kernel_pca, name) for name in ("gamma", "degree", "coef0"))


@pytest.mark.parametrize(
    ("kernel", "expected"),
    [
        (KernelType.LINEAR, {}),
        (KernelType.POLY, {"gamma": 1.0, "degree": 2, "coef0": 1.0}),
        (KernelType.RBF, {"gamma": 1.0}),
    ],
)
def test_default_kernel_parameters(
    kernel: KernelType, expected: dict[str, object]
) -> None:
    assert model(kernel).kernel_parameters == expected
    assert model(kernel, kernel_parameters={}).kernel_parameters == expected


def test_partial_polynomial_parameters_preserve_input() -> None:
    parameters = MappingProxyType({"coef0": 0.0})
    kernel_pca = model("poly", kernel_parameters=parameters)

    assert kernel_pca.kernel_parameters == {"gamma": 1.0, "degree": 2, "coef0": 0.0}
    assert dict(parameters) == {"coef0": 0.0}


def test_polynomial_kernel_matches_explicit_feature_space_pca() -> None:
    gamma, coef0 = 0.5, 0.7
    kernel_pca = model(
        "poly", kernel_parameters={"gamma": gamma, "coef0": coef0, "degree": 2}
    )
    kernel_pca.fit(TRAIN)

    def feature_map(features: tuple[tuple[float, ...], ...]) -> np.ndarray:
        x, y = np.asarray(features).T
        return np.column_stack(
            (
                np.full(len(x), coef0),
                np.sqrt(2 * gamma * coef0) * x,
                np.sqrt(2 * gamma * coef0) * y,
                gamma * x**2,
                np.sqrt(2) * gamma * x * y,
                gamma * y**2,
            )
        )

    pca = PCA(n_components=1).fit(feature_map(TRAIN))
    mapped = feature_map(QUERY)
    reconstructed = pca.inverse_transform(pca.transform(mapped))

    assert kernel_pca.score(QUERY) == pytest.approx(
        np.linalg.norm(mapped - reconstructed, axis=1)
    )


def test_rbf_residuals_use_train_centered_kernel_and_normalized_projections() -> None:
    kernel_pca = model("rbf", kernel_parameters={"gamma": 0.3})
    kernel_pca.fit(TRAIN)
    kernel = rbf_kernel(TRAIN, gamma=0.3)
    centering = np.eye(len(TRAIN)) - np.ones((len(TRAIN), len(TRAIN))) / len(TRAIN)
    eigenvalues, eigenvectors = np.linalg.eigh(centering @ kernel @ centering)
    cross = rbf_kernel(QUERY, TRAIN, gamma=0.3)
    centered_cross = (
        cross - kernel.mean(axis=0) - cross.mean(axis=1, keepdims=True) + kernel.mean()
    )
    projection = centered_cross @ eigenvectors[:, -1] / np.sqrt(eigenvalues[-1])
    expected = np.sqrt(1 - 2 * cross.mean(axis=1) + kernel.mean() - projection**2)

    assert kernel_pca.score(QUERY) == pytest.approx(expected)


@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf"])
def test_scoring_is_batch_independent_and_does_not_refit(kernel: str) -> None:
    kernel_pca = model(kernel, standardize_features=True)
    kernel_pca.fit(TRAIN)
    mean = kernel_pca.scaler.mean_.copy()
    eigenvalues = kernel_pca.pca.eigenvalues_.copy()
    scores = kernel_pca.score(QUERY)

    assert scores == pytest.approx(tuple(kernel_pca.score((row,))[0] for row in QUERY))
    np.testing.assert_array_equal(kernel_pca.scaler.mean_, mean)
    np.testing.assert_array_equal(kernel_pca.pca.eigenvalues_, eigenvalues)
    assert isinstance(scores, tuple)
    assert all(np.isfinite(score) and score >= 0 for score in scores)


@pytest.mark.parametrize(
    ("kernel", "kernel_parameters"),
    [
        (KernelType.LINEAR, {}),
        (KernelType.POLY, {"gamma": 0.4, "degree": 3, "coef0": 0.2}),
        (KernelType.RBF, {"gamma": 0.4}),
    ],
)
def test_save_and_load_preserves_kernel_scores(
    kernel: KernelType, kernel_parameters: dict[str, object], tmp_path: Path
) -> None:
    kernel_pca = model(
        kernel, standardize_features=True, kernel_parameters=kernel_parameters
    )
    kernel_pca.fit(TRAIN)
    path = tmp_path / "model.pkl"
    kernel_pca.save(path)
    logger = Logger().for_component("mark_ad_benchmark.models.kernel_pca")
    loaded = KernelPCAModel.load(path, logger)

    assert loaded.score(QUERY) == pytest.approx(kernel_pca.score(QUERY))
    assert loaded.kernel is kernel
    assert loaded.kernel_parameters == kernel_parameters
    assert loaded.logger is logger
    with path.open("rb") as file:
        parameters = pickle.load(file)["parameters"]
    assert parameters == {
        "n_components": 1,
        "kernel": kernel.value,
        "standardize_features": True,
        "kernel_parameters": kernel_parameters,
    }
    assert type(parameters["kernel"]) is str


def test_load_revalidates_saved_configuration(tmp_path: Path) -> None:
    kernel_pca = model("rbf", kernel_parameters={"gamma": 0.4})
    kernel_pca.fit(TRAIN)
    path = tmp_path / "model.pkl"
    kernel_pca.save(path)
    with path.open("rb") as file:
        payload = pickle.load(file)
    payload["parameters"]["kernel_parameters"]["degree"] = 3
    with path.open("wb") as file:
        pickle.dump(payload, file)

    with pytest.raises(ValueError, match="does not support kernel parameter degree"):
        KernelPCAModel.load(
            path, Logger().for_component("mark_ad_benchmark.models.kernel_pca")
        )


def test_kernel_components_can_exceed_input_feature_width() -> None:
    kernel_pca = model("rbf", n_components=3)
    kernel_pca.fit(TRAIN)

    assert len(kernel_pca.pca.eigenvalues_) == 3
    assert len(kernel_pca.score(QUERY)) == len(QUERY)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"n_components": 0}, "n_components"),
        ({"n_components": True}, "n_components"),
        ({"standardize_features": "true"}, "standardize_features"),
    ],
)
def test_rejects_invalid_parameters(
    parameters: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        model(**parameters)


@pytest.mark.parametrize("kernel", ["sigmoid", "POLY", "", None, True, []])
def test_rejects_unknown_kernel_names(kernel: object) -> None:
    with pytest.raises(ValueError, match="kernel must be linear, poly, or rbf"):
        model(kernel)


@pytest.mark.parametrize(
    ("kernel", "parameters"),
    [
        ("linear", {"gamma": 1.0}),
        ("linear", {"degree": 3}),
        ("linear", {"coef0": 1.0}),
        ("rbf", {"degree": 3}),
        ("rbf", {"coef0": 1.0}),
        ("linear", {"unknown": 1}),
        ("poly", {"unknown": 1}),
        ("rbf", {"unknown": 1}),
    ],
)
def test_rejects_unsupported_kernel_parameters(
    kernel: str, parameters: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="does not support kernel parameter"):
        model(kernel, kernel_parameters=parameters)


@pytest.mark.parametrize("parameters", [None, [], "gamma", 1])
def test_rejects_non_mapping_kernel_parameters(parameters: object) -> None:
    with pytest.raises(ValueError, match="kernel_parameters must be a mapping"):
        model(kernel_parameters=parameters)


@pytest.mark.parametrize("kernel", ["poly", "rbf"])
@pytest.mark.parametrize(
    "gamma", [0, -1, float("nan"), float("inf"), float("-inf"), True, "1", None]
)
def test_rejects_invalid_gamma(kernel: str, gamma: object) -> None:
    with pytest.raises(ValueError, match="gamma must be finite and positive"):
        model(kernel, kernel_parameters={"gamma": gamma})


@pytest.mark.parametrize("degree", [0, -1, 1.5, True, "2", None, float("inf")])
def test_rejects_invalid_degree(degree: object) -> None:
    with pytest.raises(ValueError, match="degree must be a positive integer"):
        model("poly", kernel_parameters={"degree": degree})


@pytest.mark.parametrize("coef0", [-1, float("nan"), float("inf"), True, "1", None])
def test_rejects_invalid_coef0(coef0: object) -> None:
    with pytest.raises(ValueError, match="coef0 must be finite and nonnegative"):
        model("poly", kernel_parameters={"coef0": coef0})


@pytest.mark.parametrize(
    "features", [(), ((),), ((1.0,), (1.0, 2.0)), ((float("nan"),),)]
)
def test_rejects_invalid_training_features(features: tuple) -> None:
    with pytest.raises(ValueError, match="Kernel PCA requires"):
        model().fit(features)


def test_rejects_incompatible_scoring_width_and_nonfinite_values() -> None:
    kernel_pca = model()
    kernel_pca.fit(TRAIN)

    with pytest.raises(ValueError, match="incompatible width"):
        kernel_pca.score(((1.0,),))
    with pytest.raises(ValueError, match="finite feature values"):
        kernel_pca.score(((float("inf"), 1.0),))


def test_rejects_components_at_training_sample_count() -> None:
    with pytest.raises(ValueError, match="fewer than the training rows"):
        model(n_components=len(TRAIN)).fit(TRAIN)


def test_unfitted_model_cannot_score_or_save(tmp_path: Path) -> None:
    kernel_pca = model()
    with pytest.raises(ValueError, match="fitted before scoring"):
        kernel_pca.score(QUERY)
    with pytest.raises(ValueError, match="fitted before saving"):
        kernel_pca.save(tmp_path / "model.pkl")


@pytest.mark.parametrize(
    "experiment_id",
    [
        "calit2-kernel-pca-inductive",
        "creditcard-kernel-pca-inductive-validation-f1-quantile",
    ],
)
def test_checked_in_experiments_resolve_all_three_kernel_candidates(
    experiment_id: str,
) -> None:
    experiment = Experiment.from_path(
        Path("experiments") / f"{experiment_id}.yaml",
        Logger().for_component("mark_ad_benchmark.experiment"),
    )
    models = tuple(
        experiment.create_model(parameters)
        for parameters in experiment.model_parameter_candidates()
    )

    assert tuple(candidate.kernel for candidate in models) == tuple(KernelType)
    gamma = (
        1.0 if experiment_id == "calit2-kernel-pca-inductive" else 0.03333333333333333
    )
    assert tuple(candidate.kernel_parameters for candidate in models) == (
        {},
        {"gamma": gamma, "degree": 2, "coef0": 1.0},
        {"gamma": gamma},
    )
    assert all(isinstance(candidate, KernelPCAModel) for candidate in models)
    assert experiment.execution_mode == "inductive"
