from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics.pairwise import pairwise_kernels

from mark_ad_benchmark.datasets.ingest import sha256
from mark_ad_benchmark.experiment import Experiment
from mark_ad_benchmark.logging import Logger
from mark_ad_benchmark.models import NystromKPCA, NystromKPCAModel

TRAIN = np.array(((0.0, 1.0), (1.0, 0.0), (2.0, 2.0), (3.0, 1.0), (4.0, 3.0)))
QUERY = np.array(((0.5, 1.5), (5.0, -1.0)))


def estimator(**kwargs: object) -> NystromKPCA:
    return NystromKPCA(
        n_components=1,
        n_landmarks=3,
        kernel="rbf",
        kernel_params={"gamma": 0.4},
        random_state=42,
        **kwargs,
    )


@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf"])
@pytest.mark.parametrize("nystrom_solver", ["eigh", "svd"])
@pytest.mark.parametrize("pca_solver", ["eigh", "svd"])
def test_study_equations_and_solver_equivalence(
    kernel: str, nystrom_solver: str, pca_solver: str
) -> None:
    model = NystromKPCA(
        n_components=1,
        n_landmarks=3,
        kernel=kernel,
        nystrom_solver=nystrom_solver,
        pca_solver=pca_solver,
        random_state=42,
    )
    projected = model.fit_transform(TRAIN)
    cross = pairwise_kernels(
        TRAIN, model.landmarks_, metric=kernel, **model.kernel_params
    )
    landmark_kernel = pairwise_kernels(
        model.landmarks_, metric=kernel, **model.kernel_params
    )
    z = cross @ model.normalizer_
    centered = z - z.mean(axis=0)
    covariance = centered.T @ centered / len(TRAIN)
    expected_values = np.linalg.eigvalsh(covariance)[::-1]

    np.testing.assert_allclose(
        z @ z.T,
        cross @ np.linalg.pinv(landmark_kernel) @ cross.T,
        atol=1e-10,
    )
    assert model.landmarks_.shape == (3, TRAIN.shape[1])
    assert model.normalizer_.shape[0] == 3
    np.testing.assert_allclose(model.feature_mean_, z.mean(axis=0))
    np.testing.assert_allclose(projected, centered @ model.components_)
    np.testing.assert_allclose(model.transform(TRAIN), projected)
    np.testing.assert_allclose(model.feature_explained_variance_, expected_values[:1])
    np.testing.assert_allclose(model.feature_total_variance_, np.trace(covariance))
    np.testing.assert_allclose(
        model.feature_explained_variance_ratio_,
        expected_values[:1] / np.trace(covariance),
    )

    reference = NystromKPCA(
        n_components=1,
        n_landmarks=3,
        kernel=kernel,
        random_state=42,
    ).fit_transform(TRAIN)
    np.testing.assert_allclose(
        projected @ projected.T, reference @ reference.T, atol=1e-10
    )


def test_all_components_reconstruct_and_transform_does_not_refit() -> None:
    model = NystromKPCA(
        n_components=None,
        n_landmarks=3,
        kernel="rbf",
        kernel_params={"gamma": 0.4},
        random_state=42,
        standardize_features=True,
        covariance_normalization="sample",
        pca_solver="svd",
    )
    projected = model.fit_transform(TRAIN)
    mean = model.feature_mean_.copy()
    landmarks = model.landmarks_.copy()
    scaler_mean = model.scaler_.mean_.copy()
    centered = model._centered_features(TRAIN)

    np.testing.assert_allclose(projected @ model.components_.T, centered, atol=1e-12)
    np.testing.assert_allclose(
        model.transform(QUERY),
        np.vstack([model.transform(row[None, :]) for row in QUERY]),
    )
    np.testing.assert_array_equal(model.feature_mean_, mean)
    np.testing.assert_array_equal(model.landmarks_, landmarks)
    np.testing.assert_array_equal(model.scaler_.mean_, scaler_mean)
    assert model.transform(np.empty((0, 2))).shape == (0, 3)
    assert model.feature_explained_variance_ratio_.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_benchmark_residuals_and_persistence(kernel: str, tmp_path: Path) -> None:
    logger = Logger().for_component("mark_ad_benchmark.models.nystrom_kpca")
    model = NystromKPCAModel(
        n_components=1,
        n_landmarks=3,
        kernel=kernel,
        standardize_features=True,
        random_state=42,
        logger=logger,
    )
    model.fit(TRAIN)
    centered = model.estimator._centered_features(QUERY)
    expected = np.linalg.norm(
        centered - model.estimator.transform(QUERY) @ model.estimator.components_.T,
        axis=1,
    )
    assert model.score(QUERY) == pytest.approx(expected)
    assert model.score(QUERY) == pytest.approx(
        tuple(model.score(row[None, :])[0] for row in QUERY)
    )

    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = NystromKPCAModel.load(path, logger)
    assert loaded.logger is logger
    assert loaded.estimator.logger is logger
    assert loaded.score(QUERY) == pytest.approx(model.score(QUERY))
    np.testing.assert_allclose(
        loaded.estimator.transform(QUERY), model.estimator.transform(QUERY)
    )


def test_save_uses_current_estimator_settings(tmp_path: Path) -> None:
    logger = Logger().for_component("mark_ad_benchmark.models.nystrom_kpca")
    model = NystromKPCAModel(
        logger=logger, n_components=1, n_landmarks=3, random_state=42
    )
    model.estimator.n_landmarks = 4
    model.fit(TRAIN)
    path = tmp_path / "model.pkl"
    model.save(path)

    loaded = NystromKPCAModel.load(path, logger)
    assert loaded.estimator.n_landmarks == len(loaded.estimator.landmarks_) == 4
    assert loaded.score(QUERY) == pytest.approx(model.score(QUERY))


def test_no_full_training_matrix_is_evaluated(monkeypatch: pytest.MonkeyPatch) -> None:
    import mark_ad_benchmark.models.nystrom_kpca as module

    original = module.pairwise_kernels
    calls = []

    def checked(left: np.ndarray, right: np.ndarray, **kwargs: object) -> np.ndarray:
        calls.append((len(left), len(right)))
        assert (len(left), len(right)) != (len(TRAIN), len(TRAIN))
        return original(left, right, **kwargs)

    monkeypatch.setattr(module, "pairwise_kernels", checked)
    original_svd = np.linalg.svd

    def checked_svd(matrix: np.ndarray, **kwargs: object) -> tuple[np.ndarray, ...]:
        assert matrix.shape != (len(TRAIN), len(TRAIN))
        assert kwargs["full_matrices"] is False
        return original_svd(matrix, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", checked_svd)
    NystromKPCA(
        n_landmarks=3,
        n_components=1,
        nystrom_solver="svd",
        pca_solver="svd",
        random_state=42,
    ).fit_transform(TRAIN)
    assert calls == [(3, 3), (5, 3)]


@pytest.mark.parametrize("solver", ["eigh", "svd"])
def test_absolute_and_relative_cutoffs_and_small_negative_values(solver: str) -> None:
    kernel = np.diag((100.0, 0.1, -1e-13))
    absolute = NystromKPCA(nystrom_solver=solver, eigenvalue_tol=0.01)._normalizer(
        kernel
    )
    relative = NystromKPCA(
        nystrom_solver=solver, eigenvalue_tol=0.01, eigenvalue_tol_mode="relative"
    )._normalizer(kernel)
    assert absolute.shape == (3, 2)
    assert relative.shape == (3, 1)


@pytest.mark.parametrize("solver", ["eigh", "svd"])
@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_landmark_truncation_and_covariance_rank(solver: str, mode: str) -> None:
    model = NystromKPCA(
        n_components=1,
        n_landmarks=3,
        kernel="linear",
        random_state=42,
        nystrom_solver=solver,
        eigenvalue_tol_mode=mode,
    )
    model.fit(TRAIN)
    assert model.normalizer_.shape == (3, 2)
    assert model.feature_explained_variance_.shape == (1,)
    with pytest.raises(ValueError, match="positive covariance rank"):
        NystromKPCA(
            n_components=2, n_landmarks=3, kernel="linear", random_state=42
        ).fit(((1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0), (5.0, 1.0)))


@pytest.mark.parametrize("solver", ["eigh", "svd"])
def test_rejects_significant_negative_landmark_spectrum(solver: str) -> None:
    model = estimator(nystrom_solver=solver)
    with pytest.raises(ValueError, match="negative eigenvalues"):
        model._normalizer(np.diag((2.0, 1.0, -0.1)))


def test_rejects_degenerate_or_uninformative_fit() -> None:
    with pytest.raises(ValueError, match="fewer than training rows"):
        NystromKPCA(n_landmarks=len(TRAIN)).fit(TRAIN)
    with pytest.raises(ValueError, match="positive centered feature-space variance"):
        NystromKPCA(n_landmarks=2, kernel="rbf").fit(np.ones((4, 2)))
    with pytest.raises(ValueError, match="no eigenvalues above cutoff"):
        NystromKPCA(n_landmarks=2, kernel="linear").fit(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="landmark effective rank"):
        NystromKPCA(n_landmarks=3, n_components=3, kernel="linear").fit(TRAIN)
    logger = Logger().for_component("mark_ad_benchmark.models.nystrom_kpca")
    full_rank_adapter = NystromKPCAModel(
        n_landmarks=3, n_components=None, logger=logger
    )
    with pytest.raises(ValueError, match="fewer components than effective rank"):
        full_rank_adapter.fit(TRAIN)
    with pytest.raises(ValueError, match="fitted before transforming"):
        full_rank_adapter.score(QUERY)
    with pytest.raises(ValueError, match="fitted before saving"):
        full_rank_adapter.save(Path("unused.pkl"))
    with pytest.raises(ValueError, match="fewer components than effective rank"):
        NystromKPCAModel(n_landmarks=3, n_components=3, logger=logger).fit(TRAIN)


def test_invalid_features_and_kernel_parameters() -> None:
    assert NystromKPCA(eigenvalue_strategy="truncate").eigenvalue_strategy == "truncate"
    with pytest.raises(ValueError, match="eigenvalue_strategy"):
        NystromKPCA(eigenvalue_strategy="clip")
    for name, parameters in (
        ("linear", {"gamma": 1.0}),
        ("poly", {"degree": 0}),
        ("rbf", {"gamma": float("nan")}),
    ):
        with pytest.raises(ValueError):
            NystromKPCA(kernel=name, kernel_params=parameters)
    model = estimator()
    with pytest.raises(ValueError, match="fitted before transforming"):
        model.transform(QUERY)
    with pytest.raises(ValueError, match="finite feature values"):
        model.fit(np.array(((1.0, 0.0), (float("inf"), 1.0))))
    model.fit(TRAIN)
    with pytest.raises(ValueError, match="incompatible width"):
        model.transform(((1.0,),))


def test_checked_in_calit2_experiment_configuration() -> None:
    experiment = Experiment.from_path(
        Path("experiments/calit2/nystrom_kpca-validation_f1.yaml"),
        Logger().for_component("mark_ad_benchmark.experiment"),
    )
    model = experiment.create_model(experiment.model_parameter_candidates()[0])
    assert isinstance(model, NystromKPCAModel)
    assert experiment.execution_mode == "inductive"
    assert model.estimator.n_landmarks == 16
    assert model.estimator.kernel == "linear"


def test_benchmark_train_and_test_with_saved_model(tmp_path: Path) -> None:
    refined = tmp_path / "refined" / "example"
    refined.mkdir(parents=True)
    rows = {
        "train": ((0, 0, 0), (1, 0, 0), (2, 2, 0), (3, 1, 0), (4, 3, 0)),
        "validation": ((1, 1, 0), (10, 10, 1)),
        "test": ((2, 1, 0), (12, 12, 1)),
    }
    for split, values in rows.items():
        (refined / f"{split}.csv").write_text(
            "timestamp,feature_0,feature_1,label\n"
            + "".join(
                f"2025-01-{i:02d},{x},{y},{label}\n"
                for i, (x, y, label) in enumerate(values, 1)
            )
        )
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "example.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "example",
                "files": {split: f"source/{split}.csv" for split in rows},
                "refined_schema": {
                    "timestamp": "timestamp",
                    "features": ["feature_0", "feature_1"],
                    "label": "label",
                },
                "checksums": {
                    "refined": {
                        split: sha256(refined / f"{split}.csv") for split in rows
                    }
                },
            }
        )
    )
    definition = {
        "id": "example-nystrom",
        "dataset": "example",
        "model": {
            "name": "nystrom_kpca",
            "parameters": {
                "n_components": 1,
                "n_landmarks": 3,
                "kernel": "rbf",
                "kernel_params": {"gamma": 0.4},
                "random_state": 42,
            },
        },
        "protocol": "inductive",
        "training": {"epochs": 1},
        "threshold": {"name": "validation_f1"},
        "selection": {"metric": "f1", "direction": "maximize"},
        "metrics": {
            "vus": {"max_window": 1, "threshold_samples": 3},
            "range": {"recall_existence_weight": 0.2},
        },
    }
    experiment = Experiment(
        definition,
        Logger().for_component("mark_ad_benchmark.experiment"),
        manifest_root=manifests,
        refined_root=tmp_path / "refined",
        runs_root=tmp_path / "runs",
    )
    result = experiment.run("nystrom-run")
    assert (result.validated.run_dir / "model.pkl").is_file()
    assert (result.test.run_dir / "test_scores.csv").is_file()
    assert result.test.test_metrics.point.f1 >= 0
