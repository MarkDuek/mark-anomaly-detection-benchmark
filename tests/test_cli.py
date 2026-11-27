from pathlib import Path

import pytest

from mark_ad_benchmark import cli
from mark_ad_benchmark.logging import ComponentLogger


@pytest.mark.parametrize("command", ["ingest", "refine"])
def test_cli_dispatches_dataset_command(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, ...]] = []
    monkeypatch.setattr(
        "mark_ad_benchmark.cli.ingest_dataset", lambda *args: calls.append(args)
    )
    monkeypatch.setattr(
        "mark_ad_benchmark.cli.refine_dataset", lambda *args: calls.append(args)
    )

    cli.main([command, "calit2"])

    expected_call = (Path("data/manifests/calit2.yaml"), Path("data/raw"))
    if command == "refine":
        expected_call += (Path("data/refined"),)
    assert calls[0][:-1] == expected_call
    assert isinstance(calls[0][-1], ComponentLogger)


def test_cli_rejects_missing_dataset_manifest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset manifest not found"):
        cli.main(["ingest", "missing"])

    assert (
        "[ERROR] Dataset manifest not found: data/manifests/missing.yaml"
        in capsys.readouterr().out
    )


def test_cli_dispatches_training_and_testing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeExperiment:
        @classmethod
        def from_path(cls, *args: object) -> "FakeExperiment":
            calls.append(args)
            return cls()

        def train(self) -> None:
            calls.append(("train",))

        def test(self, validated_run_id: str) -> None:
            calls.append(("test", validated_run_id))

    monkeypatch.setattr("mark_ad_benchmark.cli.Experiment", FakeExperiment)

    cli.main(["train", "calit2-pca-inductive"])
    cli.main(["test", "calit2-pca-inductive", "training-run"])

    assert calls[0][0] == Path("experiments/calit2-pca-inductive.yaml")
    assert calls[1] == ("train",)
    assert calls[2][0] == Path("experiments/calit2-pca-inductive.yaml")
    assert calls[3] == ("test", "training-run")


def test_cli_dispatches_visualization(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "mark_ad_benchmark.cli.visualize_runs", lambda *args: calls.append(args)
    )

    cli.main(["visualize"])

    assert len(calls) == 1
    assert isinstance(calls[0][0], ComponentLogger)


def test_cli_dispatches_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "mark_ad_benchmark.cli.publish_runs", lambda *args: calls.append(args)
    )

    cli.main(["publish"])

    assert len(calls) == 1
    assert isinstance(calls[0][0], ComponentLogger)
