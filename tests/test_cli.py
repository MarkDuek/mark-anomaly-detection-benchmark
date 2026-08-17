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
