from pathlib import Path

import pytest

from mark_ad_benchmark import cli


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
    assert calls == [expected_call]
