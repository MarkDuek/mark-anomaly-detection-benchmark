from pathlib import Path

from mark_ad_benchmark.logging import ConsoleOutput, FileOutput, Logger, LogOutput


class CapturingOutput(LogOutput):
    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(message)


def test_logger_dispatches_each_level_to_added_outputs() -> None:
    first_output = CapturingOutput()
    second_output = CapturingOutput()
    logger = Logger(first_output)
    logger.add_output(second_output)

    logger.info("loaded dataset")
    logger.warning("missing optional field")
    logger.error("ingestion failed")

    expected_messages = [
        "[INFO] loaded dataset",
        "[WARNING] missing optional field",
        "[ERROR] ingestion failed",
    ]
    assert first_output.messages == expected_messages
    assert second_output.messages == expected_messages


def test_console_output_writes_to_stdout(capsys: object) -> None:
    ConsoleOutput().log("[INFO] ready")

    assert capsys.readouterr().out == "[INFO] ready\n"


def test_file_output_appends_messages(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.log"
    output = FileOutput(path)

    output.log("[INFO] ready")
    output.log("[ERROR] failed")

    assert path.read_text() == "[INFO] ready\n[ERROR] failed\n"


def test_logger_applies_most_specific_configured_component_level(
    tmp_path: Path,
) -> None:
    path = tmp_path / "logging.yaml"
    path.write_text(
        "default_level: info\n"
        "component_levels:\n"
        "  mark_ad_benchmark.datasets: warning\n"
        "  mark_ad_benchmark.datasets.ingest.CalIt2Loader: error\n"
    )
    output = CapturingOutput()
    logger = Logger.from_config(path, output)

    logger.for_component("mark_ad_benchmark.datasets.ingest").info("skipped")
    logger.for_component("mark_ad_benchmark.datasets.ingest").warning("included")
    logger.for_component("mark_ad_benchmark.datasets.ingest.CalIt2Loader").warning(
        "skipped"
    )
    logger.for_component("mark_ad_benchmark.datasets.ingest.CalIt2Loader").error(
        "included"
    )

    assert output.messages == ["[WARNING] included", "[ERROR] included"]
