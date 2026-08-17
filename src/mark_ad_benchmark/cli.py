import argparse
from pathlib import Path

from mark_ad_benchmark.datasets.ingest import ingest_dataset
from mark_ad_benchmark.datasets.refine import refine_dataset
from mark_ad_benchmark.logging import ConsoleOutput, Logger


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("ingest", "refine"):
        subparsers.add_parser(command).add_argument("dataset_id")
    args = parser.parse_args(argv)
    manifest_path = Path("data/manifests") / f"{args.dataset_id}.yaml"
    logger = Logger.from_config(Path("config/logging.yaml"), ConsoleOutput())
    component_logger = logger.for_component(__name__)
    if not manifest_path.is_file():
        message = f"Dataset manifest not found: {manifest_path}"
        component_logger.error(message)
        raise FileNotFoundError(message)
    component_logger.info(f"Running {args.command} for dataset: {args.dataset_id}")

    if args.command == "ingest":
        ingest_dataset(
            manifest_path,
            Path("data/raw"),
            logger.for_component("mark_ad_benchmark.datasets.ingest"),
        )
    elif args.command == "refine":
        refine_dataset(
            manifest_path,
            Path("data/raw"),
            Path("data/refined"),
            logger.for_component("mark_ad_benchmark.datasets.refine"),
        )
    else:
        message = f"Unknown command: {args.command}"
        component_logger.error(message)
        raise ValueError(message)
