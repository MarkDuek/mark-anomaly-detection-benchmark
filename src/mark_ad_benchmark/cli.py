import argparse
from pathlib import Path

from mark_ad_benchmark.datasets.ingest import ingest_dataset
from mark_ad_benchmark.datasets.refine import refine_dataset
from mark_ad_benchmark.experiment import Experiment
from mark_ad_benchmark.logging import ConsoleOutput, Logger
from mark_ad_benchmark.visualization import visualize_runs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("ingest", "refine"):
        subparsers.add_parser(command).add_argument("dataset_id")
    for command in ("train", "run"):
        subparsers.add_parser(command).add_argument("experiment_id")
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("experiment_id")
    test_parser.add_argument("validated_run_id")
    subparsers.add_parser("visualize")
    args = parser.parse_args(argv)
    logger = Logger.from_config(Path("config/logging.yaml"), ConsoleOutput())
    component_logger = logger.for_component(__name__)
    if args.command == "visualize":
        component_logger.info(
            "Exporting saved test-run data for the documentation explorer"
        )
        visualize_runs(
            component_logger.for_component(
                "mark_ad_benchmark.visualization.timeseries"
            ),
        )
        return
    if args.command in {"train", "test", "run"}:
        experiment_path = Path("experiments") / f"{args.experiment_id}.yaml"
        if not experiment_path.is_file():
            message = f"Experiment definition not found: {experiment_path}"
            component_logger.error(message)
            raise FileNotFoundError(message)
        experiment = Experiment.from_path(
            experiment_path,
            component_logger.for_component("mark_ad_benchmark.experiment"),
        )
        if args.command == "train":
            component_logger.info(f"Training experiment: {args.experiment_id}")
            experiment.train()
        elif args.command == "test":
            component_logger.info(f"Testing experiment: {args.experiment_id}")
            experiment.test(args.validated_run_id)
        else:
            component_logger.info(f"Running experiment: {args.experiment_id}")
            experiment.run()
        return

    manifest_path = Path("data/manifests") / f"{args.dataset_id}.yaml"
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
