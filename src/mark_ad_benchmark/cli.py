import argparse
from pathlib import Path

from mark_ad_benchmark.datasets.ingest import ingest_dataset
from mark_ad_benchmark.datasets.refine import refine_dataset
from mark_ad_benchmark.experiment import (
    run_experiment,
    test_experiment,
    train_experiment,
)
from mark_ad_benchmark.logging import ConsoleOutput, Logger


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("ingest", "refine"):
        subparsers.add_parser(command).add_argument("dataset_id")
    for command in ("train", "run"):
        subparsers.add_parser(command).add_argument("experiment_id")
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("experiment_id")
    test_parser.add_argument("training_run_id")
    args = parser.parse_args(argv)
    logger = Logger.from_config(Path("config/logging.yaml"), ConsoleOutput())
    component_logger = logger.for_component(__name__)
    if args.command in {"train", "test", "run"}:
        experiment_path = Path("experiments") / f"{args.experiment_id}.yaml"
        if not experiment_path.is_file():
            message = f"Experiment definition not found: {experiment_path}"
            component_logger.error(message)
            raise FileNotFoundError(message)
        experiment_logger = component_logger.for_component(
            "mark_ad_benchmark.experiment"
        )
        if args.command == "train":
            component_logger.info(f"Training experiment: {args.experiment_id}")
            train_experiment(experiment_path, experiment_logger)
        elif args.command == "test":
            component_logger.info(f"Testing experiment: {args.experiment_id}")
            test_experiment(experiment_path, args.training_run_id, experiment_logger)
        else:
            component_logger.info(f"Running experiment: {args.experiment_id}")
            run_experiment(experiment_path, experiment_logger)
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
