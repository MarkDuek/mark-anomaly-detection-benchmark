import argparse
from pathlib import Path

from mark_ad_benchmark.datasets.ingest import ingest_dataset
from mark_ad_benchmark.datasets.refine import refine_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("ingest", "refine"):
        subparsers.add_parser(command).add_argument("dataset_id")
    args = parser.parse_args(argv)
    manifest_path = Path("data/manifests") / f"{args.dataset_id}.yaml"

    if args.command == "ingest":
        ingest_dataset(manifest_path, Path("data/raw"))
    elif args.command == "refine":
        refine_dataset(manifest_path, Path("data/raw"), Path("data/refined"))
    else:
        raise ValueError(f"Unknown command: {args.command}")
