import argparse
from pathlib import Path

from mark_ad_benchmark.datasets.ingest import ingest_dataset
from mark_ad_benchmark.datasets.refine import refine_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("dataset_id")
    refine_parser = subparsers.add_parser("refine")
    refine_parser.add_argument("dataset_id")
    args = parser.parse_args()

    if args.command == "ingest":
        manifest_path = Path("data/manifests") / f"{args.dataset_id}.yaml"
        ingest_dataset(manifest_path, Path("data/raw"))
    if args.command == "refine":
        manifest_path = Path("data/manifests") / f"{args.dataset_id}.yaml"
        refine_dataset(manifest_path, Path("data/raw"), Path("data/refined"))


if __name__ == "__main__":
    main()
