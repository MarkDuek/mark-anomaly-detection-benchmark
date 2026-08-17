import csv
from pathlib import Path

import yaml

from mark_ad_benchmark.datasets.ingest import sha256, validate_csv


def refine_dataset(manifest_path: Path, raw_root: Path, refined_root: Path) -> None:
    manifest = yaml.safe_load(manifest_path.read_text())
    raw_schema = manifest["raw_schema"]
    refined_schema = manifest["refined_schema"]
    raw_checksums = manifest["checksums"]["raw"]
    refined_checksums = manifest.setdefault("checksums", {}).setdefault("refined", {})
    raw_dir = raw_root / manifest["id"]
    refined_dir = refined_root / manifest["id"]
    refined_dir.mkdir(parents=True, exist_ok=True)

    for split, remote_path in manifest["files"].items():
        raw_path = raw_dir / Path(remote_path).name
        if sha256(raw_path) != raw_checksums[split]:
            raise ValueError(f"Checksum mismatch for {raw_path}")
        validate_csv(raw_path, raw_schema)
        refined_path = refined_dir / f"{split}.csv"
        refine_csv(raw_path, refined_path, raw_schema, refined_schema)
        validate_csv(refined_path, refined_schema)
        refined_checksums[split] = sha256(refined_path)

    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))


def refine_csv(
    raw_path: Path,
    refined_path: Path,
    raw_schema: dict[str, object],
    refined_schema: dict[str, object],
) -> None:
    raw_columns = [
        raw_schema["timestamp"],
        *raw_schema["features"],
        raw_schema["label"],
    ]
    refined_columns = [
        refined_schema["timestamp"],
        *refined_schema["features"],
        refined_schema["label"],
    ]
    with (
        raw_path.open(newline="", encoding="utf-8") as raw_file,
        refined_path.open("w", newline="", encoding="utf-8") as refined_file,
    ):
        reader = csv.DictReader(raw_file)
        writer = csv.DictWriter(
            refined_file, fieldnames=refined_columns, lineterminator="\n"
        )
        writer.writeheader()
        for row in reader:
            writer.writerow(
                dict(zip(refined_columns, (row[column] for column in raw_columns)))
            )
