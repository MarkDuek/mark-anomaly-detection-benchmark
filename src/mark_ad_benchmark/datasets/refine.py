import csv
from pathlib import Path

import yaml

from mark_ad_benchmark.datasets.ingest import sha256, validate_csv
from mark_ad_benchmark.logging import ComponentLogger


def refine_dataset(
    manifest_path: Path,
    raw_root: Path,
    refined_root: Path,
    logger: ComponentLogger,
) -> None:
    manifest = yaml.safe_load(manifest_path.read_text())
    raw_schema = manifest["raw_schema"]
    refined_schema = manifest["refined_schema"]
    raw_checksums = manifest["checksums"]["raw"]
    refined_checksums = manifest.setdefault("checksums", {}).setdefault("refined", {})
    raw_dir = raw_root / manifest["id"]
    refined_dir = refined_root / manifest["id"]
    refined_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Refining dataset: {manifest['id']}")

    # 3. verify raw splits and project them into the refined schema (documentation excerpt; keep synchronized)
    for split, remote_path in manifest["files"].items():
        raw_path = raw_dir / Path(remote_path).name
        if sha256(raw_path) != raw_checksums[split]:
            message = f"Checksum mismatch for {raw_path}"
            logger.error(message)
            raise ValueError(message)
        normal_only = split in manifest.get("normal_only_splits", [])
        validate_csv(raw_path, raw_schema, normal_only=normal_only)
        if normal_only:
            logger.info(
                f"Assigning normal labels to declared normal-only {split} split"
            )
        refined_path = refined_dir / f"{split}.csv"
        refine_csv(
            raw_path, refined_path, raw_schema, refined_schema, normal_only=normal_only
        )
        validate_csv(refined_path, refined_schema)
        refined_checksums[split] = sha256(refined_path)
        logger.info(f"Refined {split} split: {refined_path}")

    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    logger.info(f"Completed dataset refinement: {manifest['id']}")


def refine_csv(
    raw_path: Path,
    refined_path: Path,
    raw_schema: dict[str, object],
    refined_schema: dict[str, object],
    *,
    normal_only: bool = False,
) -> None:
    raw_columns = [
        raw_schema["timestamp"],
        *raw_schema["features"],
    ]
    if not normal_only:
        raw_columns.append(raw_schema["label"])
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
            values = [row[column] for column in raw_columns]
            if normal_only:
                values.append("0")
            writer.writerow(dict(zip(refined_columns, values, strict=True)))
