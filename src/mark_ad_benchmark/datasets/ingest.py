import csv
import hashlib
import shutil
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download


def ingest_dataset(manifest_path: Path, raw_root: Path) -> None:
    manifest = yaml.safe_load(manifest_path.read_text())
    source = manifest["source"]
    if source["type"] != "huggingface":
        raise ValueError(f"Unsupported source type: {source['type']}")

    revision = source.get("revision")
    if revision is None:
        revision = HfApi().dataset_info(source["repository"]).sha
        source["revision"] = revision

    dataset_dir = raw_root / manifest["id"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    checksums = manifest.setdefault("checksums", {}).setdefault("raw", {})

    for split, remote_path in manifest["files"].items():
        downloaded_path = Path(
            hf_hub_download(
                repo_id=source["repository"],
                repo_type="dataset",
                filename=remote_path,
                revision=revision,
            )
        )
        target_path = dataset_dir / Path(remote_path).name
        if not target_path.exists():
            shutil.copy2(downloaded_path, target_path)
        validate_csv(target_path, manifest["raw_schema"])
        checksums[split] = sha256(target_path)

    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))


def validate_csv(path: Path, schema: dict[str, object]) -> None:
    expected_columns = [schema["timestamp"], *schema["features"], schema["label"]]
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")

        row_count = 0
        for row in reader:
            if any(not row[column] for column in expected_columns):
                raise ValueError(f"Missing value in {path}")
            for feature in schema["features"]:
                float(row[feature])
            if row[schema["label"]] not in {"0", "1"}:
                raise ValueError(f"Invalid label in {path}: {row[schema['label']]}")
            row_count += 1

    if row_count == 0:
        raise ValueError(f"No data rows in {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
