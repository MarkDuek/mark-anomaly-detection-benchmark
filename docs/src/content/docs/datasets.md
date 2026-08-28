---
title: Datasets
description: Time-series data used for experiments.
---

Datasets are declared by versioned YAML manifests. A manifest identifies its source, source revision, split files, raw and refined schemas, and SHA-256 checksums. Experiments load only refined data whose contents match that manifest.

## Dataset Contract

Each refined CSV has one timestamp column, ordered numeric feature columns, and one binary label column. The loader returns an immutable `DatasetSplit` containing timestamp strings, feature tuples of floats, and integer labels.

The raw schema preserves upstream column names. Refinement writes an equivalent CSV in the benchmark schema, renaming the timestamp, feature, and label columns while preserving their values and order. Both raw and refined files must have exact declared headers, nonempty rows, numeric feature values, and labels of `0` or `1`.

## Pipeline

### 1. Pin Source Revision

Every manifest uses a Hugging Face dataset source. If a manifest lacks a revision, ingestion resolves the repository revision once and writes it back to the manifest.

```python
# 1. resolve and pin the source revision (documentation excerpt; keep synchronized)
revision = source.get("revision")
if revision is None:
    revision = HfApi().dataset_info(source["repository"]).sha
    source["revision"] = revision
```

### 2. Ingest Raw Splits

For every declared split, ingestion downloads the pinned source file, validates the raw schema, and records its SHA-256 checksum.

```python
# 2. download, validate, and checksum every raw split (documentation excerpt; keep synchronized)
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
```

### 3. Refine Schema

Refinement verifies each raw checksum and schema, writes its declared refined columns, validates the result, and records the refined checksum.

```python
# 3. verify raw splits and project them into the refined schema (documentation excerpt; keep synchronized)
for split, remote_path in manifest["files"].items():
    raw_path = raw_dir / Path(remote_path).name
    if sha256(raw_path) != raw_checksums[split]:
        message = f"Checksum mismatch for {raw_path}"
        logger.error(message)
        raise ValueError(message)
    validate_csv(raw_path, raw_schema)
    refined_path = refined_dir / f"{split}.csv"
    refine_csv(raw_path, refined_path, raw_schema, refined_schema)
    validate_csv(refined_path, refined_schema)
    refined_checksums[split] = sha256(refined_path)
```

### 4. Load Required Splits

Protocols request only the splits required by their lifecycle. The loader rejects a request missing from the manifest.

```python
# 4. resolve the splits required by the active protocol (documentation excerpt; keep synchronized)
requested_splits = requested_splits or tuple(manifest["files"])
missing_splits = set(requested_splits) - manifest["files"].keys()
if missing_splits:
    message = f"Dataset manifest is missing splits: {sorted(missing_splits)}"
    logger.error(message)
    raise ValueError(message)
```

### 5. Verify And Materialize

Before a split reaches a model, the loader verifies its refined checksum and schema, then converts rows to the immutable split representation.

```python
# 5. verify and materialize each refined split (documentation excerpt; keep synchronized)
for split in requested_splits:
    path = dataset_dir / f"{split}.csv"
    if sha256(path) != checksums[split]:
        message = f"Checksum mismatch for {path}"
        logger.error(message)
        raise ValueError(message)
    validate_csv(path, schema)
    datasets[split] = load_split(path, schema)
```

## Available Datasets

| Dataset | Source | Splits | Features |
| --- | --- | --- | --- |
| [CalIt2](/datasets/calit2/) | `PLAN-Lab/mTSBench` | Train, validation, test | 2 |
| [Credit Card](/datasets/creditcard/) | `PLAN-Lab/mTSBench` | Train, validation, test | 29 |
