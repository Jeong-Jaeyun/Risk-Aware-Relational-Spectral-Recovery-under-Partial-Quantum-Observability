from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "biqmn" / "results"
CONFIGS = ROOT / "biqmn" / "configs"
OUTPUT = ROOT / "huggingface_dataset"

REGIMES = {
    "clean": (
        "hybrid_c123_regime_map_c3r_phase2_seed10_raw.csv",
        3_600,
    ),
    "partial_syndrome": (
        "partial_syndrome_c3r_phase2_seed10_raw.csv",
        14_400,
    ),
    "noisy_syndrome": (
        "noisy_syndrome_c3r_phase2_seed10_raw.csv",
        18_000,
    ),
    "partial_noisy_syndrome": (
        "partial_noisy_syndrome_c3r_phase2_seed10_raw.csv",
        10_800,
    ),
    "ambiguity_measurement_reset": (
        "ambiguity_measurement_c3r_phase2_seed10_raw.csv",
        18_000,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        columns = len(next(reader))
        rows = sum(1 for _ in reader)
    return rows, columns


def copy_configs(destination: Path) -> None:
    for source in CONFIGS.rglob("*.yaml"):
        target = destination / source.relative_to(CONFIGS)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    raw_dir = OUTPUT / "data" / "raw"
    aggregate_dir = OUTPUT / "aggregates"
    config_dir = OUTPUT / "configs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    manifest: dict[str, object] = {
        "dataset": "Uncertainty-Gated Structural Recovery under Syndrome Ambiguity",
        "case_count": 0,
        "regimes": {},
        "files": [],
    }

    for regime, (filename, expected_rows) in REGIMES.items():
        source = RESULTS / "tables" / filename
        rows, columns = csv_shape(source)
        if rows != expected_rows:
            raise ValueError(f"{filename}: expected {expected_rows} rows, found {rows}")

        target = raw_dir / filename
        shutil.copy2(source, target)
        frame = pd.read_csv(source, low_memory=False)
        frame.insert(0, "regime", regime)
        frames.append(frame)

        manifest["case_count"] = int(manifest["case_count"]) + rows
        manifest["regimes"][regime] = {
            "file": target.relative_to(OUTPUT).as_posix(),
            "rows": rows,
            "columns": columns,
        }

    if manifest["case_count"] != 64_800:
        raise ValueError(f"Expected 64,800 total cases, found {manifest['case_count']}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.insert(1, "dataset_case_index", range(len(combined)))
    for column in combined.select_dtypes(include=["object"]).columns:
        combined[column] = combined[column].astype("string")
    parquet_path = OUTPUT / "data" / "all_cases.parquet"
    combined.to_parquet(parquet_path, index=False, compression="zstd")

    aggregate_sources = [
        path
        for path in (RESULTS / "tables").glob("*phase2_seed10*.csv")
        if not path.name.endswith("_raw.csv")
    ]
    aggregate_sources.extend((RESULTS / "tables").glob("phase2_failure_linkage*.csv"))
    aggregate_sources.extend((RESULTS / "tables").glob("phase2_failure_linkage*.md"))
    for source in sorted(set(aggregate_sources)):
        shutil.copy2(source, aggregate_dir / source.name)

    copy_configs(config_dir)

    tracked_files = sorted(
        path for path in OUTPUT.rglob("*") if path.is_file() and path.name != "dataset_manifest.json"
    )
    manifest["files"] = [
        {
            "path": path.relative_to(OUTPUT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in tracked_files
    ]
    manifest["combined_table"] = {
        "path": parquet_path.relative_to(OUTPUT).as_posix(),
        "rows": len(combined),
        "columns": len(combined.columns),
    }

    with (OUTPUT / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Prepared {manifest['case_count']:,} cases in {OUTPUT}")


if __name__ == "__main__":
    main()
