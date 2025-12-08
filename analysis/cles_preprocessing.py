"""
Preprocess CLES survey CSV files into cleaned tables and metadata artifacts.

Usage:
    python analysis/cles_preprocessing.py            # process all datasets
    python analysis/cles_preprocessing.py --datasets household_2020_2021 village_2022
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_SOURCE_DIR = ROOT_DIR / "校准用数据源"
OUTPUT_DATA_DIR = ROOT_DIR / "data" / "cles"
METADATA_PATH = ROOT_DIR / "config" / "cles_variable_metadata.json"


DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "household_2020_2021": {
        "file": "CLES_2020_2021_household_panel_mapping_label.csv",
        "description": "CLES 2020-2021 household panel mapping dataset",
    },
    "village_2020_2021": {
        "file": "CLES_2020_2021_village_panel_label_mapping.csv",
        "description": "CLES 2020-2021 village panel mapping dataset",
    },
    "household_2022": {
        "file": "CLES_2022_household_mapping_label.csv",
        "description": "CLES 2022 household mapping dataset",
    },
    "village_2022": {
        "file": "CLES_2022_village_label_mapping.csv",
        "description": "CLES 2022 village mapping dataset",
    },
}


logger = logging.getLogger("cles_preprocessing")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def load_metadata_store() -> Dict[str, Dict[str, str]]:
    if METADATA_PATH.exists():
        try:
            return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Existing metadata file is corrupted. Rebuilding from scratch.")
    return {}


def save_metadata_store(store: Dict[str, Dict[str, str]]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def preprocess_dataset(dataset_id: str) -> Tuple[Path, Dict[str, str]]:
    if dataset_id not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset id: {dataset_id}")

    entry = DATASET_REGISTRY[dataset_id]
    source_path = DATA_SOURCE_DIR / entry["file"]
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {source_path}")

    logger.info("Processing %s (%s)", dataset_id, source_path.name)
    raw_df = pd.read_csv(source_path, encoding="gb18030", dtype=str)
    if raw_df.empty:
        raise ValueError(f"{source_path} is empty.")

    metadata_row = raw_df.iloc[0].fillna("").to_dict()
    data_df = raw_df.iloc[1:].reset_index(drop=True)
    data_df.replace({"": pd.NA, " ": pd.NA}, inplace=True)
    data_df = coerce_numeric_columns(data_df)

    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DATA_DIR / f"{dataset_id}.parquet"
    try:
        data_df.to_parquet(output_path, index=False)
    except ImportError as exc:
        fallback_path = OUTPUT_DATA_DIR / f"{dataset_id}.csv"
        data_df.to_csv(fallback_path, index=False)
        logger.warning(
            "pyarrow/fastparquet not available (%s). Wrote CSV fallback to %s.",
            exc,
            fallback_path,
        )
        output_path = fallback_path
    else:
        logger.info("Wrote cleaned data to %s", output_path)

    return output_path, metadata_row


def process_datasets(dataset_ids: Iterable[str]) -> None:
    metadata_store = load_metadata_store()
    for dataset_id in dataset_ids:
        output_path, metadata_row = preprocess_dataset(dataset_id)
        metadata_store[dataset_id] = {
            "source_file": DATASET_REGISTRY[dataset_id]["file"],
            "description": DATASET_REGISTRY[dataset_id]["description"],
            "columns": metadata_row,
        }
        logger.debug("Recorded metadata for %s (%d variables).", dataset_id, len(metadata_row))

    save_metadata_store(metadata_store)
    logger.info("Updated variable metadata at %s", METADATA_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CLES CSV datasets.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Dataset identifiers to process (default: all).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    dataset_ids = args.datasets or sorted(DATASET_REGISTRY.keys())
    logger.info("Target datasets: %s", ", ".join(dataset_ids))
    process_datasets(dataset_ids)


if __name__ == "__main__":
    main()

