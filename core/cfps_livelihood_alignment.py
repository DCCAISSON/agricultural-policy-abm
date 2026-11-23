from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL_PATH = PROJECT_ROOT / "results" / "cfps_calibration" / "panel_tracking_full.csv"
DEFAULT_TARGETS_PATH = PROJECT_ROOT / "config" / "calibration_targets_dual_source.json"
CANONICAL_LABELS = ("Pure_Farmer", "Pluriactive", "Non_Farmer", "Unknown")


def normalize_record_id(value: Any) -> Optional[str]:
    """Normalize household ids so raw CSV/JSON ids can be joined safely."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None

    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return text

    if pd.isna(numeric):
        return None
    if abs(numeric - round(numeric)) < 1e-9:
        return str(int(round(numeric)))
    return text


def _resolve_panel_path(panel_path: Optional[str | Path] = None) -> Path:
    path = Path(panel_path) if panel_path is not None else DEFAULT_PANEL_PATH
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_config_distribution(
    year: int,
    include_unknown: bool,
    targets_path: Optional[str | Path] = None,
) -> Dict[str, float]:
    path = Path(targets_path) if targets_path is not None else DEFAULT_TARGETS_PATH
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    livelihood_cfg = (payload or {}).get("livelihood", {}) or {}
    year_cfg = livelihood_cfg.get(f"crosssection_targets_{int(year)}", {}) or {}
    if year_cfg:
        known = {
            "Pure_Farmer": float(year_cfg.get("pure_farmer_rate", 0.0) or 0.0),
            "Pluriactive": float(year_cfg.get("pluriactive_rate", 0.0) or 0.0),
            "Non_Farmer": float(year_cfg.get("non_farmer_rate", 0.0) or 0.0),
        }
        total = sum(known.values())
        if total > 0:
            if include_unknown:
                unknown = max(0.0, 1.0 - total)
                return {
                    "Pure_Farmer": known["Pure_Farmer"],
                    "Pluriactive": known["Pluriactive"],
                    "Non_Farmer": known["Non_Farmer"],
                    "Unknown": unknown,
                }
            return {label: value / total for label, value in known.items()}

    init_cfg = (payload or {}).get("abm_initialization", {}) or {}
    dist_cfg = init_cfg.get("livelihood_distribution", {}) or {}
    if not dist_cfg:
        return {}

    known = {
        "Pure_Farmer": float(dist_cfg.get("Pure_Farmer", 0.0) or 0.0),
        "Pluriactive": float(dist_cfg.get("Pluriactive", 0.0) or 0.0),
        "Non_Farmer": float(dist_cfg.get("Non_Farmer", 0.0) or 0.0),
    }
    if include_unknown:
        known["Unknown"] = float(dist_cfg.get("Unknown", 0.0) or 0.0)
        return known

    total = sum(known.values())
    if total <= 0:
        return {}
    return {label: value / total for label, value in known.items()}


def _load_panel_slice(
    panel_path: Optional[str | Path] = None,
    year: Optional[int] = None,
) -> pd.DataFrame:
    resolved_path = _resolve_panel_path(panel_path)
    if not resolved_path.exists():
        return pd.DataFrame()

    usecols = lambda column: column in {"fid", "year", "livelihood_type", "urban_status"}
    df = pd.read_csv(resolved_path, usecols=usecols)
    df = df[df["livelihood_type"].isin(CANONICAL_LABELS)].copy()
    if "urban_status" in df.columns:
        df = df[df["urban_status"] == 0].copy()
    if year is not None:
        df = df[df["year"] == year].copy()
    return df


def load_canonical_livelihood_map(
    panel_path: Optional[str | Path] = None,
    year: int = 2010,
) -> Dict[str, str]:
    df = _load_panel_slice(panel_path=panel_path, year=year)
    if df.empty:
        return {}

    df["__fid_key__"] = df["fid"].map(normalize_record_id)
    df = df.dropna(subset=["__fid_key__"]).drop_duplicates(subset=["__fid_key__"], keep="first")
    return dict(zip(df["__fid_key__"], df["livelihood_type"]))


def _distribution_from_slice(df: pd.DataFrame, include_unknown: bool) -> Dict[str, float]:
    counts = df["livelihood_type"].value_counts(normalize=True)
    if include_unknown:
        return {label: float(counts.get(label, 0.0)) for label in CANONICAL_LABELS}

    known = {
        "Pure_Farmer": float(counts.get("Pure_Farmer", 0.0)),
        "Pluriactive": float(counts.get("Pluriactive", 0.0)),
        "Non_Farmer": float(counts.get("Non_Farmer", 0.0)),
    }
    total = sum(known.values())
    if total <= 0:
        return {}
    return {label: value / total for label, value in known.items()}


def load_canonical_livelihood_distribution(
    panel_path: Optional[str | Path] = None,
    year: int = 2010,
    include_unknown: bool = False,
) -> Dict[str, float]:
    df = _load_panel_slice(panel_path=panel_path, year=year)
    if df.empty:
        return _load_config_distribution(year=year, include_unknown=include_unknown)
    return _distribution_from_slice(df, include_unknown=include_unknown)


def load_canonical_livelihood_distribution_by_year(
    panel_path: Optional[str | Path] = None,
    years: Optional[Iterable[int]] = None,
    include_unknown: bool = False,
) -> Dict[int, Dict[str, float]]:
    df = _load_panel_slice(panel_path=panel_path, year=None)
    if df.empty:
        if years is None:
            years = [2010]
        return {
            int(year): _load_config_distribution(year=int(year), include_unknown=include_unknown)
            for year in years
            if _load_config_distribution(year=int(year), include_unknown=include_unknown)
        }

    if years is not None:
        years = {int(year) for year in years}
        df = df[df["year"].isin(years)].copy()

    distributions: Dict[int, Dict[str, float]] = {}
    for year, year_df in df.groupby("year"):
        distributions[int(year)] = _distribution_from_slice(
            year_df, include_unknown=include_unknown
        )
    return distributions


def align_feature_library_livelihoods(
    library: Dict[str, Any],
    panel_path: Optional[str | Path] = None,
    year: int = 2010,
) -> Dict[str, Any]:
    farmers = list((library or {}).get("farmers", []) or [])
    if not farmers:
        return {"matched": 0, "changed": 0, "missing": 0, "distribution": {}}

    canonical_map = load_canonical_livelihood_map(panel_path=panel_path, year=year)
    if not canonical_map:
        return {"matched": 0, "changed": 0, "missing": len(farmers), "distribution": {}}

    matched = 0
    changed = 0
    missing = 0

    for farmer in farmers:
        record_id = normalize_record_id(farmer.get("id"))
        if record_id is None:
            missing += 1
            continue
        canonical_label = canonical_map.get(record_id)
        if canonical_label is None:
            missing += 1
            continue
        matched += 1
        if farmer.get("livelihood") != canonical_label:
            farmer["livelihood"] = canonical_label
            changed += 1

    stats = library.setdefault("statistics", {})
    livelihood_series = pd.Series([farmer.get("livelihood") for farmer in farmers], dtype="object")
    livelihood_series = livelihood_series[livelihood_series.isin(CANONICAL_LABELS)]
    counts = livelihood_series.value_counts(normalize=True)
    stats["n_total"] = len(farmers)
    stats["livelihood_distribution"] = {
        label: float(counts.get(label, 0.0))
        for label in CANONICAL_LABELS
        if float(counts.get(label, 0.0)) > 0.0
    }

    metadata = library.setdefault("metadata", {})
    metadata["livelihood_label_source"] = str(_resolve_panel_path(panel_path))
    metadata["livelihood_label_year"] = int(year)
    metadata["livelihood_label_alignment"] = {
        "matched": matched,
        "changed": changed,
        "missing": missing,
    }

    return {
        "matched": matched,
        "changed": changed,
        "missing": missing,
        "distribution": stats.get("livelihood_distribution", {}),
    }
