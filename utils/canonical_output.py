from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml


CANONICAL_OUTPUT_ROOT = Path("results") / "canonical"

THREE_AXIS_COLUMNS = [
    "scri",
    "scri_status",
    "scri_retained_marginal_effect",
    "scri_leakage_rate",
    "idi",
    "idi_status",
    "idi_smallholder_retention_component",
    "idi_exit_penalty_component",
    "idi_income_component",
    "idi_distribution_component",
    "idi_exit_state_component",
    "aesi",
    "aesi_status",
    "aesi_soil_component",
    "aesi_carbon_component",
    "aesi_nitrogen_component",
]


def canonical_run_type(run_type: str) -> str:
    return str(run_type).split(":", 1)[0]


def canonical_output_dir(
    run_type: str,
    *,
    timestamp: str | None = None,
    root: Path | str = CANONICAL_OUTPUT_ROOT,
) -> Path:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(root) / canonical_run_type(run_type) / stamp


def write_config_effective(output_dir: Path, config: Mapping[str, Any]) -> Path:
    path = Path(output_dir) / "config_effective.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(config), f, allow_unicode=True, sort_keys=False)
    return path


def _history_mapping(histories: Any, *, default_scenario: str = "policy_baseline") -> dict[str, Any]:
    if histories is None:
        return {}
    if isinstance(histories, pd.DataFrame):
        return {default_scenario: histories}
    if isinstance(histories, Mapping):
        return {str(key): value for key, value in histories.items()}
    return {default_scenario: histories}


def indicators_histories_to_dataframe(
    histories: Any,
    *,
    default_scenario: str = "policy_baseline",
) -> pd.DataFrame:
    from visualization.unified_indicators import indicators_to_dataframe

    frames: list[pd.DataFrame] = []
    for scenario, history in _history_mapping(histories, default_scenario=default_scenario).items():
        if history is None:
            continue
        if isinstance(history, pd.DataFrame):
            df = history.copy()
        else:
            items = list(history)
            if not items:
                continue
            df = indicators_to_dataframe(items)
        if "scenario" not in df.columns:
            df["scenario"] = scenario
        else:
            df["scenario"] = df["scenario"].fillna(scenario)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["scenario", "step", "year", *THREE_AXIS_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def write_main_indicators(
    output_dir: Path,
    histories: Any,
    *,
    default_scenario: str = "policy_baseline",
) -> Path:
    df = indicators_histories_to_dataframe(histories, default_scenario=default_scenario)
    path = Path(output_dir) / "main_indicators.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _last_row(group: pd.DataFrame) -> pd.Series:
    sort_cols = [col for col in ("year", "step") if col in group.columns]
    if sort_cols:
        group = group.sort_values(sort_cols)
    return group.iloc[-1]


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return _finite_or_none(value.item())
    return value


def build_three_axis_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scenario",
        "final_year",
        "final_step",
        *THREE_AXIS_COLUMNS,
        "scri_mean",
        "idi_mean",
        "aesi_mean",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    source = df.copy()
    if "scenario" not in source.columns:
        source["scenario"] = "policy_baseline"

    rows: list[dict[str, Any]] = []
    for scenario, group in source.groupby("scenario", dropna=False):
        final = _last_row(group)
        row: dict[str, Any] = {
            "scenario": str(scenario),
            "final_year": final.get("year"),
            "final_step": final.get("step"),
        }
        for col in THREE_AXIS_COLUMNS:
            row[col] = final.get(col)
        for col in ("scri", "idi", "aesi"):
            if col in group.columns:
                row[f"{col}_mean"] = group[col].mean(skipna=True)
            else:
                row[f"{col}_mean"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def write_three_axis_summary(
    output_dir: Path,
    histories_or_dataframe: Any,
    *,
    default_scenario: str = "policy_baseline",
) -> tuple[Path, Path]:
    if isinstance(histories_or_dataframe, pd.DataFrame):
        df = histories_or_dataframe.copy()
    else:
        df = indicators_histories_to_dataframe(histories_or_dataframe, default_scenario=default_scenario)

    summary = build_three_axis_summary(df)
    output_dir = Path(output_dir)
    csv_path = output_dir / "three_axis_summary.csv"
    json_path = output_dir / "three_axis_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "schema_version": "v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status_note": (
            "SCRI/IDI/AESI are model-output diagnostics. SCRI may be reference_required; "
            "AESI may be soil_only when carbon/nitrogen data are unavailable."
        ),
        "rows": [
            {key: _finite_or_none(value) for key, value in row.items()}
            for row in summary.to_dict(orient="records")
        ],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return csv_path, json_path


def write_canonical_output_contract(
    output_dir: Path,
    *,
    run_type: str,
    config: Mapping[str, Any] | None = None,
    histories: Any = None,
    default_scenario: str = "policy_baseline",
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[str] = []
    if config is not None:
        artifacts.append(write_config_effective(output_dir, config).name)

    main_path = write_main_indicators(output_dir, histories, default_scenario=default_scenario)
    summary_csv, summary_json = write_three_axis_summary(
        output_dir,
        main_path_to_dataframe(main_path),
        default_scenario=default_scenario,
    )
    artifacts.extend([main_path.name, summary_csv.name, summary_json.name])
    return artifacts


def main_path_to_dataframe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
