from __future__ import annotations

import csv
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from scripts.v4_8_national_task5_core import (
    PROJECT_ROOT,
    REQUIRED_GROUPS,
    Moment,
    Parameter,
    Task5Error,
    find_project_file,
    read_csv_rows,
)


RUNTIME_SURFACE: Final = "ModelBuilder.build+MainModel.step"
RAW_RUNTIME_FIELDS: Final = (
    "runtime_seed",
    "scenario_id",
    "simulation_year",
    "year_index",
    "runtime_surface",
    "runtime_input_hash",
    "initial_state_sha256",
    "trace_artifact",
    "trace_row",
)
TARGET_RUNTIME_PROVENANCE_FIELDS: Final = (
    "runtime_observation_count",
    "runtime_provenance_path",
    "comparison_window",
    "runtime_seed",
    "scenario_id",
    "simulation_year",
    "runtime_surface",
    "runtime_input_hash",
    "trace_artifact",
    "trace_row",
)
LEGACY_PAIRED_OUTCOME_FIELDS: Final = (
    "grain_value",
    "welfare_value",
    "fiscal_efficiency_value",
    "land_rent_value",
    "transfer_rate_value",
)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_worklog(output: Path, phase: str, detail: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    line = f"{stamp},{phase},{detail}\n"
    for root in (PROJECT_ROOT / ".omo" / "evidence" / "task-5-v4_8_r1", PROJECT_ROOT / "project_management" / "evidence" / "v4_8" / "task-5"):
        root.mkdir(parents=True, exist_ok=True)
        with (root / "WORKLOG.csv").open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
    with (output / "worklog.csv").open("a", encoding="utf-8", newline="") as handle:
        handle.write(line)


def _finite(row: dict[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise Task5Error(f"{context}: missing or invalid {field}") from exc
    if not math.isfinite(value):
        raise Task5Error(f"{context}: nonfinite {field}")
    return value


def require_raw_runtime_observation(row: dict[str, str], context: str) -> None:
    missing = [field for field in RAW_RUNTIME_FIELDS if not row.get(field, "").strip()]
    if missing:
        raise Task5Error(f"{context}: missing raw runtime fields: {','.join(missing)}")
    try:
        seed = int(row["runtime_seed"])
        year = int(row["simulation_year"])
        year_index = int(row["year_index"])
    except ValueError as exc:
        raise Task5Error(f"{context}: invalid raw runtime identifier") from exc
    if seed < 0 or year < 1 or year_index < 0:
        raise Task5Error(f"{context}: invalid raw runtime identifier")
    if row["runtime_surface"] != RUNTIME_SURFACE:
        raise Task5Error(f"{context}: runtime surface must be {RUNTIME_SURFACE}")


def require_target_runtime_provenance(row: dict[str, str], context: str) -> None:
    missing = [field for field in TARGET_RUNTIME_PROVENANCE_FIELDS if not row.get(field, "").strip()]
    if missing:
        raise Task5Error(f"{context}: missing target runtime provenance fields: {','.join(missing)}")
    try:
        observation_count = int(row["runtime_observation_count"])
    except ValueError as exc:
        raise Task5Error(f"{context}: invalid runtime_observation_count") from exc
    if observation_count < 1:
        raise Task5Error(f"{context}: runtime_observation_count must be positive")
    provenance_path = Path(row["runtime_provenance_path"])
    if provenance_path.is_absolute() or ".." in provenance_path.parts:
        raise Task5Error(f"{context}: runtime_provenance_path must stay within the result package")
    if row["runtime_surface"] != RUNTIME_SURFACE:
        raise Task5Error(f"{context}: runtime surface must be {RUNTIME_SURFACE}")


def _validated_group_rows(rows: list[dict[str, str]], group: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("required_group") == group]
    if not selected:
        raise Task5Error(f"empty required validation group: {group}")
    for row in selected:
        moment_id = row.get("moment_id", "unknown")
        context = f"{group}/{moment_id}"
        require_target_runtime_provenance(row, context)
        target = _finite(row, "target_value", context)
        weight = _finite(row, "weight", context)
        _finite(row, "simulated_value", context)
        if target <= 0.0:
            raise Task5Error(f"{context}: nonpositive target_value")
        if weight <= 0.0:
            raise Task5Error(f"{context}: nonpositive weight")
    return selected


def group_error(rows: list[dict[str, str]], group: str, metric: str) -> float:
    selected = _validated_group_rows(rows, group)
    if metric == "WAPE":
        numerator = sum(abs(_finite(row, "simulated_value", group) - _finite(row, "target_value", group)) for row in selected)
        denominator = sum(abs(_finite(row, "target_value", group)) for row in selected)
        return numerator / denominator if denominator > 0.0 else math.inf
    if metric == "MAPE":
        errors = [abs(_finite(row, "simulated_value", group) - _finite(row, "target_value", group)) / abs(_finite(row, "target_value", group)) for row in selected]
        return sum(errors) / len(errors)
    raise Task5Error(f"unsupported validation metric: {metric}")


def validation_metrics(target_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metrics: list[dict[str, str]] = []
    observed_groups = {row.get("required_group", "") for row in target_rows}
    if observed_groups != set(REQUIRED_GROUPS):
        raise Task5Error(f"required validation group coverage mismatch: {sorted(observed_groups)}")
    for group in REQUIRED_GROUPS:
        primary = "MAPE" if group.startswith("N_CFPS") else "WAPE"
        threshold = 0.25 if group == "N_CFPS_PURE_FARMER" else 0.22 if group == "N_CFPS_PLURIACTIVE" else 0.15 if group == "N_CFPS_NON_FARMER" else 0.20
        value = group_error(target_rows, group, primary)
        metrics.append(
            {
                "required_group": group,
                "primary_error_metric": primary,
                "primary_error_metric_value": f"{value:.8f}",
                "threshold": f"{threshold:.8f}",
                "gate_passed": str(value <= threshold).lower(),
                "non_compensatory": "true",
            }
        )
    return metrics


def paired_worldline_rows(raw_runtime_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    paired: list[dict[str, str]] = []
    for raw in raw_runtime_rows:
        if raw.get("year_index") == "0":
            continue
        context = f"paired seed={raw.get('runtime_seed', 'unknown')} year={raw.get('simulation_year', 'unknown')}"
        require_raw_runtime_observation(raw, context)
        for field in LEGACY_PAIRED_OUTCOME_FIELDS:
            _finite(raw, field, context)
        paired.append(
            {
                **{field: raw[field] for field in RAW_RUNTIME_FIELDS},
                "pairing_method": "same_seed_initial_state",
                **{field: raw[field] for field in LEGACY_PAIRED_OUTCOME_FIELDS},
            }
        )
    if not paired:
        raise Task5Error("paired worldlines require at least one MainModel raw runtime observation")
    return paired


def worldline_rows(seeds: range, years: int, params: list[Parameter], moments: list[Moment]) -> list[dict[str, str]]:
    raise Task5Error("formula-generated paired worldlines are prohibited; use paired_worldline_rows with MainModel raw runtime observations")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes() -> dict[str, str]:
    rows = read_csv_rows(PROJECT_ROOT / "project_management" / "V4_8_PROTECTED_CONFIG_HASHES_20260811.csv")
    return {row["path"]: hash_file(PROJECT_ROOT / row["path"]) for row in rows}


def source_data_rows() -> list[dict[str, str]]:
    nfps_path = find_project_file("2009-2015household.csv")
    cfps_path = find_project_file("cfps_calibration_summary_v6_unified.csv")
    return [
        {"source_id": "NFPS_2009_2015_household", "path": str(nfps_path.relative_to(PROJECT_ROOT)), "sha256": hash_file(nfps_path), "role": "fitting_2009_2012_validation_2013_2015"},
        {"source_id": "CFPS_livelihood_v6", "path": str(cfps_path.relative_to(PROJECT_ROOT)), "sha256": hash_file(cfps_path), "role": "fitting_2010_2016_validation_2018_2022"},
    ]
