from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

from utils.canonical_manifest import STANDARD_CANONICAL_ARTIFACTS, sha256_file, write_json
from utils.canonical_output import THREE_AXIS_COLUMNS, build_three_axis_summary


AUDIT_FILE_NAME = "canonical_output_audit.json"
MANIFEST_FILE_NAME = "run_manifest.json"

REQUIRED_MAIN_COLUMNS = (
    "scenario",
    "step",
    "year",
    "runtime_fallback_event_count",
    "runtime_fallback_status",
    "scri",
    "scri_status",
    "idi",
    "idi_status",
    "aesi",
    "aesi_status",
)
REQUIRED_SUMMARY_COLUMNS = (
    "scenario",
    "final_year",
    "final_step",
    *THREE_AXIS_COLUMNS,
    "scri_mean",
    "idi_mean",
    "aesi_mean",
)
BARE_BASELINE_NAMES = {"baseline", "Baseline"}
SCRI_COMPONENT_COLUMNS = (
    "scri_retained_marginal_effect",
    "scri_leakage_rate",
)
IDI_COMPONENT_COLUMNS = (
    "idi_smallholder_retention_component",
    "idi_exit_penalty_component",
    "idi_income_component",
    "idi_distribution_component",
    "idi_exit_state_component",
)
AESI_COMPONENT_COLUMNS = (
    "aesi_soil_component",
    "aesi_carbon_component",
    "aesi_nitrogen_component",
)


def _load_json(path: Path, failures: list[str]) -> Dict[str, Any]:
    if not path.exists():
        failures.append(f"missing file: {path.name}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"invalid json: {path.name}: {exc}")
        return {}


def _read_csv(path: Path, failures: list[str]) -> pd.DataFrame:
    if not path.exists():
        failures.append(f"missing file: {path.name}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        failures.append(f"invalid csv: {path.name}: {exc}")
        return pd.DataFrame()


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except TypeError:
        return value is None


def _values_match(left: Any, right: Any, *, tolerance: float = 1e-9) -> bool:
    if _is_missing(left) and _is_missing(right):
        return True
    try:
        left_f = float(left)
        right_f = float(right)
        if math.isnan(left_f) and math.isnan(right_f):
            return True
        return abs(left_f - right_f) <= tolerance
    except Exception:
        return str(left) == str(right)


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _numeric_series(
    df: pd.DataFrame,
    *column_names: str,
    default: float = 0.0,
) -> pd.Series:
    for name in column_names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series([default] * len(df), index=df.index, dtype="float64")


def _final_rows_from_main(main_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in main_df.groupby("scenario", dropna=False):
        sort_cols = [col for col in ("year", "step") if col in group.columns]
        if sort_cols:
            group = group.sort_values(sort_cols)
        row = group.iloc[-1].copy()
        row["scenario"] = scenario
        rows.append(row)
    return pd.DataFrame(rows)


def _check_required_files(output_dir: Path, failures: list[str]) -> None:
    for name in (MANIFEST_FILE_NAME, *STANDARD_CANONICAL_ARTIFACTS):
        if not (output_dir / name).exists():
            failures.append(f"missing canonical artifact: {name}")


def _check_manifest(
    output_dir: Path,
    manifest: Mapping[str, Any],
    failures: list[str],
    warnings: list[str],
) -> None:
    if not manifest:
        return
    if manifest.get("canonical_runtime") is not True:
        failures.append("manifest canonical_runtime is not true")
    if not manifest.get("run_type"):
        failures.append("manifest run_type is missing")
    if not manifest.get("script"):
        failures.append("manifest script is missing")
    if not manifest.get("claim_boundaries"):
        warnings.append("manifest claim_boundaries is empty")

    run_mode = str(manifest.get("run_mode", "") or "")
    if run_mode not in {"debug", "smoke", "formal"}:
        failures.append(f"manifest run_mode is invalid or missing: {run_mode!r}")
    run_parameters = manifest.get("run_parameters") or {}
    if not isinstance(run_parameters, Mapping):
        failures.append("manifest run_parameters must be an object")
        run_parameters = {}
    citation_readiness = manifest.get("citation_readiness") or {}
    if run_mode == "formal":
        steps = _as_int_or_none(run_parameters.get("steps"))
        if steps is None or steps < 30:
            failures.append("formal run requires run_parameters.steps >= 30")
        if str(run_parameters.get("scenario_set", "") or "").strip().lower() == "smoke":
            failures.append("formal run cannot use scenario_set=smoke")
        if _as_bool(run_parameters.get("workflow_lite")):
            failures.append("formal run cannot use workflow_lite outputs as citation-ready evidence")
        if citation_readiness.get("can_cite_numeric_results") is not True:
            failures.append("formal run citation_readiness.can_cite_numeric_results must be true")
    elif citation_readiness.get("can_cite_numeric_results") is True:
        warnings.append(f"{run_mode or 'unknown'} run is marked citation-ready; verify run_mode metadata")

    alias_audit = manifest.get("policy_alias_audit") or {}
    alias_conflicts = alias_audit.get("conflicts") or []
    if alias_audit.get("has_conflicts") or alias_conflicts:
        message = (
            "policy alias conflict: grain_subsidies_rate and legacy subsidy_rate differ; "
            "formal outputs must use grain_subsidies_rate as the canonical value"
        )
        if run_mode == "formal":
            failures.append(message)
        else:
            warnings.append(message)

    for scenario in manifest.get("scenarios", []):
        canonical = str(scenario.get("canonical_semantics", ""))
        name = str(scenario.get("name", ""))
        if canonical in BARE_BASELINE_NAMES or name in BARE_BASELINE_NAMES:
            failures.append(f"bare baseline scenario in manifest: {name or canonical}")

    artifacts = [str(name) for name in manifest.get("artifacts", [])]
    fingerprints = manifest.get("artifact_fingerprints", {}) or {}
    for artifact in artifacts:
        path = output_dir / artifact
        if not path.exists():
            failures.append(f"manifest artifact missing on disk: {artifact}")
            continue
        record = fingerprints.get(artifact)
        if record is None:
            warnings.append(f"artifact has no fingerprint: {artifact}")
            continue
        if record.get("exists") is not True:
            failures.append(f"artifact fingerprint records missing file: {artifact}")
            continue
        if path.is_file() and record.get("sha256") and record["sha256"] != sha256_file(path):
            failures.append(f"artifact sha256 mismatch: {artifact}")

    for name in STANDARD_CANONICAL_ARTIFACTS:
        if name not in artifacts:
            warnings.append(f"standard canonical artifact not listed in manifest: {name}")

    run_fingerprint = manifest.get("run_fingerprint") or {}
    if not run_fingerprint.get("config_sha256"):
        warnings.append("manifest run_fingerprint.config_sha256 is missing")
    git_info = run_fingerprint.get("git") or {}
    if git_info.get("available") and git_info.get("dirty"):
        warnings.append("git working tree was dirty when manifest was written")


def _check_main_indicators(
    main_df: pd.DataFrame,
    failures: list[str],
    warnings: list[str],
    *,
    run_mode: str,
) -> None:
    if main_df.empty:
        failures.append("main_indicators.csv is empty")
        return
    missing = [col for col in REQUIRED_MAIN_COLUMNS if col not in main_df.columns]
    if missing:
        failures.append(f"main_indicators.csv missing required columns: {missing}")
        return
    scenarios = {str(value) for value in main_df["scenario"].dropna().unique()}
    for scenario in scenarios:
        if scenario in BARE_BASELINE_NAMES:
            failures.append(f"bare baseline scenario in main_indicators.csv: {scenario}")

    fallback_counts = pd.to_numeric(main_df["runtime_fallback_event_count"], errors="coerce").fillna(0)
    fallback_total = int(fallback_counts.sum())
    fallback_status_dirty = (
        main_df["runtime_fallback_status"].astype(str).str.strip().str.lower() != "clean"
    ).any()
    if fallback_total > 0:
        message = (
            f"runtime fallback events present in main_indicators.csv: {fallback_total}; "
            "formal outputs require zero audited runtime fallbacks"
        )
        if run_mode == "formal":
            failures.append(message)
        else:
            warnings.append(message)
    elif fallback_status_dirty:
        warnings.append("runtime_fallback_status is not clean despite zero fallback event count")

    if "scri_status" in main_df.columns:
        computed_scri = main_df["scri_status"].astype(str).str.strip().str.lower() == "computed"
        missing_components = [col for col in SCRI_COMPONENT_COLUMNS if col not in main_df.columns]
        if computed_scri.any() and missing_components:
            failures.append(f"SCRI computed rows require component columns: {missing_components}")
        elif computed_scri.any():
            scri_values = _numeric_series(main_df, "scri", default=float("nan"))
            component_values = main_df.loc[computed_scri, list(SCRI_COMPONENT_COLUMNS)].apply(
                pd.to_numeric,
                errors="coerce",
            )
            if scri_values.loc[computed_scri].isna().any() or component_values.isna().any(axis=1).any():
                failures.append(
                    "SCRI computed rows contain missing SCRI value or components; "
                    "use scri_status=reference_required/no_fiscal_input when SCRI is incomplete"
                )
            total_subsidy = _numeric_series(main_df, "total_subsidy", default=0.0).fillna(0)
            if (computed_scri & (total_subsidy <= 0)).any():
                failures.append("SCRI computed rows require positive total_subsidy")

        no_fiscal_scri = main_df["scri_status"].astype(str).str.strip().str.lower() == "no_fiscal_input"
        if no_fiscal_scri.any():
            total_subsidy = _numeric_series(main_df, "total_subsidy", default=0.0).fillna(0)
            if (no_fiscal_scri & (total_subsidy > 0)).any():
                failures.append("SCRI no_fiscal_input rows cannot have positive total_subsidy")

    if "aesi_status" in main_df.columns and "aesi" in main_df.columns:
        incomplete = main_df["aesi_status"].astype(str).isin({"unavailable", "reference_required", "missing"})
        if incomplete.any() and (main_df.loc[incomplete, "aesi"].fillna(0) == 0).all():
            warnings.append("AESI has incomplete status with zero/NaN values; do not treat missing environmental axes as measured zero")
        computed_aesi = main_df["aesi_status"].astype(str).str.strip().str.lower() == "computed"
        missing_components = [col for col in AESI_COMPONENT_COLUMNS if col not in main_df.columns]
        if computed_aesi.any() and missing_components:
            failures.append(f"AESI computed rows require component columns: {missing_components}")
        elif computed_aesi.any():
            aesi_values = _numeric_series(main_df, "aesi", default=float("nan"))
            component_values = main_df.loc[computed_aesi, list(AESI_COMPONENT_COLUMNS)].apply(
                pd.to_numeric,
                errors="coerce",
            )
            if aesi_values.loc[computed_aesi].isna().any() or component_values.isna().any(axis=1).any():
                failures.append(
                    "AESI computed rows require finite soil, carbon, and nitrogen components; "
                    "use aesi_status=soil_only/partial when environmental axes are incomplete"
                )

    if "idi_status" in main_df.columns:
        computed_idi = main_df["idi_status"].astype(str).str.strip().str.lower() == "computed"
        missing_components = [col for col in IDI_COMPONENT_COLUMNS if col not in main_df.columns]
        if computed_idi.any() and missing_components:
            failures.append(f"IDI computed rows require component columns: {missing_components}")
        elif computed_idi.any():
            component_values = main_df.loc[computed_idi, list(IDI_COMPONENT_COLUMNS)].apply(
                pd.to_numeric,
                errors="coerce",
            )
            incomplete_rows = component_values.isna().any(axis=1)
            if incomplete_rows.any():
                failures.append(
                    "IDI computed rows contain missing components; "
                    "use idi_status=partial/unavailable when inclusion evidence is incomplete"
                )

            n_farmers = _numeric_series(main_df, "n_farmers", "farmer_count", default=0.0).fillna(0)
            if {"n_pure_farmer", "n_pluriactive", "n_non_farmer"} <= set(main_df.columns):
                counted = (
                    pd.to_numeric(main_df["n_pure_farmer"], errors="coerce").fillna(0)
                    + pd.to_numeric(main_df["n_pluriactive"], errors="coerce").fillna(0)
                    + pd.to_numeric(main_df["n_non_farmer"], errors="coerce").fillna(0)
                )
            else:
                counted = pd.Series([0] * len(main_df), index=main_df.index)
            retention = _numeric_series(
                main_df,
                "idi_smallholder_retention_component",
                default=float("nan"),
            )
            pure_rate = _numeric_series(main_df, "pure_farmer_rate", default=float("nan"))
            fake_zero_retention = (
                computed_idi
                & (n_farmers > 0)
                & (retention.fillna(-1) == 0)
                & (counted != n_farmers)
                & (pure_rate.fillna(0) <= 0)
            )
            if fake_zero_retention.any():
                failures.append(
                    "IDI computed rows report zero smallholder retention without closed livelihood counts; "
                    "do not treat default zero pure_farmer fields as observed inclusion evidence"
                )

            household_net_income = _numeric_series(
                main_df,
                "all_household_net_income",
                default=0.0,
            ).fillna(0)
            income_component = _numeric_series(
                main_df,
                "idi_income_component",
                default=float("nan"),
            )
            fake_income_component = computed_idi & income_component.notna() & (household_net_income <= 0)
            if fake_income_component.any():
                failures.append(
                    "IDI computed rows contain an income component without positive all_household_net_income"
                )


def _check_three_axis_summary(
    main_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    summary_json: Mapping[str, Any],
    failures: list[str],
) -> None:
    if main_df.empty or summary_df.empty:
        return
    missing = [col for col in REQUIRED_SUMMARY_COLUMNS if col not in summary_df.columns]
    if missing:
        failures.append(f"three_axis_summary.csv missing required columns: {missing}")
        return

    main_scenarios = {str(value) for value in main_df["scenario"].dropna().unique()}
    summary_scenarios = {str(value) for value in summary_df["scenario"].dropna().unique()}
    if main_scenarios != summary_scenarios:
        failures.append(f"scenario mismatch between main indicators and summary: {main_scenarios} vs {summary_scenarios}")

    expected = build_three_axis_summary(main_df)
    expected_by_scenario = {str(row["scenario"]): row for row in expected.to_dict(orient="records")}
    actual_by_scenario = {str(row["scenario"]): row for row in summary_df.to_dict(orient="records")}
    for scenario, expected_row in expected_by_scenario.items():
        actual_row = actual_by_scenario.get(scenario)
        if actual_row is None:
            continue
        for column in ("final_year", "final_step", *THREE_AXIS_COLUMNS, "scri_mean", "idi_mean", "aesi_mean"):
            if column in expected_row and column in actual_row and not _values_match(expected_row[column], actual_row[column]):
                failures.append(f"three_axis_summary.csv mismatch for {scenario}.{column}")

    if summary_json and summary_json.get("schema_version") != "v1":
        failures.append("three_axis_summary.json schema_version must be v1")

    rows = summary_json.get("rows", []) if summary_json else []
    if not isinstance(rows, list):
        failures.append("three_axis_summary.json rows must be a list")
        return
    if len(rows) != len(summary_df):
        failures.append("three_axis_summary.json row count does not match csv")
        return
    summary_csv_by_scenario = {
        str(row["scenario"]): row
        for row in summary_df.to_dict(orient="records")
    }
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            failures.append(f"three_axis_summary.json row {index} must be an object")
            continue
        missing_json_keys = [col for col in REQUIRED_SUMMARY_COLUMNS if col not in row]
        if missing_json_keys:
            failures.append(f"three_axis_summary.json row {index} missing required keys: {missing_json_keys}")
            continue
        scenario = str(row.get("scenario"))
        csv_row = summary_csv_by_scenario.get(scenario)
        if csv_row is None:
            failures.append(f"three_axis_summary.json row {index} scenario not found in csv: {scenario}")
            continue
        for column in REQUIRED_SUMMARY_COLUMNS:
            if not _values_match(row.get(column), csv_row.get(column)):
                failures.append(f"three_axis_summary.json mismatch for {scenario}.{column}")


def validate_canonical_output(output_dir: Path | str) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    failures: list[str] = []
    warnings: list[str] = []

    _check_required_files(output_dir, failures)
    manifest = _load_json(output_dir / MANIFEST_FILE_NAME, failures)
    _check_manifest(output_dir, manifest, failures, warnings)
    run_mode = str(manifest.get("run_mode", "") or "")

    main_df = _read_csv(output_dir / "main_indicators.csv", failures)
    summary_df = _read_csv(output_dir / "three_axis_summary.csv", failures)
    summary_json = _load_json(output_dir / "three_axis_summary.json", failures)

    _check_main_indicators(main_df, failures, warnings, run_mode=run_mode)
    _check_three_axis_summary(main_df, summary_df, summary_json, failures)

    status = "fail" if failures else "warn" if warnings else "pass"
    return {
        "schema_version": "v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "checked_files": [MANIFEST_FILE_NAME, *STANDARD_CANONICAL_ARTIFACTS],
    }


def write_canonical_output_audit(output_dir: Path | str) -> Path:
    output_dir = Path(output_dir)
    report = validate_canonical_output(output_dir)
    return write_json(output_dir / AUDIT_FILE_NAME, report)
