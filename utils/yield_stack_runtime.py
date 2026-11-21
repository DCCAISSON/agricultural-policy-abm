"""
@yield_mode legacy_stack (ACTIVE)

This is the CURRENT ACTIVE yield path. It computes final yield by multiplying
a base yield through a stack of modifiers (technology, scale, soil, service,
irrigation, disaster, climate, random). This is NOT a calibrated production
function (Cobb-Douglas or otherwise).

@provenance The multiplier values are a mix of:
  - auxiliary driver panel data (irrigation, disaster) from NBS
  - config values from master_config.yaml (technology, scale, soil, service)
  - some values may be stipulated rather than data-estimated

@diagnostic SHADOW mode: production_function_shadow or cobb_douglas.shadow_compare_enabled
  - Cobb-Douglas: Y = A * Land^alpha * Labor^beta * Capital^gamma * exp(climate)
  - Uses NFPS-estimated elasticities from config/master_config.yaml
  - Runs in parallel with legacy_stack; outputs difference decomposition
  - Does not replace the returned active yield

@planned ACTIVE mode: production_function_active
  - Would replace legacy_stack as the primary yield path
  - Only after shadow output is stable and validated against NFPS moments
  - NOT IMPLEMENTED

config entry: production.yield_mode in master_config.yaml
  - legacy_stack (current, default)
  - production_function_shadow (diagnostic shadow trace)
  - production_function_active (not implemented)
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVER_PANEL = (
    PROJECT_ROOT
    / "data"
    / "official_grain_anchors"
    / "national_auxiliary_driver_panel.csv"
)


def crop_key(crop_type: Any) -> str:
    value = str(getattr(crop_type, "value", crop_type) or "").strip().lower()
    if value in {"grain", "crop_type.grain"} or "grain" in value:
        return "grain"
    if "cash" in value:
        return "cash_crop"
    return value or "unknown"


def _finite_float(value: Any, default: float = math.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _get_mapping(root: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    value = root.get(key, {}) if isinstance(root, Mapping) else {}
    return value if isinstance(value, Mapping) else {}


def _get_model_config(model: Any) -> Mapping[str, Any]:
    config = getattr(model, "config", {}) if model is not None else {}
    return config if isinstance(config, Mapping) else {}


def _get_yield_stack_config(model: Any) -> Mapping[str, Any]:
    production = _get_mapping(_get_model_config(model), "production")
    stack = _get_mapping(production, "yield_stack")
    return stack


def _get_forecast_context(model: Any) -> Mapping[str, Any]:
    config = _get_model_config(model)
    candidates = (
        getattr(model, "forecast_context", None),
        _get_mapping(config, "forecast_context"),
        _get_mapping(config, "forecasting"),
        _get_mapping(_get_yield_stack_config(model), "forecast_context"),
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def resolve_forecast_origin_year(model: Any) -> int | None:
    """Return the forecast origin year when an origin-aware audit supplies one."""
    for value in (
        getattr(model, "forecast_origin_year", None),
        _get_forecast_context(model).get("forecast_origin_year"),
        _get_forecast_context(model).get("origin_year"),
    ):
        year = _finite_float(value)
        if math.isfinite(year) and year > 1900:
            return int(year)
    return None


def resolve_as_of_year(model: Any) -> int | None:
    """Return the last year whose auxiliary drivers are allowed to be observed."""
    for value in (
        getattr(model, "as_of_year", None),
        _get_forecast_context(model).get("as_of_year"),
        resolve_forecast_origin_year(model),
    ):
        year = _finite_float(value)
        if math.isfinite(year) and year > 1900:
            return int(year)
    return None


def resolve_model_year(model: Any) -> int:
    for attr in ("current_year", "year"):
        value = getattr(model, attr, None)
        year = _finite_float(value)
        if math.isfinite(year) and year > 1900:
            return int(year)
    start_year = int(
        _finite_float(
            _get_mapping(_get_model_config(model), "experiment").get("start_year", 2010),
            2010,
        )
    )
    step = int(_finite_float(getattr(model, "step_count", 0), 0))
    return start_year + step


@lru_cache(maxsize=8)
def _load_auxiliary_panel(path_text: str) -> tuple[dict[str, float], ...]:
    path = Path(path_text)
    if not path.exists():
        return tuple()
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            year = _finite_float(row.get("year"))
            if not math.isfinite(year):
                continue
            rows.append(
                {
                    "year": float(int(year)),
                    "effective_irrigation_to_grain_sown_area_ratio": _finite_float(
                        row.get("effective_irrigation_to_grain_sown_area_ratio")
                    ),
                    "crop_disaster_affected_to_grain_sown_area_ratio": _finite_float(
                        row.get("crop_disaster_affected_to_grain_sown_area_ratio")
                    ),
                }
            )
    return tuple(sorted(rows, key=lambda item: item["year"]))


def _driver_panel_path(model: Any) -> Path:
    stack = _get_yield_stack_config(model)
    panel_cfg = _get_mapping(stack, "auxiliary_driver_panel")
    raw = panel_cfg.get("path", str(DEFAULT_DRIVER_PANEL))
    path = Path(str(raw))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _runtime_max_observed_year(model: Any) -> int:
    as_of_year = resolve_as_of_year(model)
    if as_of_year is not None:
        return int(as_of_year)
    stack = _get_yield_stack_config(model)
    panel_cfg = _get_mapping(stack, "auxiliary_driver_panel")
    return int(_finite_float(panel_cfg.get("max_runtime_observed_year", 2021), 2021))


def _select_driver_record(model: Any, year: int) -> tuple[dict[str, float] | None, str]:
    panel_path = _driver_panel_path(model)
    rows = list(_load_auxiliary_panel(str(panel_path)))
    if not rows:
        return None, "missing_auxiliary_driver_panel"
    capped_year = min(int(year), _runtime_max_observed_year(model))
    usable = [row for row in rows if int(row["year"]) <= capped_year]
    if not usable:
        usable = [rows[0]]
    selected = usable[-1]
    mode = "observed_training_year" if int(selected["year"]) == int(year) else "training_hold_forward"
    if int(year) < int(rows[0]["year"]):
        mode = "earliest_training_year_backcast"
    return dict(selected), mode


def _rows_through_as_of(model: Any, as_of_year: int) -> list[dict[str, float]]:
    rows = list(_load_auxiliary_panel(str(_driver_panel_path(model))))
    return [row for row in rows if int(row["year"]) <= int(as_of_year)]


def _row_at_or_before(rows: list[dict[str, float]], year: int) -> dict[str, float] | None:
    usable = [row for row in rows if int(row["year"]) <= int(year)]
    return usable[-1] if usable else (rows[0] if rows else None)


def _clip_ratio(value: float) -> float:
    if not math.isfinite(value):
        return value
    return max(0.0, min(1.0, value))


def _linear_project(rows: list[dict[str, float]], column: str, target_year: int) -> tuple[float, str]:
    values = [
        (float(row["year"]), _finite_float(row.get(column)))
        for row in rows
        if math.isfinite(_finite_float(row.get(column)))
    ]
    if not values:
        return math.nan, "training_trend_unavailable_no_finite_values"
    if int(target_year) <= int(values[-1][0]):
        selected = _row_at_or_before(
            [{"year": year, column: value} for year, value in values],
            int(target_year),
        )
        if selected is not None:
            return _finite_float(selected.get(column)), "observed_as_of_year"
    if len(values) < 2:
        return values[-1][1], "training_trend_fallback_hold_forward_insufficient_history"
    x = [item[0] for item in values]
    y = [item[1] for item in values]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denom = sum((item - x_mean) ** 2 for item in x)
    if denom <= 0:
        return y[-1], "training_trend_fallback_hold_forward_zero_variance"
    slope = sum((year - x_mean) * (value - y_mean) for year, value in values) / denom
    projected = y_mean + slope * (float(target_year) - x_mean)
    return _clip_ratio(float(projected)), "training_trend_projected_from_as_of_history"


def _policy_target_project(
    model: Any,
    *,
    rows: list[dict[str, float]],
    column: str,
    target_year: int,
    as_of_year: int,
) -> tuple[float, str]:
    stack = _get_yield_stack_config(model)
    target_cfg = _get_mapping(stack, "projection_policy_targets")
    column_cfg = _get_mapping(target_cfg, column)
    if not column_cfg:
        value, status = _linear_project(rows, column, target_year)
        return value, f"policy_target_missing_fallback_{status}"
    target_value = _finite_float(column_cfg.get("target_value"))
    target_year_cfg = int(_finite_float(column_cfg.get("target_year", target_year), target_year))
    if not math.isfinite(target_value):
        value, status = _linear_project(rows, column, target_year)
        return value, f"policy_target_value_missing_fallback_{status}"
    origin_row = _row_at_or_before(rows, as_of_year)
    origin_value = _finite_float(origin_row.get(column) if origin_row is not None else math.nan)
    if not math.isfinite(origin_value) or target_year_cfg <= as_of_year:
        return _clip_ratio(target_value), "policy_target_direct"
    share = max(0.0, min(1.0, (float(target_year) - float(as_of_year)) / (target_year_cfg - as_of_year)))
    projected = origin_value + (target_value - origin_value) * share
    return _clip_ratio(float(projected)), "policy_target_interpolated_from_as_of_year"


def project_auxiliary_driver_value(
    model: Any,
    *,
    column: str,
    target_year: int,
    as_of_year: int | None = None,
    projection_mode: str = "hold_forward",
) -> dict[str, Any]:
    """Project one auxiliary driver using only information visible at as_of_year."""
    resolved_as_of = int(as_of_year if as_of_year is not None else resolve_as_of_year(model) or _runtime_max_observed_year(model))
    rows = _rows_through_as_of(model, resolved_as_of)
    if not rows:
        return {
            "value": math.nan,
            "status": "missing_auxiliary_driver_panel",
            "as_of_year": resolved_as_of,
            "projection_mode": projection_mode,
            "source": str(_driver_panel_path(model)),
        }
    mode = str(projection_mode or "hold_forward")
    if mode == "training_trend":
        value, status = _linear_project(rows, column, int(target_year))
    elif mode == "policy_target_path":
        value, status = _policy_target_project(
            model,
            rows=rows,
            column=column,
            target_year=int(target_year),
            as_of_year=resolved_as_of,
        )
    else:
        selected_year = int(target_year) if int(target_year) <= resolved_as_of else resolved_as_of
        selected = _row_at_or_before(rows, selected_year)
        value = _finite_float(selected.get(column) if selected is not None else math.nan)
        status = "origin_hold_forward" if int(target_year) > resolved_as_of else "observed_as_of_year"
    return {
        "value": value,
        "status": status,
        "as_of_year": resolved_as_of,
        "projection_mode": mode,
        "source": str(_driver_panel_path(model)),
    }


def project_yield_driver_modifiers(
    config_or_model: Any,
    *,
    target_year: int,
    as_of_year: int | None = None,
    projection_mode: str = "hold_forward",
    climate_shock: float = 0.0,
) -> dict[str, Any]:
    """Project grain irrigation/disaster modifiers under a forecast information set."""
    model = (
        config_or_model
        if hasattr(config_or_model, "config")
        else SimpleNamespace(config=config_or_model if isinstance(config_or_model, Mapping) else {})
    )
    irrigation = project_auxiliary_driver_value(
        model,
        column="effective_irrigation_to_grain_sown_area_ratio",
        target_year=int(target_year),
        as_of_year=as_of_year,
        projection_mode=projection_mode,
    )
    disaster = project_auxiliary_driver_value(
        model,
        column="crop_disaster_affected_to_grain_sown_area_ratio",
        target_year=int(target_year),
        as_of_year=as_of_year,
        projection_mode=projection_mode,
    )
    stack = _get_yield_stack_config(model)
    irrigation_cfg = _get_mapping(stack, "irrigation")
    disaster_cfg = _get_mapping(stack, "disaster_loss")
    irrigation_ratio = _finite_float(irrigation.get("value"))
    if math.isfinite(irrigation_ratio):
        baseline_ratio = _finite_float(
            irrigation_cfg.get("baseline_ratio"),
            _finite_float(
                _load_auxiliary_panel(str(_driver_panel_path(model)))[0].get(
                    "effective_irrigation_to_grain_sown_area_ratio"
                ),
                irrigation_ratio,
            ),
        )
        irrigation_modifier = 1.0 + _finite_float(irrigation_cfg.get("elasticity", 0.08), 0.08) * (
            irrigation_ratio - baseline_ratio
        )
        irrigation_modifier = max(
            _finite_float(irrigation_cfg.get("min_modifier", 0.96), 0.96),
            min(_finite_float(irrigation_cfg.get("max_modifier", 1.04), 1.04), irrigation_modifier),
        )
    else:
        baseline_ratio = math.nan
        irrigation_modifier = math.nan
    affected_ratio = _finite_float(disaster.get("value"))
    disaster_source = str(disaster.get("source", ""))
    disaster_status = str(disaster.get("status", ""))
    if not math.isfinite(affected_ratio):
        fallback_weight = _finite_float(disaster_cfg.get("climate_shock_fallback_weight", 0.6), 0.6)
        affected_ratio = max(0.0, -float(climate_shock or 0.0)) * fallback_weight
        disaster_status = "climate_shock_fallback"
        disaster_source = "model.climate_shock"
    disaster_modifier = 1.0 - _finite_float(disaster_cfg.get("severity_weight", 0.12), 0.12) * max(
        0.0,
        affected_ratio,
    )
    disaster_modifier = max(
        _finite_float(disaster_cfg.get("min_modifier", 0.85), 0.85),
        min(_finite_float(disaster_cfg.get("max_modifier", 1.0), 1.0), disaster_modifier),
    )
    combined = (
        irrigation_modifier * disaster_modifier
        if math.isfinite(irrigation_modifier) and math.isfinite(disaster_modifier)
        else math.nan
    )
    return {
        "target_year": int(target_year),
        "as_of_year": int(irrigation.get("as_of_year", as_of_year or target_year)),
        "projection_mode": str(projection_mode or "hold_forward"),
        "irrigation_ratio": irrigation_ratio,
        "irrigation_baseline_ratio": baseline_ratio,
        "irrigation_modifier": irrigation_modifier,
        "irrigation_status": str(irrigation.get("status", "")),
        "irrigation_source": str(irrigation.get("source", "")),
        "disaster_affected_ratio": affected_ratio,
        "disaster_loss_modifier": disaster_modifier,
        "disaster_loss_status": disaster_status,
        "disaster_loss_source": disaster_source,
        "combined_auxiliary_modifier": combined,
        "claim_boundary": (
            "Auxiliary-driver projection uses only information visible at as_of_year. "
            "training_trend and policy_target_path are audit/projection rules, not validation-period refits."
        ),
    }


def resolve_irrigation_modifier(model: Any, crop_type: Any, year: int | None = None) -> dict[str, Any]:
    key = crop_key(crop_type)
    if key != "grain":
        return {
            "modifier": 1.0,
            "ratio": math.nan,
            "status": "not_applicable_non_grain",
            "source": "not_applicable",
        }

    stack = _get_yield_stack_config(model)
    irrigation_cfg = _get_mapping(stack, "irrigation")
    if not bool(irrigation_cfg.get("enabled", True)):
        return {
            "modifier": math.nan,
            "ratio": math.nan,
            "status": "disabled",
            "source": "production.yield_stack.irrigation.enabled=false",
        }

    runtime_year = resolve_model_year(model) if year is None else int(year)
    record, mode = _select_driver_record(model, runtime_year)
    if record is None:
        return {
            "modifier": math.nan,
            "ratio": math.nan,
            "status": mode,
            "source": str(_driver_panel_path(model)),
        }

    ratio = _finite_float(record.get("effective_irrigation_to_grain_sown_area_ratio"))
    if not math.isfinite(ratio):
        return {
            "modifier": math.nan,
            "ratio": math.nan,
            "status": "missing_effective_irrigation_ratio",
            "source": str(_driver_panel_path(model)),
        }

    baseline_ratio = _finite_float(
        irrigation_cfg.get("baseline_ratio"),
        _finite_float(_load_auxiliary_panel(str(_driver_panel_path(model)))[0].get(
            "effective_irrigation_to_grain_sown_area_ratio"
        ), ratio),
    )
    elasticity = _finite_float(irrigation_cfg.get("elasticity", 0.08), 0.08)
    min_modifier = _finite_float(irrigation_cfg.get("min_modifier", 0.96), 0.96)
    max_modifier = _finite_float(irrigation_cfg.get("max_modifier", 1.04), 1.04)
    modifier = 1.0 + elasticity * (ratio - baseline_ratio)
    modifier = max(min_modifier, min(max_modifier, modifier))
    return {
        "modifier": modifier,
        "ratio": ratio,
        "baseline_ratio": baseline_ratio,
        "status": mode,
        "source": str(_driver_panel_path(model)),
    }


def resolve_disaster_loss_modifier(
    model: Any,
    crop_type: Any,
    *,
    year: int | None = None,
    climate_shock: float = 0.0,
) -> dict[str, Any]:
    key = crop_key(crop_type)
    if key != "grain":
        return {
            "modifier": 1.0,
            "affected_ratio": math.nan,
            "status": "not_applicable_non_grain",
            "source": "not_applicable",
        }

    stack = _get_yield_stack_config(model)
    disaster_cfg = _get_mapping(stack, "disaster_loss")
    if not bool(disaster_cfg.get("enabled", True)):
        return {
            "modifier": math.nan,
            "affected_ratio": math.nan,
            "status": "disabled",
            "source": "production.yield_stack.disaster_loss.enabled=false",
        }

    runtime_year = resolve_model_year(model) if year is None else int(year)
    record, mode = _select_driver_record(model, runtime_year)
    affected_ratio = math.nan
    source = str(_driver_panel_path(model))
    if record is not None:
        affected_ratio = _finite_float(
            record.get("crop_disaster_affected_to_grain_sown_area_ratio")
        )
    if not math.isfinite(affected_ratio):
        fallback_weight = _finite_float(disaster_cfg.get("climate_shock_fallback_weight", 0.6), 0.6)
        affected_ratio = max(0.0, -float(climate_shock or 0.0)) * fallback_weight
        mode = "climate_shock_fallback"
        source = "model.climate_shock"

    severity_weight = _finite_float(disaster_cfg.get("severity_weight", 0.12), 0.12)
    min_modifier = _finite_float(disaster_cfg.get("min_modifier", 0.85), 0.85)
    max_modifier = _finite_float(disaster_cfg.get("max_modifier", 1.0), 1.0)
    modifier = 1.0 - severity_weight * max(0.0, affected_ratio)
    modifier = max(min_modifier, min(max_modifier, modifier))
    return {
        "modifier": modifier,
        "affected_ratio": affected_ratio,
        "status": mode,
        "source": source,
    }


def finalize_yield_stack(
    *,
    model: Any,
    actor_type: str,
    crop_type: Any,
    area_mu: float,
    base_yield_per_mu: float,
    raw_base_yield_kg: float,
    non_climate_yield_kg: float,
    tech_progress_factor: float,
    scale_factor: float,
    soil_factor: float,
    ability_or_efficiency_multiplier: float,
    technology_multiplier: float,
    structural_multiplier: float,
    service_bonus_multiplier: float,
    service_signal: float,
    mechanization_signal: float,
    scale_progress: float,
    professional_signal: float = math.nan,
    management_signal: float = math.nan,
    coordination_signal: float = math.nan,
    climate_shock: float = 0.0,
    climate_modifier: float = 1.0,
    resilience_buffer: float = 0.0,
    random_factor: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    production_cfg: Mapping[str, Any] = {}
    cd_cfg: Mapping[str, Any] = {}
    yield_mode = "legacy_stack"
    try:
        production_cfg = _get_mapping(model.config, "production")
        cd_cfg = _get_mapping(production_cfg, "cobb_douglas")
        yield_mode = str(production_cfg.get("yield_mode", "legacy_stack"))
    except Exception:
        pass

    raw_shadow_enabled = cd_cfg.get("shadow_compare_enabled", False)
    shadow_compare_enabled = (
        str(raw_shadow_enabled).strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(raw_shadow_enabled, str)
        else bool(raw_shadow_enabled)
    )
    cd_shadow_enabled = yield_mode == "production_function_shadow" or shadow_compare_enabled

    year = resolve_model_year(model)
    irrigation = resolve_irrigation_modifier(model, crop_type, year)
    disaster = resolve_disaster_loss_modifier(
        model,
        crop_type,
        year=year,
        climate_shock=climate_shock,
    )
    irrigation_modifier = _finite_float(irrigation.get("modifier"), math.nan)
    disaster_modifier = _finite_float(disaster.get("modifier"), math.nan)
    applied_irrigation = irrigation_modifier if math.isfinite(irrigation_modifier) else 1.0
    applied_disaster = disaster_modifier if math.isfinite(disaster_modifier) else 1.0
    final_yield = (
        max(0.0, float(non_climate_yield_kg or 0.0))
        * applied_irrigation
        * float(climate_modifier or 1.0)
        * applied_disaster
        * float(random_factor or 1.0)
    )
    final_yield = max(0.0, final_yield)

    # Cobb-Douglas shadow is diagnostic: it never replaces final_yield.
    # Elasticities default to the NFPS-estimated values in config/master_config.yaml.
    # Inputs remain proxy multipliers, not direct land/labor/capital measurements.
    cd_yield = 0.0
    cd_yield_per_mu = 0.0
    shadow_cd_status = "disabled"
    if cd_shadow_enabled:
        try:
            alpha = float(cd_cfg.get("alpha", 0.7026))
            beta = float(cd_cfg.get("beta", 0.2547))
            gamma = float(cd_cfg.get("gamma", 0.0173))
            land_input = max(0.01, float(scale_factor or 1.0))
            labor_input = max(0.01, float(technology_multiplier or 1.0))
            capital_input = max(0.01, float(service_bonus_multiplier or 1.0))
            cd_yield_per_mu = (
                float(base_yield_per_mu or 390.0)
                * (land_input ** alpha)
                * (labor_input ** beta)
                * (capital_input ** gamma)
            )
            cd_yield = (
                cd_yield_per_mu
                * float(area_mu or 0.0)
                * applied_irrigation
                * float(climate_modifier or 1.0)
                * applied_disaster
                * float(random_factor or 1.0)
            )
            cd_yield = max(0.0, cd_yield)
            shadow_cd_status = "computed_proxy_shadow"
        except Exception:
            cd_yield = 0.0
            cd_yield_per_mu = 0.0
            shadow_cd_status = "failed_proxy_shadow"

    trace = {
        "actor_type": actor_type,
        "crop_type": str(getattr(crop_type, "value", crop_type)),
        "area_mu": float(area_mu or 0.0),
        "base_yield_per_mu": float(base_yield_per_mu or 0.0),
        "raw_base_yield_kg": float(raw_base_yield_kg or 0.0),
        "tech_progress_factor": float(tech_progress_factor or 1.0),
        "scale_factor": float(scale_factor or 1.0),
        "soil_factor": float(soil_factor or 1.0),
        "ability_or_efficiency_multiplier": float(ability_or_efficiency_multiplier or 1.0),
        "technology_multiplier": float(technology_multiplier or 1.0),
        "structural_multiplier": float(structural_multiplier or 1.0),
        "service_bonus_multiplier": float(service_bonus_multiplier or 1.0),
        "service_signal": float(service_signal or 0.0),
        "mechanization_signal": float(mechanization_signal or 0.0),
        "scale_progress": float(scale_progress or 0.0),
        "professional_signal": float(professional_signal),
        "management_signal": float(management_signal),
        "coordination_signal": float(coordination_signal),
        "irrigation_modifier": irrigation_modifier,
        "irrigation_ratio": _finite_float(irrigation.get("ratio"), math.nan),
        "irrigation_status": str(irrigation.get("status", "unknown")),
        "irrigation_source": str(irrigation.get("source", "")),
        "climate_shock": float(climate_shock or 0.0),
        "climate_modifier": float(climate_modifier or 1.0),
        "resilience_buffer": float(resilience_buffer or 0.0),
        "disaster_loss_modifier": disaster_modifier,
        "disaster_affected_ratio": _finite_float(disaster.get("affected_ratio"), math.nan),
        "disaster_loss_status": str(disaster.get("status", "unknown")),
        "disaster_loss_source": str(disaster.get("source", "")),
        "random_factor": float(random_factor or 1.0),
        "non_climate_multiplier": (
            float(non_climate_yield_kg / raw_base_yield_kg)
            if raw_base_yield_kg > 0
            else 0.0
        ),
        "final_yield_kg": float(final_yield or 0.0),
        "final_yield_per_mu": float(final_yield / area_mu) if area_mu > 0 else 0.0,
        "active_yield_kg": float(final_yield or 0.0),
        "active_yield_per_mu": float(final_yield / area_mu) if area_mu > 0 else 0.0,
        "step": int(_finite_float(getattr(model, "step_count", 0), 0)),
        "year": int(year),
        "forecast_origin_year": resolve_forecast_origin_year(model),
        "as_of_year": resolve_as_of_year(model),
        "claim_boundary": (
            "Shared production-side yield stack runtime. Irrigation and disaster "
            "drivers use training-window auxiliary national data with hold-forward "
            "semantics unless separately supplied and validated."
        ),
        "yield_mode": yield_mode,
        "shadow_cd_enabled": bool(cd_shadow_enabled),
        "shadow_cd_status": shadow_cd_status,
        "shadow_cd_yield_kg": float(cd_yield) if cd_yield > 0 else None,
        "shadow_cd_yield_per_mu": float(cd_yield_per_mu) if cd_yield > 0 else None,
        "shadow_cd_diff_vs_legacy": float(cd_yield - final_yield) if cd_yield > 0 else None,
        "shadow_cd_ratio_vs_legacy": (
            float(cd_yield / final_yield) if cd_yield > 0 and final_yield > 0 else None
        ),
        "shadow_cd_input_basis": "proxy_multipliers_scale_technology_service",
        "shadow_cd_claim_boundary": (
            "diagnostic only; active returned yield remains legacy_stack final_yield. "
            "Inputs are proxy multipliers, not direct land/labor/capital measurements."
        ),
    }
    return final_yield, trace


__all__ = [
    "crop_key",
    "finalize_yield_stack",
    "project_auxiliary_driver_value",
    "project_yield_driver_modifiers",
    "resolve_disaster_loss_modifier",
    "resolve_as_of_year",
    "resolve_forecast_origin_year",
    "resolve_irrigation_modifier",
    "resolve_model_year",
]
