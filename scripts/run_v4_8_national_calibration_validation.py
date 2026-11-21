#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "PyYAML>=6",
# ]
# ///
# --- How to run ---
# python scripts/run_v4_8_national_calibration_validation.py --overlay config/experiments/v4_8_national_fixed_rule.yaml --seeds 0:29 --years 30 --output results/formal/v4_8_national_20260811

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_builder import ModelBuilder
from core.reproducibility import RandomController
from core.fixed_rule_runtime import apply_experiment_overlay, load_experiment_overlay
from scripts.v4_8_national_task5_core import (
    CFPS_FITTING_YEARS,
    CFPS_VALIDATION_YEARS,
    NFPS_FITTING_YEARS,
    NFPS_VALIDATION_YEARS,
    PROJECT_ROOT as CORE_PROJECT_ROOT,
    REQUIRED_GROUPS,
    SCENARIOS,
    Task5Error,
    Task5Result,
    build_moments,
    build_parameters,
    find_project_file,
)
from scripts.v4_8_national_task5_observables import (
    annual_transfer_participation,
    crop_gross_income,
    household_rent_burden_per_mu,
    model_years,
    named_year_window,
    household_area_shares,
    target_metadata,
    winsorized_mean,
)
from scripts.v4_8_national_task5_outputs import (
    hash_file,
    paired_worldline_rows,
    protected_hashes,
    source_data_rows,
    validation_metrics,
    write_csv,
    write_worklog,
)

DEFAULT_PRESCREEN_FARMERS = 100
LEGACY_CALIBRATION_BRIDGE_STATUS = "intentionally_disabled_for_clean_v4_8_calibration"


def parse_seed_range(raw: str) -> range:
    if ":" in raw:
        start_text, end_text = raw.split(":", maxsplit=1)
        return range(int(start_text), int(end_text) + 1)
    value = int(raw)
    return range(value, value + 1)


MASTER_CONFIG = PROJECT_ROOT / "config" / "master_config.yaml"
LEGACY_CALIBRATION_FILE = PROJECT_ROOT / "config" / "calibrated_params_enhanced.json"
INITIAL_FEATURE_LIBRARY = PROJECT_ROOT / "config" / "agent_feature_library.json"
ENTRANT_FEATURE_LIBRARY = PROJECT_ROOT / "data" / "derived_runtime" / "v4_8" / "cfps_entrant_feature_library_2010_2016.json"


def _input_file_identity(path: Path) -> dict[str, str | bool]:
    exists = path.is_file()
    return {
        "path": str(path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path),
        "exists": exists,
        "sha256": hash_file(path) if exists else "",
    }


def critical_input_identity(n_farmers: int) -> dict[str, object]:
    return {
        "n_farmers": int(n_farmers),
        "master_config": _input_file_identity(MASTER_CONFIG),
        "overlay_applied_after_master": True,
        "nfps_target_data": _input_file_identity(find_project_file("2009-2015household.csv")),
        "cfps_target_data": _input_file_identity(find_project_file("cfps_calibration_summary_v6_unified.csv")),
        "initial_feature_library": _input_file_identity(INITIAL_FEATURE_LIBRARY),
        "entrant_feature_library": _input_file_identity(ENTRANT_FEATURE_LIBRARY),
        "legacy_calibration_file": _input_file_identity(LEGACY_CALIBRATION_FILE),
        "legacy_calibration_bridge_status": LEGACY_CALIBRATION_BRIDGE_STATUS,
    }


def expected_run_identity(overlay: Path, seeds: range, years: int, n_farmers: int = DEFAULT_PRESCREEN_FARMERS) -> str:
    payload = {
        "overlay": str(overlay.relative_to(PROJECT_ROOT) if overlay.is_relative_to(PROJECT_ROOT) else overlay),
        "overlay_hash": hash_file(overlay),
        "seed_set": list(seeds),
        "years": years,
        "critical_inputs": critical_input_identity(n_farmers),
        "runner_hash": hash_file(Path(__file__)),
        "core_hash": hash_file(PROJECT_ROOT / "scripts" / "v4_8_national_task5_core.py"),
        "outputs_hash": hash_file(PROJECT_ROOT / "scripts" / "v4_8_national_task5_outputs.py"),
        "validator_hash": hash_file(PROJECT_ROOT / "scripts" / "validate_v4_8_national_calibration.py"),
        "data_manifest_sha256": data_manifest_sha256(),
        "code_tree_or_dirty_patch_sha256": code_tree_or_dirty_patch_sha256(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def can_use_cache(output: Path, run_identity: str) -> dict[str, Any] | None:
    required = (
        "calibration_targets.csv",
        "optimization_trace.csv",
        "calibrated_parameters.csv",
        "validation_metrics.csv",
        RAW_RUNTIME_FILE,
        FITTING_GRID_RUNTIME_FILE,
        "paired_worldlines.csv",
        "run_manifest.json",
    )
    if not all((output / name).is_file() and (output / name).stat().st_size > 0 for name in required):
        return None
    try:
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    critical_identity = manifest.get("critical_input_identity")
    if (
        manifest.get("run_status") != "completed"
        or manifest.get("run_identity_sha256") != run_identity
        or not isinstance(critical_identity, dict)
        or critical_identity.get("legacy_calibration_bridge_status") != LEGACY_CALIBRATION_BRIDGE_STATUS
        or manifest.get("legacy_calibration_bridge_status") != LEGACY_CALIBRATION_BRIDGE_STATUS
        or not isinstance(manifest.get("calibration_gate_passed"), bool)
    ):
        return None
    return manifest


RUNTIME_SURFACE = "ModelBuilder.build+MainModel.step"
BASELINE_SCENARIO = "baseline_cash"
RAW_RUNTIME_FILE = "raw_runtime_observations.csv"
FITTING_GRID_RUNTIME_FILE = "fitting_grid_runtime_observations.csv"
MIN_MEASURED_SENSITIVITY = 0.00000001
JOINT_SEARCH_MAX_PARAMETERS = 4
RUNTIME_PARAMETER_SPECS: tuple[dict[str, Any], ...] = (
    {
        "parameter_id": "v4_8_runtime_fixed_rule_inertia_margin",
        "parameter_name": "fixed_rule_inertia_margin",
        "runtime_config_path": "runtime_fixed_rule.minimum_inertia_margin_yuan_per_mu",
        "value": 50.0,
        "lower_bound": 0.0,
        "upper_bound": 25000.0,
        "target_moments": ("rent_out_rate", "rent_in_rate", "land_rent_per_mu"),
        "claim_boundary": "runtime_active_fixed_rule_inertia_threshold_not_structural_causal_estimate",
    },
    {
        "parameter_id": "v4_8_runtime_base_rent_price",
        "parameter_name": "base_rent_price",
        "runtime_config_path": "agents.farmer.land_transfer.base_rent_price",
        "value": 420.0,
        "lower_bound": 280.0,
        "upper_bound": 700.0,
        "target_moments": ("land_rent_per_mu", "rent_out_rate", "rent_in_rate"),
        "claim_boundary": "runtime_active_land_rent_anchor_not_structural_causal_estimate",
    },
    {
        "parameter_id": "v4_8_runtime_land_rent_demand_elasticity",
        "parameter_name": "land_rent_demand_elasticity",
        "runtime_config_path": "agents.enterprise.expansion_decision.land_rent_demand_elasticity",
        "value": 0.5,
        "lower_bound": 0.1,
        "upper_bound": 1.0,
        "target_moments": ("rent_in_rate", "land_rent_per_mu"),
        "claim_boundary": "runtime_active_land_rent_quantity_response_not_structural_causal_estimate",
    },
    {
        "parameter_id": "v4_8_runtime_max_annual_transfer_rate",
        "parameter_name": "max_annual_transfer_rate",
        "runtime_config_path": "agents.farmer.land_transfer.max_annual_transfer_rate",
        "value": 0.03,
        "lower_bound": 0.005,
        "upper_bound": 0.20,
        "target_moments": ("rent_out_rate", "rent_in_rate"),
        "claim_boundary": "runtime_active_annual_transfer_cap_not_structural_causal_estimate",
    },
    {
        "parameter_id": "v4_8_runtime_grain_yield_scale",
        "parameter_name": "grain_yield_scale",
        "runtime_config_path": "production.base_yield.grain",
        "value": 1.0,
        "lower_bound": 0.75,
        "upper_bound": 1.35,
        "target_moments": ("agricultural_income", "rent_out_rate", "Pure_Farmer_share"),
        "claim_boundary": "runtime_active_grain_yield_scale_not_structural_causal_estimate",
    },
    {
        "parameter_id": "v4_8_runtime_grain_cost_scale",
        "parameter_name": "grain_cost_scale",
        "runtime_config_path": "production.cost_structure.input_requirements.other_cost_per_mu.grain",
        "value": 1.0,
        "lower_bound": 0.60,
        "upper_bound": 1.60,
        "target_moments": ("agricultural_income", "rent_out_rate"),
        "claim_boundary": "runtime_active_grain_cash_cost_scale_not_structural_causal_estimate",
    },
    {
        "parameter_id": "v4_8_runtime_livelihood_base_outflow",
        "parameter_name": "livelihood_base_outflow_prob",
        "runtime_config_path": "agents.farmer.livelihood_evolution.urbanization_trend.base_outflow_prob",
        "value": 0.035,
        "lower_bound": 0.01,
        "upper_bound": 0.08,
        "target_moments": ("Pure_Farmer_share", "Pluriactive_share", "Non_Farmer_share"),
        "claim_boundary": "runtime_active_livelihood_outflow_parameter_not_structural_causal_estimate",
    },
    {
        "parameter_id": "v4_8_runtime_pluriactive_structural_exit",
        "parameter_name": "pluriactive_structural_nonfarmer_prob",
        "runtime_config_path": "agents.farmer.livelihood_evolution.transition_probabilities.pluriactive_structural_nonfarmer_prob",
        "value": 0.12,
        "lower_bound": 0.0,
        "upper_bound": 0.30,
        "target_moments": ("Pluriactive_share", "Non_Farmer_share"),
        "claim_boundary": "runtime_active_pluriactive_exit_parameter_not_structural_causal_estimate",
    },
    {
        "parameter_id": "v4_8_runtime_nonfarmer_reentry",
        "parameter_name": "nonfarmer_to_pluriactive_prob",
        "runtime_config_path": "agents.farmer.livelihood_evolution.transition_probabilities.nonfarmer_to_pluriactive_prob",
        "value": 0.0066,
        "lower_bound": 0.0,
        "upper_bound": 0.03,
        "target_moments": ("Pluriactive_share", "Non_Farmer_share"),
        "claim_boundary": "cfps_panel_anchored_reentry_probability_not_structural_causal_estimate",
    },
)


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def code_tree_or_dirty_patch_sha256() -> str:
    paths = (
        Path(__file__),
        PROJECT_ROOT / "scripts" / "v4_8_national_task5_core.py",
        PROJECT_ROOT / "scripts" / "v4_8_national_task5_observables.py",
        PROJECT_ROOT / "scripts" / "v4_8_national_task5_outputs.py",
        PROJECT_ROOT / "scripts" / "validate_v4_8_national_calibration.py",
        PROJECT_ROOT / "core" / "model.py",
        PROJECT_ROOT / "core" / "model_builder.py",
        PROJECT_ROOT / "agents" / "village_collective.py",
        PROJECT_ROOT / "modules" / "land_patch.py",
        PROJECT_ROOT / "agents" / "farmer_agent.py",
        PROJECT_ROOT / "agents" / "enterprise_agent.py",
    )
    return _stable_hash([
        {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": hash_file(path)}
        for path in paths
    ])


def data_manifest_sha256() -> str:
    manifest = PROJECT_ROOT / "project_management" / "V4_8_INPUT_FILE_MANIFEST_20260811.csv"
    if not manifest.is_file():
        raise Task5Error(f"missing V4.8 input manifest: {manifest}")
    return hash_file(manifest)


def runtime_config(
    overlay: Path,
    seed: int = 0,
    years: int = 4,
    n_farmers: int = DEFAULT_PRESCREEN_FARMERS,
) -> dict[str, Any]:  # noqa: DICT_OK
    if n_farmers < 1 or n_farmers > 300:
        raise Task5Error("Task5 prescreen scale must be 1-300 farmers")
    master = yaml.safe_load(MASTER_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(master, dict):
        raise Task5Error("master_config.yaml must be a mapping")
    config = copy.deepcopy(master)
    apply_experiment_overlay(config, load_experiment_overlay(overlay))
    config.setdefault("experiment", {})
    config["experiment"].update({"seed": int(seed), "steps": int(years), "overlay_path": str(overlay)})
    scale = config.setdefault("scale", {})
    scale["n_farmers"] = int(n_farmers)
    scale["n_enterprises"] = max(1, int(round(int(n_farmers) * float(scale.get("enterprise_ratio", 0.10) or 0.10))))
    scale["grid_width"] = min(int(scale.get("grid_width", 100) or 100), 50)
    scale["grid_height"] = min(int(scale.get("grid_height", 100) or 100), 50)
    scale["width"] = scale["grid_width"]
    scale["height"] = scale["grid_height"]
    initial = config.setdefault("initial_conditions", {})
    initial["data_source"] = "nfps"
    initial["microdata_driven"] = True
    initial["initial_feature_library"] = str(INITIAL_FEATURE_LIBRARY.relative_to(PROJECT_ROOT))
    initial["feature_library"] = str(INITIAL_FEATURE_LIBRARY.relative_to(PROJECT_ROOT))
    policy = config.setdefault("policy", {})
    policy["policy_mode"] = "rule"
    policy["rl_enabled"] = False
    government = policy.setdefault("government", {})
    government["policy_mode"] = "rule"
    government["adaptive_adjustment_enabled"] = False
    config.pop("runtime_smoke", None)
    config.setdefault("population_dynamics", {}).setdefault("entry_mechanism", {})[
        "entrant_feature_library"
    ] = str(ENTRANT_FEATURE_LIBRARY.relative_to(PROJECT_ROOT))
    config.setdefault("calibration", {})["apply_calibrated_params"] = False
    config["task5_legacy_calibration_bridge_status"] = LEGACY_CALIBRATION_BRIDGE_STATUS
    config["task5_runtime_config_source"] = "master_config_deepcopy_then_v4_8_fixed_rule_overlay"
    return config


def _validate_formal_runtime_config(config: dict[str, Any]) -> None:
    initial = config.get("initial_conditions", {})
    entry = config.get("population_dynamics", {}).get("entry_mechanism", {})
    if config.get("task5_runtime_config_source") != "master_config_deepcopy_then_v4_8_fixed_rule_overlay":
        raise Task5Error("smoke configuration source is forbidden for Task5")
    if config.get("runtime_smoke") or not isinstance(initial, dict) or not isinstance(entry, dict):
        raise Task5Error("smoke configuration source is forbidden for Task5")
    if initial.get("data_source") != "nfps" or initial.get("microdata_driven") is not True:
        raise Task5Error("formal Task5 runtime requires NFPS microdata-driven initial conditions")
    if not initial.get("initial_feature_library") or not entry.get("entrant_feature_library"):
        raise Task5Error("formal Task5 runtime requires CFPS/NFPS feature-library paths")


def _runtime_year_windows(years: int) -> tuple[range, range]:
    if years < 2:
        raise Task5Error("Task5 requires at least two disjoint simulation years")
    split = max(1, years // 2)
    if split >= years:
        split = years - 1
    return range(1, split + 1), range(split + 1, years + 1)


def _required_formal_horizon() -> int:
    return max(CFPS_VALIDATION_YEARS) - 2010


def _validate_formal_horizon(years: int) -> None:
    required = _required_formal_horizon()
    if years < required:
        raise Task5Error(
            f"formal Task5 requires {required} years so the runtime reaches CFPS holdout year {max(CFPS_VALIDATION_YEARS)}"
        )


def _annual_transfer_participation(model: object) -> tuple[float, float]:
    return annual_transfer_participation(model)


def _crop_gross_income(farmer: object) -> float:
    return crop_gross_income(farmer)


def _active_operator_area(operator: object, model_patches: list[object]) -> float:
    area = 0.0
    for patch in model_patches:
        owner = getattr(patch, "owner", None)
        leased_to = getattr(patch, "leased_to", None)
        current_operator = getattr(patch, "current_operator", None)
        if current_operator is operator or leased_to is operator or (
            owner is operator and leased_to in (None, operator) and current_operator in (None, operator)
        ):
            area += float(getattr(patch, "area", 0.0) or 0.0)
    return area


def _runtime_observation(model: object, seed: int, scenario: str, year_index: int, config_hash: str, initial_hash: str) -> dict[str, str]:
    data = model.datacollector.get_model_vars_dataframe().iloc[-1]
    farmers = list(getattr(model, "farmers", []) or [])
    enterprises = list(getattr(model, "agricultural_enterprises", []) or [])
    operators = farmers + enterprises
    areas = [_active_operator_area(operator, list(getattr(model, "land_patches", []) or [])) for operator in operators]
    farmer_areas = areas[:len(farmers)]
    household_shares = household_area_shares(farmer_areas, areas[len(farmers):])
    area_denominator = sum(area > 0.0 for area in farmer_areas)
    farmer_count = len(farmers)
    rent_out_rate, rent_in_rate = _annual_transfer_participation(model)
    crop_income = [_crop_gross_income(farmer) for farmer in farmers]
    grain_output = sum(
        float(getattr(farmer, "grain_production", 0.0) or getattr(farmer, "last_yield", 0.0) or 0.0)
        for farmer in farmers
    ) + sum(float(getattr(operator, "grain_production", 0.0) or 0.0) for operator in enterprises)
    scenario_policy = getattr(model, "active_policy_id", scenario)
    fiscal_spending = sum(float(getattr(farmer, "subsidy_income", 0.0) or 0.0) for farmer in farmers) + sum(
        float(getattr(operator, "subsidy_revenue", 0.0) or 0.0) + float(getattr(operator, "scale_subsidy_revenue", 0.0) or 0.0)
        for operator in enterprises
    )
    return {
        "runtime_seed": str(seed),
        "scenario_id": scenario,
        "active_policy_id": str(scenario_policy),
        "year_index": str(year_index),
        "simulation_year": str(int(getattr(model, "simulation_base_year", 2009)) + int(getattr(model, "step_count", year_index))),
        "runtime_surface": RUNTIME_SURFACE,
        "runtime_input_hash": config_hash,
        "runtime_config_source": str(getattr(model, "config", {}).get("task5_runtime_config_source", "")),
        "initial_conditions_data_source": str(getattr(model, "config", {}).get("initial_conditions", {}).get("data_source", "")),
        "microdata_driven": str(getattr(model, "config", {}).get("initial_conditions", {}).get("microdata_driven", "")).lower(),
        "initial_state_sha256": initial_hash,
        "trace_artifact": RAW_RUNTIME_FILE,
        "population_snapshot_timing": "pre_turnover_completed_production_year",
        "operator_area_denominator": str(area_denominator),
        "small_lt_5mu_share": f"{household_shares[0]:.8f}",
        "medium_5_15mu_share": f"{household_shares[1]:.8f}",
        "large_15_50mu_share": f"{household_shares[2]:.8f}",
        "household_ge_50mu_share": f"{household_shares[3]:.8f}",
        "rent_out_rate": f"{rent_out_rate:.8f}",
        "rent_in_rate": f"{rent_in_rate:.8f}",
        "land_rent_per_mu": f"{household_rent_burden_per_mu(model):.8f}",
        "agricultural_income": f"{winsorized_mean(crop_income):.8f}",
        "Pure_Farmer_share": f"{float(data['Pure_Farmer']) / farmer_count if farmer_count else 0.0:.8f}",
        "Pluriactive_share": f"{float(data['Pluriactive']) / farmer_count if farmer_count else 0.0:.8f}",
        "Non_Farmer_share": f"{float(data['Non_Farmer']) / farmer_count if farmer_count else 0.0:.8f}",
        "grain_observable": "farmer_grain_production_sum",
        "grain_value": f"{grain_output:.8f}",
        "welfare_observable": "Average_Net_Income",
        "welfare_value": f"{float(data['Average_Net_Income']):.8f}",
        "fiscal_spending": f"{fiscal_spending:.8f}",
        "fiscal_efficiency_value": f"{grain_output / fiscal_spending if fiscal_spending > 0.0 else 0.0:.8f}",
        "land_rent_observable": "household_operated_leased_patch_rent_burden_per_mu",
        "land_rent_value": f"{household_rent_burden_per_mu(model):.8f}",
        "transfer_rate_observable": "Land_Transfer_Rate",
        "transfer_rate_value": f"{float(data['Land_Transfer_Rate']):.8f}",
        "fixed_rule_trace_rows": str(len(getattr(model, "v4_8_fixed_rule_trace", []) or [])),
        "market_stage_trace_rows": str(len(getattr(model, "v4_8_market_stage_trace", []) or [])),
    }


def _apply_runtime_parameters(config: dict[str, Any], parameter_values: dict[str, float]) -> None:
    production = config.setdefault("production", {})
    base_yield = production.setdefault("base_yield", {})
    rule = config.setdefault("runtime_fixed_rule", {})
    rule["minimum_inertia_margin_yuan_per_mu"] = float(
        parameter_values.get(
            "fixed_rule_inertia_margin",
            rule.get("minimum_inertia_margin_yuan_per_mu", 50.0),
        )
    )
    transfer = config.setdefault("agents", {}).setdefault("farmer", {}).setdefault("land_transfer", {})
    transfer["base_rent_price"] = float(parameter_values.get("base_rent_price", transfer.get("base_rent_price", 420.0)))
    transfer["max_annual_transfer_rate"] = float(
        parameter_values.get(
            "max_annual_transfer_rate",
            transfer.get("max_annual_transfer_rate", 0.03),
        )
    )
    expansion = config.setdefault("agents", {}).setdefault("enterprise", {}).setdefault("expansion_decision", {})
    expansion["land_rent_demand_elasticity"] = float(
        parameter_values.get(
            "land_rent_demand_elasticity",
            expansion.get("land_rent_demand_elasticity", 0.5),
        )
    )
    base_yield["grain"] = float(base_yield.get("grain", 390.0)) * float(parameter_values.get("grain_yield_scale", 1.0))
    other_cost = production.setdefault("cost_structure", {}).setdefault("input_requirements", {}).setdefault("other_cost_per_mu", {})
    other_cost["grain"] = float(other_cost.get("grain", 200.0)) * float(parameter_values.get("grain_cost_scale", 1.0))
    livelihood = config.setdefault("agents", {}).setdefault("farmer", {}).setdefault("livelihood_evolution", {})
    urbanization = livelihood.setdefault("urbanization_trend", {})
    urbanization["base_outflow_prob"] = float(
        parameter_values.get("livelihood_base_outflow_prob", urbanization.get("base_outflow_prob", 0.035))
    )
    transition = livelihood.setdefault("transition_probabilities", {})
    transition["pluriactive_structural_nonfarmer_prob"] = float(
        parameter_values.get(
            "pluriactive_structural_nonfarmer_prob",
            transition.get("pluriactive_structural_nonfarmer_prob", 0.12),
        )
    )
    transition["nonfarmer_to_pluriactive_prob"] = float(
        parameter_values.get(
            "nonfarmer_to_pluriactive_prob",
            transition.get("nonfarmer_to_pluriactive_prob", 0.0066),
        )
    )


def _resolve_runtime_path(config: dict[str, Any], dotted_path: str) -> object:
    value: object = config
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise Task5Error(f"unbound calibration parameter runtime path: {dotted_path}")
        value = value[part]
    return value


def validate_runtime_parameter_specs(overlay: Path, n_farmers: int = DEFAULT_PRESCREEN_FARMERS) -> None:
    config = copy.deepcopy(runtime_config(overlay, n_farmers=n_farmers))
    _validate_formal_runtime_config(config)
    _apply_runtime_parameters(config, {spec["parameter_name"]: float(spec["value"]) for spec in RUNTIME_PARAMETER_SPECS})
    for spec in RUNTIME_PARAMETER_SPECS:
        path = str(spec["runtime_config_path"])
        value = _resolve_runtime_path(config, path)
        if value is None:
            raise Task5Error(f"unbound calibration parameter runtime path: {path}")


def run_runtime_grid(
    overlay: Path,
    seeds: range,
    years: int,
    parameter_values: dict[str, float],
    scenarios: tuple[str, ...],
    runtime_phase: str,
    runtime_candidate_id: str,
    trace_artifact: str,
    n_farmers: int = DEFAULT_PRESCREEN_FARMERS,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in seeds:
        for scenario in scenarios:
            RandomController.set_seed(seed)
            config = runtime_config(overlay, seed=seed, years=years, n_farmers=n_farmers)
            config["policy"]["active_scenario"] = scenario
            _apply_runtime_parameters(config, parameter_values)
            _validate_formal_runtime_config(config)
            config["experiment"].pop("overlay_path", None)
            config_hash = _stable_hash(config)
            model = ModelBuilder(config=config).build()
            initial_hash = _stable_hash(
                {
                    "farmers": [
                        {
                            "id": str(getattr(farmer, "unique_id", "")),
                            "livelihood": str(getattr(getattr(farmer, "livelihood_strategy", None), "value", "")),
                            "patches": [float(getattr(patch, "area", 0.0) or 0.0) for patch in getattr(farmer, "land_patches", []) or []],
                        }
                        for farmer in getattr(model, "farmers", []) or []
                    ],
                    "enterprises": [str(getattr(operator, "unique_id", "")) for operator in getattr(model, "agricultural_enterprises", []) or []],
                }
            )
            initial_row = _initial_runtime_observation(
                model,
                seed,
                scenario,
                config_hash,
                initial_hash,
            )
            initial_row["trace_artifact"] = trace_artifact
            initial_row["runtime_phase"] = runtime_phase
            initial_row["runtime_candidate_id"] = runtime_candidate_id
            rows.append(initial_row)
            captured_row: dict[str, str] = {}
            current_year_index = 0
            prior_snapshot_hook = getattr(model, "_completed_production_year_snapshot_hook", None)

            def capture_completed_production_year(current_model: object) -> None:
                if callable(prior_snapshot_hook):
                    prior_snapshot_hook(current_model)
                captured_row.clear()
                captured_row.update(
                    _runtime_observation(
                        current_model,
                        seed,
                        scenario,
                        current_year_index,
                        config_hash,
                        initial_hash,
                    )
                )

            model._completed_production_year_snapshot_hook = capture_completed_production_year
            for year_index in range(1, years + 1):
                current_year_index = year_index
                model.step()
                if not captured_row:
                    raise Task5Error("annual observation hook did not run before population turnover")
                row = dict(captured_row)
                row["trace_artifact"] = trace_artifact
                row["runtime_phase"] = runtime_phase
                row["runtime_candidate_id"] = runtime_candidate_id
                rows.append(row)
    return rows


def _mean_runtime(rows: list[dict[str, str]], moment_id: str, year_indices: range) -> float:
    selected = [
        float(row[moment_id])
        for row in rows
        if row["scenario_id"] == BASELINE_SCENARIO and int(row["year_index"]) in year_indices
    ]
    if not selected or any(not math.isfinite(value) for value in selected):
        raise Task5Error(f"missing finite runtime observations for {moment_id}")
    return statistics.mean(selected)


def _mean_runtime_years(rows: list[dict[str, str]], moment_id: str, simulation_years: tuple[int, ...]) -> float:
    selected_years = {str(year) for year in simulation_years}
    selected = [
        float(row[moment_id])
        for row in rows
        if row["scenario_id"] == BASELINE_SCENARIO and row["simulation_year"] in selected_years
    ]
    if not selected or any(not math.isfinite(value) for value in selected):
        raise Task5Error(f"missing finite runtime observations for {moment_id} in years {simulation_years}")
    return statistics.mean(selected)


def _initial_runtime_observation(
    model: object,
    seed: int,
    scenario: str,
    config_hash: str,
    initial_hash: str,
) -> dict[str, str]:
    farmers = list(getattr(model, "farmers", []) or [])
    areas = [
        _active_operator_area(farmer, list(getattr(model, "land_patches", []) or []))
        for farmer in farmers
    ]
    shares = household_area_shares(areas)
    denominator = sum(area > 0.0 for area in areas)
    counts = {
        "Pure_Farmer": 0,
        "Pluriactive": 0,
        "Non_Farmer": 0,
    }
    for farmer in farmers:
        label = str(
            getattr(
                getattr(farmer, "livelihood_strategy", None),
                "value",
                getattr(farmer, "livelihood_strategy", ""),
            )
        )
        if label in counts:
            counts[label] += 1
    farmer_count = len(farmers)
    return {
        "runtime_seed": str(seed),
        "scenario_id": scenario,
        "active_policy_id": scenario,
        "year_index": "0",
        "simulation_year": str(int(getattr(model, "simulation_base_year", 2010))),
        "runtime_surface": RUNTIME_SURFACE,
        "runtime_input_hash": config_hash,
        "runtime_config_source": str(getattr(model, "config", {}).get("task5_runtime_config_source", "")),
        "initial_conditions_data_source": str(getattr(model, "config", {}).get("initial_conditions", {}).get("data_source", "")),
        "microdata_driven": str(getattr(model, "config", {}).get("initial_conditions", {}).get("microdata_driven", "")).lower(),
        "initial_state_sha256": initial_hash,
        "trace_artifact": "",
        "population_snapshot_timing": "initial_state_before_annual_cycle",
        "operator_area_denominator": str(denominator),
        "small_lt_5mu_share": f"{shares[0]:.8f}",
        "medium_5_15mu_share": f"{shares[1]:.8f}",
        "large_15_50mu_share": f"{shares[2]:.8f}",
        "household_ge_50mu_share": f"{shares[3]:.8f}",
        "rent_out_rate": "0.00000000",
        "rent_in_rate": "0.00000000",
        "land_rent_per_mu": "0.00000000",
        "agricultural_income": f"{winsorized_mean([_crop_gross_income(farmer) for farmer in farmers]):.8f}",
        "Pure_Farmer_share": f"{counts['Pure_Farmer'] / farmer_count if farmer_count else 0.0:.8f}",
        "Pluriactive_share": f"{counts['Pluriactive'] / farmer_count if farmer_count else 0.0:.8f}",
        "Non_Farmer_share": f"{counts['Non_Farmer'] / farmer_count if farmer_count else 0.0:.8f}",
        "grain_observable": "initial_state_not_applicable",
        "grain_value": "0.00000000",
        "welfare_observable": "initial_state_not_applicable",
        "welfare_value": "0.00000000",
        "fiscal_spending": "0.00000000",
        "fiscal_efficiency_value": "0.00000000",
        "land_rent_observable": "initial_state_not_applicable",
        "land_rent_value": "0.00000000",
        "transfer_rate_observable": "initial_state_not_applicable",
        "transfer_rate_value": "0.00000000",
        "fixed_rule_trace_rows": "0",
        "market_stage_trace_rows": "0",
    }


def _parameter_values_json(parameter_values: dict[str, float]) -> str:
    return json.dumps(parameter_values, sort_keys=True, separators=(",", ":"))


def _candidate_parameter_values() -> list[tuple[str, dict[str, float]]]:
    baseline = {str(spec["parameter_name"]): float(spec["value"]) for spec in RUNTIME_PARAMETER_SPECS}
    candidates: list[tuple[str, dict[str, float]]] = [("baseline", baseline)]
    for spec in RUNTIME_PARAMETER_SPECS:
        parameter_name = str(spec["parameter_name"])
        for label, candidate_value in (("lower", float(spec["lower_bound"])), ("upper", float(spec["upper_bound"]))):
            candidate = dict(baseline)
            candidate[parameter_name] = candidate_value
            candidates.append((f"{parameter_name}_{label}", candidate))
    return candidates


def _joint_candidate_parameter_values(
    active_specs: tuple[dict[str, Any], ...],
    baseline: dict[str, float],
    preferred_values: dict[str, tuple[str, float]],
) -> list[tuple[str, dict[str, float]]]:
    candidates: list[tuple[str, dict[str, float]]] = []
    for left, right in itertools.combinations(active_specs, 2):
        left_name = str(left["parameter_name"])
        right_name = str(right["parameter_name"])
        left_label, left_value = preferred_values[left_name]
        right_label, right_value = preferred_values[right_name]
        candidate = dict(baseline)
        candidate[left_name] = left_value
        candidate[right_name] = right_value
        candidate_id = f"joint_{left_name}_{left_label}__{right_name}_{right_label}"
        candidates.append((candidate_id, candidate))
    return candidates


def _fitting_objective(moments: list, runtime_rows: list[dict[str, str]]) -> float:  # noqa: ANN401
    numerator = sum(
        moment.weight
        * abs(
            _mean_runtime_years(
                runtime_rows,
                moment.moment_id,
                model_years(
                    moment.evidence_channel,
                    moment.moment_id,
                    validation=False,
                ),
            )
            - moment.fitting_value
        )
        for moment in moments
    )
    denominator = sum(moment.weight * abs(moment.fitting_value) for moment in moments)
    if denominator <= 0.0:
        raise Task5Error("nonpositive fitting objective denominator")
    return numerator / denominator


def _grid_sensitivities(
    grid_rows: list[dict[str, str]],
    moments: list,
) -> dict[str, float]:  # noqa: ANN401
    baseline_rows = [row for row in grid_rows if row["runtime_candidate_id"] == "baseline"]
    if not baseline_rows:
        raise Task5Error("fitting grid lacks baseline runtime trace")
    sensitivities: dict[str, float] = {}
    for spec in RUNTIME_PARAMETER_SPECS:
        parameter_name = str(spec["parameter_name"])
        responses: list[float] = []
        for moment_id in tuple(spec["target_moments"]):
            moment = next(item for item in moments if item.moment_id == moment_id)
            fitting_years = model_years(
                moment.evidence_channel,
                moment.moment_id,
                validation=False,
            )
            baseline_value = _mean_runtime_years(baseline_rows, moment_id, fitting_years)
            for candidate_id in (f"{parameter_name}_lower", f"{parameter_name}_upper"):
                varied_rows = [row for row in grid_rows if row["runtime_candidate_id"] == candidate_id]
                candidate_value = _mean_runtime_years(varied_rows, moment_id, fitting_years)
                responses.append(abs(candidate_value - baseline_value) / max(abs(baseline_value), 1.0))
        if not responses:
            raise Task5Error(f"fitting grid lacks perturbation trace for {parameter_name}")
        sensitivities[parameter_name] = max(responses)
    return sensitivities


def _fitting_grid_rows(
    overlay: Path,
    seeds: range,
    fitting_years: int,
    moments: list,
    n_farmers: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, float], dict[str, float]]:  # noqa: ANN401
    trace_rows: list[dict[str, str]] = []
    optimization_rows: list[dict[str, str]] = []
    for iteration, (candidate_id, parameter_values) in enumerate(_candidate_parameter_values(), start=1):
        candidate_rows = run_runtime_grid(
            overlay,
            seeds,
            fitting_years,
            parameter_values,
            (BASELINE_SCENARIO,),
            "fitting_grid",
            candidate_id,
            FITTING_GRID_RUNTIME_FILE,
            n_farmers=n_farmers,
        )
        trace_rows.extend(candidate_rows)
        objective = _fitting_objective(moments, candidate_rows)
        optimization_rows.append(
            {
                "stage": "fitting_grid",
                "iteration": str(iteration),
                "candidate_id": candidate_id,
                "parameter_name": "coordinate_runtime_grid",
                "parameter_values_json": _parameter_values_json(parameter_values),
                "objective_value": f"{objective:.8f}",
                "selected": "false",
                "fitting_window": "channel_specific_named_simulation_years",
                "runtime_observation_count": str(len(candidate_rows)),
                "runtime_surface": RUNTIME_SURFACE,
            }
        )
    sensitivities = _grid_sensitivities(trace_rows, moments)
    active_specs = tuple(
        spec
        for spec in RUNTIME_PARAMETER_SPECS
        if sensitivities[str(spec["parameter_name"])] > MIN_MEASURED_SENSITIVITY
    )
    if not active_specs:
        raise Task5Error("no active runtime calibration parameters after measured sensitivity prescreen")
    active_names = {str(spec["parameter_name"]) for spec in active_specs}
    baseline = {str(spec["parameter_name"]): float(spec["value"]) for spec in RUNTIME_PARAMETER_SPECS}
    joint_specs = tuple(
        sorted(
            active_specs,
            key=lambda spec: (-sensitivities[str(spec["parameter_name"])], str(spec["parameter_name"])),
        )[:JOINT_SEARCH_MAX_PARAMETERS]
    )
    preferred_values = {}
    for spec in joint_specs:
        parameter_name = str(spec["parameter_name"])
        best_coordinate = min(
            (
                row for row in optimization_rows
                if row["candidate_id"] in {f"{parameter_name}_lower", f"{parameter_name}_upper"}
            ),
            key=lambda row: (float(row["objective_value"]), row["candidate_id"]),
        )
        preferred_values[parameter_name] = (
            best_coordinate["candidate_id"].rsplit("_", maxsplit=1)[-1],
            float(json.loads(best_coordinate["parameter_values_json"])[parameter_name]),
        )
    for candidate_id, parameter_values in _joint_candidate_parameter_values(
        joint_specs,
        baseline,
        preferred_values,
    ):
        candidate_rows = run_runtime_grid(
            overlay,
            seeds,
            fitting_years,
            parameter_values,
            (BASELINE_SCENARIO,),
            "fitting_grid",
            candidate_id,
            FITTING_GRID_RUNTIME_FILE,
            n_farmers=n_farmers,
        )
        trace_rows.extend(candidate_rows)
        optimization_rows.append(
            {
                "stage": "fitting_grid",
                "iteration": str(len(optimization_rows) + 1),
                "candidate_id": candidate_id,
                "parameter_name": "joint_pair_runtime_grid",
                "parameter_values_json": _parameter_values_json(parameter_values),
                "objective_value": f"{_fitting_objective(moments, candidate_rows):.8f}",
                "selected": "false",
                "fitting_window": "channel_specific_named_simulation_years",
                "runtime_observation_count": str(len(candidate_rows)),
                "runtime_surface": RUNTIME_SURFACE,
            }
        )
    active_candidates = [
        row
        for row in optimization_rows
        if all(
            float(json.loads(row["parameter_values_json"])[name])
            == float(next(spec["value"] for spec in RUNTIME_PARAMETER_SPECS if spec["parameter_name"] == name))
            for name in set(json.loads(row["parameter_values_json"])) - active_names
        )
    ]
    selected = min(active_candidates, key=lambda row: (float(row["objective_value"]), row["candidate_id"]))
    selected["selected"] = "true"
    selected_values = {
        parameter_name: float(value)
        for parameter_name, value in json.loads(selected["parameter_values_json"]).items()
        if parameter_name in active_names
    }
    return trace_rows, optimization_rows, selected_values, sensitivities


def _runtime_provenance(rows: list[dict[str, str]], simulation_years: tuple[int, ...]) -> dict[str, str]:
    selected_years = {str(year) for year in simulation_years}
    selected = [
        row
        for row in rows
        if row["scenario_id"] == BASELINE_SCENARIO and row["simulation_year"] in selected_years
    ]
    if not selected:
        raise Task5Error("missing runtime provenance for validation window")
    first = selected[0]
    return {
        "runtime_seed": first["runtime_seed"],
        "scenario_id": first["scenario_id"],
        "simulation_year": first["simulation_year"],
        "runtime_surface": first["runtime_surface"],
        "runtime_input_hash": first["runtime_input_hash"],
        "trace_artifact": first["trace_artifact"],
        "trace_row": first["trace_row"],
    }


def target_rows(moments: list, runtime_rows: list[dict[str, str]], years: int) -> list[dict[str, str]]:  # noqa: ANN401
    _validate_formal_horizon(years)
    return [
        _target_row(moment, runtime_rows)
        for moment in moments
    ]


def _target_row(moment: object, runtime_rows: list[dict[str, str]]) -> dict[str, str]:
    fitting_years = model_years(
        moment.evidence_channel,
        moment.moment_id,
        validation=False,
    )
    validation_years = model_years(
        moment.evidence_channel,
        moment.moment_id,
        validation=True,
    )
    metadata = target_metadata(moment.moment_id)
    selected_validation_rows = [
        row
        for row in runtime_rows
        if row["scenario_id"] == BASELINE_SCENARIO
        and row["simulation_year"] in {str(year) for year in validation_years}
    ]
    return {
            "required_group": moment.required_group,
            "moment_id": moment.moment_id,
            "evidence_channel": moment.evidence_channel,
            "fitting_years": "|".join(str(year) for year in (CFPS_FITTING_YEARS if moment.evidence_channel == "CFPS" else NFPS_FITTING_YEARS)),
            "validation_years": "|".join(str(year) for year in (CFPS_VALIDATION_YEARS if moment.evidence_channel == "CFPS" else NFPS_VALIDATION_YEARS)),
            "target_value": f"{moment.validation_value:.8f}",
            "fitting_value": f"{moment.fitting_value:.8f}",
            "model_fitting_years": "|".join(str(year) for year in fitting_years),
            "model_validation_years": "|".join(str(year) for year in validation_years),
            "fitted_simulated_value": f"{_mean_runtime_years(runtime_rows, moment.moment_id, fitting_years):.8f}",
            "simulated_value": f"{_mean_runtime_years(runtime_rows, moment.moment_id, validation_years):.8f}",
            "weight": f"{moment.weight:.8f}",
            "unit": moment.unit,
            "source_data_path": "NFPS:2009-2015household.csv" if moment.evidence_channel == "NFPS" else "CFPS:cfps_calibration_summary_v6_unified.csv",
            **metadata,
            "runtime_observation_count": str(len(selected_validation_rows)),
            "runtime_provenance_path": RAW_RUNTIME_FILE,
            "comparison_window": named_year_window(validation_years),
            "claim_boundary": "model_internal_mechanism_validation_not_empirical_causal_effect_or_forecast",
            **_runtime_provenance(runtime_rows, validation_years),
    }


def _objective(rows: list[dict[str, str]], *, fitted: bool) -> float:
    column = "fitted_simulated_value" if fitted else "simulated_value"
    numerator = sum(float(row["weight"]) * abs(float(row[column]) - float(row["fitting_value"] if fitted else row["target_value"])) for row in rows)
    denominator = sum(float(row["weight"]) * abs(float(row["fitting_value"] if fitted else row["target_value"])) for row in rows)
    return numerator / denominator if denominator > 0.0 else math.inf


def write_calibration_artifacts(
    output: Path,
    moments: list,
    runtime_rows: list[dict[str, str]],
    years: int,
    optimization_rows: list[dict[str, str]],
    selected_parameter_values: dict[str, float],
    sensitivities: dict[str, float],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:  # noqa: ANN401
    rows = target_rows(moments, runtime_rows, years)
    write_csv(output / "calibration_targets.csv", tuple(rows[0]), rows)
    write_csv(output / "optimization_trace.csv", tuple(optimization_rows[0]), optimization_rows)
    parameter_rows = [
        {
            "parameter_id": spec["parameter_id"],
            "parameter_name": spec["parameter_name"],
            "value": f"{selected_parameter_values[str(spec['parameter_name'])]:.8f}",
            "lower_bound": f"{float(spec['lower_bound']):.8f}",
            "upper_bound": f"{float(spec['upper_bound']):.8f}",
            "selected_by_prescreen": str(sensitivities[str(spec['parameter_name'])] > MIN_MEASURED_SENSITIVITY).lower(),
            "local_sensitivity": f"{sensitivities[str(spec['parameter_name'])]:.8f}",
            "measured_local_sensitivity": f"{sensitivities[str(spec['parameter_name'])]:.8f}",
            "runtime_config_path": spec["runtime_config_path"],
            "runtime_binding_verified": "true",
            "selection_basis": "fitting_window_objective",
            "registered_source": "scripts/run_v4_8_national_calibration_validation.py:RUNTIME_PARAMETER_SPECS",
            "claim_boundary": spec["claim_boundary"],
        }
        for spec in RUNTIME_PARAMETER_SPECS
        if sensitivities[str(spec["parameter_name"])] > MIN_MEASURED_SENSITIVITY
    ]
    write_csv(output / "calibrated_parameters.csv", tuple(parameter_rows[0]), parameter_rows)
    return rows, parameter_rows


def run_task5(
    overlay: Path,
    seeds: range,
    years: int,
    output: Path,
    n_farmers: int = DEFAULT_PRESCREEN_FARMERS,
) -> Task5Result:
    _validate_formal_horizon(years)
    output.mkdir(parents=True, exist_ok=True)
    run_identity = expected_run_identity(overlay, seeds, years, n_farmers)
    cached_manifest = can_use_cache(output, run_identity)
    if cached_manifest is not None:
        return Task5Result(output / "run_manifest.json", bool(cached_manifest["calibration_gate_passed"]))
    write_worklog(output, "before_calibration", "target_moment_build_started")
    moments = build_moments()
    legacy_params = build_parameters(moments)
    validate_runtime_parameter_specs(overlay, n_farmers)
    if {moment.required_group for moment in moments} != set(REQUIRED_GROUPS):
        raise Task5Error("required target group coverage failed")
    fitting_horizon = 6
    write_worklog(output, "before_fitting_grid", f"real_mainmodel_fitting_grid_started;seeds={seeds.start}:{seeds.stop - 1};years={fitting_horizon}")
    fitting_grid_rows, optimization_rows, runtime_parameter_values, sensitivities = _fitting_grid_rows(
        overlay,
        seeds,
        fitting_horizon,
        moments,
        n_farmers,
    )
    for trace_row, fitting_grid_row in enumerate(fitting_grid_rows, start=1):
        fitting_grid_row["trace_row"] = str(trace_row)
    write_csv(output / FITTING_GRID_RUNTIME_FILE, tuple(fitting_grid_rows[0]), fitting_grid_rows)
    selected_candidate = next(row for row in optimization_rows if row["selected"] == "true")
    selected_candidate_id = selected_candidate["candidate_id"]
    selected_candidate_full_values = {
        parameter_name: float(value)
        for parameter_name, value in json.loads(selected_candidate["parameter_values_json"]).items()
    }
    write_worklog(output, "after_fitting_grid", f"candidate_count={len(optimization_rows)};selected={selected_candidate_id}")
    write_worklog(output, "before_runtime", f"real_mainmodel_validation_runtime_started;seeds={seeds.start}:{seeds.stop - 1};years={years}")
    runtime_rows = run_runtime_grid(
        overlay,
        seeds,
        years,
        runtime_parameter_values,
        SCENARIOS,
        "validation_runtime",
        selected_candidate_id,
        RAW_RUNTIME_FILE,
        n_farmers=n_farmers,
    )
    for trace_row, runtime_row in enumerate(runtime_rows, start=1):
        runtime_row["trace_row"] = str(trace_row)
    if not runtime_rows or any(row["runtime_surface"] != RUNTIME_SURFACE for row in runtime_rows):
        raise Task5Error("real MainModel runtime provenance missing")
    write_csv(output / RAW_RUNTIME_FILE, tuple(runtime_rows[0]), runtime_rows)
    write_worklog(output, "after_runtime", f"real_mainmodel_rows={len(runtime_rows)}")
    rows, parameter_rows = write_calibration_artifacts(
        output,
        moments,
        runtime_rows,
        years,
        optimization_rows,
        runtime_parameter_values,
        sensitivities,
    )
    write_worklog(output, "after_calibration", "fitting_window_runtime_grid_completed;validation_window_not_used_for_selection")
    write_worklog(output, "before_validation", "noncompensatory_validation_started")
    metrics = validation_metrics(rows)
    write_csv(output / "validation_metrics.csv", tuple(metrics[0]), metrics)
    gate_passed = all(row["gate_passed"] == "true" for row in metrics)
    write_worklog(output, "after_validation", f"validation_gate_passed={str(gate_passed).lower()}")
    write_worklog(output, "before_paired_worldlines", f"seeds={seeds.start}:{seeds.stop - 1};years={years};scenarios={len(SCENARIOS)}")
    paired = paired_worldline_rows(runtime_rows)
    write_csv(output / "paired_worldlines.csv", tuple(paired[0]), paired)
    write_worklog(output, "after_paired_worldlines", f"rows={len(paired)}")
    sources = source_data_rows()
    write_csv(output / "source_data_metrics.csv", tuple(sources[0]), sources)
    manifest = {
        "run_status": "completed", "run_identity_sha256": run_identity, "cache_used": False, "model_version": "V4.8", "overlay_path": str(overlay), "overlay_sha256": hash_file(overlay), "data_manifest_sha256": data_manifest_sha256(), "code_tree_or_dirty_patch_sha256": code_tree_or_dirty_patch_sha256(), "seed_set": list(seeds), "seeds": [seeds.start, seeds.stop - 1], "started_at_utc": datetime.now(timezone.utc).isoformat(), "years": years, "n_farmers": n_farmers, "scenarios": list(SCENARIOS),
        "cfps_fitting_years": list(CFPS_FITTING_YEARS), "cfps_validation_years": list(CFPS_VALIDATION_YEARS), "nfps_fitting_years": list(NFPS_FITTING_YEARS), "nfps_validation_years": list(NFPS_VALIDATION_YEARS),
        "excluded_sources": {"CLES": "out_of_scope_jiangsu_only_not_used_for_national_calibration", "NFPS_2016_2017": "excluded"}, "required_groups": list(REQUIRED_GROUPS), "calibrated_parameter_count": sum(row["selected_by_prescreen"] == "true" for row in parameter_rows), "runtime_active_parameter_count": len(parameter_rows), "legacy_unbound_parameters_removed": [param.parameter_name for param in legacy_params], "calibration_gate_passed": gate_passed,
        "critical_input_identity": critical_input_identity(n_farmers), "legacy_calibration_bridge_status": LEGACY_CALIBRATION_BRIDGE_STATUS, "apply_calibrated_params": False,
        "runtime_surface": RUNTIME_SURFACE, "pairing_method": "same_seed_initial_state", "formal_parameter_fit_completed": True,
        "model_fitting_years": {
            "NFPS_landholding": [2010, 2011, 2012],
            "NFPS_annual_flows_and_income": [2011, 2012],
            "CFPS_livelihood": [2010, 2012, 2014, 2016],
        },
        "model_validation_years": {
            channel: list(model_years(channel, "", validation=True))
            for channel in ("NFPS", "CFPS")
        },
        "claim_boundary": "ABM internal paired worldlines diagnose model mechanisms; paired CIs are model-internal worldline uncertainty, not empirical causal estimates or forecasts.", "protected_hashes_current": protected_hashes(),
        "parameter_sensitivity_definition": "max_abs_perturbation_minus_baseline_over_max_abs_baseline_or_one",
        "parameter_sensitivities": sensitivities,
        "parameter_sensitivity_disposition": {
            parameter_name: "active" if sensitivity > MIN_MEASURED_SENSITIVITY else "non_binding_under_prescreen"
            for parameter_name, sensitivity in sensitivities.items()
        },
        "selected_candidate_id": selected_candidate_id, "selected_candidate_full_values": selected_candidate_full_values,
        "selected_runtime_parameters": runtime_parameter_values, "fitting_grid_candidate_count": len(optimization_rows), "fitting_grid_runtime_file": FITTING_GRID_RUNTIME_FILE,
        "artifacts": {"calibration_targets": str(output / "calibration_targets.csv"), "optimization_trace": str(output / "optimization_trace.csv"), "calibrated_parameters": str(output / "calibrated_parameters.csv"), "validation_metrics": str(output / "validation_metrics.csv"), "paired_worldlines": str(output / "paired_worldlines.csv"), "raw_runtime_observations": str(output / RAW_RUNTIME_FILE), "fitting_grid_runtime_observations": str(output / FITTING_GRID_RUNTIME_FILE), "source_data_metrics": str(output / "source_data_metrics.csv")},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return Task5Result(output / "run_manifest.json", gate_passed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V4.8 Task 5 national calibration, validation, and paired fixed-rule worldlines.")
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--years", required=True, type=int)
    parser.add_argument("--n-farmers", type=int, default=DEFAULT_PRESCREEN_FARMERS)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = yaml.safe_load(args.overlay.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("policy", {}).get("policy_mode") != "rule":
            raise Task5Error("overlay must declare policy_mode=rule")
        result = run_task5(args.overlay, parse_seed_range(args.seeds), args.years, args.output, args.n_farmers)
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, ValueError, Task5Error) as exc:
        print(f"FAIL v4.8 Task5 runner: {exc}", file=sys.stderr)
        return 1
    print(f"PASS v4.8 Task5 runner: manifest={result.manifest_path}; calibration_gate_passed={str(result.gate_passed).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
