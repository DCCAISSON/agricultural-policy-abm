from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from core.model import MainModel
from core.model_builder import ModelBuilder
from core.reproducibility import RandomController
from modules.production_succession_accounting import ProductionSuccessionAccount
from scripts.run_v4_8_national_calibration_validation import (
    _apply_runtime_parameters,
    _runtime_observation,
    _stable_hash,
)
from scripts.run_v4_9_production_succession_accounting import _observe_final_use
from scripts.u2_unified_paper_model_accounts import (
    allocation_hhi,
    classify_household_action,
    classify_professional_action,
    land_stats,
    mass_by_action,
)
from scripts.v5_r6_runtime import runtime_config


__all__ = (
    "U2_SCENARIOS",
    "classify_household_action",
    "classify_professional_action",
    "run_scenario",
    "selected_parameters",
    "u2_runtime_config",
)
PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_OVERLAY: Final = PROJECT_ROOT / "config/experiments/v4_8_national_fixed_rule.yaml"
DEFAULT_COHORT: Final = PROJECT_ROOT / "data/derived_runtime/v5_r6/nfps_fixed_empirical_cohort_2010_k600.json"
DEFAULT_R6_MANIFEST: Final = PROJECT_ROOT / "results/formal/v5_r6_limited_smm_calibration_20260817/run_manifest.json"


@dataclass(frozen=True, slots=True)
class U2Scenario:
    scenario_id: str
    label_cn: str
    service_capacity_scale: float


@dataclass(frozen=True, slots=True)
class U2Run:
    annual_rows: tuple[dict[str, Any], ...]
    allocation_rows: tuple[dict[str, Any], ...]


class U2RuntimeError(RuntimeError):
    pass


U2_SCENARIOS: Final = (
    U2Scenario("baseline_service_capacity", "\u57fa\u51c6\u670d\u52a1\u5bb9\u91cf", 1.0),
    U2Scenario("half_service_capacity", "\u670d\u52a1\u5bb9\u91cf\u51cf\u534a", 0.5),
)


def selected_parameters(manifest_path: Path = DEFAULT_R6_MANIFEST) -> dict[str, float]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = payload.get("selected_parameters") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise U2RuntimeError("R6 selected parameters are missing")
    return {str(key): float(value) for key, value in raw.items()}


def _scale_service_capacity(config: dict[str, Any], scale: float) -> None:
    production = config.setdefault("production", {})
    service = production.setdefault("agricultural_service", {})
    economics = service.setdefault("provider_economics", {})
    capacity = economics.setdefault("initial_capacity", {})
    for bound in ("min", "max"):
        capacity[bound] = float(capacity[bound]) * scale


def u2_runtime_config(
    seed: int,
    years: int,
    *,
    service_capacity_scale: float = 1.0,
    cohort_path: Path = DEFAULT_COHORT,
    overlay_path: Path = DEFAULT_OVERLAY,
    parameters: dict[str, float] | None = None,
) -> dict[str, Any]:
    if years < 1:
        raise U2RuntimeError("U2 runtime requires at least one year")
    if not 0.0 < service_capacity_scale <= 1.0:
        raise U2RuntimeError("service capacity scale must be in (0, 1]")
    config = runtime_config(overlay_path, cohort_path, seed, years)
    _apply_runtime_parameters(config, parameters or selected_parameters())
    _scale_service_capacity(config, service_capacity_scale)
    config["unified_paper_model"] = {
        "enabled": True,
        "service_environment_only": True,
        "game_enabled": False,
        "runtime_stage": "U2_unified_paper_model_runtime",
    }
    config["mechanism_switches"] = {
        **config.get("mechanism_switches", {}),
        "use_learning_farmer": False,
        "use_learning_enterprise": False,
    }
    config["policy"]["policy_mode"] = "rule"
    config["policy"]["rl_enabled"] = False
    config["policy"].setdefault("government", {})["adaptive_adjustment_enabled"] = False
    config["population_dynamics"]["paper_model_exit_mode"] = "livelihood_exit_only"
    config["u2_runtime_config_source"] = "master_copy_plus_fixed_rule_overlay_plus_r6_parameters_plus_fixed_cohort"
    return config


def _initial_state_hash(model: MainModel) -> str:
    payload = [
        {
            "id": str(getattr(farmer, "unique_id", "")),
            "source": str(getattr(farmer, "source_representation_weight_q", "")),
            "patches": [float(getattr(patch, "area", 0.0) or 0.0) for patch in getattr(farmer, "land_patches", []) or []],
        }
        for farmer in model.farmers
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_scenario(
    scenario: U2Scenario,
    *,
    seed: int = 17,
    years: int = 2,
    cohort_path: Path = DEFAULT_COHORT,
    overlay_path: Path = DEFAULT_OVERLAY,
    parameters: dict[str, float] | None = None,
) -> U2Run:
    RandomController.set_seed(seed)
    config = u2_runtime_config(
        seed,
        years,
        service_capacity_scale=scenario.service_capacity_scale,
        cohort_path=cohort_path,
        overlay_path=overlay_path,
        parameters=parameters,
    )
    config_hash = _stable_hash(config)
    annual_rows: list[dict[str, Any]] = []
    allocation_rows: list[dict[str, Any]] = []
    with Path(os.devnull).open("w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            model = ModelBuilder(config=config).build()
            initial_hash = _initial_state_hash(model)
            scheduled_agents = tuple(getattr(model.schedule, "agents", ()) or ())
            registered_agents = tuple((getattr(model, "agent_registry", {}) or {}).values())
            provider_in_scheduler_count = sum(
                any(provider is agent for agent in scheduled_agents)
                for provider in model.service_providers
            )
            provider_in_registry_count = sum(
                any(provider is agent for agent in registered_agents)
                for provider in model.service_providers
            )
            provider_role_count = sum(
                getattr(provider, "service_role", "") == "capacity_matching_environment"
                for provider in model.service_providers
            )
            account = ProductionSuccessionAccount()
            setattr(model, "production_succession_account", account)
            population_manager = getattr(model, "population_manager", None)
            if population_manager is None:
                raise U2RuntimeError("population manager is missing")
            setattr(population_manager.land_bank, "production_succession_account", account)
            prior_hook = getattr(model, "_completed_production_year_snapshot_hook", None)
            captured: dict[str, str] = {}

            def capture(current_model: MainModel) -> None:
                if callable(prior_hook):
                    _ = prior_hook(current_model)
                year = int(current_model.simulation_base_year + current_model.step_count)
                _observe_final_use(current_model, account, year)
                captured.clear()
                captured.update(_runtime_observation(current_model, seed, scenario.scenario_id, current_model.step_count, config_hash, initial_hash))

            setattr(model, "_completed_production_year_snapshot_hook", capture)
            for _ in range(years):
                trace_start = len(getattr(model, "v4_8_fixed_rule_trace", []) or [])
                weights = {
                    str(getattr(operator, "unique_id", "")): float(getattr(operator, "representation_weight_q", 1.0) or 1.0)
                    for operator in [*model.farmers, *model.agricultural_enterprises]
                }
                household_ids = {
                    str(getattr(farmer, "unique_id", "")) for farmer in model.farmers
                }
                professional_ids = {
                    str(getattr(enterprise, "unique_id", ""))
                    for enterprise in model.agricultural_enterprises
                }
                active_household_mass = sum(weights[operator_id] for operator_id in household_ids)
                active_professional_mass = sum(
                    weights[operator_id] for operator_id in professional_ids
                )
                model.step()
                if not captured:
                    raise U2RuntimeError("annual observation hook did not run")
                year = int(model.simulation_base_year + model.step_count)
                trace = list(getattr(model, "v4_8_fixed_rule_trace", []) or [])[trace_start:]
                household_mass = mass_by_action(trace, weights, "household")
                professional_mass = mass_by_action(trace, weights, "scale")
                land = land_stats(model, year)
                allocations = list(land.get("allocation_trace", []) or [])
                for allocation in allocations:
                    allocation_rows.append({"scenario_id": scenario.scenario_id, "runtime_seed": seed, "simulation_year": year, **allocation})
                allocation_rows.append({"scenario_id": scenario.scenario_id, "runtime_seed": seed, "simulation_year": year, "trace_type": "annual_account", "area_unit": "mu", "released_area_mu": float(land.get("released_area_mu", 0.0) or 0.0), "allocated_area_mu": float(land.get("allocated_area_mu", 0.0) or 0.0), "unabsorbed_area_mu": float(land.get("unabsorbed_area_mu", 0.0) or 0.0), "allocation_hhi": allocation_hhi(allocations)})
                stats = dict(getattr(model, "service_market_stats", {}) or {})
                operators = [*model.farmers, *model.agricultural_enterprises]
                annual_rows.append({
                    "scenario_id": scenario.scenario_id,
                    "scenario_label_cn": scenario.label_cn,
                    "runtime_seed": seed,
                    "simulation_year": year,
                    "runtime_input_hash": config_hash,
                    "initial_state_sha256": initial_hash,
                    "active_household_mass": active_household_mass,
                    "active_professional_mass": active_professional_mass,
                    "self_farm_mass": household_mass["SelfFarm"],
                    "rent_out_mass": household_mass["RentOut"],
                    "household_expand_mass": household_mass["Expand"],
                    "shrink_exit_mass": household_mass["ShrinkOrExit"],
                    "professional_maintain_mass": professional_mass["Maintain"],
                    "professional_expand_mass": professional_mass["Expand"],
                    "professional_shrink_mass": professional_mass["Shrink"],
                    "professional_exit_mass": professional_mass["ExitLocalOperation"],
                    "released_land_mu": float(land.get("released_area_mu", 0.0) or 0.0),
                    "eligible_weighted_demand_mu": float(land.get("eligible_weighted_demand_mu", 0.0) or 0.0),
                    "allocated_land_mu": float(land.get("allocated_area_mu", 0.0) or 0.0),
                    "unabsorbed_land_mu": float(land.get("unabsorbed_area_mu", 0.0) or 0.0),
                    "allocation_hhi": allocation_hhi(allocations),
                    "service_demand_mu": float(stats.get("service_demand_area", 0.0) or 0.0),
                    "service_matched_mu": float(stats.get("service_matched_area", 0.0) or 0.0),
                    "service_capacity_mu": sum(float(getattr(provider, "current_capacity", 0.0) or 0.0) for provider in model.service_providers),
                    "service_price_or_cost_yuan_per_mu": float(stats.get("total_service_fee", 0.0) or 0.0) / max(1e-12, float(stats.get("total_service_area", 0.0) or 0.0)),
                    "service_provider_profit_total": float(stats.get("provider_profit_total", 0.0) or 0.0),
                    "service_provider_count": len(model.service_providers),
                    "service_environment_role_count": provider_role_count,
                    "service_provider_in_scheduler_count": provider_in_scheduler_count,
                    "service_provider_in_registry_count": provider_in_registry_count,
                    "learning_enabled": False,
                    "game_enabled": False,
                    "grain_area_mu": sum(float(getattr(operator, "grain_area", 0.0) or 0.0) for operator in operators),
                    "grain_output_kg": float(captured["grain_value"]),
                    "agricultural_income_yuan_per_household": float(captured["agricultural_income"]),
                    "off_farm_income_yuan": sum(float(getattr(farmer, "off_farm_income", 0.0) or 0.0) for farmer in model.farmers),
                    "rent_income_yuan": sum(float(getattr(farmer, "land_rental_income", 0.0) or 0.0) for farmer in model.farmers),
                    "professional_profit_yuan": sum(float(getattr(enterprise, "profit", 0.0) or 0.0) for enterprise in model.agricultural_enterprises),
                    "fiscal_spending_yuan": float(captured["fiscal_spending"]),
                })
    return U2Run(tuple(annual_rows), tuple(allocation_rows))
