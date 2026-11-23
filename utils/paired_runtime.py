from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from core.model_builder import ModelBuilder
from core.reproducibility import RandomController, build_numpy_rng
from utils.policy_runtime import (
    apply_policy_runtime,
    apply_scale_subsidy_runtime,
    assert_policy_runtime_matches,
    snapshot_policy_runtime_from_config,
    snapshot_policy_runtime_from_model,
)
from visualization.unified_indicators import UnifiedIndicators, collect_indicators_from_model


@dataclass
class RuntimeScenario:
    name: str
    description: str = ""
    policy_scenario: str = "baseline_cash"
    policy_overrides: Dict[str, Any] = field(default_factory=dict)
    scale_subsidy: Dict[str, Any] = field(default_factory=dict)
    rl_enabled: Optional[bool] = None
    ai_mechanism: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShockWindow:
    start_year: int
    duration: int
    magnitude: float
    label: str = "temporary_yield_shock"

    @property
    def end_year(self) -> int:
        return self.start_year + self.duration - 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "start_year": self.start_year,
            "duration": self.duration,
            "end_year": self.end_year,
            "magnitude": self.magnitude,
        }


def load_config(config_path: Path | str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_fixed_farmer_configs(n_farmers: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = build_numpy_rng(seed, "paired_runtime_farmer_configs")
    configs: List[Dict[str, Any]] = []
    for i in range(n_farmers):
        if i < n_farmers * 0.29:
            livelihood_type = "Pure_Farmer"
            land_area = rng.lognormal(mean=2.5, sigma=0.5)
            off_farm_wage = rng.uniform(0.4, 0.7)
        elif i < n_farmers * 0.64:
            livelihood_type = "Pluriactive"
            land_area = rng.lognormal(mean=1.8, sigma=0.6)
            off_farm_wage = rng.uniform(0.8, 1.2)
        else:
            livelihood_type = "Non_Farmer"
            land_area = rng.lognormal(mean=1.0, sigma=0.8)
            off_farm_wage = rng.uniform(1.2, 1.8)

        configs.append(
            {
                "unique_id": f"farmer_{i}",
                "initial_land_area": float(land_area),
                "initial_capital": float(rng.lognormal(mean=9.2, sigma=0.6)),
                "farming_ability": float(rng.uniform(0.7, 1.3)),
                "off_farm_wage_level": float(off_farm_wage),
                "risk_aversion": float(rng.uniform(1.5, 3.0)),
                "age": int(rng.randint(25, 65)),
                "initial_livelihood": livelihood_type,
            }
        )
    return configs


def apply_farmer_configs_to_model(model: Any, farmer_configs: Sequence[Dict[str, Any]]) -> List[str]:
    from agents.farmer_agent import LivelihoodStrategy

    farmers = [agent for agent in model.schedule.agents if hasattr(agent, "farming_ability")]
    tracked_ids: List[str] = []
    for farmer, config in zip(farmers, farmer_configs):
        if hasattr(farmer, "land_patches") and farmer.land_patches:
            target_area = float(config["initial_land_area"])
            for patch in farmer.land_patches:
                patch.area = target_area / max(len(farmer.land_patches), 1)

        farmer.capital = float(config["initial_capital"])
        farmer.farming_ability = float(config["farming_ability"])
        farmer.off_farm_wage_level = float(config["off_farm_wage_level"])
        farmer.base_risk_aversion = float(config["risk_aversion"])
        farmer.risk_aversion = float(config["risk_aversion"])
        farmer.age = int(config["age"])

        initial_livelihood = config.get("initial_livelihood", "Pluriactive")
        if hasattr(farmer, "livelihood_strategy"):
            matched = False
            for livelihood in LivelihoodStrategy:
                if livelihood.value == initial_livelihood:
                    farmer.livelihood_strategy = livelihood
                    matched = True
                    break
            if not matched:
                farmer.livelihood_strategy = LivelihoodStrategy.PLURIACTIVE

        farmer.tracked_id = config["unique_id"]
        tracked_ids.append(farmer.tracked_id)
    return tracked_ids


def extract_farmer_state(farmer: Any) -> Dict[str, Any]:
    land_area = 0.0
    if hasattr(farmer, "land_patches") and farmer.land_patches:
        land_area = float(sum(getattr(patch, "area", 0.0) for patch in farmer.land_patches))

    livelihood = "Unknown"
    if hasattr(farmer, "livelihood_strategy"):
        livelihood = getattr(farmer.livelihood_strategy, "value", str(farmer.livelihood_strategy))

    income = float(getattr(farmer, "last_income", 0.0) or getattr(farmer, "total_income", 0.0) or 0.0)

    return {
        "tracked_id": getattr(farmer, "tracked_id", str(getattr(farmer, "unique_id", "unknown"))),
        "land_area": land_area,
        "capital": float(getattr(farmer, "capital", 0.0)),
        "income": income,
        "livelihood": livelihood,
        "farming_ability": float(getattr(farmer, "farming_ability", 0.0)),
        "age": int(getattr(farmer, "age", 0) or 0),
        "is_farming": land_area > 0.5 and livelihood != "Non_Farmer",
    }


def indicators_to_dict(indicators: UnifiedIndicators) -> Dict[str, Any]:
    if is_dataclass(indicators):
        record = asdict(indicators)
    else:
        record = dict(vars(indicators))

    record["total_subsidy_amount"] = indicators.total_subsidy_amount
    record["gini_coefficient"] = indicators.gini_coefficient
    return record


def summarize_governance_workload(model: Any) -> Dict[str, Any]:
    villages = list(getattr(model, "village_collectives", []) or [])
    if not villages:
        return {
            "n_village_collectives": 0,
            "governance_registered_farmers": 0,
            "governance_transfers_this_step": 0,
            "governance_total_service_fee": 0.0,
            "governance_integrated_land_area": 0.0,
            "governance_subsidy_pre_review_coverage": 0.0,
            "governance_subsidy_pre_review_approval_rate": 0.0,
            "governance_cross_village_contracts": 0,
            "governance_cross_village_transfer_area": 0.0,
            "governance_cross_village_partners": 0.0,
            "governance_transfer_success_rate": 0.0,
            "governance_objective_function_value": 0.0,
            "governance_annual_service_fee": 0.0,
            "governance_accumulated_service_fee": 0.0,
        }

    stats = [village.get_statistics() for village in villages if hasattr(village, "get_statistics")]
    if not stats:
        return {
            "n_village_collectives": len(villages),
            "governance_registered_farmers": 0,
            "governance_transfers_this_step": 0,
            "governance_total_service_fee": 0.0,
            "governance_integrated_land_area": 0.0,
            "governance_subsidy_pre_review_coverage": 0.0,
            "governance_subsidy_pre_review_approval_rate": 0.0,
            "governance_cross_village_contracts": 0,
            "governance_cross_village_transfer_area": 0.0,
            "governance_cross_village_partners": 0.0,
            "governance_transfer_success_rate": 0.0,
            "governance_objective_function_value": 0.0,
            "governance_annual_service_fee": 0.0,
            "governance_accumulated_service_fee": 0.0,
        }

    def _avg(key: str, nested_key: str | None = None) -> float:
        values = []
        for item in stats:
            payload = item.get(key, 0.0) if nested_key is None else (item.get(key, {}) or {}).get(nested_key, 0.0)
            try:
                values.append(float(payload or 0.0))
            except (TypeError, ValueError):
                continue
        return float(sum(values) / len(values)) if values else 0.0

    def _sum(key: str, nested_key: str | None = None) -> float:
        total = 0.0
        for item in stats:
            payload = item.get(key, 0.0) if nested_key is None else (item.get(key, {}) or {}).get(nested_key, 0.0)
            try:
                total += float(payload or 0.0)
            except (TypeError, ValueError):
                continue
        return float(total)

    return {
        "n_village_collectives": len(stats),
        "governance_registered_farmers": int(round(_sum("registered_farmers"))),
        "governance_transfers_this_step": int(round(_sum("transfers_this_step"))),
        "governance_total_service_fee": _sum("total_service_fee"),
        "governance_integrated_land_area": _sum("integrated_land_area"),
        "governance_subsidy_pre_review_coverage": _avg("subsidy_pre_review_coverage"),
        "governance_subsidy_pre_review_approval_rate": _avg("subsidy_pre_review_approval_rate"),
        "governance_cross_village_contracts": int(round(_sum("cross_village_contracts"))),
        "governance_cross_village_transfer_area": _sum("cross_village_transfer_area"),
        "governance_cross_village_partners": _avg("cross_village_partners"),
        "governance_transfer_success_rate": _avg("economy_state", "transfer_success_rate"),
        "governance_objective_function_value": _avg("objective_function_value"),
        "governance_annual_service_fee": _sum("economy_state", "annual_service_fee"),
        "governance_accumulated_service_fee": _sum("economy_state", "accumulated_service_fee"),
    }


def overlay_shocks_on_worldline(worldline: Any, shocks: Sequence[ShockWindow]) -> Any:
    worldline_copy = deepcopy(worldline)
    for shock in shocks:
        for year in range(shock.start_year, shock.start_year + shock.duration):
            index = year - 1
            if index < 0:
                continue
            if hasattr(worldline_copy, "climate_shocks"):
                if index >= len(worldline_copy.climate_shocks):
                    continue
                worldline_copy.climate_shocks[index] = True
                worldline_copy.climate_severities[index] = max(
                    float(worldline_copy.climate_severities[index]),
                    abs(float(shock.magnitude)),
                )
            elif isinstance(worldline_copy, dict):
                if index >= len(worldline_copy.get("climate_shocks", [])):
                    continue
                worldline_copy["climate_shocks"][index] = True
                worldline_copy["climate_severities"][index] = max(
                    float(worldline_copy["climate_severities"][index]),
                    abs(float(shock.magnitude)),
                )
    return worldline_copy


def worldline_to_dict(worldline: Any) -> Dict[str, Any]:
    if hasattr(worldline, "to_dict"):
        payload = worldline.to_dict()
    elif isinstance(worldline, dict):
        payload = deepcopy(worldline)
    else:
        raise TypeError(f"Unsupported worldline type: {type(worldline)!r}")
    return payload


def worldline_hash(worldline: Any) -> str:
    payload = worldline_to_dict(worldline)
    content = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def hash_farmer_configs(farmer_configs: Sequence[Dict[str, Any]]) -> str:
    content = json.dumps(list(farmer_configs), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def get_worldline_identifier(worldline: Any) -> str:
    return str(getattr(worldline, "hash_id", "")) or worldline_hash(worldline)


def hash_model_microstate(model: Any) -> str:
    farmers = []
    for agent in model.schedule.agents:
        if hasattr(agent, "farming_ability"):
            farmers.append(extract_farmer_state(agent))
    farmers.sort(key=lambda item: item["tracked_id"])
    content = json.dumps(farmers, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def build_model_for_scenario(
    base_config: Dict[str, Any],
    *,
    scenario: RuntimeScenario,
    seed: int,
    steps: int,
    n_farmers: Optional[int] = None,
    worldline: Optional[Any] = None,
    farmer_configs: Optional[Sequence[Dict[str, Any]]] = None,
    policy_dashboard_path: Optional[Path | str] = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str]:
    cfg = deepcopy(base_config)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["steps"] = int(steps)
    cfg.setdefault("policy", {})
    cfg.setdefault("scale", {})
    cfg.setdefault("results", {})
    if n_farmers is not None:
        cfg["scale"]["n_farmers"] = int(n_farmers)
    if policy_dashboard_path is not None:
        cfg["results"]["policy_dashboard_path"] = str(policy_dashboard_path)

    apply_policy_runtime(
        cfg,
        scenario_id=scenario.policy_scenario,
        overrides=deepcopy(scenario.policy_overrides),
    )
    if scenario.scale_subsidy:
        apply_scale_subsidy_runtime(
            cfg,
            enabled=scenario.scale_subsidy.get("enabled"),
            base_rate=scenario.scale_subsidy.get("base_rate"),
            base_rate_multiplier=scenario.scale_subsidy.get("base_rate_multiplier"),
        )
    if scenario.rl_enabled is not None:
        cfg["policy"]["rl_enabled"] = bool(scenario.rl_enabled)

    expected_policy = snapshot_policy_runtime_from_config(cfg)

    RandomController.set_seed(int(seed))

    builder = ModelBuilder(cfg)
    model = builder.build()
    assert_policy_runtime_matches(model, expected_policy)

    if worldline is not None and hasattr(model, "environment"):
        model.environment.set_worldline(worldline, get_worldline_identifier(worldline))

    if farmer_configs is not None:
        apply_farmer_configs_to_model(model, farmer_configs)

    return model, cfg, expected_policy, hash_model_microstate(model)


def run_model_collect_history(
    model: Any,
    *,
    scenario_name: str,
    steps: int,
    worldline_id: Optional[str] = None,
    tracked_farmer_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    aggregate_history: List[Dict[str, Any]] = []
    tracked_history: Dict[str, List[Dict[str, Any]]] = {
        farmer_id: [] for farmer_id in (tracked_farmer_ids or [])
    }

    def record_current(year_index: int) -> None:
        indicators = collect_indicators_from_model(model)
        aggregate_record = indicators_to_dict(indicators)
        aggregate_record.update(
            {
                "scenario_name": scenario_name,
                "year_index": year_index,
                "worldline_id": worldline_id,
                "policy_runtime": snapshot_policy_runtime_from_model(model),
            }
        )
        aggregate_record.update(summarize_governance_workload(model))
        aggregate_history.append(aggregate_record)

        if tracked_history:
            lookup = {}
            for agent in model.schedule.agents:
                tracked_id = getattr(agent, "tracked_id", None)
                if tracked_id:
                    lookup[tracked_id] = agent
            for tracked_id in tracked_history:
                farmer = lookup.get(tracked_id)
                if farmer is None:
                    tracked_history[tracked_id].append(
                        {
                            "tracked_id": tracked_id,
                            "year_index": year_index,
                            "scenario_name": scenario_name,
                            "alive": False,
                            "worldline_id": worldline_id,
                        }
                    )
                    continue
                state = extract_farmer_state(farmer)
                state.update(
                    {
                        "year_index": year_index,
                        "scenario_name": scenario_name,
                        "alive": True,
                        "worldline_id": worldline_id,
                    }
                )
                tracked_history[tracked_id].append(state)

    record_current(0)
    for year_index in range(1, steps + 1):
        model.step()
        record_current(year_index)

    return aggregate_history, tracked_history
