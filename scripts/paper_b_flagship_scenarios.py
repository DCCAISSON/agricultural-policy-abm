# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
# --- How to run ---
# python -m pytest -q tests/test_paper_b_flagship_counterfactual.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypeAlias, cast

import yaml


ROOT: Final = Path(__file__).resolve().parents[1]
SSP_CONFIG: Final = ROOT / "config" / "ssp_scenarios.yaml"
SUPPORT_LEVELS: Final = (0, 100, 200, 350, 500)
COORDINATION_LEVELS: Final = {"C0": 0.5, "C1": 1.0}
REFERENCE_SCENARIO_ID: Final = "ssp2_s350_c1"
BASE_SUPPORT_YUAN_PER_MU: Final = 350.0
BASE_DECLINE_RATE: Final = -0.0265
ConfigScalar: TypeAlias = str | int | float | bool | None
ConfigValue: TypeAlias = ConfigScalar | dict[str, "ConfigValue"] | list["ConfigValue"]
Config: TypeAlias = dict[str, ConfigValue]


class RuntimeEnvironment(Protocol):
    land_rental_price: float
    market_prices: dict[str, float]


class RuntimeModel(Protocol):
    config: Config
    environment: RuntimeEnvironment


@dataclass(frozen=True, slots=True)
class SSPPathway:
    code: str
    name: str
    gdp_growth_rate: float
    urbanization_rate_change: float
    grain_yield_growth_rate: float
    cash_yield_growth_rate: float
    input_cost_growth_rate: float
    base_rental_index: float
    rental_growth_rate: float


@dataclass(frozen=True, slots=True)
class FlagshipScenario:
    scenario_id: str
    ssp_code: str
    ssp_name: str
    support_yuan_per_mu: int
    coordination_code: str
    coordination_scale: float
    gdp_growth_rate: float
    rural_decline_rate: float
    grain_yield_growth_rate: float
    cash_yield_growth_rate: float
    input_cost_growth_rate: float
    base_rental_index: float
    rental_growth_rate: float


class FlagshipScenarioError(RuntimeError):
    pass


def load_ssp_pathways(path: Path = SSP_CONFIG) -> tuple[SSPPathway, ...]:
    payload = cast(Config, yaml.safe_load(path.read_text(encoding="utf-8")))
    keys = (
        "ssp1_green_development",
        "ssp2_middle_road",
        "ssp3_regional_rivalry",
        "ssp4_inequality",
        "ssp5_fossil_development",
    )
    output: list[SSPPathway] = []
    for index, key in enumerate(keys, start=1):
        row = payload.get(key)
        if not isinstance(row, dict):
            raise FlagshipScenarioError(f"SSP configuration is missing: {key}")
        economic = _section(row, "economic")
        technology = _section(row, "technology")
        land = _section(row, "land_market")
        output.append(
            SSPPathway(
                code=f"SSP{index}",
                name=str(row.get("name", f"SSP{index}")),
                gdp_growth_rate=_number(economic, "gdp_growth_rate"),
                urbanization_rate_change=_number(
                    economic, "urbanization_rate_change"
                ),
                grain_yield_growth_rate=_number(
                    technology, "yield_growth_grain"
                ),
                cash_yield_growth_rate=_number(
                    technology, "yield_growth_cash"
                ),
                input_cost_growth_rate=_number(
                    technology, "input_cost_growth"
                ),
                base_rental_index=_number(land, "base_rental"),
                rental_growth_rate=_number(land, "rental_growth"),
            )
        )
    return tuple(output)


def flagship_scenarios() -> tuple[FlagshipScenario, ...]:
    pathways = load_ssp_pathways()
    middle = next(pathway for pathway in pathways if pathway.code == "SSP2")
    scenarios: list[FlagshipScenario] = []
    for pathway in pathways:
        decline = BASE_DECLINE_RATE * (
            pathway.urbanization_rate_change
            / middle.urbanization_rate_change
        )
        for support in SUPPORT_LEVELS:
            for coordination_code, scale in COORDINATION_LEVELS.items():
                scenario_id = (
                    f"{pathway.code.lower()}_s{support}_{coordination_code.lower()}"
                )
                scenarios.append(
                    FlagshipScenario(
                        scenario_id=scenario_id,
                        ssp_code=pathway.code,
                        ssp_name=pathway.name,
                        support_yuan_per_mu=support,
                        coordination_code=coordination_code,
                        coordination_scale=scale,
                        gdp_growth_rate=pathway.gdp_growth_rate,
                        rural_decline_rate=decline,
                        grain_yield_growth_rate=pathway.grain_yield_growth_rate,
                        cash_yield_growth_rate=pathway.cash_yield_growth_rate,
                        input_cost_growth_rate=pathway.input_cost_growth_rate,
                        base_rental_index=(
                            pathway.base_rental_index / middle.base_rental_index
                        ),
                        rental_growth_rate=pathway.rental_growth_rate,
                    )
                )
    return tuple(scenarios)


def apply_scenario_config(config: Config, scenario: FlagshipScenario) -> None:
    policy = _mutable_section(config, "policy")
    policy["overrides"] = {
        "grain_subsidies_rate": (
            scenario.support_yuan_per_mu / BASE_SUPPORT_YUAN_PER_MU
        ),
        "base_subsidy_per_mu": BASE_SUPPORT_YUAN_PER_MU,
        "grain_price_floor": 1.2,
        "disable_grain_intervention": False,
        "subsidy_type": "area_based",
        "delivery_mode": "cash",
        "subsidy_target": "operator",
        "payment_lag": 0,
    }
    population = _mutable_section(config, "population_dynamics")
    population["land_matching_capacity_scale"] = scenario.coordination_scale
    macro_anchor = _mutable_section(population, "macro_anchor")
    macro_anchor["average_annual_decline_rate"] = scenario.rural_decline_rate
    _mutable_section(config, "simulation")["gdp_growth_rate"] = (
        scenario.gdp_growth_rate
    )
    production = _mutable_section(config, "production")
    technology = _mutable_section(production, "technology_progress")
    technology["grain_annual_yield_growth_rate"] = (
        scenario.grain_yield_growth_rate
    )
    technology["cash_crop_annual_yield_growth_rate"] = (
        scenario.cash_yield_growth_rate
    )
    transfer = _mutable_section(
        _mutable_section(_mutable_section(config, "agents"), "farmer"),
        "land_transfer",
    )
    calibrated_rent = _number(transfer, "base_rent_price")
    initial_rent = calibrated_rent * scenario.base_rental_index
    transfer["base_rent_price"] = initial_rent
    market = _mutable_section(_mutable_section(config, "environment"), "market")
    market["base_land_rent"] = initial_rent
    cost = _mutable_section(
        _mutable_section(production, "cost_structure"), "input_requirements"
    )
    other = _mutable_section(cost, "other_cost_per_mu")
    config["paper_b_flagship_scenario"] = {
        "scenario_id": scenario.scenario_id,
        "input_cost_growth_rate": scenario.input_cost_growth_rate,
        "rental_growth_rate": scenario.rental_growth_rate,
        "initial_grain_other_cost": _number(other, "grain"),
        "initial_cash_other_cost": _number(other, "cash_crop"),
        "initial_land_rent": initial_rent,
    }


def apply_annual_exogenous_path(model: RuntimeModel, year_index: int) -> None:
    metadata = _mutable_section(model.config, "paper_b_flagship_scenario")
    cost_factor = (1.0 + _number(metadata, "input_cost_growth_rate")) ** year_index
    rent_factor = (1.0 + _number(metadata, "rental_growth_rate")) ** year_index
    production = _mutable_section(model.config, "production")
    cost = _mutable_section(
        _mutable_section(production, "cost_structure"), "input_requirements"
    )
    requirements = _mutable_section(cost, "other_cost_per_mu")
    requirements["grain"] = _number(metadata, "initial_grain_other_cost") * cost_factor
    requirements["cash_crop"] = _number(
        metadata, "initial_cash_other_cost"
    ) * cost_factor
    rent = _number(metadata, "initial_land_rent") * rent_factor
    agents = _mutable_section(model.config, "agents")
    farmer = _mutable_section(agents, "farmer")
    _mutable_section(farmer, "land_transfer")["base_rent_price"] = rent
    environment = _mutable_section(model.config, "environment")
    _mutable_section(environment, "market")["base_land_rent"] = rent
    model.environment.land_rental_price = rent
    model.environment.market_prices["land_rent"] = rent
    model.environment.market_prices["land_rental"] = rent


def _section(row: Config, key: str) -> Config:
    value = row.get(key)
    if not isinstance(value, dict):
        raise FlagshipScenarioError(f"SSP section is missing: {key}")
    return value


def _mutable_section(row: Config, key: str) -> Config:
    value = row.setdefault(key, {})
    if not isinstance(value, dict):
        raise FlagshipScenarioError(f"runtime section is not a mapping: {key}")
    return value


def _number(row: Config, key: str) -> float:
    value = row.get(key)
    if not isinstance(value, int | float):
        raise FlagshipScenarioError(f"SSP value is not numeric: {key}")
    return float(value)
