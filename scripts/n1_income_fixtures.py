from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from agents.farmer_agent import CropType, FarmerAgent, LivelihoodStrategy
from core.model_builder import ModelBuilder
from core.reproducibility import RandomController
from scripts.run_v4_8_national_fixed_rule_runtime_smoke import runtime_config

ROOT: Final = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class HandCase:
    role: str
    owned_mu: float
    rented_mu: float
    labor_days: float
    capital: float
    job_shock: float
    grain_net: float
    cash_net: float
    exit_net: float


HAND_CASES: Final = (
    HandCase("owned", 10, 0, 400, 100000, 0, 21840, 19000, 19000),
    HandCase("rented", 0, 10, 400, 100000, 0, 17840, 15000, 15000),
    HandCase("mixed", 6, 4, 400, 100000, 0, 20240, 17400, 17400),
    HandCase("labor_shortage", 10, 0, 20, 100000, 0, -660, -13500, 7000),
    HandCase("capital_shortage", 10, 0, 400, 1000, 0, 21840, 19000, 19000),
    HandCase("nonfarm_limited", 10, 0, 400, 100000, 1, 6840, 4000, 4000),
)


def hand_farmer(case: HandCase) -> FarmerAgent:
    RandomController.set_seed(17)
    config = runtime_config(ROOT / "config/experiments/v4_8_national_fixed_rule.yaml")
    model = ModelBuilder(config=config).build()
    model.config["runtime_fixed_rule"]["accounting_basis"] = "action_conditioned_v1"
    model.config["capitalization"] = {"base_rate": 0.0}
    production = model.config["production"]
    production["base_yield"] = {"grain": 500.0, "cash_crop": 1000.0}
    production["technology_progress"] = {"annual_yield_growth": 0.0}
    production["yield_factors"] = {
        "soil_quality": {"min_effect": 1.0, "max_effect": 1.0},
        "farming_ability": {"min_multiplier": 1.0, "max_multiplier": 1.0},
        "technology": {"max_improvement": 0.0},
        "structural_efficiency": {
            "farmer_grain_scale_mechanization_weight": 0.0,
            "farmer_grain_service_weight": 0.0,
            "farmer_grain_service_mechanization_weight": 0.0,
            "farmer_grain_professional_weight": 0.0,
            "farmer_cash_scale_weight": 0.0,
            "farmer_cash_service_weight": 0.0,
            "farmer_cash_professional_weight": 0.0,
        },
    }
    production["cost_structure"] = {
        "use_detailed_cost": True,
        "input_prices": {"small_tools": 0.0},
        "input_requirements": {
            "seed": {"grain": 0.0, "cash_crop": 0.0},
            "fertilizer": {"grain": 0.0, "cash_crop": 0.0},
            "pesticide": {"grain": 0.0, "cash_crop": 0.0},
            "labor_days": {"grain": 10.0, "grain_farmer": 10.0, "cash_crop": 25.0, "cash_crop_farmer": 25.0},
            "other_cost_per_mu": {"grain": 900.0, "cash_crop": 1100.0},
        },
    }
    env = model.environment
    env.config = model.config
    env.active_policy = {
        "base_subsidy_per_mu": 200.0, "grain_subsidies_rate": 1.0,
        "delivery_mode": "cash", "subsidy_type": "area_based", "subsidy_target": "operator",
    }
    env.market_prices.update({"crop_price": 1.4, "cash_crop": 1.0, "wage_rate": 150.0, "land_rent": 400.0})
    env.land_rental_price = 400.0
    model.labor_market.current_agricultural_wage = 100.0
    farmer = model.farmers[0]
    farmer.livelihood_strategy = LivelihoodStrategy.PLURIACTIVE
    farmer.total_labor_units = case.labor_days
    farmer.capital = case.capital
    farmer.farming_ability = 1.0
    farmer.off_farm_wage_level = 1.0
    farmer.productivity_shock_enabled = bool(case.job_shock)
    farmer.productivity_non_farm_job_shock = case.job_shock
    farmer.productivity_wage_penalty = 0.0
    farmer.productivity_adoption_lag = 0.0
    farmer.planned_crop = CropType.GRAIN
    farmer.fixed_rule_previous_action = "maintain_grain"
    farmer.fixed_rule_hired_labor_available_days = 50.0 if case.role == "labor_shortage" else 1000.0
    farmer.hired_labor_cost = 0.0
    farmer.labor_reduction_ratio = 0.0
    farmer.service_contracts = []
    farmer.service_yield_bonus_multiplier = 1.0
    owned, rented = model.land_patches[:2]
    owned.owner, owned.leased_to, owned.current_operator, owned.area = farmer, None, farmer, case.owned_mu
    rented.owner = model.farmers[-1]
    rented.leased_to = rented.current_operator = farmer
    rented.area = case.rented_mu
    rented.rental_price = 400.0 * case.rented_mu
    rented.lease_contract_years = 3
    model.land_patches = [owned, rented]
    farmer.land_patches = [owned]
    farmer.leased_in_patches = [rented]
    model.invalidate_land_tenure_caches()
    return farmer
