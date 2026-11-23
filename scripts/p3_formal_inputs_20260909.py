# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mesa", "pyyaml", "pydantic>=2"]
# ///
# How to run: python -B -X utf8 scripts/p3_formal_inputs_20260909.py
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.fixed_rule_runtime import initialize_initial_household_tenure, rental_rights_holder
from core.model_builder import ModelBuilder
from core.p3_project_contracts import P3InputsMissing, P3Program, PreservedProject
from core.reproducibility import RandomController
from scripts.u2_unified_paper_model_runtime import DEFAULT_COHORT, u2_runtime_config

OUTPUT: Final = ROOT / "results/empirical/three_paper_execution_20260908/national/p3_formal_inputs_20260909_r1"
RULES: Final = ("area_universal", "scale_eligibility", "family_service", "area_service_mix")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    RandomController.set_seed(17)
    config = u2_runtime_config(17, 1)
    config["runtime_fixed_rule"].update(initial_tenure_mode="A", accounting_basis="action_conditioned_v1")
    production = config["production"]
    structural = production["yield_factors"]["structural_efficiency"]
    for key in ("farmer_grain_service_weight", "farmer_grain_service_mechanization_weight", "farmer_cash_service_weight"):
        structural[key] = 0.0
    production["agricultural_service"]["apply_direct_yield_bonus_multiplier"] = False
    ecology = production.setdefault("ecological", dict(config["agents"]["fiscal_system"].get("ecological", {})))
    ecology["service_management_recovery_bonus"] = 0.0
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        model = ModelBuilder(config=config).build()
        initialize_initial_household_tenure(model)
    cohort = model.initial_household_cohort
    land = sum(float(p.area) for p in model.land_patches)
    rights = sum(float(f.initial_rental_rights_area_mu) for f in cohort)
    weights = sum(float(f.representation_weight_q) for f in cohort)
    assert len(cohort) == 600 and len(model.agricultural_enterprises) == 60
    assert abs(land - rights) < 1e-6 and weights == 600
    assert all(e.get_operated_land_area() == 0 for e in model.agricultural_enterprises)
    base_area = sum(float(p.area) for p in model.land_patches if not p.abandoned
                    and p.current_operator is not None and rental_rights_holder(p) in cohort)
    scale_area = sum(f.get_operated_land_area() for f in cohort if f.get_operated_land_area() >= 80)
    base_reserve = base_area * 83
    remaining_before_cost = 8_000_000 - base_reserve
    households = [{"household_id": str(f.unique_id), "representation_weight_q": f.representation_weight_q,
        "source_representation_weight_q": f.source_representation_weight_q,
        "rights_area_mu": f.initial_rental_rights_area_mu, "operated_area_mu": f.get_operated_land_area()}
        for f in cohort]
    with (OUTPUT / "common_initial_households.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(households[0]))
        writer.writeheader()
        writer.writerows(households)
    project = PreservedProject(project_id="land_protection_2024_condition", basis="rights_area",
        rate=83, price_year=2024, to_model=1,
        source="parent_authorized_2024_new_baseline_not_verified_national_payment_mean",
        requires_cultivation=True)
    quantities = []
    missing_by_rule: dict[str, str] = {}
    for rule in RULES:
        area_ceiling = scale_area * 350 if rule == "scale_eligibility" else land * 350
        if rule == "family_service":
            area_ceiling = 0.0
        share = .5 if rule == "area_service_mix" else (0.0 if rule == "family_service" else 1.0)
        quantities.append({"rule": rule, "initial_households": 600, "initial_enterprises": 60,
            "physical_land_mu": land, "representation_weight_sum": weights,
            "budget_2024_yuan": 8_000_000, "budget_per_physical_mu_2024_yuan": 8_000_000 / land,
            "budget_status": "candidate_8000000_pending_author_not_frozen",
            "historical_scale_basis": "300*26.5+80*500=47950_mu_conflicts_with_15000_mu_comment_not_actual_data_no_rescaling",
            "grain_reserve_default_budget_yuan_separately_reported": 2_800_000,
            "grain_reserve_package_scope": "pending_parent_not_excluded_from_total_fiscal_reporting",
            "preserved_eligible_initial_mu": base_area, "preserved_reserve_2024_yuan": base_reserve,
            "implementation_cost_2024_yuan": None, "trial_budget_before_unknown_cost_2024_yuan": remaining_before_cost,
            "trial_budget_formula": "8000000 - preserved_reserve - implementation_cost_C",
            "area_pool_share": share, "area_claim_ceiling_at_nominal350": area_ceiling,
            "area_only_full_payment_cost_threshold_if350_is_2024": remaining_before_cost - area_ceiling / share if share else None,
            "expected_area_claims_yuan": None, "expected_service_claims_yuan": None,
            "expected_area_payment_ratio": None, "expected_service_payment_ratio": None,
            "status": "INITIAL_QUANTITIES_VERIFIED_FORECASTS_WAIT_CURRENCY_AND_COST_DISPOSITION"})
        pending = P3Program(rule=rule, model_price_year=2024, trial_price_year=None, trial_to_model=None,
            service_2019_to_model=None, price_conversion_source=None, base_budget_yuan=base_reserve,
            trial_budget_yuan=None, implementation_cost_yuan=None, expected_area_claims_yuan=None,
            expected_service_claims_yuan=None, preserved_projects=(project,), scenario_id=f"p3_{rule}")
        with (OUTPUT / f"{rule}.pending_program.json").open("x", encoding="utf-8") as stream:
            json.dump(pending.model_dump(), stream, ensure_ascii=False, indent=2)
        try:
            pending.require_ready()
        except P3InputsMissing as error:
            missing_by_rule[rule] = str(error)
        else:
            raise AssertionError("pending program must not be runnable")
    with (OUTPUT / "budget_initial_quantities.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(quantities[0]))
        writer.writeheader()
        writer.writerows(quantities)
    with (OUTPUT / "common_initial_state_pre_currency_config.json").open("x", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
    receipt = {"status": "INITIAL_QUANTITIES_ONLY_NOT_FROZEN", "seed": 17, "worldline_steps_run": 0,
        "budget_scope": "600_household_represented_land_P3_direct_support_package_not_national_extrapolation",
        "budget_identity": "candidate_8000000_pending_author_not_frozen",
        "historical_scale_basis": "master_config comments: 300 households*26.5 mu +80 enterprises*500 mu=47950 mu; conflicts with 15000 mu comment; historical rationale not actual data; no mechanical rescaling",
        "grain_reserve_default_budget_yuan_separately_reported": 2_800_000,
        "grain_reserve_package_scope": "pending_parent_not_excluded_from_total_fiscal_reporting",
        "annual_total_formula_2024_yuan": "8000000 * 1.05 ** (t - 1)", "growth_type": "real_no_extra_inflation",
        "land_mu": land, "rights_mu": rights, "weight_sum": weights,
        "cohort_source": DEFAULT_COHORT.relative_to(ROOT).as_posix(),
        "cohort_sha256": hashlib.sha256(DEFAULT_COHORT.read_bytes()).hexdigest(),
        "initial_households_sha256": hashlib.sha256((OUTPUT / "common_initial_households.csv").read_bytes()).hexdigest(),
        "source_weighting": "patch_area_already_representation_scaled_no_second_q_multiplication",
        "service_providers": len(model.service_providers),
        "service_capacity_mu": sum(p.current_capacity for p in model.service_providers),
        "population_module_present": model.population_manager is not None,
        "grain_reserve_scope_recommendation": "retain_module_separate_visible_actual_fiscal_account_package_scope_pending_parent",
        "missing": ["Hegel_currency_mapping", "public_implementation_cost_or_authorized_conditional_disposition",
                    "four_rules_actual_initial_behavior_forecasts", "parent_freeze"],
        "formal_matrices_started": 0, "missing_by_rule": missing_by_rule}
    with (OUTPUT / "initial_quantity_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
