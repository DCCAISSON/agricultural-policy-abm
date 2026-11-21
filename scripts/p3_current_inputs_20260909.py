# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mesa", "pyyaml", "pydantic>=2"]
# ///
# How to run: python -B -X utf8 scripts/p3_current_inputs_20260909.py
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.fixed_rule_runtime import choose_action_income, initialize_initial_household_tenure, rental_rights_holder
from core.model import MainModel
from core.model_builder import ModelBuilder
from core.p3_organization import best_crop_accounts, organization_accounts
from core.p3_project_contracts import P3Program, PreservedProject
from core.reproducibility import RandomController
from scripts.national_real2024 import OUTPUT as MONEY_INPUTS, apply_real2024
from scripts.p3_formal_inputs_20260909 import OUTPUT as INITIAL_INPUTS, RULES

OUTPUT: Final = INITIAL_INPUTS.parent / "p3_formal_inputs_20260909_r3"
COST_RATE: Final = .008027


@dataclass(frozen=True, slots=True)
class Forecast:
    actor_id: str
    actor_type: str
    action: str
    organization: str
    grain_area_mu: float
    service_area_mu: float
    quoted_service_fee_yuan: float
    area_claim_yuan: float
    service_claim_yuan: float


def common_model(seed: int = 17) -> MainModel:
    config = json.loads((INITIAL_INPUTS / "common_initial_state_pre_currency_config.json").read_text(encoding="utf-8"))
    config["runtime_fixed_rule"].update(inertia_basis="yuan_per_mu", minimum_inertia_margin_yuan_per_mu=50.0)
    config["experiment"]["seed"] = seed
    apply_real2024(config)
    assert config["runtime_fixed_rule"]["minimum_inertia_margin_yuan_per_mu"] == 50
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink):
        RandomController.set_seed(seed)
        model = ModelBuilder(config=config).build()
        initialize_initial_household_tenure(model)
    assert len(model.initial_household_cohort) == 600
    assert len(model.agricultural_enterprises) == 60
    return model


def program(model: MainModel, rule: str, trial_pool: float, label: str) -> P3Program:
    money = json.loads((MONEY_INPUTS / "p3_currency_inputs.json").read_text(encoding="utf-8"))
    base_area = sum(p.area for p in model.land_patches if rental_rights_holder(p) in model.initial_household_cohort)
    return P3Program(rule=rule, **{key: money[key] for key in (
        "model_price_year", "trial_price_year", "trial_to_model", "service_2019_to_model", "price_conversion_source")},
        base_budget_yuan=base_area * money["preserved_rate_2024"], trial_budget_yuan=trial_pool,
        implementation_cost_yuan=trial_pool * COST_RATE, implementation_cost_rate=COST_RATE,
        expected_area_claims_yuan=0, expected_service_claims_yuan=0, scenario_id=f"{label}__{rule}__cost_0p8027pct",
        preserved_projects=(PreservedProject(project_id="land_protection_2024_condition", basis="rights_area",
            rate=83, price_year=2024, to_model=1, requires_cultivation=True,
            source="author_new_2024_mechanism_baseline_not_observed_payment_mean"),))


def forecast(model: MainModel, spec: P3Program) -> tuple[P3Program, tuple[Forecast, ...]]:
    # Iterate only expected receipts; the real actors, land and accounts are not enacted here.
    land = sum(p.area for p in model.land_patches)
    spec = spec.model_copy(update={"expected_area_claims_yuan": land * 350,
                                   "expected_service_claims_yuan": land * 100 * spec.service_2019_to_model})
    for _iteration in range(30):
        model.config.setdefault("fiscal_system", {})["p3_support"] = spec.model_dump()
        rows: list[Forecast] = []
        for actor in [*model.farmers, *model.agricultural_enterprises]:
            household = actor in model.farmers
            if household:
                choices = organization_accounts(actor)
                accounts = best_crop_accounts(choices)
            else:
                accounts = tuple(actor.expected_fixed_rule_account(action) for action in
                                 ("maintain_grain", "expand_cash", "contract"))
            action = choose_action_income(actor, accounts)
            selected = next(account for account in accounts if account.action == action)
            organization = next(item.mode for item in choices if item.account is selected) if household else "enterprise"
            grain = selected.grain_area_mu
            area_claim = grain * 350 if (spec.rule in ("area_universal", "area_service_mix") or
                (spec.rule == "scale_eligibility" and grain >= 80)) else 0.0
            service_claim = min(.3 * selected.service_cost_yuan,
                100 * spec.service_2019_to_model * min(grain, selected.service_area_mu)) if (
                household and spec.rule in ("family_service", "area_service_mix") and grain > 0) else 0.0
            rows.append(Forecast(str(actor.unique_id), "household" if household else "enterprise", action,
                organization, grain, selected.service_area_mu, selected.service_cost_yuan, area_claim, service_claim))
        area, service = math.fsum(row.area_claim_yuan for row in rows), math.fsum(row.service_claim_yuan for row in rows)
        converged = (math.isclose(area, spec.expected_area_claims_yuan, abs_tol=1e-7)
                     and math.isclose(service, spec.expected_service_claims_yuan, abs_tol=1e-7))
        spec = spec.model_copy(update={"expected_area_claims_yuan": area, "expected_service_claims_yuan": service})
        if converged:
            model.config["fiscal_system"]["p3_support"] = spec.model_dump()
            return spec, tuple(rows)
    raise RuntimeError("Expected claims do not converge; do not freeze this input")


def main() -> None:
    OUTPUT.mkdir(exist_ok=False)
    model = common_model()
    cohort_path = ROOT / "data/derived_runtime/v5_r6/nfps_fixed_empirical_cohort_2010_k600.json"
    patches = []
    for patch in sorted(model.land_patches, key=lambda item: str(item.patch_id)):
        actor = patch.current_operator
        crop = getattr(getattr(actor, "planned_crop", None), "value", None)
        eligible = actor is not None and not patch.abandoned and patch.area > 0 and crop == "Grain"
        patches.append({"patch_id": str(patch.patch_id), "area_mu": patch.area,
            "raw_patch_crop": getattr(getattr(patch, "current_crop", None), "value", None),
            "operator_id": str(actor.unique_id) if actor is not None else None,
            "operator_planned_crop": crop, "abandoned": patch.abandoned,
            "eligible_pre_policy_grain": eligible, "eligible_grain_area_mu": patch.area if eligible else 0.0})
    a0 = math.fsum(row["eligible_grain_area_mu"] for row in patches)
    from core.p3_project_claims import production_records
    independently_qualified = math.fsum(record.area_mu for actor in [*model.farmers, *model.agricultural_enterprises]
                                       for record in production_records(actor) if record.grain)
    assert math.isclose(a0, 2508.3, abs_tol=1e-7) and math.isclose(a0, independently_qualified, abs_tol=1e-7)
    assert model.step_count == 0
    with (OUTPUT / "fiscal_reference_patches.json").open("x", encoding="utf-8") as stream:
        json.dump(patches, stream, indent=2)
    with (OUTPUT / "common_pre_policy_config.json").open("x", encoding="utf-8") as stream:
        json.dump(model.config, stream, indent=2)
    receipt = {"status": "FROZEN_PRE_POLICY_FISCAL_REFERENCE_AND_PRICE_INPUTS", "main_A0": a0,
        "independent_qualification_sum_mu": independently_qualified, "reference_seed": 17, "reference_step": 0,
        "reference_patch_count": len(patches), "trial_pool_year1": 877905.0, "annual_real_growth_factor": 1.05,
        "all_policy_and_20_paired_seed_budget_scope": "one_shared_reference_no_recalculation_from_seed_or_policy_area",
        "initial_crop_status": "existing_model_mechanism_not_CLES_national_observation_no_recalibration",
        "nfps_sow_food_not_used": True,
        "patch_reference_sha256": hashlib.sha256((OUTPUT / "fiscal_reference_patches.json").read_bytes()).hexdigest(),
        "reference_config_sha256": hashlib.sha256((OUTPUT / "common_pre_policy_config.json").read_bytes()).hexdigest(),
        "reference_rate": 350, "reference_price_year": 2024, "trial_pool_formula": "frozen_A0 * 350 * 1.05**(t-1)",
        "budget_scope": "trial_pool_only_base_and_public_implementation_cost_separate",
        "cost_rate": COST_RATE, "source_exact_ratio": 20.5 / 2554, "rounding": "author_0.8027_percent_used",
        "cost_reserve_formula": "cost_rate * trial_pool_budget", "actual_cost_formula": "cost_rate * actual_trial_paid",
        "cost_sensitivity_rates": [0.0, COST_RATE, 2 * COST_RATE],
        "wide_budget_label": "wide8m_new_2024_condition_not_historical_same_scale",
        "grain_reserve_default_budget_separate_yuan": 2800000, "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "inertia_yuan_per_mu": 50, "currency_input": str(MONEY_INPUTS / "p3_currency_inputs.json")}
    with (OUTPUT / "input_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    summaries, household_rows = [], []
    for rule in RULES:
        model = common_model()
        spec, forecasts = forecast(model, program(model, rule, receipt["trial_pool_year1"], "main_frozenA0_2508p3"))
        summaries.append({"scenario_id": spec.scenario_id, "main_A0": a0, "reference_rate_2024": 350,
            "trial_pool_budget": spec.trial_budget_yuan, "base_budget_separate": spec.base_budget_yuan,
            "implementation_reserve_separate": spec.implementation_cost_yuan,
            "expected_area_claims": spec.expected_area_claims_yuan, "expected_service_claims": spec.expected_service_claims_yuan,
            "expected_area_ratio": min(1, spec.pool_budget("area") / spec.expected_area_claims_yuan) if spec.expected_area_claims_yuan else 0,
            "expected_service_ratio": min(1, spec.pool_budget("service") / spec.expected_service_claims_yuan) if spec.expected_service_claims_yuan else 0,
            "forecast_scope": "pre_market_expected_choice_not_realized_payment", "inertia_yuan_per_mu": 50})
        household_rows.extend({"scenario_id": spec.scenario_id, **asdict(row)} for row in forecasts)
        for rate, label in ((0.0, "cost_0pct"), (COST_RATE, "cost_0p8027pct"), (2 * COST_RATE, "cost_1p6054pct")):
            sensitivity = spec.model_copy(update={"implementation_cost_rate": rate,
                "implementation_cost_yuan": spec.trial_budget_yuan * rate,
                "scenario_id": f"main_frozenA0_2508p3__{rule}__{label}"})
            with (OUTPUT / f"{rule}.{label}.program.json").open("x", encoding="utf-8") as stream:
                json.dump(sensitivity.model_dump(), stream, indent=2)
    budget_path = [{"budget_condition": label, "step": step, "trial_pool_budget": pool * 1.05 ** (step - 1),
        "cost_rate": rate, "implementation_reserve_separate": pool * 1.05 ** (step - 1) * rate,
        "pool_scope": "trial_only_not_rebased_on_policy_or_seed_area"}
        for label, pool in (("main_frozenA0_2508p3", 877905.0), ("wide8m_new2024", 8000000.0))
        for step in range(1, 31) for rate in (0.0, COST_RATE, 2 * COST_RATE)]
    for name, rows in (("fiscal_reference_patches.csv", patches), ("four_rule_forecasts.csv", summaries),
                       ("actor_forecasts.csv", household_rows), ("shared_annual_budget_paths.csv", budget_path)):
        with (OUTPUT / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"receipt": receipt, "forecasts": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
