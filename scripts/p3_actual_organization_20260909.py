# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "mesa", "pyyaml", "pydantic>=2"]
# ///
# How to run: python -B -X utf8 scripts/p3_actual_organization_20260909.py
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Final, Literal
from unittest.mock import patch

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model import MainModel
from core.fixed_rule_runtime import ActionIncomeAccount
from core.p3_project_contracts import P3Program, ration
from core.p3_project_payments import freeze_previous_claims
from agents.farmer_agent import FarmerAgent
from agents.enterprise_agent import AgriculturalEnterpriseAgent
from modules.agricultural_service import ServiceProvider
from modules.population_dynamics_v2 import PopulationManagerV2
from scripts.p3_current_inputs_20260909 import COST_RATE, OUTPUT as INPUTS, common_model, program

OUTPUT: Final = INPUTS.parent / "p3_actual_organization_20260909_r7"


def main() -> None:
    OUTPUT.mkdir(exist_ok=False)
    protected = ROOT / "config/master_config.yaml"
    protected_hash = hashlib.sha256(protected.read_bytes()).hexdigest()
    model = common_model()
    reference = json.loads((INPUTS / "input_receipt.json").read_text(encoding="utf-8"))
    initial_ids = tuple(str(f.unique_id) for f in model.initial_household_cohort)
    households, tasks, annual, payments, providers = [], [], [], [], []
    harvest_calls, consumption_calls, exit_calls, expectations = [], [], [], []
    ratio_calls, capacity_calls, capacity_timing = [], [], []
    capacity_seen: set[tuple[int, str, str]] = set()
    release_seen: set[int] = set()
    candidate_context: tuple[str, str] | None = None
    harvest, consume = FarmerAgent.harvest_and_settle, FarmerAgent.make_consumption_decision
    close_provider, exit_household = ServiceProvider.end_of_year_settlement, PopulationManagerV2._process_agent_exit
    expected_ratio = P3Program.expected_payment_ratio
    farmer_account, enterprise_account = FarmerAgent.expected_fixed_rule_account, AgriculturalEnterpriseAgent.expected_fixed_rule_account
    remaining_capacity, refresh_area = ServiceProvider.get_remaining_capacity, MainModel._refresh_total_land_area
    source_paths = ("core/model.py", "agents/farmer_agent.py", "modules/population_dynamics_v2.py",
        "modules/agricultural_service.py", "core/fixed_rule_runtime.py", "core/p3_project_payments.py",
        "core/p3_service_settlement.py", "core/p3_project_claims.py", "core/p3_project_contracts.py",
        "agents/enterprise_agent.py", "core/p3_organization.py", "scripts/p3_current_inputs_20260909.py", __file__)
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}

    def observed_farmer_account(actor: FarmerAgent, action: str, area: float | None = None) -> ActionIncomeAccount:
        nonlocal candidate_context
        previous = candidate_context
        candidate_context = (action, str(getattr(actor, "p3_service_mode", "")))
        try:
            return farmer_account(actor, action, area)
        finally:
            candidate_context = previous

    def observed_enterprise_account(actor: AgriculturalEnterpriseAgent, action: str, area: float | None = None) -> ActionIncomeAccount:
        nonlocal candidate_context
        previous = candidate_context
        candidate_context = (action, "enterprise")
        try:
            return enterprise_account(actor, action, area)
        finally:
            candidate_context = previous

    def observed_ratio(spec: P3Program, actor_id: str, pool: Literal["area", "service"], nominal: float) -> float:
        value = expected_ratio(spec, actor_id, pool, nominal)
        if candidate_context is not None:
            total = spec.expected_area_claims_yuan if pool == "area" else spec.expected_service_claims_yuan
            own = (spec.previous_actor_pool_nominal or {}).get(actor_id, {}).get(pool, 0.0)
            denominator = total if spec.previous_actor_pool_nominal is None else total - own + nominal
            ratio_calls.append({"step": model.step_count, "actor_id": actor_id,
                "action": candidate_context[0], "organization": candidate_context[1], "pool": pool,
                "stage": model.v4_8_market_stage_trace[-1]["stage"], "timestamp_ns": time.perf_counter_ns(),
                "source_step": spec.expectation_source_step, "pool_budget_yuan": spec.pool_budget(pool),
                "signal_pool_total_yuan": total, "previous_actor_pool_nominal_yuan": own,
                "candidate_nominal_yuan": nominal, "expected_denominator_yuan": denominator,
                "expected_payment_ratio": value, "expected_payment_yuan": nominal * value})
        return value

    def observed_refresh_area(current: MainModel) -> None:
        if current.step_count not in release_seen:
            release_seen.add(current.step_count)
            for provider in current.service_providers:
                capacity_timing.append({"step": current.step_count, "event": "after_release_before_market_coordinator",
                    "timestamp_ns": time.perf_counter_ns(), "provider_id": provider.unique_id,
                    "service_area_used": provider.service_area_used, "capacity_mu": provider.current_capacity,
                    "contracts": [{"area": c.area, "year_signed": c.year_signed, "duration": c.duration}
                        for c in provider.active_contracts]})
        refresh_area(current)

    def observed_remaining_capacity(provider: ServiceProvider) -> float:
        value = remaining_capacity(provider)
        year = model.simulation_base_year + model.step_count
        expired = math.fsum(c.area for c in provider.active_contracts if c.is_expired(year))
        assert expired == 0, (model.step_count, provider.unique_id, expired)
        trace = model.v4_8_market_stage_trace
        stage = trace[-1]["stage"] if trace and trace[-1]["step"] == model.step_count else "annual_start"
        key = (model.step_count, provider.unique_id, stage)
        if key not in capacity_seen:
            capacity_seen.add(key)
            capacity_calls.append({"step": model.step_count, "provider_id": provider.unique_id, "stage": stage,
                "timestamp_ns": time.perf_counter_ns(), "service_area_used": provider.service_area_used,
                "remaining_capacity_mu": value, "capacity_mu": provider.current_capacity,
                "expired_contract_area_mu": expired, "candidate_context": candidate_context})
        return value

    def observed_harvest(farmer: FarmerAgent) -> None:
        before = {key: getattr(farmer, key) for key in ("capital", "off_farm_income", "land_rental_income", "net_income")}
        harvest(farmer)
        after = {key: getattr(farmer, key) for key in ("capital", "off_farm_income", "land_rental_income", "net_income",
            "agricultural_income", "labor_sale_income", "subsidy_income", "gross_income", "total_expense",
            "realized_off_farm_labor_days", "off_farm_wage_level")}
        harvest_calls.append({"household_id": str(farmer.unique_id), "step": model.step_count,
            "year": model.simulation_base_year + model.step_count, "in_active": farmer in model.farmers,
            "before": before, "after": after, "daily_wage": farmer._get_effective_non_farm_daily_wage(),
            "rental_contracts": [{"patch_id": str(p.patch_id), "rent_yuan": p.rental_price,
                "remaining_years": p.lease_contract_years, "tenant_id": str(p.leased_to.unique_id)}
                for p in farmer.leased_out_patches], "scope": "harvest_before_consumption_and_ex_post_payment"})

    def observed_consumption(farmer: FarmerAgent) -> None:
        before = farmer.capital
        consume(farmer)
        consumption_calls.append({"household_id": str(farmer.unique_id), "step": model.step_count,
            "year": model.simulation_base_year + model.step_count, "capital_before": before,
            "capital_after": farmer.capital, "consumption_yuan": farmer.total_consumption_expenditure,
            "consumer_surplus_not_cash": farmer.consumer_surplus})

    def observed_provider(provider: ServiceProvider, year: int) -> float:
        before, revenue, area = provider.capital, provider.annual_revenue, provider.service_area_used
        depreciation = provider.large_tractors * 12000 + provider.combine_harvesters * 20000 + provider.drone_sprayers * 3000
        profit = close_provider(provider, year)
        providers.append({"step": model.step_count, "provider_id": provider.unique_id,
            "settlement_year": provider.last_settlement_year, "profit": profit, "capital": provider.capital,
            "current_capacity_mu": provider.current_capacity, "capital_before": before,
            "revenue_before_reset": revenue, "area_before_reset": area,
            "cost_by_existing_formula": depreciation * 1.1 + area * 70,
            "revenue_after_reset": provider.annual_revenue})
        return profit

    def observed_exit(manager: PopulationManagerV2, farmer: FarmerAgent, reason: str) -> None:
        exit_calls.append({"household_id": str(farmer.unique_id), "step": model.step_count,
            "year": manager.current_year, "reason": reason, "age": farmer.age,
            "capital_before": farmer.capital, "income_before": farmer.net_income})
        exit_household(manager, farmer, reason)
        exit_calls[-1]["capital_after"] = farmer.capital

    def snapshot(current: MainModel) -> None:
        government = current.government
        spec = government.p3_batch.program
        step = current.step_count
        active_ids = {str(f.unique_id) for f in current.farmers}
        for farmer in current.initial_household_cohort:
            households.append({"scenario_id": spec.scenario_id, "step": step, "household_id": farmer.unique_id,
                "in_operating_list_at_production": str(farmer.unique_id) in active_ids,
                "fixed_cohort_member": True, "weight_q": farmer.representation_weight_q,
                "action": getattr(farmer, "fixed_rule_current_action", ""),
                "organization": getattr(farmer, "p3_service_mode", ""),
                "production_exit": bool(getattr(farmer, "fixed_rule_exit_pending", False)),
                "rights_area_mu": farmer.initial_rental_rights_area_mu,
                "operated_mu": farmer.get_operated_land_area(), "grain_output_kg": farmer.grain_production,
                "cash_output_kg": farmer.cash_crop_production, "off_farm_income_yuan": farmer.off_farm_income,
                "land_rental_income_yuan": farmer.land_rental_income, "net_income_yuan": farmer.net_income,
                "consumption_yuan": farmer.total_consumption_expenditure,
                "capital_yuan": farmer.capital, "private_service_cost_yuan": farmer.service_fee_expense,
                "subsidy_paid_yuan": farmer.subsidy_income,
                "year": current.simulation_base_year + step,
                "prior_population_exit_reason": next((r["reason"] for r in exit_calls
                    if r["household_id"] == str(farmer.unique_id) and r["step"] < step), None)})
        for farmer in [*current.initial_household_cohort,
                       *(f for f in current.farmers if str(f.unique_id) not in initial_ids)]:
            for index, contract in enumerate(farmer.service_contracts):
                if contract.p3_application_step != step:
                    continue
                tasks.append({"scenario_id": spec.scenario_id, "step": step, "household_id": farmer.unique_id,
                    "fixed_cohort_member": str(farmer.unique_id) in initial_ids,
                    "contract_index": index, "provider_id": contract.service_provider_id,
                    "crop": contract.crop_type, "season": contract.production_season,
                    "area_mu": contract.area, "fee_per_mu": contract.fee_per_mu,
                    "private_invoice_yuan": contract.fee_per_mu * contract.area,
                    "private_paid_yuan": contract.p3_paid_yuan, "fulfilled": contract.p3_fulfilled,
                    "paid_step": contract.p3_payment_step, "coverage_ratio": contract.coverage_ratio,
                    "plot_areas_mu": json.dumps(contract.plot_areas_mu, sort_keys=True),
                    "direct_yield_multiplier": farmer.service_yield_bonus_multiplier})
        payments.extend(dict(row) for row in government.p3_project_payments)
        runtime = government.current_subsidy_runtime
        base = math.fsum(row["paid_yuan"] for row in government.p3_project_payments if row["layer"] == "base")
        trial = math.fsum(row["paid_yuan"] for row in government.p3_project_payments if row["layer"] == "trial")
        bureau = current.grain_reserve_bureau
        release_revenue = math.fsum(record["revenue"] for record in bureau.release_history if record["step"] == step)
        annual.append({"scenario_id": spec.scenario_id, "step": step,
            "expected_area_claims": spec.expected_area_claims_yuan,
            "expected_service_claims": spec.expected_service_claims_yuan,
            "expected_claims_source": "frozen_initial_actor_forecasts" if step == 1 else "previous_others_plus_current_candidate_nominal",
            "initial_cohort_count": len(current.initial_household_cohort), "base_budget": spec.base_budget_yuan,
            "base_paid": base, "base_unspent": spec.base_budget_yuan - base,
            "trial_budget": spec.trial_budget_yuan, "trial_paid": trial, "trial_unspent": spec.trial_budget_yuan - trial,
            "implementation_rate": COST_RATE, "implementation_reserve": runtime["implementation_reserved_yuan"],
            "implementation_cost": runtime["implementation_cost_yuan"],
            "implementation_unspent": runtime["implementation_unspent_yuan"],
            "reserve_bureau_gross_expenditure": bureau.annual_expenditure,
            "reserve_bureau_reported_release_revenue": release_revenue,
            "total_fiscal_gross_expenditure": base + trial + runtime["implementation_cost_yuan"] + bureau.annual_expenditure,
            "total_fiscal_net_expenditure": base + trial + runtime["implementation_cost_yuan"] + bureau.annual_expenditure - release_revenue,
            "service_matched_area": current.service_market_stats["service_matched_area"],
            "service_unmatched_area": current.service_market_stats["service_unmatched_demand_area"],
            "scope": "main_frozenA0_two_year_integration_not_20_seed_matrix"})

    model._completed_production_year_snapshot_hook = snapshot
    with (patch.object(FarmerAgent, "harvest_and_settle", observed_harvest),
          patch.object(FarmerAgent, "expected_fixed_rule_account", observed_farmer_account),
          patch.object(AgriculturalEnterpriseAgent, "expected_fixed_rule_account", observed_enterprise_account),
          patch.object(P3Program, "expected_payment_ratio", observed_ratio),
          patch.object(ServiceProvider, "get_remaining_capacity", observed_remaining_capacity),
          patch.object(MainModel, "_refresh_total_land_area", observed_refresh_area),
          patch.object(FarmerAgent, "make_consumption_decision", observed_consumption),
          patch.object(ServiceProvider, "end_of_year_settlement", observed_provider),
          patch.object(PopulationManagerV2, "_process_agent_exit", observed_exit),
          (OUTPUT / "runtime.log").open("x", encoding="utf-8") as stream, contextlib.redirect_stdout(stream)):
        for step in (1, 2):
            pool = reference["trial_pool_year1"] * reference["annual_real_growth_factor"] ** (step - 1)
            if step == 1:
                spec = P3Program.model_validate(json.loads((INPUTS / "area_service_mix.cost_0p8027pct.program.json").read_text()))
            else:
                spec = freeze_previous_claims(model.government,
                    program(model, "area_service_mix", pool, "main_frozenA0_2508p3"))
            model.config.setdefault("fiscal_system", {})["p3_support"] = spec.model_dump()
            expectations.append({"step": step, "year": model.simulation_base_year + step,
                "source_step": 0 if step == 1 else step - 1, "recorded_before_model_step": True,
                "source": "frozen_initial_actor_forecasts" if step == 1 else "previous_others_plus_current_candidate_nominal",
                "program": spec.model_dump()})
            for provider in model.service_providers:
                capacity_timing.append({"step": step, "event": "before_model_step",
                    "timestamp_ns": time.perf_counter_ns(), "provider_id": provider.unique_id,
                    "service_area_used": provider.service_area_used, "capacity_mu": provider.current_capacity,
                    "contracts": [{"area": c.area, "year_signed": c.year_signed, "duration": c.duration}
                        for c in provider.active_contracts]})
            with (OUTPUT / f"year_{step}_predecision_program.json").open("x", encoding="utf-8") as stream_before:
                json.dump(expectations[-1], stream_before, indent=2)
            start = time.perf_counter()
            model.step()
            annual[-1]["step_elapsed_seconds"] = time.perf_counter() - start
            active_after = {str(f.unique_id) for f in model.farmers}
            for row in households[-600:]:
                row["in_operating_list_after_demography"] = row["household_id"] in active_after
            assert tuple(str(f.unique_id) for f in model.initial_household_cohort) == initial_ids
            with (OUTPUT / f"year_{step}_actual_records.json").open("x", encoding="utf-8") as checkpoint:
                json.dump({"status": "completed_annual_step_not_full_run_acceptance", "step": step,
                    "households": households[-600:], "tasks": [row for row in tasks if row["step"] == step],
                    "fiscal": annual[-1], "project_payments": model.government.p3_project_payments,
                    "household_settlements": [r for r in harvest_calls if r["step"] == step],
                    "household_consumption": [r for r in consumption_calls if r["step"] == step],
                    "candidate_expectations": [r for r in ratio_calls if r["step"] == step],
                    "capacity_timing": [r for r in capacity_timing if r["step"] == step],
                    "capacity_reads": [r for r in capacity_calls if r["step"] == step],
                    "provider_settlements": [r for r in providers if r["step"] == step],
                    "population_exits": [r for r in exit_calls if r["step"] == step]}, checkpoint, indent=2)
            print(f"Completed actual step {step}: 600 fixed households recorded", file=sys.stderr, flush=True)
    assert len(households) == 1200
    assert tasks and all(row["fulfilled"] and row["paid_step"] == row["step"] for row in tasks)
    assert all(math.isclose(row["private_paid_yuan"], row["private_invoice_yuan"], abs_tol=1e-6) for row in tasks)
    assert all(row["direct_yield_multiplier"] == 1 for row in tasks)
    assert any(not row["in_operating_list_at_production"] for row in households if row["step"] == 2)
    assert all(math.isclose(row["implementation_cost"], COST_RATE * row["trial_paid"], abs_tol=1e-6) for row in annual)
    assert protected_hash == hashlib.sha256(protected.read_bytes()).hexdigest()
    assert source_hashes == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}
    assert all(row["reason"] != "death" for row in exit_calls)
    assert len(providers) == 2 * len(model.service_providers)
    for row in providers:
        assert row["settlement_year"] == model.simulation_base_year + row["step"]
        assert math.isclose(row["area_before_reset"], math.fsum(t["area_mu"] for t in tasks
            if t["provider_id"] == row["provider_id"] and t["step"] == row["step"]), rel_tol=0, abs_tol=1e-6)
        assert math.isclose(row["capital"] - row["capital_before"], row["profit"], abs_tol=1e-6)
        assert math.isclose(row["revenue_before_reset"] - row["cost_by_existing_formula"], row["profit"], abs_tol=1e-6)
        assert math.isclose(row["revenue_before_reset"], math.fsum(t["private_paid_yuan"] for t in tasks
            if t["provider_id"] == row["provider_id"] and t["step"] == row["step"]), abs_tol=1e-6)
    for row in harvest_calls:
        assert math.isclose(row["after"]["capital"] - row["before"]["capital"], row["after"]["net_income"], abs_tol=1e-6)
    for row in consumption_calls:
        assert math.isclose(row["capital_before"] - row["capital_after"], row["consumption_yuan"], abs_tol=1e-6)
    for step in (1, 2):
        cohort_calls = [r for r in harvest_calls if r["step"] == step and r["household_id"] in initial_ids]
        assert len(cohort_calls) == 600 and {r["household_id"] for r in cohort_calls} == set(initial_ids)
    harvest_by_key = {(r["step"], r["household_id"]): r for r in harvest_calls}
    for row in households:
        before = harvest_by_key[row["step"], row["household_id"]]["before"]["capital"]
        assert math.isclose(row["capital_yuan"] - before, row["net_income_yuan"] - row["consumption_yuan"], abs_tol=1e-6)
    rate_summary = []
    for step in (1, 2):
        spec = P3Program.model_validate(expectations[step - 1]["program"])
        for pool in ("area", "service"):
            actual_claims = math.fsum(r["nominal_claim_yuan"] for r in payments if r["step"] == step and r["pool"] == pool)
            actual_ratio = ration(spec.pool_budget(pool), actual_claims)
            calls = [r for r in ratio_calls if r["step"] == step and r["pool"] == pool]
            assert calls
            for row in calls:
                row["actual_pool_nominal_yuan"] = actual_claims
                row["actual_payment_ratio"] = actual_ratio
                row["rate_prediction_error_actual_minus_expected"] = actual_ratio - row["expected_payment_ratio"]
                row["candidate_equivalent_payment_error_yuan"] = row["candidate_nominal_yuan"] * (actual_ratio - row["expected_payment_ratio"])
                assert row["source_step"] == step - 1
                assert math.isclose(row["expected_payment_ratio"], ration(row["pool_budget_yuan"], row["expected_denominator_yuan"]), rel_tol=0, abs_tol=1e-12)
            rate_summary.append({"step": step, "pool": pool, "candidate_rate_call_count": len(calls),
                "expected_ratio_min": min(r["expected_payment_ratio"] for r in calls),
                "expected_ratio_max": max(r["expected_payment_ratio"] for r in calls),
                "actual_ratio": actual_ratio, "actual_pool_nominal_yuan": actual_claims,
                "rate_error_min": min(r["rate_prediction_error_actual_minus_expected"] for r in calls),
                "rate_error_max": max(r["rate_prediction_error_actual_minus_expected"] for r in calls),
                "scope": "range_over_actual_candidate_calls_not_one_common_forecast_or_selected_action_error"})
        releases = {r["provider_id"]: r for r in capacity_timing if r["step"] == step and r["event"] == "after_release_before_market_coordinator"}
        first_reads = {}
        for row in capacity_calls:
            if row["step"] == step:
                first_reads.setdefault(row["provider_id"], row)
        assert len(first_reads) == len(model.service_providers)
        for provider_id, row in first_reads.items():
            assert row["timestamp_ns"] > releases[provider_id]["timestamp_ns"]
            assert row["service_area_used"] == 0 and row["expired_contract_area_mu"] == 0
    for name, rows in (("household_settlement_calls.json", harvest_calls), ("household_consumption_calls.json", consumption_calls),
                       ("population_exit_calls.json", exit_calls), ("expected_claims_log.json", expectations),
                       ("capacity_timing.json", capacity_timing), ("capacity_reads.json", capacity_calls)):
        with (OUTPUT / name).open("x", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2)
    for name, rows in (("fixed_household_annual.csv", households), ("actual_service_tasks.csv", tasks),
                       ("fiscal_annual.csv", annual), ("project_payments.csv", payments), ("provider_settlements.csv", providers)):
        with (OUTPUT / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name, rows in (("candidate_expected_payment_rates.csv", ratio_calls), ("annual_expected_actual_rates.csv", rate_summary)):
        with (OUTPUT / name).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    expectation_rule = {
        "name": "lagged_others_plus_current_candidate_nominal",
        "author_source": "project_management/evidence/next_execution_batch_20260907/AUTHOR_INSTITUTIONAL_DECISIONS_20260909.md",
        "initialization": "first year uses frozen predecision total nominal claims; no max(own,total)",
        "later_denominator": "previous_pool_total - previous_actor_pool_nominal + current_candidate_actor_pool_nominal",
        "aggregation": "all verified nominal trial claims by recipient and pool, across plots/projects; excludes base; never paid_yuan",
        "timing": "freeze_previous_claims from completed prior PaymentBatch before next model.step; candidate functions read only frozen P3Program",
        "actual_payment": "current nominal claim times min(1,pool_budget/current_total_verified_nominal); zero budget/claims gives zero",
        "candidate_log": "candidate_expected_payment_rates.csv",
        "candidate_log_scope": "actual expected_fixed_rule_account calls with stage/action/organization; candidate-equivalent error is not observed actor cash error",
        "yearly_rate_log": "annual_expected_actual_rates.csv", "two_cycle_check": "Mc 30-year runner; two years cannot establish a repeated two-cycle",
        "perfect_frozen_controls": "lightweight key comparisons, not common prerequisites for baseline or matrix",
        "predecision_programs": "expected_claims_log.json;year_1_predecision_program.json;year_2_predecision_program.json",
    }
    receipt = {"status": "PASS_EXECUTOR_R7_EXPECTATION_AND_FIRST_CAPACITY_GATES_PENDING_DIRAC", "years": 2,
        "fixed_household_rows": len(households), "private_task_rows": len(tasks), "project_payment_rows": len(payments),
        "initial_cohort_count_each_year": 600, "protected_master_sha256": protected_hash,
        "cost_rate": COST_RATE, "trial_pool_year1": reference["trial_pool_year1"],
        "trial_pool_year2": reference["trial_pool_year1"] * reference["annual_real_growth_factor"],
        "main_budget_status": "frozen_seed17_A0_2508p3_shared_across_all_paired_seeds", "formal_matrix_started": False,
        "source_sha256": source_hashes, "original_r4_preserved": True, "original_r5_preserved": True,
        "original_r6_preserved": True,
        "annual_task_cost_gate": "current-year provider area equals same-step fulfilled task area; tolerance 1e-6",
        "expectation_rule": expectation_rule,
        "predecision_expectation_log": "expected_claims_log.json;year_1_predecision_program.json;year_2_predecision_program.json",
        "independent_actual_payment_log": "project_payments.csv", "provider_bilateral_log": "provider_settlements.csv",
        "h0_harvest_log": "household_settlement_calls.json", "h0_consumption_log": "household_consumption_calls.json",
        "population_exit_log": "population_exit_calls.json", "source_frozen_after_this_run": True,
        "thirty_year_release": "AUTHOR_RULE_AUTHORIZED_PENDING_R7_INDEPENDENT_GATES_AND_PARENT_SAME_HASH_RECEIPT"}
    with (OUTPUT / "receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
