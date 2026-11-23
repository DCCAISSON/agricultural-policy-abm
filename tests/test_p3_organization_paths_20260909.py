from __future__ import annotations

import pytest

from core.fixed_rule_runtime import evaluate_fixed_rule_operators
from modules.agricultural_service import CustodyType, ServiceContract, ServiceMarket, ServiceProvider
from tests.test_p3_project_payments_20260909 import prepared, program


def test_frozen_fiscal_reference_is_shared_without_replacing_runtime_seed() -> None:
    import json
    from scripts.p3_current_inputs_20260909 import OUTPUT, common_model, program as current_program
    reference = json.loads((OUTPUT / "input_receipt.json").read_text(encoding="utf-8"))
    model = common_model(seed=23)

    spec = current_program(model, "area_universal", reference["trial_pool_year1"], "main_frozenA0_2508p3")

    assert model.config["experiment"]["seed"] == 23
    assert reference["reference_seed"] == 17
    assert reference["main_A0"] == 2508.3
    assert spec.trial_budget_yuan == 877905
    assert spec.implementation_cost_yuan == pytest.approx(7046.943435)
    assert model.config["runtime_fixed_rule"]["inertia_basis"] == "yuan_per_mu"
    assert model.config["runtime_fixed_rule"]["minimum_inertia_margin_yuan_per_mu"] == 50


def test_formal_two_year_records_reconcile_private_and_public_accounts() -> None:
    import csv
    from math import fsum
    from scripts.p3_actual_organization_20260909 import OUTPUT
    with (OUTPUT / "fixed_household_annual.csv").open(encoding="utf-8-sig", newline="") as stream:
        households = list(csv.DictReader(stream))
    with (OUTPUT / "actual_service_tasks.csv").open(encoding="utf-8-sig", newline="") as stream:
        tasks = list(csv.DictReader(stream))
    with (OUTPUT / "project_payments.csv").open(encoding="utf-8-sig", newline="") as stream:
        payments = list(csv.DictReader(stream))
    with (OUTPUT / "fiscal_annual.csv").open(encoding="utf-8-sig", newline="") as stream:
        fiscal = list(csv.DictReader(stream))

    ids = [{row["household_id"] for row in households if row["step"] == str(step)} for step in (1, 2)]

    assert len(households) == 1200 and len(ids[0]) == 600 and ids[0] == ids[1]
    assert {row["model_price_year"] for row in payments} == {"2024"}
    assert tasks and all(row["fulfilled"] == "True" and row["paid_step"] == row["step"] for row in tasks)
    for ledger in fiscal:
        step = ledger["step"]
        actors = [row for row in households if row["step"] == step]
        paid = [row for row in payments if row["step"] == step]
        trial = fsum(float(row["paid_yuan"]) for row in paid if row["layer"] == "trial")
        assert float(ledger["implementation_cost"]) == pytest.approx(trial * .008027)
        assert float(ledger["implementation_reserve"]) == pytest.approx(float(ledger["trial_budget"]) * .008027)
        assert fsum(float(row["private_service_cost_yuan"]) for row in actors) == pytest.approx(
            fsum(float(row["private_paid_yuan"]) for row in tasks if row["step"] == step))
        assert float(ledger["trial_budget"]) == pytest.approx(877905 * 1.05 ** (int(step) - 1))
        assert float(ledger["total_fiscal_gross_expenditure"]) == pytest.approx(
            fsum(float(row["paid_yuan"]) for row in paid) + float(ledger["implementation_cost"])
            + float(ledger["reserve_bureau_gross_expenditure"]))
    second = fiscal[1]
    assert second["expected_claims_source"] == "previous_others_plus_current_candidate_nominal"
    for pool in ("area", "service"):
        assert float(second[f"expected_{pool}_claims"]) == pytest.approx(fsum(
            float(row["nominal_claim_yuan"]) for row in payments if row["step"] == "1" and row["pool"] == pool))
    absent = [row for row in households if row["step"] == "2" and row["in_operating_list_at_production"] == "False"]
    assert absent and all(float(row["grain_output_kg"]) == 0 for row in absent)
    assert fsum(float(row["off_farm_income_yuan"]) for row in absent) > 0
    assert fsum(float(row["consumption_yuan"]) for row in absent) > 0


def test_same_batch_service_matches_reserve_provider_capacity() -> None:
    from modules.agricultural_service import ServiceType
    farmer = prepared(program("family_service"))
    model = farmer.model
    provider = ServiceProvider("limited_provider", model, position=farmer.pos, initial_capacity=10)
    model.service_providers = [provider]
    market = ServiceMarket(model, model.config["production"].get("agricultural_service", {}))
    demand = {"farmer": farmer, "area": 6, "service_type": ServiceType.SINGLE_SERVICE,
              "max_price_per_mu": 10000}

    matches = market.match_service_demands([demand, dict(demand)], [provider])

    assert len(matches) == 1
    assert sum(match["contract"].area for match in matches) <= provider.get_remaining_capacity()
    assert provider.service_area_used == 0


def test_unaffordable_service_is_not_matched_for_expected_public_reimbursement() -> None:
    from modules.agricultural_service import ServiceType
    farmer = prepared(program("family_service"))
    farmer.capital = 0
    model = farmer.model
    provider = ServiceProvider("private_cash_provider", model, position=farmer.pos, initial_capacity=100)
    market = ServiceMarket(model, model.config["production"].get("agricultural_service", {}))
    demand = {"farmer": farmer, "area": 6, "service_type": ServiceType.SINGLE_SERVICE,
              "max_price_per_mu": 10000}

    matches = market.match_service_demands([demand], [provider])

    assert matches == []


def test_p3_compares_self_operation_and_outsourcing_before_crop_choice() -> None:
    farmer = prepared(program("family_service"))
    model = farmer.model
    model.config["production"]["agricultural_service"]["enabled"] = True
    provider = ServiceProvider("choice_provider", model, position=farmer.pos, initial_capacity=100)
    model.service_providers = [provider]
    model.service_market = ServiceMarket(model, model.config["production"]["agricultural_service"])

    evaluate_fixed_rule_operators(model.config, [farmer], [], 2011)

    alternatives = farmer.p3_organization_accounts
    assert {(item.account.action, item.mode) for item in alternatives} == {
        ("maintain_grain", "self"), ("maintain_grain", "outsource"),
        ("expand_cash", "self"), ("expand_cash", "outsource"), ("exit", "transfer"),
    }
    for item in alternatives:
        if item.mode == "self":
            assert item.account.service_cost_yuan == 0
        if item.mode == "outsource":
            assert item.account.service_cost_yuan > 0
    selected = farmer.fixed_rule_selected_account
    assert any(item.account is selected and item.mode == farmer.p3_service_mode for item in alternatives)


@pytest.mark.parametrize("mode,has_demand", [("self", False), ("outsource", True)])
def test_selected_organization_mode_controls_actual_service_demand(mode: str, has_demand: bool) -> None:
    farmer = prepared(program("family_service"))
    farmer.model.config["production"]["agricultural_service"]["enabled"] = True
    farmer.p3_service_mode = mode

    demand = farmer.model._get_farmer_service_demand(farmer)

    assert (demand is not None) == has_demand


def test_cash_crop_task_is_paid_privately_without_grain_service_subsidy() -> None:
    from agents.farmer_agent import CropType
    farmer = prepared(program("family_service"))
    farmer.planned_crop = CropType.CASH_CROP
    farmer.expected_yield = 10000
    farmer.current_production_area = 10
    model = farmer.model
    provider = ServiceProvider("cash_crop_provider", model, position=farmer.pos, initial_capacity=100)
    model.service_providers = [provider]
    contract = ServiceContract(str(farmer.unique_id), provider.unique_id, 10, "grain", CustodyType.PARTIAL, 100)

    model._apply_service_contract({"farmer": farmer, "provider": provider, "contract": contract})
    farmer.harvest_and_settle()
    model.government.verify_subsidy_claims()
    model.government.distribute_subsidies()

    assert contract.p3_fulfilled and contract.p3_paid_yuan == 1000
    assert contract.crop_type == "cash_crop"
    assert model.government.total_subsidy_paid == 100
    assert all(row["layer"] == "base" for row in model.government.p3_project_payments)


def test_p3_paid_provider_settles_despite_environment_only_switch() -> None:
    from tests.test_p3_project_payments_20260909 import service
    farmer = prepared(program("family_service"))
    model = farmer.model
    model.config["unified_paper_model"] = {"enabled": True, "service_environment_only": True}
    provider = service(farmer)
    farmer.harvest_and_settle()
    before = provider.capital
    revenue = provider.annual_revenue
    depreciation = provider.large_tractors * 12000 + provider.combine_harvesters * 20000 + provider.drone_sprayers * 3000
    cost = depreciation * 1.1 + provider.service_area_used * 70

    model._settle_service_providers_end_of_year()

    assert revenue == pytest.approx(sum(c.p3_paid_yuan for c in farmer.service_contracts))
    assert provider.last_settlement_year == model.simulation_base_year + model.step_count
    assert provider.capital - before == pytest.approx(revenue - cost)
    assert provider.last_settlement_profit == pytest.approx(revenue - cost)
    assert provider.annual_revenue == 0


@pytest.mark.parametrize("signed_offset,duration,retained_area", [(-1, 1, 0), (0, 1, 6), (-1, 2, 6)])
def test_p3_first_capacity_read_releases_only_expired_occupancy(
    signed_offset: int, duration: int, retained_area: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    farmer = prepared(program("family_service"))
    model = farmer.model
    farmer.p3_service_mode = "self"
    provider = ServiceProvider("annual_provider", model, position=farmer.pos, initial_capacity=10)
    model.service_providers = [provider]
    model.service_market = ServiceMarket(model, model.config["production"].get("agricultural_service", {}))
    year = model.simulation_base_year + model.step_count + 1
    contract = ServiceContract(str(farmer.unique_id), provider.unique_id, 6, "grain",
        CustodyType.PARTIAL, 100, duration=duration, year_signed=year + signed_offset)
    provider.active_contracts = [contract]
    provider.service_area_used = 6

    def first_capacity_read() -> None:
        assert provider.get_remaining_capacity() == 10 - retained_area
        raise RuntimeError("capacity probe completed")

    monkeypatch.setattr(model, "_refresh_total_land_area", first_capacity_read)
    with pytest.raises(RuntimeError, match="capacity probe completed"):
        model.step()

    assert provider.service_area_used == retained_area
    assert provider.get_remaining_capacity() == 10 - retained_area
    assert provider.active_contracts == ([contract] if retained_area else [])
    depreciation = provider.large_tractors * 12000 + provider.combine_harvesters * 20000 + provider.drone_sprayers * 3000
    assert provider.end_of_year_settlement(year) == pytest.approx(-depreciation * 1.1 - retained_area * 70)


def test_p3_consumption_debits_cash_not_consumer_surplus() -> None:
    farmer = prepared(program())
    farmer.disposable_income = 30000
    before = farmer.capital

    farmer.make_consumption_decision()

    assert farmer.total_consumption_expenditure > 0
    assert farmer.capital - before == pytest.approx(-farmer.total_consumption_expenditure)


def test_livelihood_only_exit_does_not_force_elderly_death() -> None:
    from modules.population_dynamics_v2 import PopulationManagerV2
    from scripts.n1_income_fixtures import ROOT
    farmer = prepared(program())
    model = farmer.model
    model.config["population_dynamics"].update(enabled=True, paper_model_exit_mode="livelihood_exit_only")
    farmer.age = 80
    manager = PopulationManagerV2(feature_library_path=str(
        ROOT / "data/derived_runtime/v4_8/cfps_entrant_feature_library_2010_2016.json"))
    manager.model = model

    remaining, stats = manager._execute_exits([farmer], 1)

    assert stats == {"death": 0, "urbanization": 1}
    assert remaining == []
