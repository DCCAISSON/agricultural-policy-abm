from __future__ import annotations

from dataclasses import replace

import pytest

from core.p3_project_contracts import P3InputsMissing, P3Program, PreservedProject
from scripts.n1_income_fixtures import HAND_CASES, hand_farmer
from core.p3_project_contracts import P3AccountingError, ProductionRecord
from agents.farmer_agent import FarmerAgent
from modules.agricultural_service import ServiceContract, ServiceProvider, CustodyType, ServiceType


def prepared(spec: P3Program) -> FarmerAgent:
    farmer = hand_farmer(HAND_CASES[0])
    farmer.expected_yield = 5000.0
    farmer.model.farmers = [farmer]
    farmer.model.agricultural_enterprises = []
    farmer.model.config["fiscal_system"] = {"p3_support": spec.model_dump()}
    farmer.model.government.begin_annual_cycle()
    return farmer


def service(farmer: FarmerAgent, duration: int = 1) -> ServiceProvider:
    provider = ServiceProvider("p3_provider", farmer.model, initial_capacity=100)
    farmer.model.service_providers = [provider]
    for task in (ServiceType.TILLAGE, ServiceType.HARVESTING):
        contract = ServiceContract(str(farmer.unique_id), provider.unique_id, 2, "grain",
            CustodyType.PARTIAL, 200, duration=duration, service_type=task)
        farmer.model._apply_service_contract({"farmer": farmer, "provider": provider, "contract": contract})
    return provider


def program(rule: str = "area_universal", budget: float = 3500.0) -> P3Program:
    return P3Program(
        rule=rule,
        model_price_year=2010,
        trial_price_year=2010,
        trial_to_model=1.0,
        service_2019_to_model=0.8,
        price_conversion_source="constructed_test_factor_not_an_estimate",
        base_budget_yuan=1000.0,
        trial_budget_yuan=budget,
        implementation_cost_yuan=0.0,
        expected_area_claims_yuan=3500.0,
        expected_service_claims_yuan=500.0,
        preserved_projects=(PreservedProject(
            project_id="land_protection", basis="rights_area", rate=10.0,
            price_year=2010, to_model=1.0,
            source="constructed_test_contract_not_observed_policy",
        ),),
    )


def test_zero_trial_budget_preserves_base_through_actual_payment_entry() -> None:
    farmer = hand_farmer(HAND_CASES[0])
    model = farmer.model
    model.farmers = [farmer]
    model.agricultural_enterprises = []
    model.config["fiscal_system"] = {"p3_support": program(budget=0.0).model_dump()}
    government = model.government
    government.begin_annual_cycle()
    before = farmer.capital

    expected = farmer.expected_fixed_rule_account("maintain_grain").support_yuan
    assert farmer.capital == before
    farmer.submit_subsidy_claim()
    government.verify_subsidy_claims()
    assert farmer.capital == before
    government.distribute_subsidies()
    assert expected == pytest.approx(100.0)
    assert farmer.capital - before == pytest.approx(100.0)
    assert sum(row["paid_yuan"] for row in government.p3_project_payments) == pytest.approx(100.0)
    assert government.total_arrears_outstanding == 0.0
    assert government.remaining_budget == pytest.approx(900.0)


def test_service_requires_settled_cash_and_caps_tasks_on_same_plot_season() -> None:
    farmer = prepared(program("family_service"))
    provider = service(farmer)
    assert sum(c.p3_paid_yuan for c in farmer.service_contracts) == 0
    assert all(c.p3_fulfilled for c in farmer.service_contracts)
    assert provider.annual_revenue == pytest.approx(800)
    farmer.harvest_and_settle()
    government = farmer.model.government
    government.verify_subsidy_claims()
    government.distribute_subsidies()
    rows = [r for r in government.p3_project_payments if r["pool"] == "service"]
    assert sum(c.p3_paid_yuan for c in farmer.service_contracts) == pytest.approx(800)
    assert sum(r["paid_yuan"] for r in rows) == pytest.approx(min(800 * .3, 2 * 100 * .8))


def test_mix_keeps_half_pools_separate_and_does_not_carry_arrears() -> None:
    farmer = prepared(program("area_service_mix", 1000))
    service(farmer)
    farmer.harvest_and_settle()
    government = farmer.model.government
    government.verify_subsidy_claims()
    government.distribute_subsidies()
    rows = government.p3_project_payments
    assert sum(r["paid_yuan"] for r in rows if r["pool"] == "area") == pytest.approx(500)
    assert sum(r["paid_yuan"] for r in rows if r["pool"] == "service") == pytest.approx(160)
    assert government.current_subsidy_runtime["trial_unspent_yuan"]["service"] == pytest.approx(340)
    assert government.total_arrears_outstanding == 0


def test_exit_after_verification_keeps_original_recipient_and_payment_is_idempotent() -> None:
    farmer = prepared(program())
    farmer.submit_subsidy_claim()
    government = farmer.model.government
    government.verify_subsidy_claims()
    farmer.model.farmers = []
    before = farmer.capital
    government.distribute_subsidies()
    government.distribute_subsidies()
    assert farmer.capital - before == pytest.approx(3600)
    assert {r["recipient_id"] for r in government.p3_project_payments} == {str(farmer.unique_id)}


def test_base_gap_fails_before_any_cash_payment() -> None:
    farmer = prepared(program().model_copy(update={"base_budget_yuan": 99}))
    before = farmer.capital
    farmer.submit_subsidy_claim()
    with pytest.raises(P3AccountingError, match="funding gap"):
        farmer.model.government.verify_subsidy_claims()
    assert farmer.capital == before


def test_two_seasons_qualify_annually_but_rates_apply_to_eligible_seasons() -> None:
    spec = program(budget=0).model_copy(update={"base_budget_yuan": 20000,
        "preserved_projects": (
            PreservedProject(project_id="early", basis="grain_area", rate=120, price_year=2010,
                to_model=1, source="constructed", threshold_mu=50),
            PreservedProject(project_id="local", basis="grain_area", rate=150, price_year=2010,
                to_model=1, source="constructed", threshold_mu=50, qualification_period="season",
                eligible_seasons=("early", "late")),
        )})
    farmer = prepared(spec)
    patch = farmer.model.land_patches[0]
    patch.area = 30
    farmer.p3_production_records = tuple(ProductionRecord(plot_id=str(patch.patch_id), season=s,
        area_mu=30, harvest_kg=10000, source="constructed_two_season_input").model_dump()
        for s in ("early", "late"))
    farmer.submit_subsidy_claim()
    farmer.model.government.verify_subsidy_claims()
    farmer.model.government.distribute_subsidies()
    assert farmer.model.government.total_subsidy_paid == pytest.approx(60 * 120)


def test_order_delivery_is_not_full_harvest_or_additional_sale_revenue() -> None:
    spec = program(budget=0).model_copy(update={"preserved_projects": (
        PreservedProject(project_id="order", basis="order_delivery", rate=.2,
            price_year=2010, to_model=1, source="constructed"),)})
    farmer = prepared(spec)
    patch = farmer.model.land_patches[0]
    farmer.p3_production_records = (ProductionRecord(plot_id=str(patch.patch_id), season="annual_grain",
        area_mu=10, harvest_kg=5000, qualified_delivery_kg=2000, source="constructed").model_dump(),)
    before, sales = farmer.capital, farmer.agricultural_income
    farmer.submit_subsidy_claim()
    farmer.model.government.verify_subsidy_claims()
    farmer.model.government.distribute_subsidies()
    assert farmer.capital - before == pytest.approx(400)
    assert farmer.agricultural_income == sales


@pytest.mark.parametrize("rate", [0, 350, 525])
def test_b19_rate_source_does_not_change_p3_fixed350_contract(rate: float) -> None:
    with pytest.raises(P3AccountingError):
        P3Program.model_validate(program().model_dump() | {"nominal_rate": 525})
    farmer = prepared(P3Program.model_validate(program(budget=10000).model_dump() | {
        "rate_origin": "b19_scenario", "nominal_rate": rate,
        "scenario_id": f"b19_s{rate}_m1_v1", "scenario_rate_source": "constructed_B19_runner_scenario"}))
    farmer.model.config["capitalization"] = {"base_rate": .4}
    assert farmer.model.environment._get_subsidy_capitalization_rate("farmer", 10) > 0
    farmer.submit_subsidy_claim()
    farmer.model.government.verify_subsidy_claims()
    farmer.model.government.distribute_subsidies()
    assert farmer.model.government.total_subsidy_paid == pytest.approx(100 + 10 * rate)


def test_scale_threshold_applies_to_all_eligible_area() -> None:
    farmer = hand_farmer(replace(HAND_CASES[0], owned_mu=80.0))
    model = farmer.model
    model.farmers = [farmer]
    model.agricultural_enterprises = []
    spec = program("scale_eligibility", 30000.0).model_copy(update={
        "expected_area_claims_yuan": 28000.0,
    })
    model.config["fiscal_system"] = {"p3_support": spec.model_dump()}
    government = model.government
    government.begin_annual_cycle()
    farmer.submit_subsidy_claim()

    government.verify_subsidy_claims()
    government.distribute_subsidies()

    trial = [row for row in government.p3_project_payments if row["layer"] == "trial"]
    assert sum(row["nominal_claim_yuan"] for row in trial) == pytest.approx(80.0 * 350.0)
    assert sum(row["paid_yuan"] for row in trial) == pytest.approx(28000.0)


def test_missing_base_or_price_inputs_block_without_assuming_zero() -> None:
    farmer = hand_farmer(HAND_CASES[0])
    spec = program().model_copy(update={"preserved_projects": None, "model_price_year": None})
    farmer.model.config["fiscal_system"] = {"p3_support": spec.model_dump()}
    before = farmer.capital

    with pytest.raises(P3InputsMissing):
        farmer.model.government.begin_annual_cycle()

    assert farmer.capital == before


def test_five_year_contract_has_no_duration_yield_or_soil_bonus() -> None:
    outcomes = []
    for duration in (1, 5):
        farmer = prepared(program("family_service"))
        for patch in farmer.model.land_patches:
            patch.lease_contract_years = duration
        service(farmer, duration)
        farmer.model._update_ecological_feedback()
        outcomes.append((farmer.service_yield_bonus_multiplier,
                         tuple(p.soil_quality for p in farmer.model.land_patches)))
    assert outcomes[0] == outcomes[1]


def test_unpaid_service_does_not_receive_reimbursement() -> None:
    farmer = prepared(program("family_service"))
    service(farmer)
    farmer.submit_subsidy_claim()
    farmer.model.government.verify_subsidy_claims()
    farmer.model.government.distribute_subsidies()
    assert farmer.model.government.total_subsidy_paid == pytest.approx(100)


def test_expected_service_support_is_not_working_capital() -> None:
    from core.fixed_rule_runtime import expected_support
    farmer = prepared(program("family_service"))
    before = farmer.capital
    value = expected_support(farmer, 10, "farmer", service_cost_yuan=800, service_area_mu=2)
    assert value == pytest.approx(260)
    assert farmer.capital == before
    farmer.capital = 100
    with pytest.raises(P3AccountingError, match="working capital"):
        service(farmer)
    assert farmer.service_contracts == []


def test_two_recipients_are_prorated_together_not_first_come() -> None:
    farmer = prepared(program(budget=700))
    other = farmer.model.land_patches[1].owner
    patch = farmer.model.land_patches[1]
    patch.area = 10
    patch.current_operator = other
    patch.leased_to = None
    from agents.farmer_agent import CropType
    other.planned_crop = CropType.GRAIN
    farmer.model.farmers = [farmer, other]
    farmer.submit_subsidy_claim()
    other.submit_subsidy_claim()
    farmer.model.government.verify_subsidy_claims()
    farmer.model.government.distribute_subsidies()
    trial = [r for r in farmer.model.government.p3_project_payments if r["layer"] == "trial"]
    assert [r["paid_yuan"] for r in trial] == pytest.approx([350, 350])


@pytest.mark.parametrize("legacy_target", ["operator", "owner"])
def test_rights_owner_and_enterprise_operator_receive_different_projects(legacy_target: str) -> None:
    farmer = hand_farmer(HAND_CASES[0])
    model = farmer.model
    enterprise = model.agricultural_enterprises[0]
    patch = model.land_patches[0]
    patch.current_operator = patch.leased_to = enterprise
    model.farmers = [farmer]
    model.config["fiscal_system"] = {"p3_support": program().model_dump()}
    enterprise.p3_production_records = (ProductionRecord(plot_id=str(patch.patch_id), season="annual_grain",
        area_mu=10, harvest_kg=5000, source="constructed_enterprise_production").model_dump(),)
    model.government.begin_annual_cycle()
    model.environment.active_policy["subsidy_target"] = legacy_target
    farmer.submit_subsidy_claim()
    enterprise.submit_subsidy_claim()
    model.government.verify_subsidy_claims()
    before = farmer.capital, enterprise.capital
    model.government.distribute_subsidies()
    assert farmer.capital - before[0] == pytest.approx(100)
    assert enterprise.capital - before[1] == pytest.approx(3500)


def test_missing_order_delivery_blocks_not_imputed_as_zero() -> None:
    spec = program().model_copy(update={"preserved_projects": (
        PreservedProject(project_id="order", basis="order_delivery", rate=.2,
            price_year=2010, to_model=1, source="constructed"),)})
    farmer = prepared(spec)
    before = farmer.capital
    with pytest.raises(P3InputsMissing, match="delivery missing"):
        farmer.submit_subsidy_claim()
    assert farmer.capital == before
