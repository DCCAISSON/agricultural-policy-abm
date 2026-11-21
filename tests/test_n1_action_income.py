from dataclasses import replace

import pytest

from scripts.n1_income_fixtures import HAND_CASES, hand_farmer
from agents.farmer_agent import CropType
from agents.enterprise_agent import CropType as EnterpriseCropType
from core.fixed_rule_runtime import enact_fixed_rule_action, submit_action_income_bids, fixed_rule_operator_inertia


@pytest.mark.parametrize("case", HAND_CASES, ids=lambda case: case.role)
def test_six_hand_calculated_accounts(case):
    farmer = hand_farmer(case)
    accounts = [farmer.expected_fixed_rule_account(action) for action in ("maintain_grain", "expand_cash", "exit")]
    assert [a.net_income for a in accounts] == pytest.approx([case.grain_net, case.cash_net, case.exit_net])
    assert accounts[2].rent_out_yuan == 400 * case.owned_mu
    assert accounts[2].rent_in_yuan == 0
    assert accounts[0].support_yuan == pytest.approx(1840)
    assert accounts[1].support_yuan == 0
    assert accounts[2].feasible
    if case.role in {"labor_shortage", "capital_shortage"}:
        assert not accounts[0].feasible
        assert not accounts[1].feasible
    else:
        assert accounts[0].feasible


def test_common_income_does_not_change_crop_ranking_or_side_effects():
    farmer = hand_farmer(HAND_CASES[0])
    prior = (farmer.capital, farmer.hired_labor_days, farmer.model.environment.annual_capitalization_loss)
    before = [farmer.expected_fixed_rule_account(a) for a in ("maintain_grain", "expand_cash", "exit")]
    farmer.fixed_rule_common_income_yuan = 10000
    after = [farmer.expected_fixed_rule_account(a) for a in ("maintain_grain", "expand_cash", "exit")]
    assert [a.net_income - b.net_income for a, b in zip(after, before)] == [10000] * 3
    assert (farmer.capital, farmer.hired_labor_days, farmer.model.environment.annual_capitalization_loss) == prior


def test_expansion_changes_complete_cost_and_resource_requirements():
    farmer = hand_farmer(HAND_CASES[0])
    current = farmer.expected_fixed_rule_account("maintain_grain")
    expanded = farmer.expected_fixed_rule_account("maintain_grain", 20.0)
    assert expanded.sales_yuan > current.sales_yuan
    assert expanded.production_cost_yuan > current.production_cost_yuan
    assert expanded.family_labor_days > current.family_labor_days
    assert expanded.rent_in_yuan == 4000
    assert expanded.working_capital_required > current.working_capital_required


def test_cost_and_risk_are_deducted_once_in_their_own_accounts():
    farmer = hand_farmer(HAND_CASES[0])
    farmer.fixed_rule_conversion_cost_yuan = 300.0
    farmer.fixed_rule_risk_penalty_yuan = {"expand_cash": 700.0}
    cash = farmer.expected_fixed_rule_account("expand_cash")
    assert cash.net_income == 18700
    assert cash.score == 18000
    assert cash.working_capital_required == cash.cash_cost


def test_zero_family_labor_is_not_replaced_by_a_365_day_default():
    farmer = hand_farmer(replace(HAND_CASES[0], labor_days=0))
    account = farmer.expected_fixed_rule_account("maintain_grain")
    assert account.family_labor_days == 0
    assert account.off_farm_yuan == 0
    assert account.hired_labor_days == 100


def test_fixed_cash_choice_survives_the_production_stage():
    farmer = hand_farmer(HAND_CASES[0])
    enact_fixed_rule_action(farmer, "expand_cash", "household")
    farmer.make_production_decisions()
    assert farmer.planned_crop == CropType.CASH_CROP


def test_unfilled_labor_request_is_not_booked_as_a_real_payment():
    farmer = hand_farmer(HAND_CASES[3])
    farmer.fixed_rule_labor_settled = True
    farmer.hired_labor_days = farmer.hired_labor_cost = 0.0
    cost = farmer._calculate_detailed_cost(CropType.GRAIN, 10)
    assert cost["cash_total_cost"] == 9000
    assert cost["hired_labor_cost"] == 0
    assert farmer.hired_labor_days == 0
    assert farmer.requested_hired_labor_days == 90


def test_exit_settles_nonfarm_income_without_subletting_someone_elses_land():
    farmer = hand_farmer(HAND_CASES[1])
    model = farmer.model
    model.farmers = [farmer]
    model.agricultural_enterprises = []
    farmer.cash_net_income = 0
    enact_fixed_rule_action(farmer, "exit", "household")
    model._run_fixed_rule_production_stage()
    assert farmer.cash_net_income == 15000
    assert farmer.grain_production == 0
    assert farmer.land_rental_income == 0
    assert farmer.land_rental_expense == 0
    assert all(p.leased_to is not farmer for p in model.land_patches)


def test_enterprise_has_actual_operating_accounts_instead_of_missing_farmer_fields():
    farmer = hand_farmer(HAND_CASES[0])
    enterprise = farmer.model.agricultural_enterprises[0]
    rented = farmer.model.land_patches[1]
    rented.area = 10
    rented.leased_to = rented.current_operator = enterprise
    rented.rental_price = 4000
    enterprise.leased_patches = [rented]
    farmer.model.invalidate_land_tenure_caches()
    account = enterprise.expected_fixed_rule_account("maintain_grain")
    assert account.sales_yuan > 0
    assert account.production_cost_yuan > 0
    assert account.rent_in_yuan == 4000
    assert account.off_farm_yuan == 0
    assert account.management_cost_yuan > 0
    contract = enterprise.expected_fixed_rule_account("contract")
    assert contract.area_mu == 0
    assert contract.rent_out_yuan == contract.off_farm_yuan == 0


def test_labor_planning_uses_the_same_crop_requirements_as_costs():
    farmer = hand_farmer(HAND_CASES[3])
    farmer.model.config["production"]["cost_structure"]["input_requirements"]["labor_days"]["grain_farmer"] = 14
    account = farmer.expected_fixed_rule_account("maintain_grain")
    profile = farmer.estimate_production_labor_profile(CropType.GRAIN, 10)
    assert profile["total_labor_need"] == account.family_labor_days + account.hired_labor_days


def test_small_farmer_labor_gap_is_submitted_and_not_fabricated_when_unfilled():
    farmer = hand_farmer(HAND_CASES[3])
    model = farmer.model
    model.farmers = [farmer]
    model.agricultural_enterprises = []
    farmer.planned_crop = CropType.GRAIN
    model._conduct_labor_market()
    assert model.labor_market.market_history[-1]["total_demand"] == 90
    assert farmer.hired_labor_days == farmer.hired_labor_cost == 0
    assert farmer.fixed_rule_labor_fulfillment == pytest.approx(0.1)


def test_enterprise_realized_cost_keeps_actual_labor_payment():
    farmer = hand_farmer(HAND_CASES[0])
    enterprise = farmer.model.agricultural_enterprises[0]
    enterprise._fixed_rule_preview_area = 10.0
    enterprise.crop_portfolio = {EnterpriseCropType.GRAIN: 0.4, EnterpriseCropType.CASH_CROP: 0.6}
    enterprise.fixed_rule_labor_settled = True
    enterprise.hired_labor_days = 7.0
    enterprise.hired_labor_cost = 735.0
    enterprise.calculate_production_cost()
    assert enterprise.hired_labor_days == 7.0
    assert enterprise.hired_labor_cost == 735.0


def test_enterprise_cash_choice_survives_production_subject_to_grain_floor():
    farmer = hand_farmer(HAND_CASES[0])
    enterprise = farmer.model.agricultural_enterprises[0]
    enterprise._fixed_rule_preview_area = 10.0
    enact_fixed_rule_action(enterprise, "expand_cash", "scale")
    planned = enterprise.crop_portfolio[EnterpriseCropType.GRAIN]
    enterprise.make_production_decisions()
    assert enterprise.crop_portfolio[EnterpriseCropType.GRAIN] == planned


def test_decision_year_matches_the_production_year():
    farmer = hand_farmer(HAND_CASES[0])
    model = farmer.model
    model.farmers = [farmer]
    model.agricultural_enterprises = []
    model.step_count = 1
    model.config["experiment"]["smoke_years"]["start"] = model.simulation_base_year
    model._run_fixed_rule_decisions()
    assert model.v4_8_fixed_rule_trace[-1]["year"] == model.simulation_base_year + 1


def test_grain_expansion_uses_a_complete_income_bid_without_relabeling_cash():
    farmer = hand_farmer(HAND_CASES[0])
    model = farmer.model
    parcel = model.land_patches[1]
    parcel.end_lease()
    parcel.area = 2.0
    model.land_clearing_house.add_supply_offer(parcel, parcel.owner, 800.0)
    enact_fixed_rule_action(farmer, "maintain_grain", "household")
    farmer.fixed_rule_bid_accounts = []
    submit_action_income_bids(farmer, 2.0)
    assert farmer.fixed_rule_current_action == "maintain_grain"
    assert len(model.land_clearing_house.demand_offers) == 1
    row = farmer.fixed_rule_bid_accounts[0]
    assert row["after_rent_in_yuan"] - row["before_rent_in_yuan"] == 800.0
    assert row["after_working_capital_required"] > row["before_working_capital_required"]


def test_exit_from_direct_farming_does_not_erase_the_household():
    farmer = hand_farmer(HAND_CASES[0])
    model = farmer.model
    model.farmers = [farmer]
    enact_fixed_rule_action(farmer, "exit", "household")
    model._finalize_fixed_rule_exits()
    assert farmer in model.farmers


def test_engineering_inertia_is_identical_for_both_accounting_rules():
    farmer = hand_farmer(HAND_CASES[0])
    rule = farmer.model.config["runtime_fixed_rule"]
    rule["minimum_inertia_margin_yuan_per_mu"] = 50.0
    rule["inertia_basis"] = "yuan_per_mu"
    new_margin = fixed_rule_operator_inertia(farmer)
    rule["accounting_basis"] = "legacy_lagged_v1"
    assert fixed_rule_operator_inertia(farmer) == new_margin == 500.0


def test_delivery_arithmetic_includes_realized_farm_labor_sales_once():
    from scripts.build_n1_income_delivery import COMPONENTS, arithmetic_check

    row = {realized: "0" for _, realized, _ in COMPONENTS}
    row.update(net_income_yuan="700", labor_sale_income_yuan="700")
    assert arithmetic_check(row, True) == 0


def test_real_case_selection_does_not_use_realized_outcomes():
    from scripts.build_n1_income_delivery import OUTPUT_ROOT, read_rows, select_initial_cases

    initial = read_rows(OUTPUT_ROOT / "n1_runtime_20260908_v1/initial_actors.csv")
    selected = select_initial_cases(initial)
    changed = [{**row, "action": "exit", "net_income_yuan": "-9999999"} for row in initial]
    assert [r["operator_id"] for r in select_initial_cases(changed)] == [r["operator_id"] for r in selected]
    assert len({r["operator_id"] for r in selected}) == 6


@pytest.mark.parametrize("action,crop", [("maintain_grain", CropType.GRAIN), ("expand_cash", CropType.CASH_CROP)])
def test_reentry_after_exit_restores_a_real_crop_enum(action, crop):
    farmer = hand_farmer(HAND_CASES[0])
    enact_fixed_rule_action(farmer, "exit", "household")
    farmer.last_crop_choice = None
    enact_fixed_rule_action(farmer, action, "household")
    assert farmer.planned_crop is crop
