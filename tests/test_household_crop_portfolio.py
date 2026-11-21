import numpy as np
import pytest

from agents.farmer_agent import CropType
from core.fixed_rule_runtime import prepare_preproduction_cash
from core.p3_project_claims import production_records
from core.p3_project_contracts import PreservedProject
from modules.agricultural_service import ServiceMarket, ServiceProvider
from scripts.n1_income_fixtures import HAND_CASES, hand_farmer
from tests.test_p3_project_payments_20260909 import program


def household(share=.6, *, spec=None, margin=1e7):
    farmer = hand_farmer(HAND_CASES[0])
    model = farmer.model
    model.farmers, model.agricultural_enterprises = [farmer], []
    model.config["fiscal_system"] = {"p3_support": (spec or program()).model_dump()}
    model.step_count = 1
    model.config["runtime_fixed_rule"].update(
        household_crop_portfolio=True, minimum_inertia_margin_yuan_per_mu=margin)
    model.household_initial_grain_shares = {} if share is None else {str(farmer.unique_id): share}
    model.government.begin_annual_cycle()
    return farmer


def test_six_to_seven_decision_harvest_cash_and_next_year():
    project = PreservedProject(project_id="seven", basis="grain_area", threshold_mu=7,
        rate=800, price_year=2010, to_model=1, source="test")
    spec = program(budget=0).model_copy(update={"preserved_projects": (project,), "base_budget_yuan": 10000})
    f = household(spec=spec, margin=0)
    m = f.model
    m.environment.market_prices["cash_crop"] = 1.5
    m._run_actor_land_market_planning_stage()
    accounts = {a.action: a for a in f.fixed_rule_action_accounts}
    assert accounts["portfolio_keep"].support_yuan == 0
    assert accounts["portfolio_up"].support_yuan == 5600
    assert f.fixed_rule_selected_account is accounts["portfolio_up"]
    assert f.planned_crop is CropType.MIXED
    assert f.household_planned_grain_share == .7
    m._run_actor_labor_planning_stage()
    m._conduct_service_market()
    m._conduct_labor_market()
    m._run_fixed_rule_production_stage()
    assert (f.grain_area, f.cash_crop_area) == pytest.approx((7, 3))
    assert f.cash_production_cost == pytest.approx(7*900 + 3*1100)
    assert f.grain_production > 0 and f.cash_crop_production > 0
    assert f.last_income_settlement["sales_yuan"] == pytest.approx(
        f.grain_production * f._get_output_price_per_kg("crop_price")
        + f.cash_crop_production * f._get_output_price_per_kg("cash_crop"))
    assert [(r.plot_id, r.area_mu) for r in production_records(f)] == [(None, 7)]
    before = f.capital
    m.government.verify_subsidy_claims()
    assert f.capital == before
    m.government.distribute_subsidies()
    assert f.capital - before == pytest.approx(5600)
    assert [(r["plot_id"], r["paid_yuan"]) for r in m.government.p3_project_payments if r["paid_yuan"]] == [(None, 5600)]
    m.step_count += 1
    m.household_initial_grain_shares[str(f.unique_id)] = .1
    m._run_fixed_rule_decisions()
    assert f.household_previous_grain_share == .7
    assert f.household_candidate_grain_shares["portfolio_down"] == pytest.approx(.6)


@pytest.mark.parametrize("share,expected", [(0, [0, .1, 1]), (1, [0, .9, 1]), (None, [0, 1])])
def test_endpoints_unique_and_unmatched_has_no_imputed_share(share, expected):
    f = household(share)
    f.model._run_fixed_rule_decisions()
    assert sorted(f.household_candidate_grain_shares.values()) == pytest.approx(expected)
    if share is not None:
        assert f.household_planned_grain_share == share


def test_disabled_uses_only_original_accounts():
    f = household()
    del f.model.config["runtime_fixed_rule"]["household_crop_portfolio"]
    f.model._run_fixed_rule_decisions()
    assert [a.action for a in f.fixed_rule_action_accounts] == ["maintain_grain", "expand_cash", "exit"]
    assert not hasattr(f, "household_candidate_grain_shares")


@pytest.mark.parametrize("resource", ["working_capital", "hired_labor"])
def test_infeasible_portfolios_exit_without_production(resource):
    f = household()
    if resource == "working_capital":
        f.capital = 0
    else:
        f.total_labor_units = f.fixed_rule_hired_labor_available_days = 0
    f.model._run_fixed_rule_decisions()
    assert all(resource in a.constraint_reason for a in f.fixed_rule_action_accounts if a.action != "exit")
    assert f.fixed_rule_current_action == "exit"
    f.model._conduct_labor_market()
    f.model._run_fixed_rule_production_stage()
    assert f.grain_area == f.cash_crop_area == f.agricultural_income == 0
    assert f.household_actual_grain_share is None
    assert not production_records(f)


def test_mixed_service_before_labor_and_actual_payment():
    f = household(spec=program("family_service"))
    m = f.model
    cfg = m.config["production"]["agricultural_service"]
    cfg["enabled"] = True
    cfg.setdefault("provider_economics", {}).update(cash_mode="operating_cash", initial_cash_yuan=20000)
    m.service_providers = [ServiceProvider("provider", m, initial_capacity=100)]
    m.service_market = ServiceMarket(m, cfg)
    m.config["runtime_fixed_rule"]["preproduction_cash"] = dict(
        mode="reserved_no_credit", funds_basis="capital_as_liquid_operating_balance")
    f.age, f.total_labor_units = 65, 100
    m._run_actor_land_market_planning_stage()
    assert f.planned_crop is CropType.MIXED
    unserved = f.estimate_production_labor_profile()["total_labor_need"]
    prepare_preproduction_cash(m)
    # Exercise the selected share with an outsourcing contract, independent of mode ranking.
    f.p3_service_mode = "outsource"
    m._conduct_service_market()
    contract, = f.service_contracts
    assert contract.crop_type == "mixed" and contract.p3_paid_yuan == 0
    m._conduct_labor_market()
    assert f.fixed_rule_labor_need < unserved
    assert f.requested_hired_labor_days == pytest.approx(max(0, f.fixed_rule_labor_need - f.family_labor_used))
    m._run_fixed_rule_production_stage()
    assert contract.p3_paid_yuan == pytest.approx(f.service_fee_expense)
    assert f.last_preproduction_cash_account["actual_paid_yuan"] == pytest.approx(f.total_expense)
    m.government.verify_subsidy_claims()
    m.government.distribute_subsidies()
    row, = [r for r in m.government.p3_project_payments if r["pool"] == "service"]
    assert row["plot_id"] is None
    assert row["paid_yuan"] == pytest.approx(min(.3 * contract.p3_paid_yuan * .6, 6 * 80))


def test_shared_crop_shocks_leave_same_next_draw(monkeypatch):
    import agents.farmer_agent as module
    f = household()
    f.model._run_fixed_rule_decisions()
    draws = []
    for crop in (CropType.GRAIN, CropType.CASH_CROP, CropType.MIXED):
        rng = np.random.default_rng(17)
        calls = []
        def normal(actor, loc, scale):
            calls.append((loc, scale))
            return rng.normal(loc, scale)
        monkeypatch.setattr(module, "np_normal", normal)
        monkeypatch.setattr(f, "_get_climate_shock", lambda loc, scale: normal(f, loc, scale))
        f.planned_crop = crop
        f.make_production_decisions()
        draws.append((calls, rng.random()))
    assert draws[0] == draws[1] == draws[2]
    assert len(draws[0][0]) == 2
    traces = f.last_yield_component_traces
    for key in ("climate_shock", "random_factor"):
        assert traces["grain"][key] == traces["cash_crop"][key]
