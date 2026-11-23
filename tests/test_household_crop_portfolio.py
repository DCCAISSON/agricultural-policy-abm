import numpy as np
import pytest
import hashlib
import json
from pathlib import Path

from agents.farmer_agent import CropType
from core.fixed_rule_runtime import (
    NFPS_HOUSEHOLD_ADJUSTMENT_PROXY, fixed_rule_operator_inertia, prepare_preproduction_cash,
)
from core.p3_project_claims import production_records
from core.p3_project_contracts import PreservedProject
from modules.agricultural_service import ServiceMarket, ServiceProvider
from scripts.n1_income_fixtures import HAND_CASES, hand_farmer
from tests.test_p3_project_payments_20260909 import program


@pytest.mark.parametrize("target",[0.,.6,1.])
def test_estimated_mix_is_paid_and_uses_real_labor(target):
    from types import SimpleNamespace
    from core.fixed_rule_runtime import continuous_farm_plan
    a=SimpleNamespace(area_limit=10,sowing_intensity=2.,family_days=3.,crop_mix_target=target)
    for gs,cs,aid in [(600,900,0),(900,900,0),(600,1100,0),(600,900,80)]:
        args=(gs,cs,100,150,3,6,20,aid)
        p=continuous_farm_plan(a,10,10,700,0,args,False)
        g,c=p['grain'],p['crop']
        assert 0<g+c<20 and g/(g+c)==pytest.approx(target)
        assert p['cost']<=700+1e-7 and p['hired']==pytest.approx(max(0,3*g+6*c-3))
        assert p['cultivated']==pytest.approx((g+c)/2)
        zero=continuous_farm_plan(a,10,10,0,0,args,False)
        assert zero['cost']==zero['cultivated']==0


def test_affordable_area_and_midpoint():
    from types import SimpleNamespace
    from core.fixed_rule_runtime import continuous_farm_plan
    from modules.land_clearing_house import LandClearingHouse
    a=SimpleNamespace(sowing_intensity=1.,family_days=100.,area_limit=30.)
    inputs=(200.,50.,100.,100.,1.,1.,20.,0.)
    p=continuous_farm_plan(a,10.,3.,600.,20.,inputs,True)
    assert (p['grain'],p['offer'],p['demand'])==pytest.approx((6.,3.,0.))
    after=continuous_farm_plan(a,8.,1.,600.,0.,inputs,False)
    assert after['cultivated']==6. and 8.-after['cultivated']==2.
    p=continuous_farm_plan(a,10.,3.,3000.,20.,inputs,True)
    assert p['demand']>0 and p['offer']==0
    def order(i,side,q,price):
        return dict(id=i,region=1,side=side,quantity=q,price=price)
    d=LandClearingHouse.clear_area_orders([order('s','sell',4,10),order('t','sell',3,25),
        order('u','sell',3,95),order('b','buy',3,75),order('c','buy',5,35)],2011,3)
    assert sum(x['area'] for x in d)==7 and {x['rent'] for x in d}=={30.}
    assert all(x['end']==2013 and x['seller']!='u' for x in d)


def test_fixed_area_government_budget_and_previous_state():
    from agents.government_agent import GovernmentAgent
    pre={11:{'grain_kg':100,'grain_gap':1,'leased_mu':2},12:{'grain_kg':40,'grain_gap':2,'leased_mu':1}}
    sar=GovernmentAgent.announce_area_support(None,2011,pre,50.)
    claims,regions={'a':60,'b':40},{'a':11,'b':12}
    paid=GovernmentAgent.settle_area_support(None,sar,claims,regions)
    assert paid=={'a':30.,'b':20.} and sar['state']==pre and not sar['training']
    for action in ('scenario_region_gap','scenario_region_transfer'):
        s=GovernmentAgent.announce_area_support(None,2011,pre,50.,action)
        assert sum(GovernmentAgent.settle_area_support(None,s,claims,regions).values())==pytest.approx(50.)


@pytest.mark.parametrize("kind",["AGRI_CORPORATION","COOPERATIVE"])
def test_potential_organization_must_win_land_before_harvest(kind):
    from types import SimpleNamespace as S
    from core.model import MainModel
    from agents.enterprise_agent import AgriculturalEnterpriseAgent
    from agents.government_agent import GovernmentAgent
    from modules.land_clearing_house import LandClearingHouse
    org=S(unique_id='org')
    AgriculturalEnterpriseAgent.initialize_area_business(org,kind,11,2000.,4.)
    household=S(unique_id='hh',area_kind='household',base_area=6.,lease_right=4.,zone=11,
                area_limit=6.,sowing_intensity=1.,family_days=0.,active_area_business=True,
                balance=0.,reserve=0.,overhead=0.,consumption=0.,wage_income=0.)
    bankrupt=S(unique_id='loss')
    AgriculturalEnterpriseAgent.initialize_area_business(bankrupt,kind,11,1.,4.)
    econ=dict(seed=1,actors=[household,org,bankrupt],inputs={11:(500.,10.,100.,100.,1.,1.)},
              yields={11:200.},target={11:500.},rate=0.,budget=100.,wage=20.,term=2,
              contracts=[],rents={11:50.},state={11:dict(grain_kg=0.)})
    m=S(step_count=0,area_economy=econ,government=GovernmentAgent.__new__(GovernmentAgent),
        land_clearing_house=LandClearingHouse())
    MainModel._step_area_economy(m)
    rows,orders,deals,sar=m.area_last
    assert len(deals)==1 and deals[0]['buyer']=='org' and deals[0]['area']==4.
    assert rows[1]['grain_kg']>0 and rows[1]['surplus']>0 and rows[1]['base_area']==0.
    assert rows[2]['exited'] and rows[2]['grain_kg']==0.
    MainModel._step_area_economy(m)
    rows,_,deals,_=m.area_last
    assert not deals and rows[0]['rent_received']==rows[1]['rent_paid']>0
    assert rows[1]['rented_out']==0. and not rows[2]['exited']


def test_nfps_preperiod_adjustment_proxy():
    """Optional source check; prints aggregates only, never uses post-2010 outcomes."""
    import pandas as pd
    from scripts.prepare_household_portfolio_reference import MANIFEST, RAW

    if not MANIFEST.exists() or not RAW.exists():
        pytest.skip("restricted NFPS source or frozen cohort manifest not present")
    job = json.loads(MANIFEST.read_text(encoding="utf-8"))["jobs"][8]
    payload = Path(job["cohort"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == job["cohort_sha256"]
    cohort = pd.DataFrame(json.loads(payload)["farmers"])
    years = pd.read_csv(RAW, usecols=["year"]).year
    # Preserve source row numbers for the frozen cohort, read only pre-period crop fields.
    raw = pd.read_csv(RAW, usecols=["year", "hhid", "id", "hhnumbers", "sow_food", "sow_cash"],
                      dtype={"hhid": "string", "id": "string"},
                      skiprows=set(years.index[~years.isin([2009, 2010])] + 1))
    raw.index = years.index[years.isin([2009, 2010])]
    origin = raw.loc[cohort.source_row - 2].reset_index(drop=True)
    assert origin.year.eq(2010).all() and origin.hhnumbers.eq(cohort.family_size).all()
    assert len(origin) == origin.hhid.nunique() == 600
    base = raw.loc[raw.year.eq(2010), "hhid"]
    origin = origin.loc[~origin.hhid.isin(base[base.duplicated(keep=False)])]
    prior = raw.loc[raw.year.eq(2009)]
    prior = prior.loc[~prior.hhid.duplicated(keep=False)]
    keys = ["hhid", "id"]
    paired = prior.dropna(subset=keys).merge(origin.dropna(subset=keys), on=keys,
                                              validate="one_to_one", suffixes=("09", "10"))
    shares = []
    for y in ("09", "10"):
        g, c = paired["sow_food" + y], paired["sow_cash" + y]
        valid = np.isfinite(g) & np.isfinite(c) & g.ge(0) & c.ge(0) & (g + c).gt(0)
        shares.append(g / (g + c).where(valid))
    previous, current = shares
    delta = (current - previous).dropna()
    previous = previous.loc[delta.index]
    # Assume movers headed for the endpoint in their observed direction.
    # Stayers contribute zero, including households already at either endpoint.
    closure = delta * 0
    up, down = delta.gt(0), delta.lt(0)
    closure.loc[up] = delta[up] / (1 - previous[up])
    closure.loc[down] = -delta[down] / previous[down]
    print(f"2009->2010: cohort=600 unambiguous_base={len(origin)} matched={len(paired)} "
          f"valid={len(delta)} movers={sum(up | down)} mean_abs_step={delta.abs().mean():.9f} "
          f"endpoint_closure_proxy={closure.mean():.9f}")
    assert len(delta) == 549 and closure.between(0, 1).all()
    assert closure.mean() == pytest.approx(NFPS_HOUSEHOLD_ADJUSTMENT_PROXY, abs=5e-10)


def household(share=.6, *, spec=None, margin=1e7, v1=False):
    farmer = hand_farmer(HAND_CASES[0])
    model = farmer.model
    model.farmers, model.agricultural_enterprises = [farmer], []
    model.config["fiscal_system"] = {"p3_support": (spec or program()).model_dump()}
    model.step_count = 1
    model.config["runtime_fixed_rule"].update(
        household_crop_portfolio=True, minimum_inertia_margin_yuan_per_mu=margin)
    if v1:
        model.config["runtime_fixed_rule"]["national_v1"] = True
        farmer.set_v1_labor(1, False)
    model.household_initial_grain_shares = {} if share is None else {str(farmer.unique_id): share}
    model.government.begin_annual_cycle()
    return farmer


def test_v1_target_to_real_crop_areas_and_next_year():
    f = household(.6, v1=True)
    m = f.model
    m.config["runtime_fixed_rule"]["household_crop_adjustment_rate"] = .5
    m.environment.market_prices["crop_price"] = 12
    m.environment.market_prices["cash_crop"] = .1
    m._run_actor_land_market_planning_stage()
    row = m.v4_8_fixed_rule_trace[-1]
    assert (row["target_grain_share"], row["adjusted_candidate_grain_share"],
            row["selected_grain_share"], row["inertia_yuan"]) == (1, .8, .8, 0)
    assert f.fixed_rule_selected_account.grain_area_mu == pytest.approx(8)
    m._conduct_labor_market()
    m._run_fixed_rule_production_stage()
    assert (f.grain_area, f.cash_crop_area) == pytest.approx((8, 2))
    m.step_count += 1
    m._run_fixed_rule_decisions()
    assert m.v4_8_fixed_rule_trace[-1]["previous_grain_share"] == pytest.approx(.8)


@pytest.mark.parametrize("stage,actual_action", [
    ("no_service", "portfolio_adjusted"), ("step_short_cash", "exit"),
    ("step_low_profit", "exit"), ("exit_best", "exit"),
])
def test_v1_two_stage_choice_against_exit(monkeypatch, stage, actual_action):
    from dataclasses import replace
    f = household(.53, v1=True)
    f.model.config["runtime_fixed_rule"]["household_crop_adjustment_rate"] = .5
    original = f.expected_fixed_rule_account
    def quote(action):
        account = original(action)
        outsourced = f.p3_service_mode == "outsource"
        # High-scoring unavailable service and unfunded grain cannot win the target comparison.
        score = {"expand_cash": 100, "exit": 200 if stage == "exit_best" else 30,
                 "maintain_grain": 1000, "portfolio_adjusted": 20 if stage == "step_low_profit" else 35}.get(action, 10)
        cost = 2 if action == "maintain_grain" or (action == "portfolio_adjusted" and stage == "step_short_cash") else 0
        return replace(account, service_area_mu=0, production_cost_yuan=cost, rent_in_yuan=0,
                       service_cost_yuan=0, management_cost_yuan=0, conversion_cost_yuan=0,
                       working_capital_available=1, hired_labor_days=0, risk_penalty_yuan=0,
                       sales_yuan=score+cost+(2000 if outsourced else 0), rent_out_yuan=0,
                       support_yuan=0, off_farm_yuan=0, common_income_yuan=0)
    monkeypatch.setattr(f, "expected_fixed_rule_account", quote)
    f.model._run_fixed_rule_decisions()
    row = f.model.v4_8_fixed_rule_trace[-1]
    assert row["target_action"] == ("exit" if stage == "exit_best" else "expand_cash")
    assert row["action"] == actual_action
    assert f.p3_service_mode == ("transfer" if actual_action == "exit" else "self")
    if stage != "exit_best":
        assert row["adjusted_candidate_grain_share"] == pytest.approx(.265)


def test_v1_missing_share_does_not_impute_prior_or_enable_inertia():
    f = household(None, v1=True)
    f.model._run_fixed_rule_decisions()
    row = f.model.v4_8_fixed_rule_trace[-1]
    assert row["previous_grain_share"] is None and row["inertia_yuan"] == 0
    assert row["action"] == row["target_action"]
    assert set(f.household_candidate_grain_shares.values()) == {0, 1}


def test_v1_organization_margin_is_separate():
    from types import SimpleNamespace
    rule = {"national_v1": True, "household_crop_adjustment_rate": .2,
            "minimum_inertia_margin_yuan_per_mu": 0}
    org = SimpleNamespace(model=SimpleNamespace(config={"runtime_fixed_rule": rule}),
                          crop_portfolio={}, get_operated_land_area=lambda: 7)
    assert fixed_rule_operator_inertia(org) == 350
    rule["organization_inertia_margin_yuan_per_mu"] = 2
    assert fixed_rule_operator_inertia(org) == 14


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


def test_v1_behavior_audit_uses_paired_runs_and_never_writes_legacy(tmp_path, monkeypatch):
    import json
    import sys
    import pandas as pd
    from scripts import audit_household_portfolio_behavior as audit

    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    source, dest = tmp_path/"legacy", tmp_path/"v1"
    (source/"reference").mkdir(parents=True)
    (source/"PROTOCOL.md").write_text("test protocol")
    frame = pd.DataFrame([(year, "a", 1, 1) for year in (2010, 2011, 2012)],
                         columns=["year", "actor_id", "grain_area", "cash_area"])
    frame.to_csv(source/"reference/reference_households.csv", index=False)
    originals = {p:p.read_bytes() for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(audit, "OUT", source)
    monkeypatch.setattr(audit, "V1_OUT", dest)
    runs = {"old": [], "v1": []}
    for seed in (103, 109):
        for rule in runs:
            run = tmp_path/f"{rule}_{seed}"
            ep = run/"episode"
            ep.mkdir(parents=True)
            runs[rule].append(str(ep))
            write(run/"audit.json", dict(seed=seed, rule=rule, ok=True, baseline=runs["old"][-1]))
            write(run/"design.json", dict(seed=seed, rule=rule, policy="base", inputs={}, share_sha256=None,
                                          runner_sha256=rule, source_hashes={"test": rule}))
            write(ep/"COMPLETED.json", dict(seed=seed, years=3, training=False))
            write(ep/"effective_config.json", dict(scale={"n_enterprises":60},
                                                  runtime_fixed_rule={"national_v1":rule=="v1"}))
            for t in (1, 2):
                year = ep/f"year_{t:02d}"
                year.mkdir()
                write(year/"actual_harvests.json", {"a": {}})
                write(year/"actual_plot_use.json", dict(year=2010+t, reconciliation={"unreconciled_operators":[]},
                    patches=[dict(operator_id="a", allocated_grain_area_mu=1, allocated_economic_crop_area_mu=int(rule=="v1"))]))
                (year/"actor_trace.jsonl").write_text('{"actor_id":"a","alternatives":[]}')

    def check(year):
        monkeypatch.setattr(sys, "argv", ["audit", str(year), "--v1", "--old", *runs["old"], "--new", *runs["v1"]])
        audit.main()

    check(2011)
    assert not json.loads((dest/"behavior_gate.json").read_text())["unlocked"]
    check(2012)
    gate = json.loads((dest/"behavior_gate.json").read_text(encoding="utf-8"))
    assert gate["rule"] == "v1" and gate["unlocked"] and not gate["failed"]
    status = pd.read_csv(dest/"policy_comparison_status.csv")
    assert len(status) == 12 and status.version.eq(gate["version"]).all()
    assert status.loc[status.policy.eq("P0"), "evaluation"].eq("base_behavior_only").all()
    assert status.incremental_grain_t.isna().all()
    assert {p:p.read_bytes() for p in source.rglob("*") if p.is_file()} == originals
    assert (dest/"behavior_2012.png").read_bytes().startswith(b"\x89PNG")

    receipt = tmp_path/"v1_103/audit.json"
    write(receipt, dict(seed=103, rule="new", ok=True, baseline=runs["old"][0]))
    with pytest.raises(ValueError, match="Not an audited"):
        check(2012)
    write(receipt, dict(seed=103, rule="v1", ok=True, baseline=runs["old"][1]))
    with pytest.raises(ValueError, match="Different paired old"):
        check(2012)
    write(receipt, dict(seed=103, rule="v1", ok=True, baseline=runs["old"][0]))
    with (dest/"behavior_2011.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="Development and holdout"):
        check(2012)
