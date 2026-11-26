from dataclasses import replace

import pytest

from core.land_succession import Operator, Plot, Scenario, Succession
from core.regional_succession_budget import RegionalBudget
from core.succession_allocation import CropOffer
from core.succession_economic_core import Activity, EconomicCore, LEGACY_ACTIVITIES, gain_bound, solve
from scripts.run_succession_economic_core import audit


def build(actors, plots, *, economics=LEGACY_ACTIVITIES, **kwargs):
    s = Scenario(owners=0,entrants=0,search_failure=0.,cash_available=1.,
                 labor_realization=1.,harvest=1.,economics=economics,**kwargs)
    return Succession.from_assets(7101,s,tuple(actors),tuple(plots),0.)


def actor(key, cash, labor, market=1, wage=0, job=0):
    return Operator(key,False,cash,labor,wage,job,market,consumption=0.)


def grain():
    return EconomicCore((Activity("wheat",True,450,2.5,620,2.5),))


def test_fixed_cost_spreads_over_viable_package():
    actors = [actor("a",0,0),actor("b",0,0),actor("t",9000,100)]
    fields = [Plot("pa","a",1,"grain",2.5),Plot("pb","b",1,"grain",2.5)]
    old = build(actors,fields,economics=None).step("none")
    assert old["matched_area"] == 0
    candidate = build(actors,fields,economics=grain(),contract_years=2)
    first = candidate.step("none",ledger=True)
    audit(candidate,*first[:3])
    assert first[0]["matched_area"] == 5
    t = next(a for a in first[2] if a["id"] == "t")
    assert (t["rent_paid"],t["fees"],t["setup"],t["profit"]) == pytest.approx((650,560,1000,315))
    nxt = candidate.step("none",ledger=True)
    audit(candidate,*nxt[:3])
    t2 = next(a for a in nxt[2] if a["id"] == "t")
    assert (nxt[0]["held_area"],t2["fees"],t2["setup"]) == (5,0,1000)


def test_unmatched_release_returns_to_self_production():
    people = [actor("a",10000,40,wage=175,job=100)]
    lots = [Plot("p","a",1,"grain",4)]
    old = build(people,lots,economics=None).step("none")
    assert old["grain_area"] == 0
    w = build(people,lots,economics=grain())
    row, f, a, _ = w.step("none",ledger=True)
    audit(w,row,f,a)
    assert (row["matched_area"],row["grain_area"],row["rent_total"]) == (0,4,0)
    assert f[0]["posted"] and f[0]["recovered"] and a[0]["recovery_cost"] == 0
    assert a[0]["sales"]-a[0]["inputs"]-a[0]["foregone_job_income"] > 0


def test_contract_term_matters_without_counting_future_policy():
    actors = [actor("a",0,0),actor("t",9000,100)]
    lots = [Plot("p","a",1,"grain",4)]
    tech = EconomicCore((Activity("wheat",True,450,2.5,700,2.5),))
    one = build(actors,lots,economics=tech)
    two = build(actors,lots,economics=tech,contract_years=2)
    assert one.step("none")["matched_area"] == 0
    first = two.step("none",ledger=True)
    audit(two,*first[:3])
    assert first[0]["matched_area"] == 4
    assert next(a for a in first[2] if a["id"]=="t")["profit"] == pytest.approx(-100)
    nxt = two.step("none",ledger=True)
    assert next(a for a in nxt[2] if a["id"]=="t")["profit"] == pytest.approx(180)


def test_market_never_offers_restricted_right_or_crosses_province():
    people = [actor("a",0,0,market=1),actor("b",10000,50,market=2)]
    w = build(people,[Plot("p1","a",1,"grain",4),Plot("p2","a",1,"grain",4,False)],
              start_cost=0,fee=0)
    row, f, a, _ = w.step("none",ledger=True)
    audit(w,row,f,a)
    assert row["matched_area"] == 0 and not f[1]["posted"]


def test_joint_cash_labor_and_hire_uses_actual_jobs_and_expense():
    econ = EconomicCore((Activity("wheat",True,100,2,80,2,.5),),30.,4.)
    w = build([actor("a",420,8,wage=50,job=8)],
              [Plot("p","a",1,"grain",4,False)],economics=econ)
    row,f,a,_ = w.step("none",ledger=True)
    audit(w,row,f,a)
    assert a[0]["hired_days"]>0 and a[0]["job_days"]>0
    assert a[0]["sales"] == pytest.approx(a[0]["retained_value"])
    assert a[0]["hired_cost"] == pytest.approx(30*a[0]["hired_days"])


def test_mixed_plan_feasible_and_uses_only_physical_land():
    a = Activity("wheat",True,100,3,100,1)
    b = Activity("vegetables",False,100,5,200,4)
    p = CropOffer("one",1.,area=4.)
    plan = solve(EconomicCore((a,b)),[p],500,6,0,0)
    assert dict((c.name,n) for c,n in plan.area["one"]) == pytest.approx({"wheat":10/3,"vegetables":2/3})
    assert plan.days == pytest.approx(6)
    assert plan.cash <= 500


def test_shadow_pruning_is_an_upper_bound_on_exact_increment():
    econ = LEGACY_ACTIVITIES
    p = [CropOffer("own",.93,area=4.)]
    new = [CropOffer("new",1.12,grant=80,credit=80,area=2.)]
    for cash,days in ((400,10),(900,8),(3000,4),(3000,20),(10000,50)):
        base = solve(econ,p,cash,days,15,5)
        trial = solve(econ,p+new,cash-200,days,15,5)
        assert trial.value-base.value-200 <= gain_bound(base,econ.activities,new,200)+1e-7


def test_cash_timing_annual_budget_and_crop_payment():
    a = actor("a",50,40)
    p = Plot("p","a",1,"grain",4,False)
    econ = EconomicCore((Activity("wheat",True,100,3,100,1),))
    w = build([a],[p],economics=econ)
    post = w.fork()
    fiscal = RegionalBudget(200,1,("grain_pre","grain_post"))
    pre_row = fiscal.step((w,),(("grain_pre",200),))[0]
    post_row = fiscal.step((post,),(("grain_post",200),))[0]
    assert (pre_row["grain_area"],post_row["grain_area"]) == pytest.approx((1,.5))
    assert (pre_row["spending"],post_row["spending"]) == pytest.approx((50,25))
    assert pre_row["rate_mu"] == post_row["rate_mu"] == 50
    w.scenario = replace(w.scenario,budget=0)
    assert w.step("none")["spending"] == 0


def test_identical_fork_retains_old_grain_area_for_bounds():
    w = build([actor("a",400,40)],[Plot("p","a",1,"grain",4,False)],economics=grain())
    first = w.step("none")
    branch = w.fork()
    assert first["retained_grain"] == pytest.approx(first["grain_area"])
    assert w.step("none") == branch.step("none")
