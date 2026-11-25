from dataclasses import replace

import pytest

from core.land_succession import AREA, POLICIES, Plot, Scenario, Succession, choose_crop


def test_cash_surplus_does_not_override_nonfarm_option():
    p=Plot("p","h",1.,"grain")
    assert choose_crop(p,10000,30,0,30)[0] is not None
    assert choose_crop(p,10000,30,400,30)==(None,0.)


@pytest.mark.parametrize("policy",POLICIES+("takeover_post","grain_pre"))
def test_stock_payments_area_and_annual_cash_rollforward(policy):
    w=Succession(301,Scenario(cash=.65,labor=.65))
    old=None
    for _ in range(8):
        a,p,h,t=w.step(policy,ledger=True)
        assert sum(z["physical_area"] for z in p)==w.land
        assert len({z["plot"] for z in p})==len(p)
        assert all(z["crop"]=="idle" or z["operator"] is not None for z in p)
        assert all(z["operator"] in w.operators or z["operator"] is None for z in p)
        assert all(z["planted_area"]<=z["physical_area"] for z in p)
        assert a["grain_area"]==sum(z["planted_area"] for z in p if z["crop"]=="grain")
        assert a["grain_kg"]==pytest.approx(sum(z["harvest"] for z in p if z["crop"]=="grain"))
        assert a["offered_area"]==a["leased_area"]+a["unfilled_area"]
        assert a["spending"]<=w.scenario.budget+1e-8
        assert sum(x["support"] for x in p)==pytest.approx(a["spending"])
        assert all(x["post_support" if policy.endswith("_pre") else "pre_support"]==0 for x in h)
        if policy=="none":
            assert a["spending"]==0.
        if policy.startswith("takeover_"):
            assert all(x["support"]==0 for x in h if x["kind"]=="owner")
        assert sum(z["rent_paid"] for z in h)==pytest.approx(sum(z["rent_income"] for z in h))
        assert sum(z["rent"] for z in t)==pytest.approx(sum(z["rent_paid"] for z in h))
        assert all(z["ask"]<=z["rent"]<=z["bid"]-z["fee"]+1e-8 for z in t if z["buyer"])
        for z in h:
            assert z["closing"]>=0 and z["farm_days"]+z["job_days"]<=z["labor_capacity"]+1e-8
            assert z["opening"]+z["income"]-z["consumption"]==pytest.approx(z["closing"],abs=1e-7)
            if old is not None:
                assert z["opening"]==old[z["id"]]
        old={z["id"]:z["closing"] for z in h}


def test_full_and_zero_search_failure_and_absent_tenants():
    s=Scenario(search_failure=0.,fee=0.)
    active=Succession(301,s).step("none",ledger=True)
    assert active[0]["leased_area"]>0 and all(x["search_open"] for x in active[3])
    closed=Succession(301,replace(s,search_failure=1.)).step("none")
    assert closed["offered_area"]>0 and closed["unfilled_area"]==closed["offered_area"]
    nobody=Succession(301,replace(s,entrants=0)).step("none")
    assert nobody["offered_area"]>0 and nobody["leased_area"]==0.
    assert Succession(301,replace(s,cash=0.)).step("none")["leased_area"]==0.


@pytest.mark.parametrize("condition",["labor_realization","cash_available"])
def test_a_valid_lease_can_fail_to_produce(condition):
    w=Succession(301,replace(Scenario(),**{condition:0.}))
    a,p,_,_=w.step("none",ledger=True)
    assert a["leased_area"]>0 and a["leased_idle"]==a["leased_area"]
    expected="labor" if condition=="labor_realization" else "cash"
    assert {x["failure"] for x in p if x["lease"]}=={expected}


def test_no_release_and_no_trade():
    w=Succession(15,Scenario(owners=1,labor_realization=1.,cash_available=1.))
    owner=w.operators["H00"]
    w.operators[owner.id]=replace(owner,labor=100.,wage=0.)
    w.cash[owner.id]=10000.
    a=w.step("none")
    assert a["offered_area"]==a["leased_area"]==0 and a["idle_area"]==0.


def test_beginnings_repeat_and_annual_state_is_real():
    a,b=(Succession(301,Scenario()) for _ in range(2))
    assert a.step("takeover_pre",ledger=True)==b.step("takeover_pre",ledger=True)
    paid,plots,actors,_=a.step("takeover_pre",ledger=True)
    assert paid["grain_area"]+paid["cash_crop_area"]+paid["idle_area"]==AREA*len(plots)
    assert any(x["opening"]!=b.operators[x["id"]].cash for x in actors)
    static=Succession(301,Scenario(feedback=False))
    static.step("none")
    _,_,a,_=static.step("none",ledger=True)
    assert all(x["opening"]==static.operators[x["id"]].cash for x in a)


def test_policy_reads_previous_observation_and_evaluation_does_not_learn():
    from scripts.run_land_succession import observation, simulate
    s=Scenario()
    world=Succession(301,s)
    initial=observation(world)
    q={initial:(0.,0.,2.)}
    before=q.copy()
    rows=simulate(301,s,"q",q,lambda *args:None,"heldout","test")
    assert q==before and rows[0]["previous_state"]==str(initial) and rows[0]["policy"]=="takeover_pre"
    for i,row in enumerate(rows):
        assert row["previous_state"]==str(observation(world))
        assert row["year"]==i+1
        world.step(row["policy"])
