import pytest

from core.land_succession import Operator, Plot, Scenario, Succession
from core.regional_succession_budget import OFF,RegionalBudget,fork_menu,public_state
from core.succession_allocation import CropOffer,variable_allocation
from scripts.prepare_succession_budget_inputs import split_lots
from scripts.run_succession_empirical_budget import settings,choose


def market_world(seed=8):
    return Succession.from_assets(seed,Scenario(search_failure=0.,cash_available=1.,
        labor_realization=1.,harvest=1.,contract_years=2,start_cost=500.),(
            Operator("owner",False,0.,0.,0.,0.,11),
            Operator("buyer",False,25000.,10.,0.,0.,11),
            Operator("neighbor",False,250000.,200.,0.,0.,12),
        ),(Plot("p","owner",1.,"grain",2.7),Plot("u","owner",1.,"grain",2.1,False)),2000.)


def test_area_rights_conserved_through_initial_split():
    p = split_lots("hh",23.0,12.3,23.,4.,30.)
    assert sum(x.area for x in p)==pytest.approx(23.)
    assert sum(x.area for x in p if x.tradable)==pytest.approx(12.3)
    assert sum(x.area for x in p if x.initial_crop=="grain")==pytest.approx(23.*23./30.)
    assert max(x.area for x in p)<=4
    assert len({x.id for x in p})==len(p)


def test_markets_only_match_province_and_verified_area_and_carry_contract():
    w = market_world()
    row,fields,actors,trades = w.step("grain_pre",ledger=True)
    assert row["matched_area"]==pytest.approx(2.7)
    assert row["grain_kg"]==pytest.approx(2.7*450.)
    signed = [r for r in trades if r["matched"]]
    assert len(signed)==1 and signed[0]["buyer"]=="buyer"
    assert signed[0]["ask"]<=signed[0]["rent"]<=signed[0]["bid"]
    assert next(r for r in fields if r["plot"]=="u")["failure"]=="restricted_right"
    assert sum(r["rent_paid"] for r in actors)==pytest.approx(sum(r["rent_income"] for r in actors))
    assert row["committed"]<=w.scenario.budget
    for a in actors:
        assert a["closing"]==pytest.approx(a["opening"]+a["income"]-a["consumption"])
    second = w.step("none")
    assert second["held_area"]==pytest.approx(2.7) and not second["matched_area"]
    assert second["spending"]==0.


def test_cash_and_labor_limit_same_variable_plot():
    offer = [CropOffer("p",1.,grant=120.,credit=120.,area=2.)]
    assert variable_allocation(offer,1120.,5.,0.,0.).crops=={"p":"grain"}
    assert variable_allocation(offer,1119.,5.,0.,0.).crops=={}
    assert variable_allocation(offer,1120.,4.99,0.,0.).crops=={}


def test_same_budget_and_branch_matches_direct_worlds():
    d = settings()[0]
    roots = tuple(market_world(i+1) for i in range(4))
    b = RegionalBudget(361,4,tuple(d["policies"][1:]))
    menu = b.menu()
    assert len(set(menu))==21 and sum(a[1] for a in menu[-1])==361
    with pytest.raises(ValueError,match="feasible"):
        b.step(roots,(("grain_pre",361),)*4)
    branches,next_worlds = fork_menu(roots,d,b,menu)
    assert all(state.year==0 for state in roots)
    for i,a in enumerate(menu):
        w = tuple(r.fork() for r in roots)
        actual = b.step(w,a)
        assert branches[i][0]==pytest.approx(sum(r["grain_kg"] for r in actual))
        assert public_state(w,d,b)==branches[i][1]
        assert [r.cash for r in w]==[r.cash for r in next_worlds[i]]


def test_year_eight_dynamic_and_stop_is_separate():
    frozen = {"table":{"1:0:0:0:0:361":{"supported":True,"q":3,"myopic":11}}}
    k = "1:0:0:0:0:361"
    assert choose("q",k,frozen,False)==(3,"supported")
    assert choose("myopic",k,frozen,False)==(11,"supported")
    assert choose("q",k,frozen,True)==(0,"forced_stop")
