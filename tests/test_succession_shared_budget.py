from copy import deepcopy

import pytest

from core.land_succession import Scenario, Succession
from scripts.run_succession_shared_budget import JointBudget, opposite_supported


@pytest.fixture
def budget():
    return JointBudget(7200,(3600,7200))


def test_joint_reserves_before_either_world_moves(budget):
    assert len(budget.feasible_moves())==13
    regions=(Succession(1,Scenario()),Succession(2,Scenario()))
    original=tuple(deepcopy(x.cash) for x in regions)
    with pytest.raises(ValueError,match="joint cap"):
        budget.advance(regions,(("grain_pre",7200),("takeover_pre",3600)))
    assert all(r.year==0 and r.cash==c for r,c in zip(regions,original))
    with pytest.raises(ValueError,match="inactive"):
        budget.advance(regions,(("none",3600),("none",0)))
    assert [r.year for r in regions]==[0,0]


def test_same_allocation_changes_effective_guarantee_and_never_recycles(budget):
    a,b=Succession(3,Scenario()),Succession(4,Scenario())
    x=budget.advance((a,b),(("grain_pre",3600),("takeover_pre",3600)))
    assert [i["guarantee_mu"] for i in x]==[18.75,18.75]
    y=budget.advance((a,b),(("grain_pre",7200),("none",0)))
    assert [i["guarantee_mu"] for i in y]==[37.5,0]
    assert all(0<=i["spending"]<=i["committed"]<=i["reservation"] for i in (*x,*y))
    assert b.observed["spending"]==0 and a.year==b.year==2
    assert a.scenario.budget==b.scenario.budget==7200


def test_no_support_reproduces_frozen_economy_and_stops_after_pulse(budget):
    s=Scenario(cash=.66,labor=.49)
    a,b=Succession(7,s),Succession(8,s)
    stand=a.fork()
    # Runtime comparison does not treat its local budget metadata as a resource.
    result=budget.advance((a,b),(("none",0),("takeover_pre",7200)))[0]
    expected=stand.step("none")
    assert all(result[key]==expected[key] for key in ("grain_kg","leased_area","owner_closing_cash"))
    assert a.cash==stand.cash and a.contracts==stand.contracts
    stop=budget.advance((a,b),(("none",0),("none",0)))
    assert all(i["reservation"]==i["committed"]==i["spending"]==0 for i in stop)


def test_training_gate_rejects_uncertain_or_same_sign_ordering():
    assert opposite_supported((-9,-2),(1,7))
    assert not opposite_supported((-9,2),(1,7))
    assert not opposite_supported((1,2),(1,7))
