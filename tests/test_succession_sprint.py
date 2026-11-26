import pytest

from core.land_succession import Operator,Plot,Scenario,Succession
from core.succession_allocation import CropOffer,refine_mixed_plan,variable_allocation
from scripts.diagnose_succession_sprint import exact_screen
from scripts.run_succession_empirical_budget import settings
from scripts.succession_sprint_support import select,support


def test_local_trial_only_adopts_strict_profitable_and_cash_feasible_mix():
    lots=[CropOffer(str(i),1.,area=2.) for i in range(5)]
    args=(6840.,100.,0.,0.)
    base=variable_allocation(lots,*args)
    after=refine_mixed_plan(lots,base,*args)
    assert set(base.crops.values())=={"grain"}
    assert set(after.crops.values())=={"grain","cash"}
    assert after.value==pytest.approx(base.value+260.)
    assert after.cash==pytest.approx(6840.)
    restricted=(6200.,100.,0.,0.)
    base=variable_allocation(lots,*restricted)
    assert refine_mixed_plan(lots,base,*restricted)==base


def test_exact_screen_matches_small_enumeration_and_keeps_labor_shortage():
    lots=[CropOffer("a",1.,area=2.),CropOffer("b",1.,area=1.5)]
    args=(2800.,13.,30.,3.)
    base=variable_allocation(lots,*args)
    fields=[dict(physical_area=x.area,quality=x.quality) for x in lots]
    label,value,_=exact_screen(fields,*args)
    assert label=="mixed_feasible" and value==pytest.approx(base.value)
    assert exact_screen(fields,2800.,5.,30.,3.)[0]=="labor"


def test_parent_count_not_action_count_for_support():
    d=settings()[0]
    old=(8,0,0,0,0,98953)
    sparse=(8,0,1,0,0,98953)
    rows=[(x,1,old,0) for x in range(16)]+[(x,1,sparse,0) for x in range(16,19)]
    picked=select(rows)
    assert len(picked)==13
    a=support(rows,picked,21,d)
    assert a["common_state_actions"]==42 and a["supported_common_state_actions"]==21
    all_rows=rows+[(19,1,sparse,0),(20,1,sparse,0)]
    after=select(all_rows,picked)
    assert picked.issubset(after) and support(all_rows,after,21,d)["coverage"]==1.


def test_only_preplant_support_relaxes_cash_before_production():
    actor=Operator("a",False,1100.,30.,0.,0.,11)
    lot=Plot("p","a",1.,"grain",2.)
    s=Scenario(search_failure=0.,cash_available=1.,labor_realization=1.,harvest=1.,budget=170.)
    pre=Succession.from_assets(2,s,(actor,),(lot,),900.)
    post=pre.fork()
    pres,_,people,_=pre.step("grain_pre",ledger=True)
    posts,_,people2,_=post.step("grain_post",ledger=True)
    assert pres["grain_kg"]==900. and people[0]["pre_support"]==170.
    assert posts["grain_kg"]==0. and people2[0]["pre_support"]==0.
