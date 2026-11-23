from types import SimpleNamespace
import pytest
from core.fixed_rule_runtime import continuous_farm_plan
from core.p3_project_claims import expected_support
from tests.test_p3_project_payments_20260909 import program,prepared


@pytest.mark.parametrize('target',[0.,.4,1.])
def test_soft_history_direction_and_physical_constraints(target):
    a=SimpleNamespace(sowing_intensity=1.,area_limit=10.,family_days=100.,crop_mix_target=target,crop_adjustment_yuan=10.)
    def plan(g=200.,c=200.,w=5.,land=10.,cash=5000.,rent=0.,trade=False,aid=0.):
        return continuous_farm_plan(a,land,land,cash,rent,(g,c,50.,50.,1.,4.,w,aid),trade)
    neutral=plan()
    assert neutral['grain']==pytest.approx(10*target)
    assert plan(g=225)['grain']==pytest.approx(10) and plan(c=225)['grain']==0
    # A verified, fully funded ordinary grain-area payment enters expected returns, not planting cash.
    spec=program(budget=20000.)
    farmer=prepared(spec)
    incentive=expected_support(farmer,1.,spec)-expected_support(farmer,0.,spec)
    assert incentive>10 and plan(g=200+incentive)['grain']==pytest.approx(10)
    assert plan(aid=83.)['grain']==pytest.approx(neutral['grain'])
    assert plan(rent=300,trade=True)['offer']==pytest.approx(10)
    assert plan(rent=1,trade=True)['offer']==0
    assert plan(cash=100)['cultivated']==pytest.approx(2)
    assert neutral['cultivated']==10
    a.family_days=0.
    assert plan(c=240,w=1)['crop']==10 and plan(c=240,w=30)['grain']==10
    constrained=plan(cash=100,w=50)['cultivated']
    a.family_days=100.
    assert plan(cash=100,w=50)['cultivated']>constrained


def test_v2_and_v3_budget_and_non_cash_cost():
    a=SimpleNamespace(sowing_intensity=2.,area_limit=10.,family_days=200.,crop_mix_target=.4)
    args=(200.,400.,20.,20.,1.,1.,10.,83.)
    old=continuous_farm_plan(a,10.,6.,6000.,0.,args,False)
    a.crop_adjustment_yuan=50.
    new=continuous_farm_plan(a,10.,6.,6000.,0.,args,False)
    assert old['grain']==pytest.approx(8) and new['crop']==pytest.approx(20)
    assert new['eligible']==6 and new['cost']==400
    assert new['expected']==pytest.approx(20*400-400+6*83-50*8)
