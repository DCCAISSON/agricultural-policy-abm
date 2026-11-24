from types import SimpleNamespace

import numpy as np
import pytest

from agents.government_agent import GovernmentAgent
from core.v4_household import farm_batch, fit_categories, uniforms
from modules.land_clearing_house import LandClearingHouse
from modules.spatial_index import area_moore_neighbors


def fixture():
    fit=dict(regions=['R'],prior_median=.5,amplitudes=np.full((12,2,19),.4),
             endpoint=np.zeros((12,2)),return_scale=10.)
    for k,n in [('direction',3),('stay',2),('rent_in',2),('rent_out',2)]:
        fit[k]=np.zeros((n,8))
    state=dict(land=[10.,10.],rights=[10.,10.],funds=[300.,500.],rent=[150.,150.],
        intensity=[1.,1.],labor=[16.,40.],workers=[2.,2.],transfer=[0.,0.],
        prior=[.6,.5],region=['R','R'],inputs=[[120.,110.,20.,30.,2.,6.,10.,83.]]*2,limit=[20.,20.])
    u=np.array([[.03,.8,.4,.2,.2],[.03,.4,.4,.2,.2]])
    return state,fit,u


def test_vectorized_households_equal_scalar_reference_and_balance_resources():
    state,fit,u=fixture()
    a=farm_batch(state,fit,u,stage=2)
    assert np.all(a['cost']<=state['funds'])
    assert np.all(a['family']<=state['labor']) and np.all(a['cultivated']<=state['land'])
    assert np.allclose(a['family']+a['hired'],2*a['grain']+6*a['crop'])
    for i in range(2):
        s={k:v[i:i+1] for k,v in state.items()}
        b=farm_batch(s,fit,u[i:i+1],stage=2)
        for k in a:
            assert np.allclose(a[k][i],b[k][0]),k
    assert np.array_equal(uniforms(17,2011,['b','a'])[::-1],uniforms(17,2011,['a','b']))


@pytest.mark.parametrize('prior',[0.,.4,1.])
def test_price_aid_and_budget_directions(prior):
    s,fit,u=fixture()
    s['prior']=[prior,prior];s['funds']=[1000.,1000.];s['labor']=[500.,500.]
    a=farm_batch(s,fit,u,stage=1)
    b=farm_batch(s,fit,u,stage=1,grain_aid=20.)
    assert np.all(b['probabilities'][:,2]>=a['probabilities'][:,2]-1e-9)
    s['inputs']=[[120,160,20,30,2,6,10,83.]]*2
    c=farm_batch(s,fit,u,stage=1)
    assert np.all(c['probabilities'][:,2]<=a['probabilities'][:,2]+1e-9)
    s['funds']=[25.,25.]
    d=farm_batch(s,fit,u,stage=1)
    assert np.all((d['cultivated']<10)&(d['cost']<=25.+1e-8))


def test_market_sparse_grid_and_fiscal_actions():
    farmers=[SimpleNamespace(unique_id=str(i),zone=4 if i<10 else 5,area_kind='household',
                              base_area=2.,lease_right=2.) for i in range(12)]
    w=area_moore_neighbors(farmers)
    assert w['neighbors'].diagonal().sum()==0 and w['neighbors'][:10,10:].nnz==0
    ij=w['neighbors'].nonzero()
    i,j=map(int,(ij[0][0],ij[1][0]))
    seller,buyer=w['ids'][i],w['ids'][j]
    offers=[dict(id=seller,side='sell',price=60.,quantity=2,region=4),
            dict(id=buyer,side='buy',price=140.,quantity=1,region=4)]
    deal=LandClearingHouse.clear_area_orders(offers,2011,1,space=w)[0]
    assert deal['slice_start']==0 and deal['slice_end']==1 and deal['rent']==100.
    g=GovernmentAgent.__new__(GovernmentAgent)
    prev={4:{'grain_gap':3.,'leased_mu':0.},5:{'grain_gap':1.,'leased_mu':1.}}
    basis=dict(rights_cultivated=10.,grain_sown=79.,leased_in=4.)
    expected=[10.,79.,0.,4.,10.,10.]
    for action,val in zip(g.AREA_ACTIONS,expected):
        dec=g.announce_area_support(2012,prev,100.,action)
        assert dec['state']==prev and dec['training'] is False
        assert g.area_support_basis(action,**basis)==val
        p=g.settle_area_support(dec,{'x':500.,'y':500.},{'x':4,'y':5})
        assert sum(p.values())==pytest.approx(100.)
    assert g.area_support_basis('scenario_scale_grain',**(basis|{'grain_sown':80.}))==80.


def test_clustered_fit_is_reproducible():
    x=np.column_stack((np.ones(18),np.linspace(0,1,18)))
    y=np.tile([0,1,2],6)
    c,se=fit_categories(x,y,np.repeat(np.arange(6),3),3)
    assert c.shape==se.shape==(3,2) and np.isfinite(c).all() and np.isfinite(se).all()


def test_tabular_update_smoke_on_synthetic_regional_transition():
    g=GovernmentAgent.__new__(GovernmentAgent)
    g.last_state=(1,0,0,1)
    g.last_action=0
    g.learning_rate=.2
    g.discount_factor=.9
    g.q_table_frozen=False
    g.q_table={g.last_state:[0.]*len(g.AREA_ACTIONS)}
    new_state=(0,1,1,0)
    g._update_q_table(5.,new_state)
    assert g.q_table[g.last_state][0]==1.
    assert len(g.q_table[new_state])==len(g.AREA_ACTIONS)
    g.q_table_frozen=True
    g._update_q_table(6.,new_state)
    assert g.q_table[g.last_state][0]==1.
