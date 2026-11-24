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


def test_repair_uses_entry_draw_and_lagged_land_risk_without_cash_covariate():
    from core.v4_household import repair_features
    state,fit,u=fixture()
    state['prior']=[np.nan,np.nan]
    state['had_operating_land']=[True,False]
    state['funds']=[500.,500.]
    fit.update(entry_shares={'all':np.array([0.,1.])},direction_by_prior=np.zeros((3,3,8)),
               stay_operator=np.zeros((2,8)),enter_landless=np.zeros((2,8)))
    fit['stay_operator'][1,0]=9.
    fit['enter_landless'][1,0]=-2.
    u[:,0]=.01
    u[:,2]=[.1,.9]
    a=farm_batch(state,fit,u,stage=4,incentive=0.)
    assert a['target'].tolist()==[0.,1.]
    assert a['grain'][0]==a['crop'][1]==0.
    assert a['crop'][0]>0 and a['grain'][1]>0
    assert a['stay_probability'][0]>.99 and a['stay_probability'][1]<.2
    state['funds']=[10.,10.]
    b=farm_batch(state,fit,u,stage=4,incentive=0.)
    np.testing.assert_array_equal(a['stay_probability'],b['stay_probability'])
    assert (b['cultivated']<a['cultivated']).all()
    x=repair_features(np.array([0.,np.nan]),np.array([1.,1.]),np.array([1.,1.]),
                      np.array([0.,0.]),np.array(['R','R']),['R'],0)
    assert (x[0,:-1]==x[1,:-1]).all() and x[:, -1].tolist()==[1.,0.]


def test_ablation_records_positive_plan_rejected_by_stay_random_draw():
    s,f,u=fixture()
    f['stay'][1,0]=-10.
    a=farm_batch(s,f,u,stage=1,diagnostic={'stay':True,'trace':True},incentive=0.)
    assert (a['stay_u']>a['stay_probability']).all()
    assert (a['plan_sown']>0).all() and not a['cultivated'].any()


def test_scale_shapley_closes_and_reverses_with_single_sided_cultivation():
    from scripts.audit_household_portfolio_behavior import area_shapley
    a=np.array([[8.,10.,2.],[0.,5.,2.],[0.,5.,2.]])
    b=np.array([[10.,8.,0.],[5.,8.,0.],[5.,8.,0.]])
    d=area_shapley(a,b)
    np.testing.assert_allclose(d.sum(axis=0),[5.,3.,-2.])
    np.testing.assert_allclose(d,-area_shapley(b,a))
    assert not d[2:,[0,2]].any()


def test_scale_pool_matches_only_household_residuals_at_regional_rent():
    actors=[SimpleNamespace(unique_id=k,zone=4,area_kind='household',
                             base_area=5.,lease_right=5.) for k in ('s','b','x','y')]
    space=area_moore_neighbors(actors)
    space['search']=False
    orders=[dict(id=k,side=side,quantity=q,price=p,region=4) for k,side,q,p in
            [('s','sell',5,60),('b','buy',1,140),('x','buy',1,90),('y','buy',4,50)]]
    old=LandClearingHouse.clear_area_orders(orders,2015,1,space=space)
    pool={4:dict(supply=3.,demand=2.,reference_rent=88.)}
    d=LandClearingHouse.clear_area_orders(orders,2015,1,space=space,external=pool,household_ids={'s','b','y'})
    assert d[:len(old)]==old
    assert len(d)==len(old)+1
    outside=d[-1]
    assert (outside['channel'],outside['area'],outside['rent'])==('external_demand',2.,old[0]['rent'])
    assert outside['slice_start']==sum(row['area'] for row in old)
    orders=[dict(id=k,side='buy',quantity=2,price=130,region=4) for k in ('b','org')]
    d=LandClearingHouse.clear_area_orders(orders,2015,1,space=space,external=pool,household_ids={'b'})
    assert len(d)==1 and d[0]['buyer']=='b' and d[0]['plot'] is None and d[0]['rent']==88.


def test_scale_participation_retains_zero_area_and_resource_constraints():
    s,f,u=fixture()
    f['stay'][1,0]=-10.
    s['funds']=[300.,0.]
    u[:,0]=.99
    flags={'stay':True,'trace':True}
    a=farm_batch(s,f,u,stage=1,diagnostic=flags,incentive=0.)
    b=farm_batch(s,f,u,stage=1,diagnostic=flags|{'economic_participation':True},incentive=0.)
    assert not a['cultivated'].any() and b['cultivated'][0]>0 and b['cultivated'][1]==0
    np.testing.assert_array_equal(a['stay_probability'],b['stay_probability'])
    assert (b['cost']<=s['funds']).all()
