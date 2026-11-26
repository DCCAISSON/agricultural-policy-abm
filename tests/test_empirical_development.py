import numpy as np
import pandas as pd

from core.v4_household import farm_batch
from scripts.run_empirical_succession import development_intensity
from tests.test_v4_area_economy import fixture


def test_pure_endpoint_can_rescue_an_unprofitable_drawn_crop_change():
    state, fit, u = fixture()
    state.update(prior=[0., 0.], inputs=[[160.,1.,20.,100.,2.,6.,10.,0.]]*2,
                 funds=[300.,0.],labor=[100.,100.])
    fit["endpoint"][:,1]=.2
    fit["amplitudes"][:]=.1
    u[:,2]=.4
    flags={"economic_participation":True,"joint_choice":True,"scores":False}
    old=farm_batch(state,fit,u,stage=4,diagnostic=flags)
    new=farm_batch(state,fit,u,stage=4,diagnostic=flags|{"pure_endpoint_fallback":True})
    assert old["grain"][0]==0
    assert new["grain"][0]>0 and not new["crop"].any()
    assert new["grain"][1]==0 and new["cost"][1]==0
    assert (new["cost"]<=state["funds"]).all()


def test_intensity_uses_early_noncohort_donors_and_predictor_only_on_opening_land():
    valid=pd.DataFrame(dict(year=[2010]*30,hhid=[str(i) for i in range(30)],smgen=[11]*30,
        area_b=[4.]*30,area_e=[2.]*30,sow_food=[2.]*30,sow_cash=[1.]*30))
    contaminated=valid.assign(year=2015,sow_food=200.)
    fixture=pd.concat([valid,contaminated],ignore_index=True)
    profile, receipt=development_intensity(fixture,{"1"},{11:"r"})
    assert profile["r",0]==1.5 and receipt["last_year"].max()==2010
    assert receipt.loc[receipt["scope"].eq("pooled"),"n"].item()==29
