import numpy as np
import pandas as pd

from core.fixed_rule_runtime import continuous_crop_target
from scripts.audit_crop_time_comparability import classify,market_evidence


def test_fractional_history_ties_and_exit_are_not_cash_crop():
    history={'mixed':(np.array([.1,.1,.2,.8,.9,.9]),np.array([.07,.11,np.nan,1.,.88,.94]))}
    assert continuous_crop_target(history,'mixed_lower',.1,0)==.07
    assert continuous_crop_target(history,'mixed_lower',.1,.99)==.11
    assert continuous_crop_target(history,'mixed_upper',.9,.4)==.94
    assert continuous_crop_target(history,'unknown',None,.3) is None


def test_annual_crop_and_land_missing_are_not_zero():
    g,c,t=map(pd.Series,[[0.,0.,0.,np.nan],[0.,0.,1.,1.],[2.,0.,1.,2.]])
    assert list(classify(g,c,t))==['other','none','cash','unknown']
    data=pd.DataFrame(dict(area_b=[5,5,5],area_e=[0,0,1],area_in=[0,0,0],area_de=[5,5,4],area_de2=[np.nan,5,4]))
    known,full=market_evidence(data)
    assert known.tolist()==[False,True,True] and full.tolist()==[False,True,False]
