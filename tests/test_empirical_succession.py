from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from core.empirical_succession import SyntheticLandLedger


def baseline():
    return [dict(actor="A",area_e=4.,verified_right=3.,sow_food=6.,sow_cash=2.),
            dict(actor="B",area_e=2.,verified_right=2.,sow_food=2.,sow_cash=0.)]


def account(actor,hold,cultivate,grain,buy,sell,paid,received):
    return dict(id=actor,region=1,operated=hold,cultivated=cultivate,
        grain=grain,crop=0.,grain_kg=grain*430,rented_in=buy,rented_out=sell,
        rent_paid=paid,rent_received=received,opening=500.,net=0.,consumption=0.,closing=500.,
        preplant_funds=100.,input_hire_cost=100. if cultivate else 0.,claim=0.,paid=0.,offer=sell)


def contract(buyer="B",year=2011,area=3.):
    return dict(plot="A",seller="A",buyer=buyer,region=1,area=area,ask=0.,bid=50.,rent=20.,
                start=year,end=year,slice_start=0.,slice_end=area)


def test_synthetic_strict_change_and_renewal_are_separate():
    ledger=SyntheticLandLedger(baseline())
    rows=[account("A",1.,0.,0.,0.,3.,0.,60.),account("B",5.,3.,4.,3.,0.,60.,0.)]
    pieces, year =ledger.advance(2011,rows,[contract()],[],0.)
    assert year["strict_synthetic_succession_mu"]==pytest.approx(3.)
    assert year["fixed_initial_grain_mu"]==pytest.approx(5.)
    assert sum(x["upper_mu"]-x["lower_mu"] for x in pieces)==pytest.approx(6.)
    assert any(x["synthetic_right_holder"] is None and x["unit"]=="A" for x in pieces)
    _, renewal=ledger.advance(2012,rows,[contract(year=2012)],[],0.)
    assert renewal["strict_synthetic_succession_mu"]==0.
    assert renewal["retained_grain_mu"]==3.


def test_unknown_external_supply_not_mislabeled_as_strict_succession():
    ledger=SyntheticLandLedger(baseline())
    rows=[account("A",4.,4.,4.,0.,0.,0.,0.),account("B",3.,3.,3.,1.,0.,20.,0.)]
    lease=contract(area=1.)|dict(seller="_area_pool_1",plot=None,channel="external_supply")
    outside=[dict(region=1,opening_area=1.,operated=0.,rented_in=0.,rented_out=1.,rent_received=20.,rent_paid=0.)]
    records,out=ledger.advance(2011,rows,[lease],outside,0.)
    assert out["strict_synthetic_succession_mu"]==0.
    assert out["unidentified_origin_or_use_mu"]==1.
    assert any(r["unit"].startswith("external:") and r["previous_assignment"] is None for r in records)


@pytest.mark.parametrize(("wrong","reason"),[
    ("rights","transferable rights"),("region","regional boundary"),
    ("book","rent_paid"),("fiscal","Fiscal budget"),("cash","cash"),
])
def test_same_ledger_refuses_unbalanced_worlds(wrong,reason):
    ledger=SyntheticLandLedger(baseline())
    rows=[account("A",1.,0.,0.,0.,3.,0.,60.),account("B",5.,3.,4.,3.,0.,60.,0.)]
    lease=deepcopy(contract())
    if wrong=="rights": lease=contract(area=3.5)
    if wrong=="region": lease["region"]=2
    if wrong=="book": rows[1]["rent_paid"]=0.
    if wrong=="fiscal": rows[1].update(claim=50.,paid=50.)
    if wrong=="cash": rows[1]["closing"]=501.
    with pytest.raises(ValueError,match=reason):
        ledger.advance(2011,rows,[lease],[],0.)


def test_contract_failure_is_not_encoded_as_no_match():
    ledger=SyntheticLandLedger(baseline())
    rows=[account("A",1.,0.,0.,0.,3.,0.,60.),account("B",5.,0.,0.,3.,0.,60.,0.)]
    _, out=ledger.advance(2011,rows,[contract()],[],0.)
    assert out["contracted_mu"]==out["contracted_not_sown_mu"]==3.
    assert out["owner_offered_unmatched_mu"]==0.


def test_replay_uses_archived_fit_only_on_identical_training_groups():
    from scripts.run_empirical_succession import use_frozen_fit

    def row(rule, action, coefficient):
        return dict(training_end=2012,rule=rule,action=action,term="intercept",coefficient=coefficient,
                    n=100,households=80,action_n=40)

    reference=pd.DataFrame([row("direction_grain","decrease_grain",3.5),
                            row("rent_in","positive",-.5)])
    refit=reference.assign(coefficient=0.)
    state={2012:{"direction_by_prior":np.zeros((3,3,1)),"rent_in":np.zeros((2,1))}}
    use_frozen_fit(state,refit,reference)
    assert state[2012]["direction_by_prior"][2,0,0]==3.5
    assert state[2012]["rent_in"][1,0]==-.5
    with pytest.raises(AssertionError):
        use_frozen_fit(state,refit.assign(households=75),reference)
