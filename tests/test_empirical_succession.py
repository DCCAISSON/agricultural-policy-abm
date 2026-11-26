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
    assert (year["strict_succession_lower_mu"], year["strict_succession_upper_mu"]) == (0., 3.)
    assert year["fixed_initial_grain_mu"]==pytest.approx(5.)
    assert sum(x["upper_mu"]-x["lower_mu"] for x in pieces)==pytest.approx(6.)
    assert any(x["synthetic_right_holder"] is None and x["unit"]=="A" for x in pieces)
    _, renewal=ledger.advance(2012,rows,[contract(year=2012)],[],0.)
    assert renewal["strict_synthetic_succession_mu"]==0.
    assert renewal["retained_grain_mu"]==3.
    assert (renewal["strict_succession_lower_mu"], renewal["strict_succession_upper_mu"]) == (0., 0.)


@pytest.mark.parametrize(("first_cultivated", "grain", "expected_lower"), [(4., 4., 0.), (6., 12., 2.)])
def test_renumbering_preserves_succession_bounds_in_both_years(first_cultivated, grain, expected_lower):
    def replay(owner):
        ledger = SyntheticLandLedger([
            dict(actor=k, area_e=4., verified_right=4., sow_food=4., sow_cash=0.)
            for k in ((owner, "B") if owner == "A" else ("B", owner))
        ])
        lease = contract(area=4.) | dict(plot=owner, seller=owner)
        # The 4-mu case reproduces the attachment exactly; the 6-mu case uses
        # two sowing rounds and a land base too small to avoid some succession.
        buyer = account("B", 8., first_cultivated, grain, 4., 0., 80., 0.)
        buyer.update(net=30., consumption=10., closing=520., claim=7., paid=5.)
        _, first = ledger.advance(2011, [
            account(owner, 0., 0., 0., 0., 4., 0., 80.), buyer,
        ], [lease], [], 5.)
        _, second = ledger.advance(2012, [
            account(owner, 4., 4., 8., 0., 0., 0., 0.),
            account("B", 4., 0., 0., 0., 0., 0., 0.),
        ], [], [], 0.)
        return first, second

    a, z = replay("A"), replay("Z")
    for left, right in zip(a, z, strict=True):
        assert (left["strict_succession_lower_mu"], left["strict_succession_upper_mu"]) == (
            right["strict_succession_lower_mu"], right["strict_succession_upper_mu"],
        ) == (expected_lower, 4.)
        assert left["prior_crop_unknown_mu"] == right["prior_crop_unknown_mu"] == 0.
        assert left["strict_synthetic_succession_mu"] == 4.
        assert right["strict_synthetic_succession_mu"] == expected_lower
    first_totals = dict(physical_operated_mu=8., annual_grain_mu_times=grain,
                        grain_kg=430. * grain, contracted_mu=4.,
                        contract_rent_yuan=80., seller_rent_yuan=80.,
                        rent_yuan_per_mu_if_traded=20., producer_income_yuan=30.,
                        claimed_yuan=7., paid_yuan=5., budget_yuan=5.)
    for key, total in first_totals.items():
        assert a[0][key] == z[0][key] == total
    for key, total in dict(physical_operated_mu=8., annual_grain_mu_times=8.,
                           grain_kg=3440., contracted_mu=0., paid_yuan=0.).items():
        assert a[1][key] == z[1][key] == total


def test_previous_external_pool_use_is_unknown_when_land_returns():
    ledger = SyntheticLandLedger(baseline())
    pool = "_area_pool_1"
    lease = contract(pool, area=2.) | dict(plot=None, channel="external_demand")
    outside = [dict(region=1, opening_area=0., operated=2., rented_in=2.,
                    rented_out=0., rent_received=0., rent_paid=40.)]
    ledger.advance(2011, [
        account("A", 2., 2., 2., 0., 2., 0., 40.),
        account("B", 2., 2., 2., 0., 0., 0., 0.),
    ], [lease], outside, 0.)
    internal = contract(year=2012, area=3.)
    buyer = account("B", 5., 5., 8., 3., 0., 60., 0.) | dict(crop=2.)
    _, summary = ledger.advance(2012, [
        account("A", 1., 1., 1., 0., 3., 0., 60.),
        buyer,
    ], [internal], [], 0.)
    assert (summary["strict_succession_lower_mu"], summary["strict_succession_upper_mu"]) == (0., 1.)
    assert summary["prior_crop_unknown_mu"] == summary["prior_external_pool_unknown_mu"] == 2.
    assert (summary["grain_on_prior_unknown_lower_mu"], summary["grain_on_prior_unknown_upper_mu"]) == (1., 2.)


def test_unknown_external_supply_not_mislabeled_as_strict_succession():
    ledger=SyntheticLandLedger(baseline())
    rows=[account("A",4.,4.,4.,0.,0.,0.,0.),account("B",3.,3.,3.,1.,0.,20.,0.)]
    lease=contract(area=1.)|dict(seller="_area_pool_1",plot=None,channel="external_supply")
    outside=[dict(region=1,opening_area=1.,operated=0.,rented_in=0.,rented_out=1.,rent_received=20.,rent_paid=0.)]
    records,out=ledger.advance(2011,rows,[lease],outside,0.)
    assert out["strict_synthetic_succession_mu"]==0.
    assert out["unidentified_origin_or_use_mu"]==1.
    assert any(r["unit"].startswith("external:") and r["previous_assignment"] is None for r in records)
    assert out["prior_crop_unknown_mu"] == 1.
    assert out["prior_external_pool_unknown_mu"] == 0.
    assert out["grain_on_prior_unknown_lower_mu"] == out["grain_on_prior_unknown_upper_mu"] == 1.


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


def test_observed_zero_grain_is_not_unknown_prior_crop():
    base=baseline()
    base[0].update(sow_food=0.,sow_cash=0.)
    ledger=SyntheticLandLedger(base)
    rows=[account("A",1.,0.,0.,0.,3.,0.,60.),account("B",5.,5.,5.,3.,0.,60.,0.)]
    _, out=ledger.advance(2011,rows,[contract()],[],0.)
    assert out["strict_succession_upper_mu"]==out["prior_crop_unknown_mu"]==0.


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
