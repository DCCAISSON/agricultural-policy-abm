"""Pre-production commitments; these fixtures are regression tests, not policy runs."""
from copy import deepcopy
from dataclasses import replace

import pytest

from scripts.n1_income_fixtures import HAND_CASES, hand_farmer
from agents.enterprise_agent import CropType as EnterpriseCrop
from core.fixed_rule_runtime import (
    prepare_preproduction_cash, production_area, preproduction_cash_enabled,
    settle_preproduction_cash,
)

MODE = dict(mode='reserved_no_credit', funds_basis='capital_as_liquid_operating_balance')


def test_v1_household_resource_inputs_and_rng():
    from agents.farmer_agent import FarmerAgent
    from core.reproducibility import RandomController, np_uniform
    m = hand_farmer(HAND_CASES[0]).model
    draws = []
    for enabled, share in ((False, .1), (True, .1), (True, .9)):
        m.config['runtime_fixed_rule']['national_v1'] = enabled
        RandomController.set_seed(17)
        m.np_random.seed(11)
        m.random.seed(11)
        row = dict(agri_ratio=share, family_size=4, labor_count=2, paid_nonfarm_proxy=True,
                   initial_working_capital_yuan=800., household_cash_reserve_yuan=500.)
        f = FarmerAgent.from_microdata('probe', m, row)
        draws.append(np_uniform(m, 0., 1.))
        if enabled:
            assert f.farming_ability == 1.
            assert f.capital == f.initial_capital == 800.
            assert f.get_farmer_available_capital_for_expansion() == f.balance_sheet_snapshot()['working_capital'] == 300.
            assert (f.total_labor_units, f._get_family_labor_capacity_for_production()) == (730., 240.)
            f.set_v1_labor(2, False)
            assert f.agricultural_labor == 360.
            f.set_v1_labor(0, None)
            assert f.total_labor_units == f._get_family_labor_capacity_for_production() == 0
            f.set_v1_labor(41, None)
            assert f.worker_count_source == 'one_worker_assumption' and f.agricultural_labor == 176.
    assert draws[0] == draws[1] == draws[2]


@pytest.mark.parametrize('regional_yield', [0., 350., None])
def test_v1_mixed_expected_and_actual_harvest_cash_and_work(regional_yield):
    from agents.farmer_agent import CropType
    f = funded_farmer(100000.)
    m = f.model
    m.config['runtime_fixed_rule'].update(national_v1=True, household_sales_ratio=dict(grain=.5, cash_crop=.75),
        grain_kg_mu_by_region={} if regional_yield is None else {'东部': regional_yield})
    f.region, f._microdata_source = None, {'region': '东部'}
    m.config['production']['random_noise'] = dict(std=0., settlement_std=0.)
    # Compare like with like: no realized disaster or irrigation shock in this hand account.
    m.config['production']['yield_stack'] = dict(irrigation={'enabled': False}, disaster_loss={'enabled': False})
    m.config['experiment']['seed'] = 103
    f._get_climate_shock = lambda *args: 0.
    f.household_size = 3
    f.set_v1_labor(2, True)
    f.household_candidate_grain_shares = {'mix': .6}
    f.household_previous_grain_share = f.household_planned_grain_share = .6
    f.planned_crop = CropType.MIXED
    expected = f.expected_fixed_rule_account('mix')
    f.labor_reduction_ratio = .5
    assert f.estimate_production_labor_profile()['service_substitution_rate'] == 0
    assert f.estimate_production_labor_profile()['total_labor_need'] == expected.family_labor_days == 160.
    prepare_preproduction_cash(m)
    m._conduct_labor_market()
    f.make_production_decisions()
    f.harvest_and_settle()
    grain, cash = (500. if regional_yield is None else regional_yield) * 6, 1000. * 4
    assert (expected.grain_output_kg, expected.cash_output_kg) == pytest.approx((grain, cash))
    assert (f.grain_production, f.cash_crop_production) == pytest.approx((grain, cash))
    s = f.last_income_settlement
    assert s['sold_output_kg'] == pytest.approx(grain*.5 + cash*.75)
    assert s['sold_output_kg'] + s['unsold_output_kg'] == pytest.approx(s['output_kg'])
    pg, pc = f._get_output_price_per_kg('crop_price'), f._get_output_price_per_kg('cash_crop')
    assert s['output_value_yuan'] == pytest.approx(grain*pg + cash*pc)
    assert s['sales_yuan'] == pytest.approx(expected.sales_yuan)
    assert f.agricultural_income == pytest.approx(grain*.5*pg + cash*.75*pc)
    assert f.capital == pytest.approx(100000. + f.gross_income - f.total_expense)


def test_v1_noise_paired_across_harvest_order():
    from agents.farmer_agent import CropType
    from core.reproducibility import np_normal, paired_yield_noise
    f = hand_farmer(HAND_CASES[0])
    m = f.model
    m.config['runtime_fixed_rule']['national_v1'] = True
    m.config['experiment']['seed'] = 109
    m.config['production']['random_noise'] = dict(std=.07, settlement_std=.03)
    f.set_v1_labor(1, None)
    f._get_climate_shock = lambda *args: 0.
    for crop in (CropType.GRAIN, CropType.CASH_CROP, CropType.GRAIN):
        np_normal(f, 0., 1.)
        f.planned_crop = crop
        f.make_production_decisions()
        assert f.last_yield_component_trace['random_factor'] == paired_yield_noise(f, 'production', 1., .07)
        f.harvest_and_settle()
        assert f.last_yield == pytest.approx(f.expected_yield * paired_yield_noise(f, 'settlement', 1., .03))


def funded_farmer(capital=1000., labor=400., rented=0.):
    f = hand_farmer(replace(HAND_CASES[0], capital=capital, labor_days=labor,
                            owned_mu=10. - rented, rented_mu=rented))
    m = f.model
    m.config['runtime_fixed_rule']['preproduction_cash'] = dict(MODE)
    m.farmers = [f]
    m.agricultural_enterprises = []
    return f


@pytest.mark.parametrize('capital', [0., 1000., 9000., 100000., -123.])
def test_cash_precedes_production_and_tenure_does_not_shrink(capital):
    f = funded_farmer(capital)
    m = f.model
    rights = [(p.owner, p.leased_to, p.current_operator, p.area) for p in m.land_patches]
    prepare_preproduction_cash(m)
    a = f.last_preproduction_cash_account
    assert a['committed_yuan'] <= max(0., capital) + 1e-8
    assert f.capital == capital
    assert f.get_operated_land_area() == 10.
    assert production_area(f) == pytest.approx(min(10., max(0., capital) / 900.))
    assert rights == [(p.owner, p.leased_to, p.current_operator, p.area) for p in m.land_patches]
    assert sum(p.preproduction_uncultivated_area_mu for p in m.land_patches) == pytest.approx(a['uncultivated_area_mu'])
    m._conduct_labor_market()
    f.make_production_decisions()
    f.harvest_and_settle()
    assert f.current_production_area == production_area(f)
    assert f.cash_production_cost == pytest.approx(production_area(f) * 900.)
    assert f.capital == pytest.approx(capital + f.gross_income - f.total_expense)
    assert a['actual_paid_yuan'] == pytest.approx(f.total_expense)
    assert a['unreceived_income_used_yuan'] == 0


def test_insufficient_rent_cash_is_payable_not_loan_and_owner_gets_payment():
    f = funded_farmer(1000., rented=10.)
    m = f.model
    owner = m.land_patches[1].owner
    prepare_preproduction_cash(m)
    a = f.last_preproduction_cash_account
    assert a['ordinary_rent_reserved_yuan'] == 1000.
    assert a['ordinary_rent_arrears_yuan'] == 3000.
    assert production_area(f) == 0.
    owner.calculate_land_rental_income()
    assert owner.land_rental_income == 1000.
    m._conduct_labor_market()
    f.make_production_decisions()
    f.harvest_and_settle()
    assert f.land_rental_expense == 1000.
    assert f.grain_production == 0.
    m.step_count += 1
    f.capital = 2000.
    prepare_preproduction_cash(m)
    assert f.last_preproduction_cash_account['ordinary_rent_due_yuan'] == 7000.
    assert f.last_preproduction_cash_account['ordinary_rent_arrears_yuan'] == 5000.


def test_hiring_is_backed_at_bid_ceiling_and_unfilled_commitment_released():
    f = funded_farmer(1000., labor=0.)
    prepare_preproduction_cash(f.model)
    a = f.last_preproduction_cash_account
    assert a['labor_reserved_yuan'] > 0
    f.model._conduct_labor_market()
    assert f.hired_labor_cost == 0
    assert a['labor_commitment_released_yuan'] == a['labor_reserved_yuan']
    f.make_production_decisions()
    f.harvest_and_settle()
    assert f.grain_production == 0
    assert a['actual_paid_yuan'] < a['initial_reserved_yuan']


def test_enterprise_budget_uses_seasonal_area_in_inputs_labor_and_management():
    f = hand_farmer(HAND_CASES[0])
    m = f.model
    m.config['runtime_fixed_rule']['preproduction_cash'] = dict(MODE)
    e = m.agricultural_enterprises[0]
    p = m.land_patches[1]
    p.area, p.leased_to, p.current_operator, p.rental_price = 10., e, e, 100.
    e.leased_patches = [p]
    e.capital = 2000.
    e.fixed_rule_current_action = 'maintain_grain'
    e.crop_portfolio = {EnterpriseCrop.GRAIN: 1., EnterpriseCrop.CASH_CROP: 0.}
    m.agricultural_enterprises = [e]
    m.farmers = []
    m.invalidate_land_tenure_caches()
    prepare_preproduction_cash(m)
    a = e.last_preproduction_cash_account
    assert 0 < production_area(e) < 10
    m._conduct_labor_market()
    e.make_production_decisions()
    e.harvest_and_settle()
    assert e.grain_area == production_area(e)
    assert e.management_cost == pytest.approx(production_area(e) * 50.)
    assert e.total_cost == a['actual_paid_yuan'] <= 2000.
    assert e.capital == pytest.approx(2000. + e.total_revenue - e.total_cost)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_opening_funds_rejected(invalid):
    f = funded_farmer(invalid)
    with pytest.raises(ValueError, match='finite'):
        prepare_preproduction_cash(f.model)


def test_default_and_explicit_assumption_and_idempotent_reservation():
    f = funded_farmer()
    cfg = f.model.config
    cfg['runtime_fixed_rule'].pop('preproduction_cash')
    assert not preproduction_cash_enabled(cfg)
    prepare_preproduction_cash(f.model)
    assert not hasattr(f, 'last_preproduction_cash_account')
    cfg['runtime_fixed_rule']['preproduction_cash'] = dict(MODE)
    prepare_preproduction_cash(f.model)
    a = deepcopy(f.last_preproduction_cash_account)
    prepare_preproduction_cash(f.model)
    assert f.last_preproduction_cash_account == a
    settle_preproduction_cash(f, 0.)
    with pytest.raises(RuntimeError, match='Duplicate'):
        settle_preproduction_cash(f, 0.)


def test_cancelled_conversion_is_a_recorded_unexecuted_plan():
    f = funded_farmer(0.)
    f.fixed_rule_conversion_expense = 123.
    prepare_preproduction_cash(f.model)
    a = f.last_preproduction_cash_account
    assert a['planned_conversion_yuan'] == a['cancelled_conversion_yuan'] == 123.
    assert a['conversion_reserved_yuan'] == f.fixed_rule_conversion_expense == 0.


def test_living_reserve_is_not_a_second_cash_deduction():
    f = funded_farmer(1000.)
    f.household_cash_reserve_yuan = 700.
    prepare_preproduction_cash(f.model)
    a = f.last_preproduction_cash_account
    assert a['opening_available_yuan'] == 300.
    assert f.capital == 1000.
    assert a['committed_yuan'] <= 300. + 1e-8
    f.model.config['agents']['farmer']['consumption_payment'] = {
        'mode': 'cash_limited_no_credit', 'funds_basis': MODE['funds_basis']}
    f.model._conduct_labor_market()
    f.make_production_decisions()
    f.harvest_and_settle()
    before = f.capital
    f.make_consumption_decision()
    paid = f.last_consumption_account['actual_cash_payment_yuan']
    assert paid == pytest.approx(min(before, f.last_consumption_account['consumption_demand_yuan']))
    assert f.capital == pytest.approx(before - paid)


def test_service_contract_and_cash_consumption_use_single_funded_expense():
    from tests.test_service_cash_closure_20261118 import setup_cash, demands
    from core.p3_service_settlement import confirm_service_settlement
    f, provider, market = setup_cash(20000.)
    m = f.model
    m.config['runtime_fixed_rule']['preproduction_cash'] = dict(MODE)
    m.config['agents']['farmer']['consumption_payment'] = {
        'mode': 'cash_limited_no_credit', 'funds_basis': MODE['funds_basis']}
    m.service_market = market
    m._get_farmer_service_demand = lambda farmer: demands(farmer, 1)[0]
    prepare_preproduction_cash(m)
    before = f.capital
    m._conduct_service_market()
    assert len(f.service_contracts) == 1
    fee = f.service_fee_expense
    assert fee > 0
    a = f.last_preproduction_cash_account
    assert a['service_reserved_yuan'] == fee
    assert a['committed_yuan'] <= a['opening_available_yuan']
    assert f.capital == before
    m._conduct_labor_market()
    f.make_production_decisions()
    f.harvest_and_settle()
    assert f.capital == pytest.approx(before + f.gross_income - f.total_expense)
    assert f.service_contracts[0].p3_paid_yuan == fee
    balance = provider.cash_balance
    confirm_service_settlement(f)
    assert provider.cash_balance == balance
    before_consumption = f.capital
    f.make_consumption_decision()
    assert f.capital == pytest.approx(before_consumption - f.last_consumption_account['actual_cash_payment_yuan'])


def test_service_offer_excludes_reserved_inputs_and_labor_before_matching():
    from tests.test_service_cash_closure_20261118 import setup_cash, demands
    f, provider, market = setup_cash(20000.)
    m = f.model
    m.config['runtime_fixed_rule']['preproduction_cash'] = dict(MODE)
    f.capital = 1000.
    m.service_market = market
    m._get_farmer_service_demand = lambda farmer: demands(farmer, 1)[0]
    prepare_preproduction_cash(m)
    balance = provider.cash_balance
    m._conduct_service_market()
    assert f.service_contracts == []
    assert provider.cash_balance == balance


def test_full_annual_path_rebuilds_funding_each_year():
    from scripts.run_v4_8_national_fixed_rule_runtime_smoke import runtime_config
    from scripts.n1_income_fixtures import ROOT
    from core.model_builder import ModelBuilder
    cfg = runtime_config(ROOT / 'config/experiments/v4_8_national_fixed_rule.yaml')
    cfg['runtime_fixed_rule'] = {'accounting_basis': 'action_conditioned_v1'}
    cfg['runtime_fixed_rule']['preproduction_cash'] = dict(MODE)
    cfg['agents']['farmer']['consumption_payment'] = {
        'mode': 'cash_limited_no_credit', 'funds_basis': MODE['funds_basis']}
    m = ModelBuilder(config=cfg).build()
    m.config['runtime_fixed_rule']['accounting_basis'] = 'action_conditioned_v1'
    m.config['runtime_fixed_rule']['preproduction_cash'] = dict(MODE)
    for _ in range(3):
        m.step()
        for operator in [*m.farmers, *m.agricultural_enterprises]:
            a = operator.last_preproduction_cash_account
            assert a['step'] == m.step_count
            assert a['status'] == 'settled'
            assert a['actual_paid_yuan'] <= a['committed_yuan'] + 1e-6
            assert a['committed_yuan'] <= a['opening_available_yuan'] + 1e-6
            assert operator.capital >= -1e-6


def test_paid_initial_lease_cannot_borrow_unreceived_harvest_cash():
    from core.fixed_rule_runtime import InitialPaidLeaseAccount
    f = hand_farmer(HAND_CASES[0])
    m = f.model
    e = m.agricultural_enterprises[0]
    p = m.land_patches[1]
    p.area, p.leased_to, p.current_operator, p.rental_price = 10., e, e, 4000.
    p.operational_transfer_channel = 'author_paid_initial_lease'
    p.lease_contract_years = 5
    e.leased_patches = [p]
    e.capital = 1000.
    e.fixed_rule_current_action = 'maintain_grain'
    e.crop_portfolio = {EnterpriseCrop.GRAIN: 1., EnterpriseCrop.CASH_CROP: 0.}
    m.config['runtime_fixed_rule'].update(preproduction_cash=dict(MODE), initial_tenure_mode='B_star_5y')
    m.initial_paid_lease_accounts = (InitialPaidLeaseAccount(p, e, f, 4000., 5),)
    m.farmers, m.agricultural_enterprises, m.step_count = [], [e], 1
    m.invalidate_land_tenure_caches()
    prepare_preproduction_cash(m)
    m._conduct_labor_market()
    e.make_production_decisions()
    e.harvest_and_settle()
    a = e.last_preproduction_cash_account
    assert a['initial_rent_reserved_yuan'] == e.initial_rent_paid_yuan == 1000.
    assert e.initial_rent_arrears_yuan == 3000.
    assert e.total_cost == 1000.
    assert e.grain_production == e.capital == 0.
    f.calculate_land_rental_income()
    assert f.land_rental_income == 1000.


@pytest.mark.parametrize('liquid,reserve', [(0., 0.), (0., 30.), (100., 30.)])
def test_microdata_explicit_liquid_stock_and_reserve(liquid, reserve):
    from agents.farmer_agent import FarmerAgent
    f = hand_farmer(HAND_CASES[0])
    g = FarmerAgent.from_microdata('cash_input', f.model, {
        'land_area_mu': 1., 'total_income': 999999.,
        'initial_working_capital_yuan': liquid, 'household_cash_reserve_yuan': reserve,
    }, config=f.model.config)
    assert g.capital == g.initial_capital == liquid
    assert g.household_cash_reserve_yuan == reserve
    assert g.balance_sheet_snapshot()['cash'] == liquid
    assert g.balance_sheet_snapshot()['working_capital'] == max(0., liquid - reserve)


def test_p3_production_claim_uses_cultivated_area_but_pure_rights_do_not():
    from tests.test_p3_project_payments_20260909 import prepared, program
    from core.p3_project_claims import production_records, project_claims
    from core.p3_project_contracts import ProductionRecord, P3AccountingError
    f = prepared(program())
    f.model.config['runtime_fixed_rule']['preproduction_cash'] = dict(MODE)
    f.capital = 1000.
    prepare_preproduction_cash(f.model)
    f.model._conduct_labor_market()
    f.make_production_decisions()
    f.harvest_and_settle()
    assert sum(r.area_mu for r in production_records(f)) == pytest.approx(1000. / 900.)
    claims = project_claims(f, program())
    assert sum(c.nominal_claim_yuan for c in claims if c.project_id == 'trial_area') == pytest.approx(350. * 1000. / 900.)
    assert sum(c.nominal_claim_yuan for c in claims if c.project_id == 'land_protection') == pytest.approx(10. * 1000. / 900.)
    rights_only = program().model_copy(update={'preserved_projects': tuple(
        p.model_copy(update={'requires_cultivation': False}) for p in program().preserved_projects)})
    rights_claims = project_claims(f, rights_only)
    assert sum(c.nominal_claim_yuan for c in rights_claims if c.project_id == 'land_protection') == 100.
    f.p3_production_records = [ProductionRecord(plot_id=str(f.model.land_patches[0].patch_id),
        season='annual_grain', area_mu=10., harvest_kg=100., source='invalid_held_area')]
    with pytest.raises(P3AccountingError):
        production_records(f)


def test_retired_ordinary_tenant_pays_arrears_before_owner_recognizes_receipt():
    f = hand_farmer(HAND_CASES[0])
    m = f.model
    e = m.agricultural_enterprises[0]
    p = m.land_patches[1]
    p.area, p.owner, p.leased_to, p.current_operator, p.rental_price = 10., f, None, f, 4000.
    e.capital = 2000.
    e.hired_labor_cost = 999.
    m.config['runtime_fixed_rule']['preproduction_cash'] = dict(MODE)
    m.config['agents']['farmer']['consumption_payment'] = {
        'mode': 'cash_limited_no_credit', 'funds_basis': MODE['funds_basis']}
    m.step_count = 2
    m.farmers, m.agricultural_enterprises = [f], []
    m.preproduction_rent_ledger = [dict(patch=p, tenant=e, recipient=f, arrears_yuan=3000., step=1)]
    m.invalidate_land_tenure_caches()
    prepare_preproduction_cash(m)
    f.calculate_land_rental_income()
    assert f.land_rental_income == 0.
    m._conduct_labor_market()
    m._run_fixed_rule_production_stage()
    assert e.capital == 0.
    assert e.last_preproduction_cash_account['actual_paid_yuan'] == 2000.
    assert m.preproduction_rent_ledger[0]['status'] == 'paid'
    assert m.preproduction_rent_ledger[0]['arrears_yuan'] == 1000.
    assert f.land_rental_income == 2000.
    assert e.grain_production == e.cash_crop_production == 0.
