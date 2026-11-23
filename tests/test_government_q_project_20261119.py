"""Government-only tabular Q connected to actual annual P3 payments."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.n1_income_fixtures import HAND_CASES, hand_farmer
from scripts.run_national_formal_revision_20261118 import observe_actual_harvests
from tests.test_p3_project_payments_20260909 import program
from core.p3_project_payments import freeze_previous_claims
from core.p3_project_contracts import configured_program


def setup(terminal=1, budget=10000.):
    f = hand_farmer(HAND_CASES[0])
    m = f.model
    m.farmers, m.agricultural_enterprises = [f], []
    m.initial_household_cohort = (f,)
    m.config['fiscal_system'] = {'p3_support': program(budget=budget).model_dump()}
    m.config['policy']['government_q_project'] = dict(enabled=True, episode_end_step=terminal,
        grain_reference_kg=5000., h0_income_reference_yuan=20000.)
    m.config['policy']['rl_reward_weights'] = dict(grain_security=.5, farmer_welfare=.3, fiscal_balance=.2)
    m.government.np_random = np.random.RandomState(123)
    m.step_count = 1
    return f


def choose(g, action):
    def selected(state, action_count=5):
        assert action_count == 3
        g.q_table.setdefault(state, [0.] * 3)
        return action
    g._choose_action_rl = selected


def settle_year(f, action, finish=True):
    m, g = f.model, f.model.government
    choose(g, action)
    with observe_actual_harvests(m):
        g.begin_annual_cycle()
        f.make_production_decisions()
        f.harvest_and_settle()
        g.end_annual_cycle()
        if finish:
            g.finish_government_q_year()
    return g


@pytest.mark.parametrize('action,rate,paid', [(0, 0., 100.), (1, 350., 3600.), (2, 525., 5350.)])
def test_actions_change_real_payments_not_budget_or_price_floor(action, rate, paid):
    f = setup()
    g = f.model.government
    floor = g.grain_price_floor
    settle_year(f, action)
    row = g.government_q_sar[0]
    assert g.p3_batch.program.nominal_rate == rate
    assert g.total_subsidy_paid == paid
    assert sum(r['paid_yuan'] for r in g.p3_project_payments) == paid
    assert row['trial_budget_yuan'] == 10000.
    assert row['outcome_observations']['fiscal_envelope_yuan'] == 11000.
    assert g.grain_price_floor == floor
    assert row['done'] is True
    assert row['state'][3:] == (1, 1)
    assert row['next_state'][3:] == (action, 0)
    assert row['q_after'] == pytest.approx(.1 * row['reward'])
    assert f.subsidy_income == paid
    assert not g.rl_enabled  # The old end-of-year RL path is not required or enabled.


def test_hard_budget_rations_high_action_and_reward_charges_actual_spending():
    f = setup(budget=1000.)
    g = settle_year(f, 2)
    row = g.government_q_sar[0]
    assert g.total_subsidy_paid == 1100.
    assert row['reward_components']['fiscal_balance'] == pytest.approx(-3 * 1100. / 2000.)
    assert g.remaining_budget >= 0
    assert row['reward'] == pytest.approx(sum(row['reward_weights'][k] * v
                                            for k, v in row['reward_components'].items()))


@pytest.mark.parametrize('action,rule,paid', [(0, 'area_universal', 100.),
    (1, 'area_universal', 1100.), (2, 'scale_eligibility', 100.)])
def test_budget_menu_binds_eligibility_and_uncapped_reward(action, rule, paid):
    f = setup()
    m = f.model
    m.config['policy']['government_q_project'].update(
        budget_area_menu=100., uncapped_income_reward=True)
    g = settle_year(f, action, finish=False)
    assert g.p3_batch.program.rule == rule
    assert g.p3_batch.program.trial_budget_yuan == 10000.
    assert g.p3_batch.program.nominal_rate == (100. if action else 0.)
    assert g.total_subsidy_paid == paid
    g._government_q_outcome['h0_mean_net_income_yuan'] = 50000.
    g.finish_government_q_year()
    row = g.government_q_sar[0]
    assert row['program_rule'] == rule
    assert row['reward_components']['farmer_welfare'] == pytest.approx(np.arcsinh(1.5))
    assert row['reward_components']['farmer_welfare'] > np.arcsinh(40000. / 20000. - 1.)


def test_budget_menu_switch_preserves_prior_claims_and_annual_feedback():
    f = setup(terminal=2)
    m = f.model
    m.config['policy']['government_q_project'].update(
        budget_area_menu=100., uncapped_income_reward=True)
    g = settle_year(f, 2)
    previous = freeze_previous_claims(g, configured_program(m))
    m.config['fiscal_system']['p3_support'] = previous.model_dump()
    m.step_count = 2
    settle_year(f, 1)
    assert g.p3_batch.program.previous_actor_pool_nominal == previous.previous_actor_pool_nominal
    assert g.p3_batch.program.expectation_source_step == 1
    assert g.government_q_sar[1]['state'] == g.government_q_sar[0]['next_state']
    assert g.total_subsidy_paid == 1100.


def test_terminal_update_does_not_bootstrap_and_repeat_finish_is_inert():
    f = setup()
    g = settle_year(f, 1, finish=False)
    assert g.government_q_sar == []
    pending = g._government_q_pending
    nxt = g._government_q_state(g._government_q_outcome, 1, 0)
    g.q_table[nxt] = [10000.] * 3
    assert g.q_table[pending['state']][1] == 0.
    g.finish_government_q_year()
    row = g.government_q_sar[0]
    assert row['q_after'] == pytest.approx(g.learning_rate * row['reward'])
    old = deepcopy(g.q_table)
    g.finish_government_q_year()
    assert g.q_table == old and len(g.government_q_sar) == 1


def test_two_year_transition_uses_previous_outcome_and_current_support():
    f = setup(terminal=2)
    g = settle_year(f, 2, finish=False)
    nxt = g._government_q_state(g._government_q_outcome, 2, 1)
    g.q_table[nxt] = [10., 0., 0.]
    g.finish_government_q_year()
    first = g.government_q_sar[0]
    assert first['q_after'] == pytest.approx(.1 * (first['reward'] + .95 * 10.))
    f.model.config['fiscal_system']['p3_support'] = freeze_previous_claims(g, configured_program(f.model)).model_dump()
    f.model.step_count = 2
    settle_year(f, 0)
    assert g.government_q_sar[1]['state'] == first['next_state']
    assert g.government_q_sar[1]['done']


def test_observation_uses_successful_harvests_and_fixed_h0_negative_income():
    f = setup()
    g = settle_year(f, 0, finish=False)
    actual = sum(r['grain_physical_kg'] for r in f.model.revision_harvest_records.values())
    f.grain_production = 1e12
    f.cash_net_income = -500.
    f.model.farmers = []
    observed = g._government_q_performance()
    assert observed['grain_output_kg'] == actual
    assert observed['h0_count'] == 1
    assert observed['h0_mean_net_income_yuan'] == -500.
    del f.model.revision_harvest_records
    with pytest.raises(RuntimeError, match='observer'):
        g._government_q_performance()


def test_exploration_has_an_independent_rng_and_frozen_q_does_not_update():
    f = setup()
    g = f.model.government
    before = deepcopy(f.model.np_random.get_state())
    g._choose_action_rl((0, 0, 0, 1, 1), action_count=3)
    after = f.model.np_random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    g.freeze_q_table()
    settle_year(f, 1)
    row = g.government_q_sar[0]
    assert not row['q_updated'] and row['q_before'] == row['q_after']


def test_complete_model_step_finishes_one_transition(monkeypatch):
    f = setup()
    m, g = f.model, f.model.government
    m.step_count = 0
    choose(g, 1)
    with observe_actual_harvests(m):
        m.step()
    assert len(g.government_q_sar) == 1
    row = g.government_q_sar[0]
    assert row['step'] == 1 and row['done']
    assert row['outcome_observations']['grain_output_kg'] == pytest.approx(
        sum(r['grain_physical_kg'] for r in m.revision_harvest_records.values()))


def test_frozen_project_real_selector_keeps_unknown_state_and_counter_unchanged():
    g = setup().model.government
    known, unknown = (1, 1, 0, 1, 2), (0, 0, 0, 2, 1)
    g.q_table = {known: [0., 1., 2.]}
    g.training_step_count = 17
    g.freeze_q_table()
    assert not g._frozen
    before = deepcopy(g.q_table)
    assert g._choose_action_rl(unknown, action_count=3) == 0
    assert g._choose_action_rl(known, action_count=3) == 2
    assert g.q_table == before
    assert g.training_step_count == 17
    with pytest.raises(ValueError, match='three actions'):
        g._choose_action_rl(unknown, action_count=5)
    g.q_table[known] = [0.] * 5
    with pytest.raises(ValueError, match='three actions'):
        g._choose_action_rl(known, action_count=3)


@pytest.mark.parametrize('action', [0, 1, 2])
def test_v1_fixed_actions_keep_the_legacy_payment_ledger_and_budget(action):
    def run(v1):
        f = setup()
        f.region = '东部'
        f.set_v1_labor(1, 0.)
        m, g = f.model, f.model.government
        m.config.setdefault('runtime_fixed_rule', {})['national_v1'] = v1
        m.config['policy']['government_q_project'].update(budget_area_menu=100., uncapped_income_reward=True)
        g.freeze_q_table()
        g._choose_action_rl = lambda state, action_count: action
        with observe_actual_harvests(m):
            g.begin_annual_cycle()
            f.make_production_decisions()
            f.harvest_and_settle()
            g.end_annual_cycle()
            g.finish_government_q_year()
        return g

    old, v1 = run(False), run(True)
    prev, row = old.government_q_sar[0], v1.government_q_sar[0]
    assert v1.p3_project_payments == old.p3_project_payments
    assert v1.total_subsidy_paid == old.total_subsidy_paid
    if action == 0:
        assert v1.total_subsidy_paid == 100.
        assert all(r['paid_yuan'] == 0. for r in v1.p3_project_payments if r['layer'] == 'trial')
    assert row['nominal_rate_yuan_per_mu'] == (100. if action else 0.)
    assert row['program_rule'] == prev['program_rule']
    assert row['state'][:5] == prev['state']
    assert row['next_state'][:5] == old._government_q_state(row['outcome_observations'], action, 0)
    assert len(row['state']) == len(row['next_state']) == 6 and row['state'][5] == 0
    assert row['state_observations']['macro_regions']['东部']['h0_grain_output_kg'] is None
    observed = row['outcome_observations']
    assert observed['macro_regions']['东部']['h0_grain_output_kg'] == observed['grain_output_kg']
    assert observed['fiscal_cost_yuan'] <= observed['fiscal_envelope_yuan']
    assert row['reward_weights'] == dict(grain_security=1., farmer_welfare=0., fiscal_balance=0.)
    assert row['reward'] == pytest.approx(2 * (observed['grain_output_kg'] / row['grain_reference_kg'] - 1))
    assert v1.q_table == {} and not row['q_updated']


def test_v1_macro_regions_use_fixed_households_and_not_organization_output():
    f = setup()
    m, g = f.model, f.model.government
    m.config.setdefault('runtime_fixed_rule', {})['national_v1'] = True
    g._government_q_references = (5000., 20000.)
    regions = ['东部', '中部', '西部', '东北', '东部']
    g._government_q_h0 = [
        SimpleNamespace(unique_id=i, region=region, cash_net_income=income, grain_area=1., last_income_settlement={'step': 1})
        for i, (region, income) in enumerate(zip(regions, [-2000., 18000., 22000., 17999., 22000.]))]
    m.revision_harvest_records = {str(i): dict(step=1, grain_physical_kg=float(i)) for i in range(6)}
    org = SimpleNamespace(unique_id=5, grain_area=2., enterprise_type='cooperative')
    m.agricultural_enterprises = [org]
    for p, a, r in zip(m.land_patches, (org, g._government_q_h0[1]), ('东北', '东部')):
        p.current_operator, p.abandoned = a, False
        p.official_region, p.official_region_scope = r, 'official_macro_region'
    m.land_clearing_house.transactions = [dict(patch=m.land_patches[0], price=180.)]
    g.v1_applicant_regions = {'5': '东北'}
    g.p3_project_payments = [dict(recipient_id='5', layer='trial', paid_yuan=8.)]
    g.current_subsidy_runtime.update(total_fiscal_cost_yuan=0., fiscal_envelope_yuan=100.)
    obs = g._government_q_performance()
    assert obs['grain_output_kg'] == 15.
    assert sum(r['h0_grain_output_kg'] for r in obs['macro_regions'].values()) == 10.
    east = obs['macro_regions']['东部']
    assert (east['h0_count'], east['h0_mean_net_income_yuan'], east['h0_grain_output_kg']) == (2, 10000., 4.)
    ne = obs['macro_regions']['东北']
    assert (ne['organization_grain_mu'], ne['yield_kg_per_mu'], ne['trial_paid_yuan']) == (2., 2.5, 8.)
    assert ne['transfer_in_mu'] == m.land_patches[0].area
    assert ne['contract_rent_yuan_per_mu'] == 180. / ne['transfer_in_mu']
    assert sum(r['grain_output_kg'] for r in obs['macro_regions'].values()) == 15.
    assert [obs['macro_regions'][r]['income_bin'] for r in ('东部', '中部', '西部', '东北')] == [0, 1, 2, 0]
    g._government_q_h0[0].region = None
    assert g._government_q_performance()['macro_regions']['unknown']['h0_count'] == 1


@pytest.mark.parametrize('action,region,share', [
    (3, '东部', .5), (4, '中部', .5), (5, '西部', .5), (6, '东北', .5),
    (3, '中部', 1/6), (3, None, 0.)])
def test_v1_announced_support_and_actual_paid_share(action, region, share):
    from core.p3_project_claims import expected_support
    f = setup(budget=1000.)
    m, g = f.model, f.model.government
    f.region = region
    f.set_v1_labor(1, 0.)
    m.config.setdefault('runtime_fixed_rule', {})['national_v1'] = True
    m.config['policy']['government_q_project']['budget_area_menu'] = 10.
    g.freeze_q_table()
    def fixed(state, action_count):
        assert (len(state), action_count) == (6, 7)
        return action
    g._choose_action_rl = fixed
    with observe_actual_harvests(m):
        g.begin_annual_cycle()
        offer = expected_support(f, 10., g.p3_batch.program)
        f.make_production_decisions()
        f.harvest_and_settle()
        g.end_annual_cycle()
        g.finish_government_q_year()
    sar = g.government_q_sar[0]
    assert sum(sar['regional_area_shares'].values()) == pytest.approx(1.)
    caps = g.current_subsidy_runtime['regional_area_caps_yuan']
    assert sum(caps.values()) == pytest.approx(1000.)
    district = region or 'unknown'
    assert caps[district] == pytest.approx(1000. * share)
    trial = [r for r in g.p3_project_payments if r['layer'] == 'trial']
    assert trial and all(r['applicant_region'] == district for r in trial)
    assert all(r['nominal_rate'] == pytest.approx(400. * share) for r in trial)
    assert sum(r['paid_yuan'] for r in trial) == pytest.approx(caps[district])
    assert offer == pytest.approx(100. + caps[district])
    assert sar['outcome_observations']['macro_regions'][district]['trial_paid_yuan'] == pytest.approx(caps[district])
    assert g.total_subsidy_paid == pytest.approx(offer)
    assert not g.q_table and sar['q_updated'] is False


def test_v1_transparent_uses_only_compact_state_and_handles_ties_and_missing():
    f = setup()
    m, g = f.model, f.model.government
    m.config.setdefault('runtime_fixed_rule', {})['national_v1'] = True
    g._government_q_references = (5000., 20000.)
    regions = {r: {'metric_bins': [-1]*6} for r in ('东部', '中部', '西部', '东北')}
    obs = dict(grain_output_kg=0., h0_mean_net_income_yuan=0., fiscal_envelope_yuan=0., macro_regions=regions)
    def choice():
        state = g._government_q_state(obs, 0, 2)
        assert len(state) == 6
        return state[-1], g.national_v1_transparent_action(state)
    assert choice() == (0, 1)
    for i, row in enumerate(regions.values(), 1):
        row['metric_bins'][:2] = [0, 1]
        assert choice() == (i, 2+i)
        row['metric_bins'][:2] = [1, 1]
    regions['东部']['metric_bins'][:2] = regions['西部']['metric_bins'][:2] = [0, 1]
    assert choice() == (0, 1)
    g.freeze_q_table()
    state = g._government_q_state(obs, 0, 2)
    assert g._choose_action_rl(state, 7) == 0 and not g.q_table
    g.q_table[state] = [0.]*6 + [1.]
    assert g._choose_action_rl(state, 7) == 6
    with pytest.raises(ValueError, match='seven actions'):
        g._choose_action_rl(state, 3)
