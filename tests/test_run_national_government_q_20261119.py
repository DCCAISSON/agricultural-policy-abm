"""Small runner contracts; no formal episodes are launched here."""
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from scripts import run_national_government_q_20261119 as runner
from tests.test_government_q_project_20261119 import setup


def test_v1_harvest_noise_does_not_follow_action_order():
    from core.reproducibility import paired_yield_noise
    model = SimpleNamespace(config={"experiment":{"seed":103}}, step_count=1)
    actor = SimpleNamespace(model=model, unique_id=12)
    grain = paired_yield_noise(actor, "grain", 1, .02)
    cash = paired_yield_noise(actor, "cash", 1, .02)
    assert grain != cash and paired_yield_noise(actor, "grain", 1, .02) == grain
    model.step_count = 2
    assert paired_yield_noise(actor, "grain", 1, .02) != grain


def test_q_json_roundtrip_no_eval_or_alias():
    table = {(0, 1, 0, 2, 30): [1., -2., 3.]}
    rows = runner.q_rows(table)
    restored = runner.decode_q(rows)
    restored[(0, 1, 0, 2, 30)][0] = 9
    assert table[(0, 1, 0, 2, 30)][0] == 1
    assert rows[0]['state'] == [0, 1, 0, 2, 30]


def test_q_serializes_numpy_update_values():
    table = {(0, 1, 0, 2, 30): [np.float64(.125), 0., -1.]}
    rows = runner.q_rows(table)
    assert type(rows[0]['values'][0]) is float
    assert runner.decode_q(rows) == table


@pytest.mark.parametrize('rows', [
    [{'state': "__import__('os')", 'values': [0., 0., 0.]}],
    [{'state': [0, 1], 'values': [0., 0., 0.]}],
    [{'state': [0, 1, 0, 0, 30], 'values': [0., float('nan'), 0.]}],
    [{'state': [0, 1, 0, 0, 30], 'values': [0., 0.]}],
    [{'state': [0, 1, 0, 0, 30], 'values': [0., 0., 0.]}] * 2,
])
def test_reject_invalid_q(rows):
    with pytest.raises(ValueError):
        runner.decode_q(rows)


@pytest.mark.parametrize('action,rate', [(None, 0.), (0, 0.), (1, 350.), (2, 525.)])
def test_actual_tiny_frozen_policy_harvest_and_payment(action, rate):
    f = setup()
    model, gov = f.model, f.model.government
    runner.configure_government(gov, {}, 31, training=False, fixed_action=action)
    count = gov.training_step_count
    rng_before = deepcopy(gov.np_random.get_state())
    with runner.old.observe_actual_harvests(model):
        gov.begin_annual_cycle()
        f.make_production_decisions()
        f.harvest_and_settle()
        gov.end_annual_cycle()
        gov.finish_government_q_year()
    sar = gov.government_q_sar[-1]
    assert sar['done'] is True and sar['q_updated'] is False
    assert len(sar['state']) == len(sar['next_state']) == 5
    assert sar['nominal_rate_yuan_per_mu'] == rate
    assert gov.q_table == {}
    assert gov.eval_unknown_state_count == 1
    assert gov.training_step_count == count
    assert f.subsidy_income == pytest.approx(100. + rate * 10.)
    assert sar['outcome_observations']['grain_output_kg'] == pytest.approx(
        sum(r['grain_physical_kg'] for r in model.revision_harvest_records.values()))
    assert f.last_income_settlement['step'] == model.step_count
    assert np.array_equal(rng_before[1], gov.np_random.get_state()[1])
    assert rng_before[2:] == gov.np_random.get_state()[2:]


def test_actual_tiny_training_table_carries_to_next_episode():
    f = setup()
    gov = f.model.government
    runner.configure_government(gov, {}, 17, training=True)
    with runner.old.observe_actual_harvests(f.model):
        gov.begin_annual_cycle()
        f.make_production_decisions()
        f.harvest_and_settle()
        gov.end_annual_cycle()
        gov.finish_government_q_year()
    table = deepcopy(gov.q_table)
    assert any(v != 0 for values in table.values() for v in values)
    other = setup().model.government
    runner.configure_government(other, table, 23, training=True, training_steps=gov.training_step_count)
    assert other.q_table == table and not other.q_table_frozen
    assert other.q_table is not table
    assert other.training_step_count == gov.training_step_count == 1


def test_actual_base_inputs_readonly():
    cfg, economics, reference, inputs = runner.load_base()
    assert cfg['scale']['n_farmers'] == 600
    assert cfg['scale']['n_enterprises'] == 60
    assert cfg['experiment']['steps'] == 30
    assert cfg['fiscal_system']['p3_support']['nominal_rate'] == 0
    assert cfg['agents']['farmer']['consumption_payment']['mode'] == 'cash_limited_no_credit'
    assert cfg['runtime_fixed_rule']['preproduction_cash']['mode'] == 'reserved_no_credit'
    assert economics['approved'] and reference['trial_pool_year1'] > 0 and inputs


def test_actual_formal_builder_preserves_q_and_allows_scenario_metadata():
    cfg, economics, _, _ = runner.load_base()
    model = runner.build_model(cfg, economics, 17)
    assert (len(model.farmers), len(model.agricultural_enterprises)) == (600, 60)
    assert model.config['policy']['government_q_project']['enabled'] is True
    assert model.config['policy']['government_q_project']['grain_reference_kg'] > 0
    assert model.config['policy']['government']['scenario_id'] == 'baseline_cash'
    assert model.config['policy']['rl_reward_weights'] == runner.WEIGHTS


def test_existing_output_refused_before_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'load_base', lambda: pytest.fail('must not load'))
    with pytest.raises(FileExistsError):
        runner.run(tmp_path)


def test_paired_initial_and_exogenous_mismatch_rejected():
    rows = [{'seed': seed, 'initial_sha256': 'i', 'initial_exogenous_sha256': 'e',
             'annual_exogenous_sha256': 'a', 'grain_reference_kg': 100.}
            for _, seed, _ in runner.PLAN]
    runner.verify_pairs(rows)
    rows[-1]['annual_exogenous_sha256'] = 'different'
    with pytest.raises(RuntimeError, match='paired seed 31'):
        runner.verify_pairs(rows)


def test_schedule_four_training_then_four_evaluations(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'load_base', lambda: ({}, {}, {}, {}))
    monkeypatch.setattr(runner, 'source_hashes', lambda: {})
    monkeypatch.setattr(runner, 'check_sources', lambda _: None)
    monkeypatch.setattr(runner, 'build_model', lambda base, econ, seed: SimpleNamespace(
        government=SimpleNamespace(q_table={}, training_step_count=0),
        config={'policy': {'government_q_project': {'grain_reference_kg': float(seed)}}}))
    calls, incoming = [], []
    counts, references = [], []

    def configure(gov, table, seed, training, action, training_steps):
        incoming.append(deepcopy(table))
        gov.q_table = deepcopy(table)
        counts.append(training_steps)
        gov.training_step_count = training_steps

    def episode(model, reference, output, seed, expected, training, full_audit):
        output.mkdir()
        calls.append((output.name, seed, training, full_audit))
        references.append(model.config['policy']['government_q_project']['grain_reference_kg'])
        if training:
            model.government.q_table[(0, 0, 0, 0, 30)] = [len(calls), 0., 0.]
            model.government.training_step_count += 30
        return dict(seed=seed, initial_sha256='i', initial_exogenous_sha256='e',
                    annual_exogenous_sha256='a', grain_reference_kg=100.)

    monkeypatch.setattr(runner, 'configure_government', configure)
    monkeypatch.setattr(runner, 'run_episode', episode)
    runner.run(tmp_path / 'new')
    assert len(calls) == 8
    assert [r[1] for r in calls] == [17, 23, 17, 23, 31, 31, 31, 31]
    assert [r[2] for r in calls] == [True] * 4 + [False] * 4
    assert [r[0] for r in calls if r[3]] == ['eval_q']
    assert incoming[0] == {}
    assert incoming[1][(0, 0, 0, 0, 30)][0] == 1
    assert all(t[(0, 0, 0, 0, 30)][0] == 4 for t in incoming[4:])
    assert counts == [0, 30, 60, 90, 120, 120, 120, 120]
    assert references == [17.] * 8
    assert (tmp_path / 'new/COMPLETED.json').is_file()


@pytest.mark.parametrize('failure', [None, 'done', 'mutated_q', 'missing_hook'])
def test_annual_loop_evidence_and_failure_guards(monkeypatch, tmp_path, failure):
    gov = SimpleNamespace(q_table={}, training_step_count=120, eval_unknown_state_count=0,
                          government_q_sar=[], current_subsidy_runtime={},
                          learning_rate=.1, discount_factor=.95, epsilon=.2, pretrain_steps=200)
    model = SimpleNamespace(government=gov, step_count=0, initial_household_cohort=[], land_patches=[],
        config={'policy': {'government_q_project': {'episode_end_step': 2, 'grain_reference_kg': 100.}}})
    def step():
        model.step_count += 1
        if failure != 'missing_hook':
            model._completed_production_year_snapshot_hook(model)
        gov.government_q_sar.append(dict(step=model.step_count, done=False if failure == 'done' else model.step_count == 2,
            state=(0, 0, 0, 0, 2), next_state=(0, 0, 0, 0, 1), reward_weights=runner.WEIGHTS,
            q_updated=False, reward=.5,
            outcome_observations=dict(grain_output_kg=100., h0_mean_net_income_yuan=200.,
                                      h0_count=2, fiscal_cost_yuan=10.)))
        if failure == 'mutated_q':
            gov.q_table[(0, 0, 0, 0, 1)] = [0., 0., 0.]
    model.step = step
    monkeypatch.setattr(runner, 'check_sources', lambda _: None)
    monkeypatch.setattr(runner.old, 'annual_actors', lambda _: {})
    monkeypatch.setattr(runner.old, 'apply_annual_exogenous_path', lambda *args: None)
    monkeypatch.setattr(runner.old, 'advance_program', lambda *args: None)
    monkeypatch.setattr(runner.old, 'observe_actual_harvests', lambda _: nullcontext())
    monkeypatch.setattr(runner.old, 'operator_snapshot', lambda _: {})
    monkeypatch.setattr(runner.old, 'exogenous_snapshot', lambda _: {'step': model.step_count})
    monkeypatch.setattr(runner.old, 'check_annual_accounts', lambda *args: {'checked': True})
    out = tmp_path / 'episode'
    if failure:
        with pytest.raises(RuntimeError):
            runner.run_episode(model, {}, out, 31, {}, False)
        assert (out / 'FAILED.json').is_file() and not (out / 'COMPLETED.json').exists()
    else:
        result = runner.run_episode(model, {}, out, 31, {}, False)
        assert result['cumulative_actual_grain_kg'] == 200.
        assert result['cumulative_h0_cash_income_yuan'] == 800.
        assert result['cumulative_fiscal_cost_yuan'] == 20.
        assert result['cumulative_reward'] == 1.
        assert result['training_step_count_after'] == 120
        assert (out / 'year_02/government_sar.json').is_file()
        assert (out / 'year_02/annual_checks.json').is_file()
        assert not (out / 'year_02/actors.jsonl').exists()
    assert not hasattr(model, '_completed_production_year_snapshot_hook')


@pytest.mark.parametrize('cfg', [
    {'mechanism_switches': {'use_learning_farmer': True}},
    {'mechanism_switches': {'use_learning_enterprise': True}},
    {'mechanism_switches': {'use_extended_government_rl': True}},
    {'policy': {'rl_enabled': True}},
    {'policy': {'policy_mode': 'ppo_active'}},
    {'policy': {'government': {'policy_mode': 'ppo_shadow'}}},
])
def test_no_other_learning(cfg):
    with pytest.raises(ValueError):
        runner.check_only_government_q(cfg)


@pytest.fixture
def serialization_failure(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    episode = root / 'train_01'
    episode.mkdir(parents=True)
    script = str(Path(runner.__file__).resolve())
    before = {script: 'old-runner', 'core': 'unchanged'}
    after = {script: 'new-runner', 'core': 'unchanged'}
    monkeypatch.setattr(runner, 'source_hashes', lambda: after)
    monkeypatch.setattr(runner, 'load_base', lambda: ({}, {}, {}, {}))
    write = runner.old.write
    write(root / 'source_hashes.json', before)
    write(root / 'inputs.json', {})
    write(root / 'reward_reference.json', {'grain_reference_kg': 100.})
    write(episode / 'FAILED.json', {'error': 'Q requires five integer state coordinates and three finite values'})
    write(episode / 'q_before.json', [])
    write(episode / 'learning_parameters.json', {'learning_rate': .1, 'discount_factor': .95})
    write(episode / 'effective_config.json', {'experiment': {'seed': 17},
        'scale': {'n_farmers': 600, 'n_enterprises': 60},
        'policy': {'government_q_project': {'episode_end_step': 30, 'grain_reference_kg': 100.}}})
    write(episode / 'initial_households.json', [{'id': str(i)} for i in range(600)])
    write(episode / 'initial_exogenous.json', {'step': 0})
    for t in range(1, 31):
        year = episode / f'year_{t:02d}'
        year.mkdir()
        write(year / 'government_sar.json', dict(step=t, state=[0, 0, 0, 0, 31-t],
            next_state=[0, 0, 0, 0, 30-t], action=0, done=t == 30, q_before=0., q_after=.1,
            q_updated=True, reward=1., reward_weights=runner.WEIGHTS, grain_reference_kg=100.,
            outcome_observations={'h0_count': 600, 'grain_output_kg': 100.,
                                  'h0_mean_net_income_yuan': 20., 'fiscal_cost_yuan': 10.}))
        write(year / 'annual_checks.json', dict(land_max_error_mu=0., grain_vs_operator_snapshot_error_kg=0.,
            rent_max_error_yuan=0., provider_residuals=[],
            cash_residuals=[{'id': str(i), 'annual_residual_yuan': 0., 'harvest_residual_yuan': 0.} for i in range(600)]))
        write(year / 'operator_snapshot.json', {'grain_output_kg': 100.})
        write(year / 'fiscal_account.json', {'total_fiscal_cost_yuan': 10.})
        write(year / 'exogenous_shocks.json', {'step': t})
    return root, before, after


def test_recovery_reconstructs_q_preserves_original_and_reuses_without_build(serialization_failure, monkeypatch):
    root, before, after = serialization_failure
    path = root / 'train_01'
    original = {str(p): runner.old.digest(p) for p in path.rglob('*.json')}
    monkeypatch.setattr(runner, 'build_model', lambda *args: pytest.fail('recovery must not build'))
    result = runner.recover_serialization_failure(root)
    assert result['training_step_count_after'] == 30
    assert result['elapsed_seconds'] is None
    assert result['discounted_reward'] == pytest.approx(sum(.95 ** t for t in range(30)))
    assert result['cumulative_actual_grain_kg'] == 3000.
    table = runner.decode_q(runner.old.read(path / 'q_after.json'))
    assert len(table) == 30 and (0, 0, 0, 0, 0) not in table
    assert all(v == [.1, 0., 0.] for v in table.values())
    assert all(runner.old.digest(Path(p)) == sha for p, sha in original.items())
    record = runner.old.read(root / 'resume_source_hashes.json')
    assert record['before'] == before and record['after'] == after
    reused, q, count = runner.reuse_completed(path, 17, True, {}, 0, 100.)
    assert q == table and count == 30 and reused['years'] == 30
    with pytest.raises(FileExistsError):
        runner.recover_serialization_failure(root)


@pytest.mark.parametrize('change', ['core', 'inputs', 'incomplete', 'q_after'])
def test_recovery_rejects_unrelated_change_or_missing_evidence(serialization_failure, change):
    root, _, after = serialization_failure
    if change == 'core':
        after['core'] = 'changed'
    elif change == 'inputs':
        (root / 'inputs.json').write_text('{"changed": "input"}', encoding='utf-8')
    elif change == 'incomplete':
        (root / 'train_01/year_30/government_sar.json').unlink()
    else:
        row_path = root / 'train_01/year_30/government_sar.json'
        row = runner.old.read(row_path)
        row['q_after'] = 9.
        import json
        row_path.write_text(json.dumps(row), encoding='utf-8')
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        runner.recover_serialization_failure(root)
    assert not (root / 'train_01/COMPLETED.json').exists()


def test_resume_skips_completed_then_passes_q_count_and_reference(serialization_failure, monkeypatch):
    root, _, _ = serialization_failure
    runner.recover_serialization_failure(root)
    built = []
    def build(base, economics, seed):
        built.append(seed)
        return SimpleNamespace(government=SimpleNamespace(), config={
            'policy': {'government_q_project': {'grain_reference_kg': 999.}}})
    monkeypatch.setattr(runner, 'build_model', build)
    def configure(gov, table, seed, training, action, training_steps):
        assert seed == 23 and training and training_steps == 30 and len(table) == 30
    monkeypatch.setattr(runner, 'configure_government', configure)
    # Stop before any second-episode step; exercise the real resume dispatch.
    def stop(model, *args, **kwargs):
        assert model.config['policy']['government_q_project']['grain_reference_kg'] == 100.
        raise RuntimeError('test stop before next episode')
    monkeypatch.setattr(runner, 'run_episode', stop)
    import json
    # A production design is obtained from the fixture's same empty load_base values.
    design = dict(plan=runner.PLAN, weights=runner.WEIGHTS, income_reference_yuan=20000,
        reference_yield_kg_per_mu=549.913549, initial_economics={}, budget_reference={},
        assumption='stipulated reward references and weights; four episodes are not convergence',
        observation_scope='five discretized observations are partially observable, not a complete Markov state',
        reward_reference='first training initial grain area; same Gref for all eight episodes',
        fixed0_meaning='zero trial area rate; preserved 83 yuan rights-area support remains')
    (root / 'design.json').write_text(json.dumps(design), encoding='utf-8')
    with pytest.raises(RuntimeError, match='test stop'):
        runner.run(root, resume=True)
    assert built == [23]
