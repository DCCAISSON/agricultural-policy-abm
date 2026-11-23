"""Generate explicit sample-area budgets, without changing default parameters."""
from pathlib import Path
import hashlib
import json
from copy import deepcopy

import pandas as pd
import yaml

from scripts.n1_income_fixtures import HAND_CASES, hand_farmer
from scripts.national_crop_budget_bridge_20261118 import accounts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/diagnostics/national_crop_budget_basis_20261118/run_01/joint_crop_budgets.csv'
OUT = ROOT / 'results/national_crop_budget_basis_20261118/run_04'
FIELDS = ['area', 'income', 'total_cost', 'manure', 'depreciation_repair', 'rent', 'hired_cost', 'labor']
PRICES = {'grain': 2.8, 'cash_crop': 2.0}


def derive(b):
    b = b.copy()
    b['nonlabor_low'] = b.total_cost-b.manure-b.depreciation_repair-b.rent-b.hired_cost
    b['nonlabor_high'] = b.total_cost-b.manure-b.rent-b.hired_cost
    b['eligible'] = (b[FIELDS].notna().all(axis=1) & b[FIELDS].ge(0).all(axis=1)
                     & b.area.gt(0) & b.nonlabor_low.ge(0))
    b['category'] = b.crop.map({'稻谷': 'grain', '蔬菜': 'cash_crop', '水果': 'cash_crop'})
    s = b.loc[b.eligible]
    summary = []
    for category, g in s.groupby('category'):
        area = float(g.area.sum())
        summary.append(dict(category=category, records=len(g), area=area,
            revenue=float(g.income.sum()/area), labor=float(g.labor.sum()/area),
            cost_low=float(g.nonlabor_low.sum()/area), cost_high=float(g.nonlabor_high.sum()/area),
            equivalent_output=float(g.income.sum()/area/PRICES[category])))
    assert {r['category'] for r in summary} == set(PRICES)
    return b, summary


def overlay(summary, bound):
    values = {r['category']: r for r in summary}
    return {'budget_basis': {'id': 'nfps_sample_area_2016_2017_'+bound,
        'output_unit': 'revenue_equivalent_kg_not_physical_yield',
        'reference_prices_yuan_per_kg': PRICES,
        'weights': 'eligible_sample_sown_area_not_national',
        'repair_cash_fraction': 1 if bound == 'high' else 0,
        'income_assumption': 'reported_production_value_fully_realized_at_reference_price',
        'machinery_assumption': 'observed_machinery_in_base_cost_additional_services_incremental'},
        'production': {'base_yield': {k: v['equivalent_output'] for k,v in values.items()},
            'cost_structure': {'use_detailed_cost': True,
                'input_requirements': {
                    **{f: {k: 0.0 for k in values} for f in ['seed','fertilizer','pesticide','small_tools']},
                    'other_cost_per_mu': {k: v['cost_'+bound] for k,v in values.items()},
                    'labor_days': {key: v['labor'] for k,v in values.items() for key in (k,k+'_farmer')}}}}}


def apply_budget(farmer, patch):
    # Apply only existing production interfaces; metadata is not a runtime switch.
    p = farmer.model.config['production']
    p['base_yield'].update(patch['production']['base_yield'])
    p['cost_structure']['use_detailed_cost'] = True
    for key, value in patch['production']['cost_structure']['input_requirements'].items():
        p['cost_structure']['input_requirements'].setdefault(key, {}).update(value)


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    b, summary = derive(pd.read_csv(SOURCE))
    patches = {bound: overlay(summary, bound) for bound in ('high', 'low')}
    outputs = []
    for case in (HAND_CASES[0], HAND_CASES[3]):
        f = hand_farmer(case)
        old = deepcopy(f.model.config['production'])
        for name in ('original', 'high', 'low'):
            f.model.config['production'] = deepcopy(old)
            if name != 'original':
                apply_budget(f, patches[name])
            outputs.extend(dict(condition=name, household=case.role, **a) for a in accounts(f))
    OUT.mkdir(parents=True)
    b.to_csv(OUT/'record_selection.csv', index=False)
    pd.DataFrame(summary).to_csv(OUT/'parameters.csv', index=False)
    s = b.loc[b.eligible]
    composition = s.groupby(['category','crop']).agg(records=('area','size'),area=('area','sum'))
    composition['weight'] = composition.area / composition.groupby(level=0).area.transform('sum')
    composition.to_csv(OUT/'composition.csv')
    pd.DataFrame(outputs).to_csv(OUT/'original_function_comparison.csv', index=False)
    paths = [SOURCE, Path(__file__), ROOT/'agents/farmer_agent.py', ROOT/'core/fixed_rule_runtime.py', ROOT/'config/master_config.yaml']
    for bound, patch in patches.items():
        path = ROOT/f'config/experiments/national_crop_budget_20261118_{bound}.yaml'
        if path.exists():
            raise FileExistsError(path)
        path.write_text(yaml.safe_dump(patch, allow_unicode=True, sort_keys=False), encoding='utf-8')
        paths.append(path)
    (OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2), encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print(composition.to_string())


if __name__ == '__main__':
    run()
