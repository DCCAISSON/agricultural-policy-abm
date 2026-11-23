"""2024 budget revision with physical rice yield and persistent annual anchors."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from scripts import national_crop_budget_overlay_20261118 as old
from scripts.national_real2024 import cpi_factor, CPI_SOURCE
from scripts.paper_b_flagship_scenarios import apply_annual_exogenous_path

ROOT = old.ROOT
OUT = old.OUT / 'real2024_revision'
FORMAL = ROOT / 'results/empirical/three_paper_execution_20260908/national/p3_formal_inputs_20260909_r3/common_pre_policy_config.json'


def build():
    config = json.loads(FORMAL.read_text(encoding='utf-8'))
    assert config['monetary_representation']['price_year'] == 2024
    prices = {k:float(v)*2 for k,v in config['environment']['market']['base_prices'].items() if k in ('grain','cash_crop')}
    raw = pd.read_csv(old.SOURCE)
    converted = raw.copy()
    converted['to_2024'] = converted.year.map(lambda y: cpi_factor(int(y)))
    for field in ('income','total_cost','manure','depreciation_repair','rent','hired_cost'):
        converted[field] = raw[field] * converted.to_2024
    records, _ = old.derive(converted)
    # Physical output must be present on the same rice budget record.
    records['eligible'] &= ~records.crop.eq('稻谷') | (records.output.notna() & records.output.ge(0))
    summaries = []
    for category,g in records.loc[records.eligible].groupby('category'):
        area = g.area.sum()
        revenue = g.income.sum()/area
        output = g.output.sum()/area if category == 'grain' else revenue/prices[category]
        summaries.append(dict(category=category, records=len(g), area=float(area),
            reported_value_2024_per_mu=float(revenue), reference_price_per_kg=prices[category],
            output_per_mu=float(output), output_unit='physical_kg' if category=='grain' else 'revenue_equivalent_kg',
            modeled_reference_sales_per_mu=float(output*prices[category]),
            sales_minus_reported_value=float(output*prices[category]-revenue),
            cost_low=float(g.nonlabor_low.sum()/area), cost_high=float(g.nonlabor_high.sum()/area),
            labor=float(g.labor.sum()/area), equivalent_output=float(output)))
    patches = {}
    for bound in ('high','low'):
        patch = old.overlay(summaries,bound)
        patch['budget_basis'].update(id='nfps_real2024_physical_grain_'+bound,
            price_year=2024, reference_prices_yuan_per_kg=prices,
            operator_comparability='enterprise_extra_machinery_and_labor_substitution_and_scale_savings_are_research_assumptions_not_NFPS_calibrated; farmer_scale_skill_service_adjustments_are_additional_assumptions',
            output_unit={'grain':'physical_kg','cash_crop':'revenue_equivalent_kg_not_physical'},
            income_assumption='grain_physical_yield_times_reference_price; cash_reported_value_realized',
            annual_anchor_application='after_scenario_setup_before_year_zero')
        patch['paper_b_flagship_scenario'] = {
            'initial_grain_other_cost':next(r['cost_'+bound] for r in summaries if r['category']=='grain'),
            'initial_cash_other_cost':next(r['cost_'+bound] for r in summaries if r['category']=='cash_crop')}
        patches[bound] = patch
    return records,summaries,patches,config


def apply_real2024_budget(config, patch):
    if config.get('monetary_representation',{}).get('price_year') != 2024:
        raise ValueError('Budget requires 2024 monetary configuration')
    old.apply_budget(SimpleNamespace(model=SimpleNamespace(config=config)),patch)
    config['paper_b_flagship_scenario'].update(patch['paper_b_flagship_scenario'])


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    records,summary,patches,config = build()
    checks=[]
    for bound,patch in patches.items():
        c=deepcopy(config)
        c['paper_b_flagship_scenario']={'input_cost_growth_rate':0.03,'rental_growth_rate':0.025,'initial_land_rent':288.09138703800744}
        apply_real2024_budget(c,patch)
        model=SimpleNamespace(config=c,environment=SimpleNamespace(market_prices={}))
        for year in (0,1,2):
            apply_annual_exogenous_path(model,year)
            for category,key in [('grain','initial_grain_other_cost'),('cash_crop','initial_cash_other_cost')]:
                actual=c['production']['cost_structure']['input_requirements']['other_cost_per_mu'][category]
                expected=patch['paper_b_flagship_scenario'][key]*1.03**year
                assert abs(actual-expected)<1e-8
                checks.append(dict(bound=bound,year_index=year,category=category,actual=actual,expected=expected))
    OUT.mkdir()
    records.to_csv(OUT/'record_selection_money2024.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'parameters_money2024.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'annual_anchor_checks.csv',index=False)
    paths=[Path(__file__),old.SOURCE,FORMAL,CPI_SOURCE,ROOT/'scripts/national_real2024.py',ROOT/'scripts/paper_b_flagship_scenarios.py']
    for bound,patch in patches.items():
        path=ROOT/f'config/experiments/national_crop_budget_20261118_real2024_{bound}.yaml'
        with path.open('x',encoding='utf-8') as f:
            yaml.safe_dump(patch,f,allow_unicode=True,sort_keys=False)
        paths.append(path)
    (OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print({year:cpi_factor(year) for year in (2016,2017)})


if __name__=='__main__':
    run()
