"""Same-row rice conditions through the original account; no cash calibration."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.n1_income_fixtures import HAND_CASES, hand_farmer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/diagnostics/national_crop_budget_basis_20261118'
OUT = BASE / 'run_03'
ACTIONS = ('maintain_grain', 'expand_cash', 'exit')


def accounts(farmer):
    rows = []
    for action in ACTIONS:
        a = farmer.expected_fixed_rule_account(action)
        rows.append(dict(asdict(a), net_income=a.net_income, cash_cost=a.cash_cost))
    return rows


def compare(farmer, row):
    production = farmer.model.config['production']
    labor = production['cost_structure']['input_requirements']['labor_days']
    before = accounts(farmer)
    old = (production['base_yield']['grain'], labor['grain'], labor['grain_farmer'])
    try:
        production['base_yield']['grain'] = float(row.output_per_mu)
        labor['grain'] = labor['grain_farmer'] = float(row.labor / row.area)
        after = accounts(farmer)
    finally:
        production['base_yield']['grain'], labor['grain'], labor['grain_farmer'] = old
    assert before[1:] == after[1:]
    assert accounts(farmer) == before
    return before, after


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    sources = [BASE / 'run_01/joint_crop_budgets.csv',
               BASE / 'run_02/pair_return_conditions.csv',
               ROOT / 'agents/farmer_agent.py', ROOT / 'core/fixed_rule_runtime.py',
               ROOT / 'scripts/n1_income_fixtures.py', Path(__file__),
               ROOT / 'config/master_config.yaml',
               ROOT / 'config/experiments/v4_8_national_fixed_rule.yaml']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    budgets = pd.read_csv(sources[0])
    eligible = budgets.crop.eq('稻谷') & budgets.family_joint & budgets.output_per_mu.gt(0) & budgets.labor.ge(0)
    selected = budgets.loc[eligible]
    assert len(selected) > 0
    baseline = []
    for case in HAND_CASES:
        rows = accounts(hand_farmer(case))
        assert all(abs(a['net_income'] - target) < 1e-6 for a, target in zip(rows, (case.grain_net, case.cash_net, case.exit_net)))
        baseline.extend(dict(role=case.role, **a) for a in rows)
    farmer = hand_farmer(HAND_CASES[0])
    results = []
    for row in selected.itertuples():
        before, after = compare(farmer, row)
        for condition, values in [('original_hand_baseline', before), ('same_row_rice_yield_total_labor', after)]:
            for a in values:
                results.append(dict(source_row=row.source_row, year=row.year, sm=row.sm, cm=row.cm, hm=row.hm,
                    condition=condition, evidence='synthetic_10mu_household_not_observed_household',
                    observed_income_per_mu=row.income_per_mu, observed_total_cost_per_mu=row.total_cost_per_mu,
                    observed_family_days_per_mu=row.family_days_per_mu,
                    observed_hired_cost_per_mu=row.hired_cost_per_mu, observed_rent_per_mu=row.rent_per_mu,
                    observed_margin_per_mu=row.reported_margin_per_mu,
                    **a))
    mapping = [
        ('output_kg', 'output/area', 'conditional_only', 'Rice only; normalized existing hand fixture; not composite grain calibration'),
        ('sales_yuan', 'income; sales_amount; byproduct', 'blocked', 'Gross production value is not cash sales; no composite crop weights'),
        ('production_cost_yuan', 'total_cost', 'blocked', 'Includes manure valuation, depreciation/repair, hired labor and rent; never assign to cash cost'),
        ('hired_labor_cost_yuan', 'hired_cost', 'reference_only', 'Already in total_cost; do not deduct twice'),
        ('rent_in_yuan', 'rent', 'reference_only', 'Already in total_cost; no household tenure reconstruction'),
        ('family_labor_days', 'labor-hired_days', 'reference_only', 'Observed family days do not identify total household labor endowment'),
        ('labor_days.grain_farmer', 'labor/area', 'conditional_only', 'Same-row total demand; model allocates family/hired labor'),
        ('off_farm_yuan', 'missing marginal employment opportunity', 'blocked', 'Use original employment cap; do not subtract family opportunity cost again'),
        ('exit', 'household land and nonfarm alternatives', 'blocked', 'Preserve original account, not an observed NFPS exit counterfactual'),
        ('cash_crop', 'composite crop composition and weights', 'blocked', 'Keep original composite category and assumptions; no vegetable substitution'),
    ]
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in sources)
    OUT.mkdir(parents=True)
    pd.DataFrame(baseline).to_csv(OUT / 'original_baseline.csv', index=False)
    pd.DataFrame(results).to_csv(OUT / 'original_function_conditions.csv', index=False)
    pd.DataFrame(mapping, columns=['model_field', 'empirical_field', 'adoption', 'reason']).to_csv(OUT / 'field_mapping.csv', index=False)
    budgets.assign(original_function_condition_eligible=eligible).to_csv(OUT / 'budget_eligibility.csv', index=False)
    (OUT / 'source_hashes.json').write_text(json.dumps(dict(hashes=hashes, budget_rows=len(budgets),
        selected_rice_rows=len(selected), excluded_from_condition=len(budgets)-len(selected),
        account_rows=len(results), raw_rows_deleted=0, formal_parameter_changes=0), indent=2), encoding='utf-8')
    print(f'{len(selected)} rice rows; {len(results)} original-function accounts; 18 baseline accounts verified')


if __name__ == '__main__':
    run()
