"""Assemble a clean public snapshot from allowlisted code and aggregate results."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.prepare_household_portfolio_reference import RAW, MANIFEST
from scripts.run_national_household_portfolio import legacy

D=Path(__file__).resolve().parent
R=D.parents[2]
S=D/'source'
O=D/'run_01'
P=D/'public'
rel=O.relative_to(R)
def digest(f): return hashlib.sha256(f.read_bytes()).hexdigest()
def save(f,obj): f.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n','utf-8')
def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)
def run(*args): return subprocess.check_output(args,cwd=S).decode('utf-8')
files=['core/model.py','core/fixed_rule_runtime.py','scripts/audit_household_portfolio_behavior.py','scripts/run_national_household_portfolio.py','tests/test_household_crop_portfolio.py']
tests=['test_government_q_project_20261119','test_household_crop_portfolio','test_n1_action_income','test_n1_lease_execution','test_national_actor_trace','test_national_plot_use_evidence','test_national_preproduction_cash_20261118','test_p3_project_payment_boundaries_20260909','test_p3_project_payments_20260909','test_run_national_government_q_20261119']
if __name__=='__main__':
    assert all(digest(R/f)==digest(S/f) for f in files)
    tree=ET.parse(O/'tests.xml')
    for suite in tree.iter('testsuite'):
        assert suite.get('errors')==suite.get('failures')==suite.get('skipped')=='0'
        suite.attrib.pop('hostname',None)
    assert len(list(tree.iter('testcase')))==192
    tree.write(O/'tests.xml',encoding='utf-8',xml_declaration=True)
    _,_,_,inputs=legacy.load_base()
    sources=[RAW,RAW.parent/'2009-2015individual.csv',RAW.parent/'household variables2009-2015.xlsx',RAW.parent/'2009-2017农户调查问卷.pdf',R/'config/experiments/national_real2024_cpi_20260909.csv']
    import pandas as pd
    import matplotlib.pyplot as plt
    structures=[]
    for v in ['baseline','candidate']:
        structures.append(pd.read_csv(D/'PRIVATE_DO_NOT_SHARE'/v/'structures.csv').assign(version=v))
    pd.concat(structures).to_csv(O/'structures.csv',index=False)
    p=pd.read_csv(O/'production_summary.csv')
    fig,ax=plt.subplots(figsize=(7,3.7))
    for (v,seed),a in p.groupby(['version','seed']):
        ax.plot(a.year,a.simulated_kg_on_reporting/1000,marker='.',label=f'{v} {seed}')
    ax.plot(a.year,a.observed_kg/1000,color='black',marker='o',label='Survey')
    ax.set(xticks=a.year,ylabel='Grain harvested (tonnes)',title='Same reporting households each year')
    ax.legend(ncol=3);fig.tight_layout();fig.savefig(O/'figures/harvest.png',dpi=150);plt.close(fig)
    copy(Path(__file__),S/D.relative_to(R)/'freeze_review.py')
    save(O/'source_manifest.json',{**{f:digest(R/f) for f in files},str((D/'freeze_review.py').relative_to(R).as_posix()):digest(Path(__file__))})
    save(O/'run_manifest.json',dict(parent='a8ef70620028b6031fedfe874cfdc7d5acd22dd6',years=list(range(2011,2016)),initial=2010,
        seeds=[103,109],repeat=103,policy='P0_rights_cultivated',training=False,households=600,organizations=60,
        protocols=dict(mixed_rate=.10,quantile_mae=.15,annual_change_error=.15,all_years_required=True),
        survey_inputs={p.name:digest(p) for p in sources},authorized_inputs=[dict(file=Path(p).name,sha256=h) for p,h in inputs.items()],
        economics=json.loads((D/'PRIVATE_DO_NOT_SHARE/candidate/inputs.json').read_bytes()),
        tests_command='python -B -X utf8 -m pytest -q '+' '.join('tests/'+t+'.py' for t in tests)+' --junitxml=OUT/run_01/tests.xml',
        software={k:__import__('importlib.metadata',fromlist=['version']).version(k) for k in ['mesa','numpy','pandas','scipy','pytest']},
        checks=dict(original_receipts_identical=10,repeat_receipts_identical=5,household_budget_rows=30000,passing_behavior_seed_years=6,total_behavior_seed_years=10)))
    for f in O.rglob('*'):
        if f.is_file(): copy(f,S/rel/f.relative_to(O))
    # Use no git objects or unreviewed historical evidence in the public repository.
    P.mkdir(exist_ok=True)
    paths=run('git','ls-files','-z').split('\0')
    for name in filter(None,paths):
        f=Path(name)
        if f.suffix=='.py' or name in ('AGENTS.md','requirements.txt','config/experiments/v4_8_national_fixed_rule.yaml','config/policy_params.json'):
            copy(S/f,P/f)
    for name in ('SCIENTIFIC_QUESTION.md','config/experiments/national_real2024_cpi_20260909.csv'):
        copy(R/name,P/name)
    for f in O.rglob('*'):
        if f.is_file(): copy(f,P/rel/f.relative_to(O))
    copy(O/'README.md',P/'README.md')
    copy(Path(__file__),P/D.relative_to(R)/'freeze_review.py')
    for f in [Path('HANDOFF.md'),Path('docs/全国模型主体规则与过程追溯审查.md')]:
        txt=(S/f).read_text('utf-8').split('\n## ',2)
        dst=P/f;dst.parent.mkdir(exist_ok=True);dst.write_text('\n## '.join(txt[:2])+'\n','utf-8')
    for f in P.rglob('*'):
        if not f.is_file() or '.git' in f.parts: continue
        assert 'PRIVATE_DO_NOT_SHARE' not in f.parts
        if f.suffix=='.png': continue
        t=f.read_text('utf-8-sig')
        assert not re.search(r'(?:[A-Za-z]:[/\\](?:Users|HHDD)|/home/[^ /]+/|gh[pousr]_[a-zA-Z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)',t,re.I),f
