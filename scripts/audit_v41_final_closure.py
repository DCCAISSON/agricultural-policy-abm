"""Reconcile F1 and the frozen controls without rerunning any control model."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_household_portfolio_behavior import audit_v41
from scripts.audit_nfps19_continuity import audit_continuity
from scripts.prepare_nfps_v3 import RAW, candidate_cohort, prepare_scale_inputs


def crop_state(grain, cash, other=None):
    missing=grain.isna()|cash.isna()
    g,c=grain.gt(1e-7),cash.gt(1e-7)
    if other is None:
        other=pd.Series(0.,index=grain.index)
    return np.select([missing,g&c,g,c,other.gt(0),other.isna()],
                     ['unclassified','mixed','grain','cash','other','unclassified'],default='unsown')


def survey_sources(panel):
    """Use observed fields only; a missing crop input does not mean no input."""
    codes=('w','r','c','s','t','o')
    prefixes=('sow','labor_d','labor_hd','labor_hf','mach')
    dictionary=pd.read_excel(RAW.parent/'household variables2009-2015.xlsx',header=None).set_index(0)[1]
    for field,term in [('sow_w','播种面积'),('labor_d_w','投工量'),('labor_hd_w','雇工'),
                       ('labor_hf_w','雇工费用'),('mach_w','机械作业费用')]:
        assert term in dictionary[field]
    cols=[f'{kind}_{crop}' for kind in prefixes for crop in codes]
    raw=pd.read_csv(RAW,usecols=['year','hhid',*cols],dtype={'hhid':'string'})
    raw=raw.loc[~raw.duplicated(['year','hhid'],False)]
    raw=panel[['year','hhid','actor']].merge(raw,on=['year','hhid'],validate='1:1')
    labor=[]
    for crop in codes:
        a,days,hire,cost,mach=(raw[f'{kind}_{crop}'] for kind in prefixes)
        eligible=a.gt(0)&days.ge(0)&hire.ge(0)&days.ge(hire)
        for y,t in raw[a.gt(0)].groupby('year'):
            i=t.index[eligible.loc[t.index]]
            j=t.index[cost.loc[t.index].ge(0)]
            k=t.index[mach.loc[t.index].ge(0)]
            labor.append(dict(year=y,crop=crop,positive_sown=len(t),valid_labor=len(i),
                joint_mu=a.loc[i].sum(),total_days=days.loc[i].sum(),hired_days=hire.loc[i].sum(),
                family_days=(days-hire).loc[i].sum(),hire_fee_n=len(j),hire_fee_nominal=cost.loc[j].sum(),
                machine_n=len(k),machine_fee_nominal=mach.loc[k].sum(),
                service_coverage='not identified by expenditure'))
    return pd.DataFrame(labor)


def survey_members(panel):
    """Separate observed activity days from the empirical ceiling used to plan."""
    cols=['year','hhid','idind','age','homefarmdays','homenonfarmdays','timeout']
    data=pd.read_csv(RAW.parent/'2009-2015individual.csv',usecols=cols,dtype={'hhid':'string','idind':'string'})
    data=data[data.age.between(16,79)&~data.duplicated(['year','hhid','idind'],False)]
    data=data.merge(panel[['year','hhid']],on=['year','hhid'],validate='m:1')
    for field in cols[4:]:
        data[field]=data[field].where(data[field].between(0,365))
    data['nonfarm_time_proxy']=data[['homenonfarmdays','timeout']].max(axis=1)
    return data.groupby('year').agg(member_records=('age','size'),families=('hhid','nunique'),
        observed_farm_days_n=('homefarmdays','count'),local_nonfarm_days_n=('homenonfarmdays','count'),
        migrant_days_n=('timeout','count'),nonfarm_time_proxy_n=('nonfarm_time_proxy','count'),
        median_nonfarm_time_proxy=('nonfarm_time_proxy','median'))


def yield_reference(chfs_2015: Path) -> pd.DataFrame:
    rows=[]
    for label,k in [('rice',1),('wheat',2),('maize',3)]:
        a,u,q=(f'b1004{suffix}_{k}' for suffix in 'def')
        with pd.io.stata.StataReader(chfs_2015) as reader:
            labels=reader.variable_labels()
        assert '播种面积' in labels[a] and '面积记录单位' in labels[u] and '总产量' in labels[q]
        d=pd.read_stata(chfs_2015,columns=[a,u,q],convert_categoricals=False)
        # CHFS 2015 questionnaire B1004e: 1 = mu. Other units are not pooled.
        mu=d[u].eq(1)
        ok=mu&d[a].gt(0)&d[q].gt(0)
        if not ok.any():
            raise ValueError(f'No verified mu-coded CHFS records for {label}')
        x=d.loc[ok,q]/d.loc[ok,a]
        rows.append(dict(crop=label,production_year=2014,source_file=chfs_2015.name,
                         area_field=a,unit_field=u,output_field=q,recorded_area_mu_only=True,
                         n=len(x),p25=x.quantile(.25),median=x.median(),p75=x.quantile(.75),
                         usage='external comparison only; never imputed to NFPS'))
    return pd.DataFrame(rows)


def audit_final(dest: Path, frozen: Path, chfs_2015: Path) -> None:
    public=dest/'run_01'
    private=dest/'PRIVATE_DO_NOT_SHARE'
    prior=frozen/'PRIVATE_DO_NOT_SHARE'
    audit_v41(dest,final=True,frozen=frozen)
    cohort=candidate_cohort()
    panel,*_=prepare_scale_inputs(cohort)
    panel['status']=crop_state(panel.sow_food,panel.sow_cash,panel.sow_oother)
    panel['area_issue']=np.select([panel.area_e.isna(),panel.area_e.lt(0),
         panel.area_e.eq(0)&(panel.sow_food+panel.sow_cash).gt(1e-7)],
         ['closing_area_missing','closing_area_negative','zero_area_with_positive_sowing'],default='usable')
    panel['positive_two_crop']=(panel.sow_food+panel.sow_cash).gt(1e-7)
    omitted=panel[panel.year.between(2011,2015)&panel.area_issue.ne('usable')].copy()
    assert len(omitted[omitted.year.eq(2015)])==69
    assert omitted[omitted.year.eq(2015)].positive_two_crop.sum()==12
    omitted[['year','actor','area_issue','positive_two_crop','area_b','area_e','area_b1','area_e1',
             'sow_food','sow_cash','sow_total','area_in1','area_de2']].to_csv(private/'land_exclusion_cases.csv',index=False)
    omitted.groupby(['year','area_issue','positive_two_crop']).size().rename('households').to_csv(public/'land_exclusion_reasons.csv')
    survey_sources(panel).to_csv(public/'labor_survey_accounts.csv',index=False)
    survey_members(panel).to_csv(public/'labor_member_coverage.csv')
    yield_reference(chfs_2015).to_csv(public/'chfs_crop_yield_reference.csv',index=False)
    history_file=private/'rice_history_19.csv'
    ids=pd.read_csv(history_file) if history_file.is_file() else audit_continuity(private)
    ids.groupby('finding').size().rename('cases').to_csv(public/'rice_history_19_summary.csv')
    receipts=[]
    for name in ('1986-2008household.csv','2003-2008individual.csv','2009-2015individual.csv',
                 '2009-2017农户调查问卷.txt','household variables2009-2015.xlsx'):
        receipts.append(dict(input=name,sha256=sha256((RAW.parent/name).read_bytes()).hexdigest()))
    receipts.extend([dict(input=chfs_2015.name,sha256=sha256(chfs_2015.read_bytes()).hexdigest()),
        dict(input='reused baseline annual_behavior.csv',
             sha256=sha256((frozen/'run_01/annual_behavior.csv').read_bytes()).hexdigest())])
    from os import environ
    from shutil import copyfile
    for name in ('production_parameters.csv','cash_budget_ranges.csv','labor_parameters.csv',
                 'liquidity_parameters.csv','initial_coverage.csv'):
        source=private/'F1_diag'/name
        if source.is_file():
            copyfile(source,public/name)
    cfps=next((RAW.parents[1]/'CFPS/2010').rglob('cfps2010adult*.dta'))
    cpi=Path(__file__).resolve().parents[1]/'config/experiments/national_real2024_cpi_20260909.csv'
    additional=[cfps,cpi]
    if environ.get('AP_ABM_CHFS2011'):
        additional.append(Path(environ['AP_ABM_CHFS2011']))
    receipts.extend(dict(input=path.name,sha256=sha256(path.read_bytes()).hexdigest()) for path in additional)
    from scripts.run_national_government_q_20261119 import load_base
    import json
    config,*_,base_hashes=load_base()
    receipts.extend(dict(input=Path(name).name,sha256=value) for name,value in base_hashes.items())
    receipts.append(dict(input='frozen merged base configuration, canonical JSON',
                         sha256=sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()))
    pd.DataFrame(receipts).to_csv(public/'additional_input_receipt.csv',index=False)

    cases=[(s,(private if s=='F1' else prior)/f'{s}_{k}') for s in ('M0','R0','R3','F1')
           for k in ('diag',301,306,311,316)]
    stock_rows=[]; transfers=[]; economics=[]; organizations=[]; compare=[]
    for spec,path in cases:
        ledger=path/'PRIVATE_DO_NOT_SHARE'
        init=pd.read_csv(ledger/'baseline.csv').set_index('actor')
        obs=panel[['year','actor','status','area_b','area_e','area_b1','area_e1','area_in1','area_de2',
                   'sow_total','sow_food','sow_cash','sow_oother']].rename(columns={'actor':'id','status':'observed_state'})
        for receipt in sorted(ledger.glob('*_both_*_accounts.csv')):
            seed=int(receipt.name.split('_')[0])
            all_actors=pd.read_csv(receipt)
            ordered=all_actors.sort_values(['id','year'])
            carry=ordered.groupby('id').closing.shift()
            np.testing.assert_allclose(ordered.loc[carry.notna(),'opening'],carry.dropna(),atol=1e-7)
            fees=all_actors.get('transaction_cost',pd.Series(0.,index=all_actors.index)).fillna(0.)
            surplus=all_actors.sales-all_actors.input_hire_cost-all_actors.rent_paid-all_actors.overhead+all_actors.rent_received-fees
            np.testing.assert_allclose(surplus,all_actors.surplus,atol=1e-7)
            np.testing.assert_allclose(surplus+all_actors.paid+all_actors.nonfarm,all_actors.net,atol=1e-7)
            hh=all_actors[all_actors.kind.eq('household')].copy()
            if 'crop_kg' in hh:
                for area,quantity in [('grain','grain_kg'),('crop','crop_kg')]:
                    positive=hh[area].gt(1e-7)
                    assert hh.loc[positive,quantity].gt(0.).all() and hh.loc[~positive,quantity].abs().lt(1e-3).all()
                    y=hh.loc[positive,quantity]/(hh.loc[positive,area]*hh.loc[positive,'weather_factor'])
                    np.testing.assert_allclose(y,y.groupby(hh.id).transform('first'),atol=1e-7)
            contracts=pd.read_csv(ledger/receipt.name.replace('_accounts','_contracts'))
            orders=pd.read_csv(ledger/receipt.name.replace('_accounts','_orders'))
            channel=contracts.get('channel',pd.Series('internal',index=contracts.index)).fillna('internal')
            for side,actor in [('in','buyer'),('out','seller')]:
                for market in ('internal','external'):
                    use=channel.eq('internal') if market=='internal' else channel.ne('internal')
                    amounts=contracts[use].groupby(['start',actor]).area.sum()
                    hh[f'{market}_{side}']=pd.MultiIndex.from_frame(hh[['year','id']]).map(amounts).fillna(0.)
                np.testing.assert_allclose(hh[f'internal_{side}']+hh[f'external_{side}'],hh[f'rented_{side}'],atol=1e-7)
            assert not orders.duplicated(['year','id']).any()
            np.testing.assert_allclose(hh.base_area+hh.rented_in-hh.rented_out,hh.operated,atol=1e-7)
            np.testing.assert_allclose(hh.cultivated+hh.idle,hh.operated,atol=1e-7)
            hh['sim_state']=crop_state(hh.grain,hh.crop)
            hh['initial_operated']=hh.id.map(init.area_e)
            data=hh.merge(obs,on=['year','id'],how='left',validate='1:1')
            states=pd.concat([data[['year','id','sim_state']],
                init.reset_index().rename(columns={'actor':'id'}).assign(year=2010,
                    sim_state=crop_state(init.sow_food,init.sow_cash,init.get('sow_oother')))
                    [['year','id','sim_state']]],ignore_index=True)
            states=states.merge(obs[['year','id','observed_state']],on=['year','id'],how='left',validate='1:1')
            states=states.sort_values(['id','year'])
            states[['from_model','from_survey']]=states.groupby('id')[['sim_state','observed_state']].shift()
            for y,d in states[states.year.gt(2010)].groupby('year'):
                use=d[['from_survey','observed_state']].notna().all(axis=1)&~d[['from_survey','observed_state']].eq('unclassified').any(axis=1)
                k=d[use]
                for source,i,j in [('survey','from_survey','observed_state'),('model','from_model','sim_state')]:
                    for (a,b),n in k.groupby([i,j]).size().items():
                        transfers.append(dict(spec=spec,seed=seed,year=y,source=source,from_state=a,to_state=b,
                                              households=n,paired_households=len(k)))
                mixed=lambda v:v.eq('mixed')
                compare.append(dict(spec=spec,seed=seed,year=y,matched_pair=len(k),
                    model_new=(~mixed(k.from_model)&mixed(k.sim_state)).sum(),
                    survey_new=(~mixed(k.from_survey)&mixed(k.observed_state)).sum(),
                    model_persist=(mixed(k.from_model)&mixed(k.sim_state)).sum(),
                    survey_persist=(mixed(k.from_survey)&mixed(k.observed_state)).sum(),
                    model_exit=(mixed(k.from_model)&~mixed(k.sim_state)).sum(),
                    survey_exit=(mixed(k.from_survey)&~mixed(k.observed_state)).sum(),
                    model_prior_mixed=mixed(k.from_model).sum(),survey_prior_mixed=mixed(k.from_survey).sum()))
            if spec in ('R3','F1'):
                data.to_csv(private/f'{spec}_land_bridge_{seed}.csv',index=False)
            for y,d in data.groupby('year'):
                stocks=d.area_b.ge(0)&d.area_e.ge(0)
                both_flows=d.area_in1.ge(0)&d.area_de2.ge(0)
                sums=d.loc[stocks]
                raw=(sums.area_e-sums.area_b).sum()
                model=(sums.operated-sums.base_area).sum()
                assert np.isclose((sums.area_b-sums.base_area).sum()+raw-model,(sums.area_e-sums.operated).sum())
                stock_rows.append(dict(spec=spec,seed=seed,year=y,matched_stock=len(sums),
                    survey_open=sums.area_b.sum(),survey_end=sums.area_e.sum(),survey_stock_change=raw,
                    model_base=sums.initial_operated.sum(),model_open=sums.base_area.sum(),model_end=sums.operated.sum(),
                    opening_gap=(sums.area_b-sums.base_area).sum(),within_year_gap=raw-model,
                    end_gap=(sums.area_e-sums.operated).sum(),survey_both_flow_n=both_flows.sum(),
                    survey_joint_flow=(d.loc[both_flows,'area_in1']-d.loc[both_flows,'area_de2']).sum(),
                    model_joint_reporter_flow=(d.loc[both_flows,'rented_in']-d.loc[both_flows,'rented_out']).sum(),
                    rights_open_n=d.area_b1.ge(0).sum(),rights_end_n=d.area_e1.ge(0).sum(),model_verified_right=d.rights.sum(),
                    model_offered=d.offer.sum(),model_idle=d.idle.sum(),model_internal_in=d.internal_in.sum(),
                    model_internal_out=d.internal_out.sum(),model_external_in=d.external_in.sum(),model_external_out=d.external_out.sum()))
                if 'agricultural_labor_requirement' in d:
                    np.testing.assert_allclose(d.family_used+d.hired_days,d.agricultural_labor_requirement,atol=1e-7)
                    np.testing.assert_allclose(d.family_used,np.minimum(d.family_capacity,d.agricultural_labor_requirement),atol=1e-7)
                    np.testing.assert_allclose(d.nonlabor_cost+d.hired_cost,d.input_hire_cost,atol=1e-7)
                    np.testing.assert_allclose(d.hired_cost,d.hired_days*config['labor_market']['base_agricultural_wage'],atol=1e-7)
                economics.append(dict(spec=spec,seed=seed,year=y,family_days=d.family_used.sum(min_count=1),
                    available_days=d.family_capacity.sum(min_count=1),hired_days=d.hired_days.sum(),
                    hired_cost=d['hired_cost'].sum(min_count=1) if 'hired_cost' in d else np.nan,
                    labor_requirement=d['agricultural_labor_requirement'].sum(min_count=1) if 'agricultural_labor_requirement' in d else np.nan,
                    material_and_embedded_machine_cost=d['nonlabor_cost'].sum(min_count=1) if 'nonlabor_cost' in d else np.nan,
                    standalone_service_cost=0.,separately_identifiable_machine_cost=np.nan,service_coverage=np.nan,
                    wages=d.nonfarm.sum(),crop_sales=d.sales.sum(),farm_surplus=d.surplus.sum(),
                    rent_paid=d.rent_paid.sum(),rent_received=d.rent_received.sum(),
                    support=d.paid.sum(),household_net=d.net.sum(),end_cash=d.closing.sum()))
                o=all_actors[(all_actors.year.eq(y))&~all_actors.kind.eq('household')]
                for kind,h in o.groupby('kind'):
                    organizations.append(dict(spec=spec,seed=seed,year=y,kind=kind,organizations=len(h),
                         bidders=h.demand.gt(1e-7).sum(),transacted=h.rented_in.gt(1e-7).sum(),
                         leased_mu=h.rented_in.sum(),operated_mu=h.operated.sum(),grain_kg=h.grain_kg.sum(),
                         surplus=h.surplus.sum(),fixed_and_management_cost=h.overhead.sum(),exits=h.exited.sum()))
    mixed=pd.DataFrame(compare)
    r=mixed.model_persist/mixed.model_prior_mixed.replace(0,np.nan)
    s=mixed.survey_persist/mixed.survey_prior_mixed.replace(0,np.nan)
    mixed['persist_gap_from_prior_count']=(mixed.model_prior_mixed-mixed.survey_prior_mixed)*(r+s)*.5
    mixed['persist_gap_from_persistence_rate']=(r-s)*(mixed.model_prior_mixed+mixed.survey_prior_mixed)*.5
    covered=r.notna()&s.notna()
    np.testing.assert_allclose((mixed.persist_gap_from_prior_count+mixed.persist_gap_from_persistence_rate)[covered],
                               (mixed.model_persist-mixed.survey_persist)[covered],atol=1e-7)
    mixed.to_csv(public/'mixed_sources.csv',index=False)
    for name,records in [('land_stock_bridge',stock_rows),('crop_transitions',transfers),
                         ('labor_cash_and_service',economics),('organization_accounts',organizations)]:
        pd.DataFrame(records).to_csv(public/f'{name}.csv',index=False)
    r0=pd.read_csv(frozen/'run_01/annual_behavior.csv')
    new=pd.read_csv(public/'annual_behavior.csv')
    compare_cols=['spec','year','seed','gate','obs_grain','sim_grain','sim_active','sim_mixed']
    left=r0[r0.spec.isin(['M0','R0','R3'])][compare_cols].sort_values(compare_cols[:3]).reset_index(drop=True)
    right=new[new.spec.isin(['M0','R0','R3'])][compare_cols].sort_values(compare_cols[:3]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left,right)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('frozen_v41',type=Path)
    parser.add_argument('chfs_2015',type=Path)
    args=parser.parse_args()
    audit_final(args.destination,args.frozen_v41,args.chfs_2015)
