"""Reconcile the paired continuous-share candidate against frozen v1 and matched survey years."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.audit_crop_time_comparability import OUT
from scripts.audit_national_crop_mix_drift import BASE,P,sha,share
from scripts.audit_household_portfolio_behavior import audit_area_receipts
from scripts.prepare_household_portfolio_reference import RAW,MANIFEST
from scripts.run_national_household_portfolio import estimate_crop_transitions
from core.fixed_rule_runtime import continuous_crop_target,crop_history_group


def verify(out=OUT):
    private=out.parent/'PRIVATE_DO_NOT_SHARE'
    raw=pd.read_csv(RAW,usecols=['year','hhid','id','sow_food','sow_cash'],dtype={'hhid':'string','id':'string'})
    roster=pd.DataFrame(json.loads(Path(json.loads(MANIFEST.read_bytes())['jobs'][8]['cohort']).read_bytes())['farmers'])
    keys=raw.iloc[roster.source_row.to_numpy()-2][['hhid','id']].assign(actor=roster.cohort_agent_id.to_numpy())
    history=estimate_crop_transitions(RAW,set(keys.hhid.dropna()),continuous=True)
    fields=['d'+str(i) for i in range(37,54) if i!=43]
    later=pd.read_csv(RAW.parent/'2016-2017household.csv',usecols=['year','id']+fields,dtype={'id':str})
    coverage=[]
    for y,g in later.groupby('year'):
        complete=g[fields].ge(0).all(axis=1);positive=complete&g[fields].sum(axis=1).gt(0)
        hit=g.id.isin(keys.id); coverage.append(dict(year=y,rows=len(g),same_code=hit.sum(),complete=complete.sum(),
            positive=positive.sum(),same_code_positive=(hit&positive).sum()))
    pd.DataFrame(coverage).to_csv(out/'later_item_coverage.csv',index=False)
    unique=raw.dropna(subset=['hhid','id']);unique=unique[~unique.duplicated(['year','hhid'],keep=False)]
    keys=keys.merge(unique[unique.year==2010][['hhid','id']],validate='m:1')
    survey=unique.merge(keys,on=['id','hhid'],validate='m:1').dropna(subset=['sow_food','sow_cash'])
    old=pd.read_csv(BASE/'behavior_before_after.csv').set_index(['version','seed','year'])
    par=pd.read_csv(BASE/'parameter_provenance.csv').set_index('parameter').value
    rates={k.split('_')[0]:float(v) for k,v in par.items() if k.endswith('_grain')}
    versions={}; metrics=[];accounts=[];segments=[];proof=[];stages=[]
    for version in ['v1','v2']:
        dest=private/version; pi=dest/'PRIVATE_DO_NOT_SHARE'
        audit_area_receipts(dest,'candidate')
        base=pd.read_csv(pi/'baseline.csv').set_index('actor')
        assert sha(pi/'baseline.csv')==sha(P/'candidate/PRIVATE_DO_NOT_SHARE/baseline.csv')
        if version=='v1':
            for f in sorted(pi.glob('*')):
                assert sha(f)==sha(P/'candidate/PRIVATE_DO_NOT_SHARE'/f.name)
                proof.append(dict(check='v1_against_frozen',artifact=f.name,sha256=sha(f)))
        for f in sorted(pi.glob('103_repeat*')):
            assert sha(f)==sha(f.with_name(f.name.replace('repeat','both')))
            proof.append(dict(check=version+'_repeat',artifact=f.name,sha256=sha(f)))
        for seed in [103,109]:
            name=f'{seed}_both_60_1.0'
            rows=pd.read_csv(pi/(name+'_accounts.csv'))
            hh=rows[rows.kind.eq('household')].copy()
            assert len(hh)==3000 and not hh.duplicated(['year','id']).any()
            b=base.reindex(hh.id)
            np.testing.assert_allclose(hh.grain+hh.crop,hh.cultivated*b.intensity.to_numpy(),atol=1e-6)
            np.testing.assert_allclose(hh.grain_kg,hh.grain*b.region.map(rates).to_numpy()*hh.weather_factor,atol=1e-5)
            cost=hh.grain*float(par.grain_other_cost_per_mu)+hh.crop*float(par.cash_crop_other_cost_per_mu)+hh.hired_days*float(par['hired wage'])
            np.testing.assert_allclose(cost,hh.input_hire_cost,atol=1e-5)
            np.testing.assert_allclose(hh.closing,hh.opening+hh.net-hh.consumption,atol=1e-5)
            assert (hh.input_hire_cost<=hh.preplant_funds+1e-5).all() and (hh.family_used<=hh.family_capacity+1e-6).all()
            versions[version,seed]=hh.set_index(['year','id']).sort_index()
            reports=pd.read_csv(dest/'behavior.csv').query('case==@name').set_index('year')
            prev=survey.loc[survey.year.eq(2010)].set_index('actor')
            prev=pd.DataFrame(dict(ob=share(prev.sow_food,prev.sow_cash),sim=share(prev.sow_food,prev.sow_cash)))
            for y,g in hh.groupby('year'):
                actual=survey[survey.year.eq(y)].merge(g.rename(columns={'id':'actor'}),on='actor',validate='1:1').set_index('actor')
                if version=='v2':
                    matched=actual.join(survey[survey.year.eq(y-1)].set_index('actor')[['sow_food','sow_cash']],rsuffix='_prior',how='inner')
                    matched=matched[(matched.sow_food+matched.sow_cash>0)&matched.target_share.notna()].copy()
                    teacher=[]
                    for r in matched.itertuples():
                        a,c=r.sow_food_prior,r.sow_cash_prior
                        teacher.append(continuous_crop_target(history,crop_history_group(a,c),a/(a+c) if a+c>0 else None,r.choice_u))
                    matched['teacher']=teacher
                    matched['observed']=share(matched.sow_food,matched.sow_cash)
                    matched=matched.dropna(subset=['teacher','observed'])
                    for group,z in [('survey_planters',matched),('both_planters',matched[matched.grain+matched.crop>1e-7]),('only_survey_plants',matched[matched.grain+matched.crop<=1e-7])]:
                        for metric,col in [('observed_share','observed'),('observed_lag_target','teacher'),('simulated_lag_target','target_share')]:
                            qs=z[col].quantile([.25,.5,.75])
                            stages.append(dict(seed=seed,year=y,group=group,measure=metric,n=len(z),p25=qs.loc[.25],p50=qs.loc[.5],p75=qs.loc[.75],
                                               quartile_mae=(qs-z.observed.quantile([.25,.5,.75])).abs().mean()))
                cur=pd.DataFrame(dict(ob=share(actual.sow_food,actual.sow_cash),sim=share(actual.grain,actual.crop,1e-7)))
                joint=cur.join(prev,lsuffix='1',rsuffix='0',how='inner').dropna(); prev=cur
                ob=(joint.ob1-joint.ob0).abs().mean(); sim=(joint.sim1-joint.sim0).abs().mean()
                r=reports.loc[y]; ref=old.loc['baseline',seed,y]
                passed=r.passed and abs(ob-sim)<=.15 and not(r.mixed_error>ref.mixed_error and r.share_mae>ref.share_mae)
                if version=='v1': assert passed==old.loc['candidate',seed,y].all_gates
                metrics.append(dict(**r.to_dict(),seed=seed,version=version,year=y,adjustment_n=len(joint),survey_change=ob,model_change=sim,all_gates=passed,
                                    survey_cash_mu=actual.sow_cash.sum(),model_cash_mu=actual.crop.sum()))
                accounts.append(dict(version=version,seed=seed,year=y,n=len(actual),**actual[['hired_days','family_used','sales','input_hire_cost','net','rent_received','rent_paid','paid','rented_in','rented_out','idle']].sum().to_dict()))
    for seed in [103,109]:
        x,y=(versions[v,seed] for v in ['v1','v2'])
        assert x.index.equals(y.index) and np.array_equal(x.choice_u,y.choice_u) and np.array_equal(x.weather_factor,y.weather_factor)
        for yr in range(2011,2016):
            a,b=x.loc[yr],y.loc[yr]; sa=a.grain+a.crop;sb=b.grain+b.crop; both=(sa>1e-7)&(sb>1e-7)
            fa=a.grain[both]/sa[both];fb=b.grain[both]/sb[both]
            e=b.grain[~both].sum()-a.grain[~both].sum();f=((fb-fa)*(sa[both]+sb[both])/2).sum();z=((sb[both]-sa[both])*(fa+fb)/2).sum()
            delta=b.grain.sum()-a.grain.sum();assert np.isclose(e+f+z,delta)
            segments.append(dict(seed=seed,year=yr,delta_grain_mu=delta,entry_exit=e,share=f,sown_scale=z,
                 leased_grain_lower=np.maximum(0,b.grain-(b.operated-b.rented_in)*base.intensity.reindex(b.index)).sum(),
                 leased_grain_upper=np.minimum(b.grain,b.rented_in*base.intensity.reindex(b.index)).sum()))
    for name,data in [('model_behavior',metrics),('model_accounts',accounts),('model_grain_decomposition',segments),('replication',proof),('observed_lag_replay',stages)]:
        pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
    t=pd.DataFrame(metrics)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,col,label in zip(axes,['share_mae','sim_grain'],['Grain-share quartile MAE','Grain sown (mu)']):
        for (v,seed),g in t.groupby(['version','seed']):
            ax.plot(g.year,g[col],marker='o',label=f'{v}, {seed}')
        if col=='share_mae':ax.axhline(.15,color='black',ls=':',label='Gate')
        else:ax.plot(g.year,g.survey_grain,color='black',ls=':',label='Survey')
        ax.set(ylabel=label,xticks=range(2011,2016));ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'paired_behavior.png',dpi=160);plt.close(fig)


if __name__=='__main__': verify()
