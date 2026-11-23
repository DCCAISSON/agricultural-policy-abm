"""Read-only audit of the 2026-11-23 frozen candidate; never call a model step."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1]
D=Path('results/diagnostics/national_crop_mix_repair_20261123')
OLD=R/D
BASE=OLD/'run_01'
P=OLD/'PRIVATE_DO_NOT_SHARE'
DEFAULT=R/'results/diagnostics/national_crop_mix_drift_audit_20261123/run_01'
LOCAL='b5a7df5bfd384c73f156cfe8f01d5475fedea6ce'
PUBLIC='b76149195a3d5a597ccdd54ada21b9e7331302e6'
PARENT='a8ef70620028b6031fedfe874cfdc7d5acd22dd6'
EPS=1e-7
STATES=['none','grain','mixed','cash','unknown']
Q=[.25,.5,.75]

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args])

def state(g,c,tol=0):
    return np.select([~(g.ge(0)&c.ge(0)),(g>tol)&(c>tol),g>tol,c>tol],
                     ['unknown','mixed','grain','cash'],default='none')

def share(g,c,tol=0):
    return g/(g+c).where((g+c>tol)&g.ge(0)&c.ge(0))

def ci(k,n):
    if not n: return np.nan,np.nan,np.nan
    p=k/n; z=1.95996398454; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; e=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p,max(0,mid-e),min(1,mid+e)

def same_members(a,b):
    pairs=[(i,j) for i,r in enumerate(a) for j,s in enumerate(b) if r[:2]==s[:2] and abs(s[2]-r[2]-1)<=1]
    rel=[a[i][0] for i,j in pairs if sum(x==i for x,y in pairs)==sum(y==j for x,y in pairs)==1]
    return 1 in rel and len(rel)>=2

def audit(out,reference=None):
    if out.exists(): raise FileExistsError(out)
    sources={}; tables=defaultdict(list)
    def use(p,expected=None):
        if p not in sources: sources[p]=sha(p)
        if expected is not None: assert sources[p]==expected,p.name
        return p
    def csv(p,**kwargs): return pd.read_csv(use(p),**kwargs)
    def read(p): return json.loads(use(p).read_bytes())
    def write(name,**fields): tables[name].append(fields)
    def save(name,df): df.to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    ref=read(BASE/'run_manifest.json')
    trees={}
    for directory,commit in [('source',LOCAL),('public',PUBLIC)]:
        repo=OLD/directory
        assert git(repo,'rev-parse','HEAD').decode().strip()==commit
        trees[directory]={x.split(b'\t',1)[1].decode():x.split(b'\t',1)[0].split()[-1].decode() for x in
                          git(repo,'ls-tree','-rz',commit).split(b'\0') if x}
        write('version_check',snapshot=directory,commit=commit,tree=git(repo,'rev-parse','HEAD^{tree}').decode().strip(),files=len(trees[directory]))
    assert git(OLD/'source','rev-parse','HEAD^').decode().strip()==PARENT
    required=[*read(BASE/'source_manifest.json'),*((D/'run_01'/p.name).as_posix() for p in BASE.glob('*') if p.is_file())]
    for f in required:
        assert f in trees['public'] and f in trees['source']
        for directory in trees:
            assert (OLD/directory/f).read_bytes()==(R/f).read_bytes(),f
            # Working-tree CRLF may differ from Git storage.
            assert git(OLD/directory,'show','HEAD:'+f).replace(b'\r\n',b'\n')==(R/f).read_bytes().replace(b'\r\n',b'\n'),f
        use(R/f)
    for f,digest in read(BASE/'source_manifest.json').items(): use(R/f,digest)
    survey_dir=R/'校准用数据源/全国农村固定观察点'
    for n,h in ref['survey_inputs'].items(): use((R/'config/experiments' if n.endswith('cpi_20260909.csv') else survey_dir)/n,h)
    prep=R/'results/diagnostics/national_formal_revision_20261118/run_01/prepared_seed17_release1/preparation_manifest.json'
    job=read(prep)['jobs'][8]; co=pd.DataFrame(read(use(Path(job['cohort']),ref['economics']['cohort_sha256']))['farmers'])
    raw=csv(survey_dir/'2009-2015household.csv',usecols='hhid id year smgen samplast sow_food sow_cash sow_total prod_food area_e'.split(),dtype={'hhid':'string','id':'string'})
    orig=raw.iloc[co.source_row.to_numpy()-2].copy();orig['actor']=co.cohort_agent_id.to_numpy()
    assert len(orig)==orig.actor.nunique()==orig.hhid.nunique()==600 and orig.year.eq(2010).all()
    roster=csv(survey_dir/'2009-2015individual.csv',usecols=['hhid','year','sex','relation','age'],dtype={'hhid':'string'})
    members=defaultdict(list)
    for h,y,s,r,a in roster.loc[roster.sex.isin([1,2])&roster.age.between(0,110),['hhid','year','sex','relation','age']].itertuples(index=False,name=None):
        members[h,y].append((r,s,a))
    raw['status']=state(raw.sow_food,raw.sow_cash)
    raw['share']=share(raw.sow_food,raw.sow_cash)
    raw['dup']=raw.duplicated(['year','hhid'],keep=False)
    donors=raw[~raw.hhid.isin(orig.hhid)]
    donor=donors[~donors.dup & donors.hhid.eq(donors.id)].copy()
    dictionary=pd.read_excel(survey_dir/'household variables2009-2015.xlsx',header=None)
    weight_labels=int(dictionary.astype(str).apply(lambda c:c.str.contains('权重|抽样概率|调查权数|设计权数|sampling weight|sample weight',case=False)).any(axis=1).sum())
    for year,d in raw.groupby('year'):
        h=donors[donors.year.eq(year)]
        write('raw_key_counts',year=year,original_rows=len(d),excluded_cohort_id_rows=len(d)-len(h),remaining=len(h),
              duplicate_key_values=h.loc[h.dup,'hhid'].nunique(),duplicate_rows=h.dup.sum(),
              bad_or_absent_identity=(~h.dup & ~h.hhid.eq(h.id).fillna(False)).sum(),
              missing_area_rows=h[['sow_food','sow_cash']].isna().any(axis=1).sum(),zero_both=h.status.eq('none').sum(),
              design_weight_label_hits=weight_labels)
    trans=csv(P/'candidate/transition_estimates.csv')
    def prevgroup(frame):
        return frame.status.replace({'none':'unplanted'}).mask(frame.status.eq('mixed') & frame.share.lt(.5),'mixed_lower').mask(frame.status.eq('mixed') & frame.share.ge(.5),'mixed_upper')
    annual={}
    for y in range(2009,2015):
        s=donor[donor.year.eq(y)].set_index('hhid')
        v=donor[donor.year.eq(y+1)].set_index('hhid')
        p=s[['smgen','status','share']].join(v[['smgen','samplast','status','share']],lsuffix='_old',how='left')
        credible=p.smgen_old.eq(p.smgen)&p.samplast.eq(1)
        credible.loc[credible]=[same_members(members[h,y],members[h,y+1]) for h in p.index[credible]]
        observed=p.status.notna()&credible
        p['status']=p.status.where(observed,'unknown')
        write('sample_coverage',population='noncohort_panel',year=y+1,prior_year=y,source_rows=len(donors[donors.year.eq(y)]),
              duplicate_rows=int(donors.loc[donors.year.eq(y),'dup'].sum()),unique_origin=len(s),
              certified_pairs=credible.sum(),known_both=(credible&p.status.ne('unknown')&p.status_old.ne('unknown')).sum(),
              unresolved_identity=(~credible).sum(),missing_crops_certified=(credible & (p.status.eq('unknown')|p.status_old.eq('unknown'))).sum(),
              next_zero_certified=(credible&p.status.eq('none')).sum())
        for a in STATES:
            group=p[p.status_old.eq(a)]
            for universe in ['all_origins','confirmed_known_next','next_planted']:
                chosen=group if universe=='all_origins' else group[group.status.isin(['grain','mixed','cash']+(['none'] if universe=='confirmed_known_next' else []))]
                for b in STATES:
                    k=chosen.status.eq(b).sum(); prob,lo,hi=ci(k,len(chosen))
                    write('transition_stability',prior_year=y,year=y+1,origin=a,destination=b,universe=universe,
                          denominator=len(chosen),count=k,probability=prob,ci95_low=lo,ci95_high=hi)
        keep=p[p.status_old.ne('unknown')&p.status.isin(['cash','mixed','grain'])].copy()
        if y==2009:
            assert len(keep)==keep.index.nunique()==12884
            pg=prevgroup(s.loc[keep.index])
            category=np.select([keep.share.eq(0),keep.share.eq(1)],[0,6],default=1+np.minimum(4,np.floor(keep.share*5))).astype(int)
            g=keep.assign(prior=pg,category=category).groupby(['prior','category']).share.agg(['size','mean']).reindex(pd.MultiIndex.from_frame(trans[['prior','category']]))
            assert np.array_equal(g['size'].fillna(0),trans['count']) and np.allclose(g['mean'],trans.target,equal_nan=True)
            write('sample_coverage',population='v1_estimation',year=2010,unique_families=len(keep),household_years=2*len(keep),effective_transitions=len(keep),duplicate_retained=0)
        annual[y]=keep.loc[keep.status_old.eq('mixed'),'status'].eq('mixed').astype(float)
        for k,frame in keep.groupby(prevgroup(s.loc[keep.index])):
            q=frame.share.quantile(Q)
            write('conditional_share_history',year=y+1,prior=k,n=len(frame),p25=q.iloc[0],median=q.iloc[1],p75=q.iloc[2],
                  mean=frame.share.mean(),cash_end=frame.status.eq('cash').mean(),grain_end=frame.status.eq('grain').mean())
        if y>2009:
            pair=pd.concat([annual[2009],annual[y]],axis=1,join='inner'); delta=pair.iloc[:,1]-pair.iloc[:,0]
            write('paired_stability',year=y+1,both_eligible=len(delta),early=pair.iloc[:,0].mean(),later=pair.iloc[:,1].mean(),
                  paired_change=delta.mean(),se=delta.std()/np.sqrt(len(delta)))
    # Exact evaluation match, deliberately not the stricter estimation link.
    un=raw.dropna(subset=['hhid','id']).copy()
    un=un[~un.duplicated(['year','hhid'],keep=False)]
    keys=orig[['hhid','id','actor']].merge(un[un.year.eq(2010)][['hhid','id']],validate='m:1')
    surv=un.merge(keys,on=['hhid','id'],validate='m:1')
    for y in range(2011,2016):
        g=surv[surv.year.eq(y)]
        write('sample_coverage',population='evaluation',year=y,cohort=600,base_keys=len(keys),matched_keys=len(g),
              unmatched_or_ambiguous=600-len(g),missing_crop=g[['sow_cash','sow_food']].isna().any(axis=1).sum(),zero_both=g.status.eq('none').sum(),
              analytic_n=g[['sow_cash','sow_food']].notna().all(axis=1).sum())
    surv=surv.dropna(subset=['sow_cash','sow_food'])
    five=surv[surv.year.between(2011,2015)].groupby('actor').size().loc[lambda x:x==5].index
    assert not surv.status.eq('unknown').any()
    frozen=csv(BASE/'behavior_before_after.csv').query("version=='candidate'").set_index(['seed','year'])
    params=csv(BASE/'parameter_provenance.csv').set_index('parameter').value
    lg,lc,kg,kc,wage=(float(params[k]) for k in ['grain_labor_days','cash_crop_labor_days','grain_other_cost_per_mu','cash_crop_other_cost_per_mu','hired wage'])
    alts=csv(P/'alternative_accounts.csv')
    assert len(alts)==30000 and not alts.duplicated(['id','seed','year','option']).any()
    assert np.allclose(alts.hire_cost,alts.hired_days*wage)
    assert np.allclose(alts.expected_net,alts.expected_sales-alts.inputs-alts.hire_cost+alts.support+alts.nonfarm+alts.rent_out-alts.rent_in)
    assert alts.groupby(['id','seed','year']).nonfarm.nunique().eq(1).all()
    for (opt,y,sd),g in alts.groupby(['option','year','seed']):
        write('alternative_budgets',option=opt,year=y,seed=sd,n=len(g),sales=g.expected_sales.mean(),
              inputs=g.inputs.mean(),family_days=g.family_days.mean(),hired_days=g.hired_days.mean(),
              hire_cost=g.hire_cost.mean(),expected_net=g.expected_net.mean(),nonfarm=g.nonfarm.mean())
    traces=[]; cohort_errors=[]
    for seed in [103,109]:
        name=f'{seed}_both_60_1.0'
        def ledger(ver,typ):
            return csv(P/ver/'PRIVATE_DO_NOT_SHARE'/f'{name}_{typ}.csv')
        a=ledger('candidate','accounts').query("kind=='household'").copy()
        old=ledger('baseline','accounts').query("kind=='household'").copy()
        contracts=ledger('candidate','contracts'); orders=ledger('candidate','orders')
        assert len(a)==len(old)==3000 and not a.duplicated(['year','id']).any()
        base=csv(P/'candidate/PRIVATE_DO_NOT_SHARE/baseline.csv').set_index('actor')
        prev=base[['sow_food','sow_cash']].rename(columns={'sow_food':'grain','sow_cash':'crop'})
        for y,g in a.groupby('year'):
            g=g.set_index('id').copy(); prev=prev.reindex(g.index)
            assert len(g)==600 and g.index.is_unique and set(g.index)==set(base.index)
            assert np.allclose(g.prior_share,share(prev.grain,prev.crop,EPS),equal_nan=True)
            cc=contracts[contracts.start.eq(y)]; oo=orders[orders.year.eq(y)]
            for side,col,who,fee in [('buy','rented_in','buyer','rent_paid'),('sell','rented_out','seller','rent_received')]:
                sums=cc.assign(payment=cc.area*cc.rent).groupby(who)[['area','payment']].sum().reindex(g.index,fill_value=0)
                assert np.allclose(g[col],sums.area) and np.allclose(g[fee],sums.payment)
                sub=oo[oo.side.eq(side)].set_index('id').reindex(g.index)
                g[side+'_quantity']=sub.quantity; g[side+'_price']=sub.price
            assert np.allclose(g.operated,g.base_area+g.rented_in-g.rented_out)
            assert np.allclose(g.input_hire_cost,g.grain*kg+g.crop*kc+g.hired_days*wage)
            assert np.allclose(g.family_used+g.hired_days,g.grain*lg+g.crop*lc)
            assert (g.family_used<=g.family_capacity+1e-6).all() and (g.input_hire_cost<=g.preplant_funds+1e-5).all()
            assert np.allclose(g.closing,g.opening+g.net-g.consumption)
            assert np.allclose(g.hired_days,(g.grain*lg+g.crop*lc-g.family_capacity).clip(lower=0))
            assert np.allclose(g.surplus,g.sales-g.input_hire_cost-g.overhead+g.rent_received-g.rent_paid)
            assert np.allclose(g.net,g.surplus+g.nonfarm+g.paid)
            g['plan_share']=share(g.planned_grain,g.planned_crop,EPS);g['actual_share']=share(g.grain,g.crop,EPS)
            g['prior_g']=prev.grain;g['prior_c']=prev.crop
            for group,block in g.groupby('prior_group'):
                tr=trans[trans.prior.eq(group)]
                if tr.empty:
                    assert group=='unknown' and block.target_share.isna().all()
                    continue
                j=np.searchsorted(tr.probability.cumsum(),block.choice_u,side='right').clip(0,len(tr)-1)
                assert np.allclose(tr.target.to_numpy()[j],block.target_share)
                g.loc[block.index,'bin_probability']=tr.probability.to_numpy()[j]
            for s in ['plan_share','actual_share']:
                both=g[s].notna()&g.target_share.notna()
                assert np.allclose(g.loc[both,s],g.loc[both,'target_share'])
            g['unrealized_grain']=(g.planned_grain-g.grain).clip(lower=0)
            g['unrealized_cash']=(g.planned_crop-g.crop).clip(lower=0)
            g['net_land_shortfall']=((g.demand-g.offer)-(g.rented_in-g.rented_out)).clip(lower=0)
            g['planned_but_zero']=g.plan_share.notna()&g.actual_share.isna()
            if y>=2014:
                traces.append(g.reset_index().assign(seed=seed))
                stages=[('previous_actual',prev.grain,prev.crop),('target_share_only',None,None),
                        ('joint_cash_labor_land_plan',g.planned_grain,g.planned_crop),
                        ('cash_alone_not_logged',None,None),('labor_alone_not_logged',None,None),
                        ('submitted_bids',None,None),('signed_contracts',None,None),('actual',g.grain,g.crop)]
                for stage,G,C in stages:
                    n={'submitted_bids':(g.buy_quantity.notna()|g.sell_quantity.notna()).sum(),
                       'signed_contracts':((g.rented_in+g.rented_out)>EPS).sum(),'target_share_only':g.target_share.count(),
                       'cash_alone_not_logged':None,'labor_alone_not_logged':None}.get(stage,600)
                    write('stage_decomposition',seed=seed,year=y,stage=stage,households=n,positive_area=None if G is None else (G+C>EPS).sum(),
                          grain_mu=None if G is None else G.sum(min_count=1),cash_mu=None if C is None else C.sum(min_count=1),
                          grain_delta_from_last_observed_area=(g.planned_grain-prev.grain).sum() if stage=='joint_cash_labor_land_plan' else ((g.grain-g.planned_grain).sum() if stage=='actual' else None),
                          cash_delta_from_last_observed_area=(g.planned_crop-prev.crop).sum() if stage=='joint_cash_labor_land_plan' else ((g.crop-g.planned_crop).sum() if stage=='actual' else None),
                          unrealized_grain_mu=g.unrealized_grain.sum() if stage=='actual' else None,
                          unrealized_cash_mu=g.unrealized_cash.sum() if stage=='actual' else None,
                          unsigned_buy_mu=g.buy_quantity.sum()-g.rented_in.sum() if stage=='signed_contracts' else None,
                          signed_physical_mu=g.rented_in.sum() if stage=='signed_contracts' else None)
                for label,mask in [('land_shortfall',g.net_land_shortfall>EPS),('no_land_shortfall',g.net_land_shortfall<=EPS)]:
                    z=g[mask]
                    write('stage_restrictions',seed=seed,year=y,group=label,n=len(z),gross_grain_shortfall=z.unrealized_grain.sum(),
                          gross_cash_shortfall=z.unrealized_cash.sum(),planned_but_zero=z.planned_but_zero.sum(),
                          cash_binding=(z.final_cash_shadow>EPS).sum(),labor_binding=(z.final_labor_shadow>EPS).sum(),
                          both_binding=((z.final_cash_shadow>EPS)&(z.final_labor_shadow>EPS)).sum())
            prev=g[['grain','crop']]
            v=surv[surv.year.eq(y)].set_index('actor').join(g,rsuffix='_sim',validate='1:1')
            n=len(v); row=frozen.loc[seed,y]; assert n==row.matched
            ss=state(v.grain,v.crop,EPS); os=v.status
            assert (ss=='mixed').sum()==row.sim_mixed and (os=='mixed').sum()==row.survey_mixed
            obs=v.share.notna();sim=v.actual_share.notna(); common=obs&sim
            oq=surv[surv.year.eq(y-1)].set_index('actor')
            high=v[v.target_share.gt(.8)]
            landless=high[high.operated.le(EPS)]
            assert np.allclose(landless.base_area,landless.rented_out) and np.allclose(landless.offer,landless.base_area)
            write('high_grain_land_exit',year=y,seed=seed,high_intent=len(high),sowed=high.actual_share.count(),
                  landless=len(landless),landless_positive_open_funds=landless.working_funds.gt(EPS).sum(),
                  offered_mu=landless.offer.sum(),leased_out_mu=landless.rented_out.sum(),
                  planned_cultivated_mu=landless.planned_cultivated.sum())
            pr=prevgroup(oq).reindex(v.index)
            for label,keep in [('same_prior',pr.eq(v.prior_group)),('different_prior',pr.notna()&pr.ne(v.prior_group)),('prior_not_matched',pr.isna())]:
                z=v[keep&common]
                write('history_agreement',year=y,seed=seed,group=label,common_sowers=len(z),
                      mean_observed_share=z.share.mean(),mean_target=z.target_share.mean(),mean_abs_error=(z.share-z.target_share).abs().mean())
            for tar,z in v.groupby(pd.cut(v.target_share,[-.001,.2,.5,.8,1.],include_lowest=True),observed=True):
                write('target_selection',year=y,seed=seed,share_interval=str(tar),households=len(z),model_sowed=z.actual_share.count(),
                      model_participation=z.actual_share.notna().mean(),survey_participation=z.share.notna().mean(),
                      zero_preplant_funds=z.preplant_funds.le(EPS).sum(),zero_operated=z.operated.le(EPS).sum())
            for who,s0 in [('survey',v.share),('candidate',v.actual_share)]:
                z=s0[v.index.isin(five)];q=z.quantile(Q)
                write('balanced_panel_check',year=y,seed=seed,who=who,matched=len(z),positive=z.count(),p25=q.iloc[0],median=q.iloc[1],p75=q.iloc[2])
            assert obs.sum()==row.survey_planted and sim.sum()==row.sim_planted
            for k in STATES[:-1]:
                for l in STATES[:-1]:
                    write('paired_states',seed=seed,year=y,survey=k,model=l,count=((os==k)&(ss==l)).sum())
            for who,G,C,stat,S,output in [('survey',v.sow_food,v.sow_cash,os,v.share,v.prod_food),('model',v.grain,v.crop,ss,v.actual_share,v.grain_kg)]:
                planted=S.notna(); vals=S.quantile(Q).to_numpy()
                assert np.allclose(vals,[row[f'{"survey" if who=="survey" else "sim"}_share_{k}'] for k in ['p25','median','p75']])
                write('behavior_metrics_by_year',seed=seed,year=y,source=who,matched=n,planted=planted.sum(),participation=planted.mean(),
                      **{k+'_n':(stat==k).sum() for k in STATES[:-1]},**{k+'_rate':(stat==k).mean() for k in STATES[:-1]},
                      mixed_given_planted=(stat=='mixed').sum()/planted.sum(),p25=vals[0],median=vals[1],p75=vals[2],
                      grain_mu=G.sum(),cash_mu=C.sum(),two_crop_mu=(G+C).sum(),two_crop_mu_per_household=(G+C).mean(),
                      all_crop_sown_mu=(v.sow_total.sum(min_count=1) if who=='survey' else (G+C).sum()),
                      all_crop_reporting=(v.sow_total.count() if who=='survey' else n),
                      all_crop_per_reporting_household=(v.sow_total.mean() if who=='survey' else (G+C).mean()),
                      production_reporting=v.prod_food.count(),kg_on_common_reporters=output[v.prod_food.notna()].sum(min_count=1),
                      kg_per_mu_on_reporters=output[v.prod_food.notna()].sum()/G[v.prod_food.notna()].sum())
            # Telescoping quantile differences isolate composition of the two positive samples.
            for name1,s in [('obs_positive',v.share),('obs_common',v.share[common]),('target_common',v.target_share[common]),
                            ('final_common',v.actual_share[common]),('final_positive',v.actual_share),
                            ('target_all_matched',v.target_share),('target_survey_positive',v.target_share[obs]),
                            ('plan_positive',v.plan_share),('target_final_positive',v.target_share[sim])]:
                q=s.quantile(Q);write('quantile_bridge',seed=seed,year=y,stage=name1,n=s.count(),p25=q.iloc[0],p50=q.iloc[1],p75=q.iloc[2])
            S=v.sow_food+v.sow_cash;T=v.grain+v.crop
            for crop,O,M in [('grain',v.sow_food,v.grain),('cash',v.sow_cash,v.crop)]:
                u=O[common]/S[common];t=M[common]/T[common]
                entry=M[sim&~obs].sum();lost=O[obs&~sim].sum()
                size=((T[common]-S[common])*(t+u)/2).sum();mix=((t-u)*(T[common]+S[common])/2).sum()
                assert np.isclose(entry-lost+size+mix,M.sum()-O.sum())
                write('area_bridge',year=y,seed=seed,crop=crop,model_only=entry,survey_only=lost,common_size=size,common_share=mix,total_model_minus_survey=M.sum()-O.sum())
            cohort_errors.append(v[['sow_food','sow_cash','grain','crop','share','target_share','plan_share','actual_share']].assign(seed=seed,year=y).reset_index())
            for ver,acct in [('baseline',old[old.year.eq(y)].set_index('id').loc[v.index]),('candidate',g.loc[v.index])]:
                ac=acct; reqd=lg*ac.grain+lc*ac.crop;cap=g.loc[v.index,'family_capacity'];hire=(reqd-cap).clip(lower=0)
                assert np.allclose(hire,ac.hired_days) and np.allclose(ac.nonfarm,g.loc[v.index,'nonfarm'])
                assert np.allclose(ac.input_hire_cost,kg*ac.grain+kc*ac.crop+wage*hire)
                write('economic_accounts_check',year=y,seed=seed,version=ver,n=n,required_labor=reqd.sum(),
                      family_days=(reqd-hire).sum(),hiring_households=(hire>EPS).sum(),hired_days=hire.sum(),hired_yuan=wage*hire.sum(),
                      cash_binding_plan=(ac.cash_shadow>EPS).sum(),nonfarm=ac.nonfarm.sum(),net_cash_income=ac.net.sum(),
                      input_hire_cost=ac.input_hire_cost.sum(),family_capacity=cap.sum())
    def portable(p):
        return p.relative_to(R).as_posix() if p.is_relative_to(R) else 'authorized_input/'+p.name
    manifest=pd.DataFrame([dict(path=portable(p),sha256=h,bytes=p.stat().st_size) for p,h in sources.items()])
    assert manifest.path.is_unique
    if reference:
        for z in pd.read_csv(reference).itertuples():
            assert manifest.set_index('path').at[z.path,'sha256']==z.sha256,z.path
        assert len(pd.read_csv(reference))==len(sources)
    out.mkdir(parents=True,exist_ok=False)
    for name,rows in tables.items(): save(name,pd.DataFrame(rows))
    save('source_manifest',manifest)
    priv=out/'PRIVATE_DO_NOT_SHARE';priv.mkdir()
    pd.concat(traces).to_csv(priv/'late_household_trace.csv',index=False)
    pd.concat(cohort_errors).to_csv(priv/'paired_errors.csv',index=False)
    print('Verified candidate, survey matches, frozen transition counts, year histories, decision draws, household and contract accounts.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path,nargs='?',default=DEFAULT)
    parser.add_argument('--verify-inputs',type=Path)
    args=parser.parse_args();audit(args.output,args.verify_inputs)
