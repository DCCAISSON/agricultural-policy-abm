"""Annual NFPS comparison and strictly forward, household-disjoint prediction diagnostics."""
import argparse
from collections import defaultdict
from itertools import product, permutations
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd

from scripts.audit_national_crop_mix_drift import R,OLD,BASE,P,LOCAL,PUBLIC,sha,git,same_members,share,ci
from core.extract_microdata_features import REGION_MAPPING
from core.fixed_rule_runtime import continuous_farm_plan

OUT=R/'results/diagnostics/national_crop_mix_data_comparability_20261123/run_01'
ST=['grain','mixed','cash','other','none','unknown']
Q=[.1,.25,.5,.75,.9]

def classify(g,c,total):
    return np.select([g.lt(0)|c.lt(0)|g.isna()|c.isna(),g.gt(0)&c.gt(0),g.gt(0),c.gt(0),total.gt(0),total.eq(0)],
                     ['unknown','mixed','grain','cash','other','none'],default='unknown')

def bucket(x):
    return np.select([x<0,x==0,x==1,x<.5],['off','cash','grain','mixed_low'],default='mixed_high')

def market_evidence(d):
    covered=d[['area_b','area_e','area_in','area_de','area_de2']].ge(0).all(axis=1)
    balanced=(d.area_b+d.area_in-d.area_de-d.area_e).abs()<1e-4
    valid=covered&balanced&d.area_b.gt(0)&d.area_in.eq(0)&d.area_de2.le(d.area_de+1e-4)
    return valid,valid&d.area_e.eq(0)&(d.area_b-d.area_de2).abs().lt(1e-4)

def backtest(train,test,continuous):
    # Each household faces a predictive distribution. No random evaluation draw.
    prob=[]; masses=[]; values=[]; errors=[]; score=[]
    for group,sub in test.groupby('oldgroup'):
        tr=train[train.oldgroup.eq(group)]
        x=tr.x.to_numpy(); yy=tr.y.to_numpy(); st=tr.state.to_numpy()
        for r in sub.itertuples():
            select=np.ones(len(x),bool)
            if continuous and group.startswith('mixed'):
                # Exact endpoints are separate. Pool both interior groups before finding neighbours.
                if len(x)==len(tr):
                    interior=train[train.oldgroup.str.startswith('mixed')]
                    x=interior.x.to_numpy();yy=interior.y.to_numpy();st=interior.state.to_numpy()
                dist=np.abs(x-r.x)
                k=int(np.ceil(np.sqrt(len(x))))
                select=dist<=np.partition(dist,k-1)[k-1]
            a=yy[select]; states=st[select]; pr=np.array([(states==v).mean() for v in ST[:-1]])
            prob.append(pr)
            a=a[a>=0]
            if not len(a): raise ValueError('empty planting donor group')
            if not continuous:
                cat=np.where(a==0,0,np.where(a==1,6,np.minimum(4,(a*5).astype(int))+1))
                aa,ct=np.unique(cat,return_counts=True);means=np.array([a[cat==j].mean() for j in aa]); w=ct/len(a)
            else: means,w=np.unique(a,return_counts=True); w=w/len(a)
            values.extend(means);masses.extend(w*sum(pr[:3]))
            if r.y>=0:
                errors.append(abs(np.dot(means,w)-r.y))
                # Proper conditional distribution score; means sorted within crop bins.
                score.append(np.dot(abs(means-r.y),w)-np.dot(w*means,2*np.cumsum(w)-1-w))
    prob=np.mean(prob,axis=0); observed=np.array([test.state.eq(v).mean() for v in ST[:-1]])
    vals=np.array(values); order=vals.argsort(); cw=np.cumsum(np.asarray(masses)[order]); cw/=cw[-1]
    pred=vals[order][np.searchsorted(cw,Q)]; actual=test.loc[test.y.ge(0),'y'].quantile(Q).to_numpy()
    return dict(n=len(test),positive=(test.y>=0).sum(),state_tv=.5*np.abs(prob-observed).sum(),
                exit_prob=prob[3:].sum(),observed_exit=observed[3:].sum(),all_stop_prob=prob[4],observed_all_stop=observed[4],
                q_mae=np.mean(np.abs(pred-actual)),conditional_mean_mae=np.mean(errors),conditional_crps=np.mean(score),
                **{f'pred_q{int(q*100)}':v for q,v in zip(Q,pred)},**{f'obs_q{int(q*100)}':v for q,v in zip(Q,actual)})

def audit(out):
    if out.exists(): raise FileExistsError(out)
    tables=defaultdict(list); hashes={}
    def path(p):
        hashes[p.name if not p.is_relative_to(R) else p.relative_to(R).as_posix()]=sha(p)
        return p
    def tab(table,**row): tables[table].append(row)
    d=R/'校准用数据源/全国农村固定观察点'
    meta=json.loads(path(BASE/'run_manifest.json').read_bytes())
    for repo,rev in [('source',LOCAL),('public',PUBLIC)]: assert git(OLD/repo,'rev-parse','HEAD').decode().strip()==rev
    for f,h in meta['survey_inputs'].items():
        p=d/f if (d/f).exists() else R/'config/experiments'/f
        assert sha(path(p))==h
    fields='sow_food sow_cash sow_total sow_oother area_b area_e area_in area_in1 area_de area_de2 hhlabornum labor_d_r fer_r mach_r'.split()
    dictionary=pd.read_excel(path(d/'household variables2009-2015.xlsx'),header=None).set_index(0)
    form=path(d/'2009-2017农户调查问卷.txt').read_text('utf8')
    assert '转包出' in form and '年初经营耕地面积' in form
    raw=pd.read_csv(path(d/'2009-2015household.csv'),usecols=['year','id','hhid','samplast','smgen']+fields,dtype={'id':str,'hhid':str})
    raw['region']=raw.smgen.map({p:z for z,ps in REGION_MAPPING.items() for p in ps}).fillna('unmapped')
    raw['scale']=pd.cut(raw.area_e,[-np.inf,0,5,15,50,np.inf],right=False,labels=['negative','lt5','5_15','15_50','50plus']).astype(str)
    raw['labor_group']=pd.cut(raw.hhlabornum,[-1,0,1,2,np.inf],labels=['0','1','2','3plus']).astype(str)
    raw['role']=np.where(raw.area_in1.isna()|raw.area_de2.isna(),'incomplete',
                np.where(raw.area_in1.gt(0),np.where(raw.area_de2.gt(0),'both','in'),np.where(raw.area_de2.gt(0),'out','neither')))
    raw['state']=classify(raw.sow_food,raw.sow_cash,raw.sow_total)
    raw['y']=share(raw.sow_food,raw.sow_cash).where(lambda s:s.notna(),-1)
    for y,g in raw.groupby('year'):
        for f in fields:
            tab('variable_by_year',year=y,name=f,question=dictionary.loc[f,1],rows=len(g),valid=g[f].ge(0).sum(),
                zeros=g[f].eq(0).sum(),missing=g[f].isna().sum(),negative=g[f].lt(0).sum())
        complete=g[['area_b','area_e','area_in','area_de']].notna().all(axis=1)
        gap=g.sow_total-g.sow_food-g.sow_cash
        tab('account_coverage',year=y,land_complete=complete.sum(),land_balanced=((g.area_b+g.area_in-g.area_de-g.area_e).abs().lt(1e-4)&complete).sum(),
            gap_n=gap.count(),positive_gap=gap.gt(1e-4).sum(),negative_gap=gap.lt(-1e-4).sum(),
            other_column_joint=(gap.notna()&g.sow_oother.notna()).sum(),other_exact=(gap-g.sow_oother).abs().lt(1e-4).sum())
    late=pd.read_excel(path(d/'household variables2016-2017.xlsx'),header=None).set_index(0)
    late_fields=['d30','d31','d35','d36']+['d'+str(i) for i in range(37,55)]
    ext=pd.read_csv(path(d/'2016-2017household.csv'),usecols=['year','id']+late_fields,low_memory=False)
    for y,g in ext.groupby('year'):
        for f in late_fields:
            tab('later_availability',year=y,field=f,definition=late.loc[f,1],rows=len(g),numeric_nonmissing=pd.to_numeric(g[f],errors='coerce').count(),unique_nonnull_ids=g.id.nunique())
    prep=R/'results/diagnostics/national_formal_revision_20261118/run_01/prepared_seed17_release1/preparation_manifest.json'
    cp=Path(json.loads(path(prep).read_bytes())['jobs'][8]['cohort'])
    assert sha(path(cp))==meta['economics']['cohort_sha256']
    co=pd.DataFrame(json.loads(cp.read_bytes())['farmers']); origin=raw.iloc[co.source_row.to_numpy()-2].copy();origin['actor']=co.cohort_agent_id.to_numpy()
    assert origin.year.eq(2010).all() and origin.hhid.nunique()==600
    un=raw.dropna(subset=['hhid','id']); un=un[~un.duplicated(['year','hhid'],keep=False)]
    keys=origin[['actor','hhid','id']].merge(un[un.year.eq(2010)][['hhid','id']],validate='m:1')
    linked=un.merge(keys,on=['id','hhid'],validate='m:1')
    five=linked[linked.year.between(2011,2015)].groupby('actor').size().loc[lambda x:x==5].index
    assert len(five)==450
    roster=pd.read_csv(path(d/'2009-2015individual.csv'),usecols=['hhid','year','sex','relation','age'],dtype={'hhid':str})
    people=defaultdict(list)
    for h,y,s,r,a in roster.loc[roster.sex.isin([1,2])&roster.age.between(0,110),['hhid','year','sex','relation','age']].itertuples(index=False,name=None):
        people[h,y].append((r,s,a))
    waves={y:g.set_index('hhid') for y,g in un[un.id.eq(un.hhid)].groupby('year')}
    pairs=[];common_donors=None
    for yr in range(2010,2016):
        s,v=waves[yr-1],waves[yr]
        pair=s[['region','scale','state','y','smgen']].rename(columns={'state':'prior','y':'x'}).join(v[['state','y','smgen','samplast']],rsuffix='_next')
        ok=pair.smgen.eq(pair.smgen_next)&pair.samplast.eq(1)
        same=ok.copy(); same.loc[ok]=[same_members(people[h,yr-1],people[h,yr]) for h in pair.index[ok]]
        next_raw=raw[raw.year.eq(yr)]; dup=next_raw.loc[next_raw.hhid.duplicated(False),'hhid']
        reason=np.select([pair.index.isin(dup),~pair.index.isin(next_raw.hhid),pair.state.isna(),pair.smgen.ne(pair.smgen_next),
               ~pair.samplast.eq(1),~same],['duplicate_next','absent_next','invalid_id','region_changed','not_confirmed_revisit','member_link_failed'],default='linked')
        pair['link']=reason; pair.loc[~same,'state']='unknown';pair.loc[~same,'y']=np.nan
        pair['year']=yr;pair['oldgroup']=bucket(pair.x)
        for r,n in pd.Series(reason).value_counts().items():tab('household_attrition',year=yr,reason=r,n=n,denom=len(pair))
        start=raw[raw.year.eq(yr-1)]
        tab('household_attrition',year=yr,reason='duplicate_origin_rows_excluded',n=start.hhid.duplicated(False).sum(),denom=len(start))
        if yr==2010:
            common_donors=pair.index[(~pair.index.isin(origin.hhid))&same&pair.y.ge(0)&pair.prior.ne('unknown')]
            assert len(common_donors)==12884
        pairs.append(pair)
    panel=pd.concat(pairs);panel.index.name='hhid'
    for sample,df in [('A_full_rows',raw),('B_balanced',linked[linked.actor.isin(five)]),
                     ('C_frozen',pd.concat([origin,linked[linked.year.ne(2010)]],ignore_index=True)),('D_estimation_ids',un[un.hhid.isin(common_donors)])]:
        for year,g in df.groupby('year'):
            valid=g.state.ne('unknown'); shares=g.y[g.y.ge(0)]
            tab('annual_distribution',sample=sample,year=year,n=len(g),unique_ids=g.hhid.nunique(),known=valid.sum(),unknown=(~valid).sum(),positive=len(shares),
                **{k+'_n':g.state.eq(k).sum() for k in ST[:-1]},**{k+'_rate':g.state.eq(k).sum()/valid.sum() for k in ST[:-1]},
                grain_positive=(g.sow_food>0).sum(),grain_positive_rate=g.sow_food.gt(0).sum()/g.sow_food.ge(0).sum(),cash_positive=g.sow_cash.gt(0).sum(),cash_positive_rate=g.sow_cash.gt(0).sum()/g.sow_cash.ge(0).sum(),
                zero_pair=(g.sow_food.eq(0)&g.sow_cash.eq(0)).sum(),**{f'p{int(q*100)}':shares.quantile(q) for q in Q},
                grain_mu=g.sow_food.sum(min_count=1),cash_mu=g.sow_cash.sum(min_count=1),all_sown=g.sow_total.sum(min_count=1),sown_mean=g.sow_total.mean(),sown_n=g.sow_total.count(),
                other_gap=(g.sow_total-g.sow_food-g.sow_cash).sum(min_count=1),operated_mean=g.area_e.mean(),operated_n=g.area_e.count(),labor_mean=g.hhlabornum.mean(),labor_n=g.hhlabornum.count())
            for dim in ['region','scale','labor_group','role']:
                for k,n in g[dim].value_counts(dropna=False).items():tab('annual_composition',sample=sample,year=year,dimension=dim,category=k,n=n,denominator=len(g))
    for y,g in panel.groupby('year'):
        for dim in ['overall','region','scale']:
            groups=[('all',g)] if dim=='overall' else g.groupby(dim)
            for name,h in groups:
                for k,z in h.groupby('prior'):
                    for dst in ST:
                        n=z.state.eq(dst).sum();p,lo,hi=ci(n,len(z))
                        tab('transition_matrix_by_year',year=y,grouping=dim,group=name,origin=k,next=dst,n=n,denominator=len(z),p=p,lo95=lo,hi95=hi)
        z=g[g.x.ge(0)&g.y.ge(0)]; change=(z.y-z.x).abs()
        for name,h in [('overall',change),*change.groupby(pd.cut(z.x,[-1,.2,.8,1],labels=['low','middle','high']))]:
            tab('share_adjustment',year=y,prior_share=name,n=len(h),mean_abs=h.mean(),median_abs=h.median(),cluster_se=h.std()/np.sqrt(len(h)))
    # Repeated-household inference on two periods, including covariance for overlapping IDs.
    a=panel[panel.year.eq(2010)]
    for yr in range(2011,2016):
        b=panel[panel.year.eq(yr)]
        for state0 in ST[:-1]:
            for dest in ST[:-1]:
                samples=[v.loc[v.prior.eq(state0)&v.state.ne('unknown'),'state'].eq(dest).astype(float) for v in (a,b)]
                if min(map(len,samples))<2:continue
                u,v=samples; influence=((v-v.mean())/len(v)).subtract((u-u.mean())/len(u),fill_value=0)
                tab('clustered_stability',year=yr,origin=state0,next=dest,delta=v.mean()-u.mean(),unique_households=len(influence),
                    cluster_se=np.sqrt((influence**2).sum()*len(influence)/(len(influence)-1)))
    weights=pd.read_csv(path(R/'project_management/evidence/v5_r/r2/nfps_calibration_weights_2010.csv'))
    w=weights[weights.national_representation_eligible];eligible=raw.iloc[w.source_row.to_numpy()-2].copy()
    assert eligible.year.eq(2010).all() and np.allclose(w.operated_area_mu,eligible.area_e)
    raw['share_band']=bucket(raw.y);origin['share_band']=bucket(origin.y);eligible['share_band']=bucket(eligible.y)
    by=['state','share_band','region','scale','labor_group','role']
    comps=[]
    for name,g,wt in [('full',raw[raw.year.eq(2010)],None),('eligible',eligible,None),('calibrated',eligible,w.representation_weight_q.to_numpy()),('cohort',origin,None)]:
        g=g.assign(mass=1. if wt is None else wt);g.mass/=g.mass.sum()
        comps.append(g.groupby(by).mass.sum().rename(name))
    joint=pd.concat(comps,axis=1).fillna(0).reset_index()
    tables['initial_cohort_comparison']=joint.to_dict('records')
    # Identical observed high-grain families for the two model paths; no target-share grouping.
    hist=linked.sort_values('year')
    for y in range(2011,2016):
        hi=hist[(hist.year==y-1)&(hist.y>.8)][['actor']].merge(hist[hist.year==y],on='actor',validate='1:1')
        known,full=market_evidence(hi)
        for seed in [103,109]:
            acts=pd.read_csv(path(P/f'candidate/PRIVATE_DO_NOT_SHARE/{seed}_both_60_1.0_accounts.csv'))
            m=hi.merge(acts[(acts.year==y)&(acts.kind=='household')],left_on='actor',right_on='id',validate='1:1',suffixes=('','_m'))
            mf=m.base_area.gt(0)&m.operated.lt(1e-7)&(m.base_area-m.rented_out).abs().lt(1e-7)
            tab('land_exit_validation',year=y,seed=seed,n=len(hi),grain_continues=hi.sow_food.gt(0).sum(),cash_only=hi.state.eq('cash').sum(),other_only=hi.state.eq('other').sum(),
                zero_food_cash=hi.y.lt(0).sum(),stop_all=hi.state.eq('none').sum(),missing_de2=hi.area_de2.isna().sum(),identifiable=known.sum(),
                coverage=known.mean(),full_lease=full.sum(),indeterminate=(~known).sum(),partial_positive=(hi.area_de2.gt(0)&hi.area_e.gt(0)).sum(),
                model_full_all=mf.sum(),model_full_identifiable=mf[known.to_numpy()].sum(),model_zero_pair=(m.grain+m.crop<1e-7).sum())
    # Regional 2016-17 crop budget comparisons; matched sales and components only.
    budget=R/'results/national_crop_budget_basis_20261118/run_04/real2024_revision/record_selection_money2024.csv'
    bb=pd.read_csv(path(budget)).query('eligible and area>0')
    for (y,c),z in bb.groupby(['year','crop']):
        s=z[z.sales_amount.ge(0)]; per=s.sales_amount*s.to_2024/s.area
        tab('crop_return_source_comparison',year=y,crop=c,rows=len(z),regions=z.sm.nunique(),sales_n=len(s),
            sales_price_nominal=(s.loc[s.sales_quantity.gt(0),'sales_amount'].sum()/s.loc[s.sales_quantity.gt(0),'sales_quantity'].sum()),
            sales_real_per_mu=(s.sales_amount*s.to_2024).sum()/s.area.sum(),sale_p25=per.quantile(.25),sale_p75=per.quantile(.75),
            nonlabor_low=z.nonlabor_low.sum()/z.loc[z.nonlabor_low.notna(),'area'].sum(),
            labor_days=z.labor.sum()/z.loc[z.labor.notna(),'area'].sum())
    # Disjoint households; all target outcomes hidden until the prediction has been constructed.
    clean=panel[panel.prior.ne('unknown')&panel.state.ne('unknown')]
    donors=clean[~clean.index.isin(origin.hhid)]
    for y in range(2011,2016):
        test=clean[(clean.year==y)&clean.index.isin(origin.hhid)]
        for window,last in [('fixed',2010),('pooled_early',min(2012,y-1)),('expanding',y-1)]:
            train=donors[donors.year.le(last)]
            for cont in [False,True]:
                tab('rolling_transition_backtest',year=y,window=window,model='continuous' if cont else 'discrete',last_training_year=last,
                    train_pairs=len(train),**backtest(train,test,cont))
    # Replay only the 2015 v1 full-lessor decision on owned land; prohibit expansion in all eight cases.
    params=pd.read_csv(path(BASE/'parameter_provenance.csv')).set_index('parameter').value
    L=[float(params[c+'_labor_days']) for c in ['grain','cash_crop']]
    K=[float(params[c+'_other_cost_per_mu']) for c in ['grain','cash_crop']]
    alt=pd.read_csv(path(P/'alternative_accounts.csv')).query("year==2015 and option=='free'").set_index(['seed','id'])
    shap=[]; budgetlog=[]
    for seed in (103,109):
        g=pd.read_csv(path(P/f'candidate/PRIVATE_DO_NOT_SHARE/{seed}_both_60_1.0_accounts.csv'))
        g=g[g.id.isin(linked[linked.year.eq(2015)].actor)&g.year.eq(2015)&g.kind.eq('household')&g.target_share.gt(.8)&g.operated.lt(1e-7)]
        base=pd.read_csv(path(P/'candidate/PRIVATE_DO_NOT_SHARE/baseline.csv')).set_index('actor')
        for r in g.itertuples():
            m=base.loc[r.id];a=alt.loc[seed,r.id];args=(a.grain_unit_sales,a.cash_unit_sales,*K,*L,float(params['hired wage']),float(params.P0))
            values={}; over=r.base_area*m.intensity
            for C,H,T in product([0,1],repeat=3):
                fam=SimpleNamespace(sowing_intensity=m.intensity,area_limit=r.base_area,crop_mix_target=r.target_share,
                    family_days=(r.family_capacity if H else max(r.family_capacity,over*max(L)+1)))
                funds=r.working_funds if C else max(r.working_funds,over*(max(K)+max(L)*args[-2])+1)
                # Rent-out opportunity: compare own production plus opportunity rent on eligible idle rights.
                rent=r.expected_rent if T else 0.
                p=continuous_farm_plan(fam,r.base_area,r.rights,funds,rent,args,bool(T))
                values[C,H,T]=p['cultivated']
                if (C,H,T)==(1,1,1):assert p['cultivated']<1e-5
                budgetlog.append(dict(id=r.id,seed=seed,cash=C,labor=H,rent=T,cultivated=p['cultivated'],expected=p['expected'],
                    rental_receipt=r.rent_received,posted_rent=r.expected_rent,deal_rent=r.rent_received/r.rented_out,working_funds=r.working_funds,
                    opening_land=r.base_area,available_days=r.family_capacity,hired=p['hired'],cost=p['cost']))
            attrib=np.zeros(3)
            for perm in permutations(range(3)):
                s=[0,0,0]
                for j in perm:
                    before=values[tuple(s)];s[j]=1;attrib[j]+=(values[tuple(s)]-before)/6
            assert np.isclose(attrib.sum(),values[1,1,1]-values[0,0,0])
            shap.append(dict(seed=seed,id=r.id,cash=attrib[0],labor=attrib[1],rent=attrib[2],unconstrained=values[0,0,0],actual=values[1,1,1]))
    sh=pd.DataFrame(shap);tables['shapley_land_exit']=sh.groupby('seed').agg(n=('id','size'),**{k:(k,'sum') for k in ['cash','labor','rent','unconstrained','actual']}).reset_index().to_dict('records')
    out.mkdir(parents=True,exist_ok=False); priv=out/'PRIVATE_DO_NOT_SHARE';priv.mkdir()
    pd.DataFrame(budgetlog).to_csv(priv/'eight_budgets.csv',index=False)
    for n,v in tables.items():pd.DataFrame(v).to_csv(out/(n+'.csv'),index=False)
    pd.DataFrame(tables['annual_distribution']).query("sample=='B_balanced'").to_csv(out/'balanced_panel_distribution.csv',index=False)
    pd.DataFrame(hashes.items(),columns=['path','sha256']).to_csv(out/'source_manifest.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path,nargs='?',default=OUT); audit(p.parse_args().out)
