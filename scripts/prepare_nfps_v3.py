"""Construct the V3 NFPS-only cohort. Raw records and link keys never leave the local workspace."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.prepare_household_portfolio_reference import RAW,MANIFEST
from core.extract_microdata_features import REGION_MAPPING

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=ROOT/'project_management/evidence/v5_r/r2/nfps_calibration_weights_2010.csv'
OUTPUT=ROOT/'results/diagnostics/national_crop_choice_soft_response_20261123'


def clean_area(frame, private_neighbors=None):
    # Identify only the two confirmed field errors; household linkage stays private.
    found=[]
    for year,col,value in [(2013,'sow_cash',400431.),(2014,'sow_food',1051055554560.)]:
        match=frame.index[frame.year.eq(year)&np.isclose(frame[col],value,rtol=1e-9)]
        assert len(match)==1
        found.append((match[0],col))
    if private_neighbors is not None:
        frame[frame.hhid.isin(frame.loc[[i for i,_ in found],'hhid'])].to_csv(private_neighbors,index=True)
    for index,col in found:
        frame.loc[index,[col,'sow_total']]=np.nan


def candidate_cohort(out=None):
    fields='year hhid id smgen area_b area_e area_e1 sow_total sow_food sow_cash sow_oother hhlabornum hhnumbers exp_live exp_live_1 area_in1 area_de2 prod_food'.split()
    raw=pd.read_csv(RAW,usecols=fields,dtype={'hhid':str,'id':str})
    dictionary=pd.read_excel(RAW.parent/'household variables2009-2015.xlsx',header=None).set_index(0)
    assert '现金性支出' in dictionary.loc['exp_live_1',1] and '转包出' in dictionary.loc['area_de2',1]
    if out is not None:
        priv=out/'PRIVATE_DO_NOT_SHARE';pub=out/'run_01'
        before=raw.groupby('year')[['sow_food','sow_cash','sow_total','sow_oother']].agg(['sum','count'])
    clean_area(raw,priv/'exception_neighbors.csv' if out is not None else None)
    weights=pd.read_csv(WEIGHTS); ix=weights.loc[weights.national_representation_eligible,'source_row'].to_numpy()-2
    e=raw.iloc[ix].copy()
    assert e.year.eq(2010).all() and np.allclose(e.area_e,weights.loc[weights.national_representation_eligible,'operated_area_mu'])
    geo={p:z for z,ps in REGION_MAPPING.items() for p in ps}
    old=pd.DataFrame(json.loads(Path(json.loads(MANIFEST.read_bytes())['jobs'][8]['cohort']).read_bytes())['farmers'])
    e['region']=e.smgen.map(geo)
    e['status']=np.select([e.sow_food.gt(0)&e.sow_cash.gt(0),e.sow_food.gt(0),e.sow_cash.gt(0),e.sow_oother.gt(0)],['mixed','grain','cash','other'],default='none')
    e['size']=pd.cut(e.area_e,[0,5,15,50,np.inf],right=False).astype(str)
    e['workers']=e.hhlabornum.clip(upper=3).fillna(-1)
    e['transfer']=np.select([e.area_in1.isna()|e.area_de2.isna(),e.area_in1.gt(0)&e.area_de2.gt(0),e.area_in1.gt(0),e.area_de2.gt(0)],['unknown','both','in','out'],default='neither')
    all2010=raw[raw.year.eq(2010)]; dup=all2010.loc[all2010.hhid.duplicated(False),'hhid']
    bad_id=e.hhid.isin(dup)|e.hhid.isna()|e.hhid.ne(e.id);bad_family=~e.hhnumbers.ge(1);bad_right=e.area_e1.lt(0)|e.area_e1.gt(e.area_e)
    pool=e[~(bad_id|bad_family|bad_right)]
    dims=['region','status','size','workers','transfer']
    groups=pool.groupby(dims,sort=True); sizes=groups.size(); quota=600*sizes/len(pool)
    n=np.floor(quota).astype(int);remaining=600-int(n.sum())
    n.loc[(quota-n).sort_values(ascending=False,kind='stable').index[:remaining]]+=1
    rng=np.random.default_rng(17)
    indices=np.concatenate([rng.choice(g.index,n.loc[key],replace=False) for key,g in groups])
    selected=e.loc[indices].sort_index().copy(); assert len(selected)==selected.hhid.nunique()==600
    cpi=pd.read_csv(ROOT/'config/experiments/national_real2024_cpi_20260909.csv')
    factor=(1+cpi.loc[cpi.year.between(2011,2024),'annual_change_percent']/100).prod()
    # Reconstruct the frozen V2 consumption reference; the sampling quotas never use these weights.
    w=weights[weights.national_representation_eligible].set_index(e.index)
    good=e.exp_live.gt(0)&e.hhnumbers.gt(0)&e.exp_live_1.between(0,e.exp_live)
    refs=pd.DataFrame(dict(size=w.farm_size,region=w.common_region,weight=w.representation_weight_q,
        share=e.exp_live_1/e.exp_live,person=e.exp_live/e.hhnumbers))[good]
    def median(g,c):
        g=g.sort_values(c)
        return g.loc[g.weight.cumsum().ge(g.weight.sum()/2),c].iloc[0]
    reference={}
    for dims0 in [['size','region'],['size'],[]]:
        for key,g in (refs.groupby(dims0) if dims0 else [((),refs)]):
            if len(g)>=25: reference[key]=(median(g,'share'),median(g,'person'))
    cost=e.exp_live_1.copy()
    for i in cost[cost.isna()].index:
        k=(w.at[i,'farm_size'],w.at[i,'common_region'])
        fraction,person=reference.get(k,reference.get(k[:1],reference[()]))
        cost.at[i]=fraction*(e.at[i,'exp_live'] if pd.notna(e.at[i,'exp_live']) else person*e.at[i,'hhnumbers'])
    assert np.allclose(cost.loc[old.source_row.to_numpy()-2].to_numpy()*factor,
                       old.economic_basis_20261021.str['annual_reported_cash_candidate_yuan2024'])
    annual=cost.loc[selected.index]*factor
    co=pd.DataFrame(dict(source_row=selected.index+2,cohort_agent_id=old.cohort_agent_id.to_numpy(),
         family_size=selected.hhnumbers.to_numpy(),region=selected.region.to_numpy(),economic_basis_20261021=[
         dict(annual_reported_cash_candidate_yuan2024=x,held_reserve_yuan2024=x/4) for x in annual]))
    if out is not None:
        co.to_json(priv/'cohort600.json',orient='records',force_ascii=False)
        after=raw.groupby('year')[['sow_food','sow_cash','sow_total','sow_oother']].agg(['sum','count'])
        pd.concat({'before':before,'after':after},axis=1).stack(0,future_stack=True).to_csv(pub/'survey_cleaning.csv')
        dist=[];tv=[]
        for dim in [[x] for x in dims]+[dims]:
            comps={k:g.groupby(dim).size()/len(g) for k,g in [('eligible',e),('selectable',pool),('original',e.loc[old.source_row.to_numpy()-2]),('new',selected)]}
            joint=pd.concat(comps,axis=1).fillna(0)
            tv.append(dict(dim='|'.join(dim),original=.5*(joint.original-joint.eligible).abs().sum(),new=.5*(joint.new-joint.eligible).abs().sum()))
            if len(dim)==1:
                for j,v in joint.iterrows():dist.append(dict(dim=dim[0],value=j,**v.to_dict()))
            else: joint.to_csv(priv/'joint.csv')
        pd.DataFrame(dist).to_csv(pub/'cohort_margins.csv',index=False)
        pd.DataFrame(tv).to_csv(pub/'cohort_distance.csv',index=False)
        pd.DataFrame([dict(raw=len(all2010),eligible=len(e),selectable=len(pool),id_excluded=bad_id.sum(),
             family_excluded_after_id=(~bad_id&bad_family).sum(),rights_excluded_after_prior=(~bad_id&~bad_family&bad_right).sum(),selected=len(co),
             old_overlap=len(set(selected.index)&set(old.source_row-2)),consumption_proxy=selected.exp_live_1.isna().sum())]).to_csv(pub/'selection.csv',index=False)
    return co


if __name__=='__main__':candidate_cohort(OUTPUT)


def prepare_v4_inputs(cohort, output, deflator, chfs=None, *, repair=False):
    """Read source definitions and estimate only on pre-outcome, non-cohort records."""
    from core.v4_household import features, repair_features, fit_categories, DIRECTIONS, FIELD_NAMES
    from scripts.run_national_household_portfolio import estimate_crop_transitions
    household = {
        'year':'reference year', 'hhid':'panel key (private)', 'id':'key crosscheck (private)',
        'samplast':'follow-up eligibility', 'smgen':'province', 'hhnumbers':'household composition',
        'hhlabornum':'laborers', 'area_b':'opening operated land', 'area_b1':'opening contracted land',
        'area_e':'closing operated land','area_e1':'contracted part of closing operated area',
        'area_in1':'confirmed leased in','area_de2':'confirmed leased out','area_de':'total decrease',
        'sow_total':'annual total sown', 'sow_food':'annual grain sown','sow_cash':'annual cash sown',
        'sow_oother':'other crops, validation only','prod_food':'grain output',
        'saving_1':'end-year deposits','saving_2':'end-year cash','saving_4':'outstanding debts, not available credit',
        'inc_wage_l':'local wage income','inc_wage_o':'migrant wage income','exp_live':'total living expenditure',
        'exp_live_1':'cash living expenditure',
    }
    for suffix in ('w','r','c','s','t','o'):
        for prefix in ('sow','prod','labor_d','labor_hd','labor_hf','rent'):
            household[f'{prefix}_{suffix}']='crop-specific input or output; see item'
    individual = {'hhid':'household linkage (private)','idind':'member identifier (private)','year':'reference year',
                  'relation':'relationship for follow-up','sex':'identity cross-check','age':'age structure and continuity',
                  'eduyears':'education, descriptive','selfhealth':'health, descriptive',
                  'homefarmdays':'realized local farm work; not availability',
                  'homenonfarmdays':'realized local nonfarm work','timeout':'outside work, overlap not assumed',
                  'outincome':'migrant income, contextual'}
    frame = pd.read_csv(RAW,usecols=list(household),dtype={'hhid':'string','id':'string'})
    members = pd.read_csv(RAW.parent/'2009-2015individual.csv',usecols=list(individual),dtype={'hhid':'string','idind':'string'})
    mapping=[]
    question = (RAW.parent/'2009-2017农户调查问卷.txt').read_text(encoding='utf-8-sig')
    for data, fields, kind, file in [(frame,household,'household',RAW),(members,individual,'member',RAW.parent/'2009-2015individual.csv')]:
        name='household' if kind=='household' else 'individual'
        dictionary=pd.read_excel(RAW.parent/f'{name} variables2009-2015.xlsx',header=None).set_index(0)[1]
        for field,role in fields.items():
            label=dictionary.get(field)
            needle=str(label).split('(')[0].split('（')[0]
            matches=[line.strip() for line in question.splitlines() if len(needle)>1 and needle in line]
            for y,g in data.groupby('year'):
                observed=g[field].notna()
                if field not in ('id','hhid','idind'): observed &= g[field].ge(0)
                mapping.append(dict(source='NFPS',file=file.name,year=y,field=field,question=matches[0] if matches else '',
                    dictionary=label,unit=('mu' if field.startswith(('area','sow')) else 'kg' if field.startswith('prod')
                        else 'day' if field.startswith(('labor_d','labor_hd')) or field.endswith('days') else
                        'yuan' if field.startswith(('saving','inc_','rent','exp_','labor_hf')) else 'see dictionary'),
                    object=kind,period='survey reference year; closing stock where specified',
                    key='year/household'+('/member' if kind=='member' else ''),role=role,
                    valid=int(observed.sum()),missing=int((~observed).sum()),n=len(g),missing_rate=float((~observed).mean()),missing_rule='blank; negative excluded for non-ID fields'))
    pd.DataFrame(mapping).to_csv(output/'field_mapping.csv',index=False)
    clean_area(frame)
    selected=frame.iloc[cohort.source_row.to_numpy()-2].copy().set_index('hhid')
    selected['actor']=cohort.cohort_agent_id.to_numpy()
    geo={p:z for z,ps in REGION_MAPPING.items() for p in ps}
    regions=sorted(set(geo.values()))
    fit={}; report=[]; transfer_report=[]; pairs=[]
    for y in (2010,2011,2012):
        paired=estimate_crop_transitions(RAW,set(selected.index),True,y,True)
        cols='hhid area_e hhlabornum saving_1 saving_2 area_in1 area_de2 sow_oother sow_total'.split()
        if repair: cols.append('area_e1')
        keep=frame.year.eq(y-1)&frame.hhid.isin(paired.hhid)
        keep2=frame.year.eq(y)&frame.hhid.isin(paired.hhid)
        p=paired.merge(frame.loc[keep,cols],on='hhid',validate='1:1').merge(
            frame.loc[keep2,['hhid','sow_total','area_in1','area_de2']],on='hhid',suffixes=('','_next'),validate='1:1')
        p['year']=y;p['region']=p.smgen.map(geo)
        pairs.append(p[p.region.notna()])
        all_pairs=pd.concat(pairs,ignore_index=True)
        ar=all_pairs.area_e.clip(lower=0).fillna(0).to_numpy()
        lag=all_pairs.prior_share.to_numpy()
        median=float(np.nanmedian(lag))
        starts=np.where(np.isfinite(lag),lag,median)
        labor=all_pairs.hhlabornum.where(all_pairs.hhlabornum.ge(0)).fillna(all_pairs.hhlabornum.median())
        liquid=all_pairs[['saving_1','saving_2']].where(lambda f:f.ge(0)).sum(axis=1,min_count=1)
        liquid=liquid.fillna(liquid.groupby(all_pairs.region).transform('median')).fillna(0)*deflator
        flow=all_pairs.area_in1.gt(0).astype(int)-all_pairs.area_de2.gt(0).astype(int)
        x=features(starts,ar,labor,liquid,flow,all_pairs.region.to_numpy(),regions,all_pairs.year.to_numpy())
        if repair:
            x=repair_features(lag,ar,labor,flow,all_pairs.region.to_numpy(),regions,all_pairs.year.to_numpy())
        cluster=all_pairs.hhid.to_numpy(dtype=str)
        diff=all_pairs.share.to_numpy()-starts
        usable=np.isfinite(diff)
        if repair: usable &= np.isfinite(lag)
        k=np.where(diff>1e-7,2,np.where(diff< -1e-7,0,1))
        group=3*np.digitize(ar,[5.,15.,30.])+np.digitize(starts,[1e-8,1.-1e-8])
        amplitudes=np.empty((12,2,19))
        ep=np.empty((12,2))
        for j,code in enumerate((0,2)):
            pool=usable&(k==code)
            endpoint=all_pairs.share.eq(0. if j==0 else 1.).to_numpy()
            relative=np.divide(np.abs(diff),starts if j==0 else 1.-starts,out=np.zeros_like(starts),where=(starts if j==0 else 1.-starts)>1e-8)
            for cell in range(12):
                local=pool&(group==cell)
                donor=local if local.sum()>=25 else pool
                ep[cell,j]=endpoint[donor].mean()
                interior=donor&~endpoint
                if not interior.any(): interior=pool&~endpoint
                amplitudes[cell,j]=np.quantile(relative[interior],np.linspace(.05,.95,19))
                transfer_report.append(dict(training_end=y,group=cell,direction=DIRECTIONS[code],n=int(local.sum()),
                    endpoint_probability=ep[cell,j],p05_fraction_of_room=amplitudes[cell,j,0],
                    median_fraction_of_room=amplitudes[cell,j,9],p95_fraction_of_room=amplitudes[cell,j,-1]))
        out=dict(amplitudes=amplitudes,endpoint=ep,regions=regions,prior_median=median,return_scale=3000.)
        names=('intercept',*FIELD_NAMES,*regions[1:],'year2011','year2012')
        rules=[
            ('direction',usable,k,3,DIRECTIONS),
            ('stay',all_pairs.sow_total_next.ge(0).to_numpy(),all_pairs.sow_total_next.gt(0).to_numpy(int),2,('not_sowing','sowing')),
            ('rent_in',all_pairs.area_in1_next.ge(0).to_numpy(),all_pairs.area_in1_next.gt(0).to_numpy(int),2,('not_observed_positive','positive')),
            ('rent_out',all_pairs.area_de2_next.ge(0).to_numpy(),all_pairs.area_de2_next.gt(0).to_numpy(int),2,('not_observed_positive','positive'))]
        if repair:
            names=tuple(k0 for k0 in names if k0!='log_liquidity')+('prior_two_crop_known',)
            right=all_pairs.area_e1.gt(0)
            operators=(all_pairs.area_e.gt(0)|right).to_numpy()
            landless=(all_pairs.area_e.eq(0)&~right).to_numpy()
            next_sown=all_pairs.share.notna().to_numpy(int)
            next_known=all_pairs.sow_total_next.ge(0).to_numpy()
            rules=rules[2:]+[(tag,group1&next_known,next_sown,2,('not_sowing_two','sowing_two'))
                             for tag,group1 in [('stay_operator',operators),('enter_landless',landless)]]
            out['direction_by_prior']=np.zeros((3,3,x.shape[1]))
            ptype=np.digitize(starts,[1e-8,1.-1e-8])
            dir_rows=[]
            for j,allowed in enumerate(((1,2),(0,1,2),(0,1))):
                keep=usable&(ptype==j)
                assert np.isin(k[keep],allowed).all()
                ydir=np.searchsorted(allowed,k[keep])
                co,se=fit_categories(x[keep],ydir,cluster[keep],len(allowed))
                out['direction_by_prior'][j,list(allowed)]=co
                for ci,action in enumerate(allowed):
                    for name,b,s in zip(names,co[ci],se[ci]):
                        dir_rows.append(dict(training_end=y,rule='direction_'+('cash','mixed','grain')[j],
                            action=DIRECTIONS[action],term=name,coefficient=b,cluster_se=s,
                            n=int(keep.sum()),households=len(set(cluster[keep])),action_n=int((ydir==ci).sum())))
            report.extend(dir_rows)
            pool=~np.isfinite(lag)&all_pairs.share.notna().to_numpy()
            out['entry_shares']={'all':np.sort(all_pairs.share.to_numpy()[pool])}
            assert len(out['entry_shares']['all'])>0
            for z in regions:
                for cell in range(4):
                    keep=pool&all_pairs.region.eq(z).to_numpy()&(np.digitize(ar,[5.,15.,30.])==cell)
                    if keep.sum()>=25: out['entry_shares'][z,cell]=np.sort(all_pairs.share.to_numpy()[keep])
                    draw=out['entry_shares'].get((z,cell),out['entry_shares']['all'])
                    transfer_report.append(dict(training_end=y,region=z,group=cell,direction='entry_share',
                         n=int(keep.sum()),n_donors=len(draw),entry_cash=(draw==0).mean(),
                         entry_mixed=((draw>0)&(draw<1)).mean(),entry_grain=(draw==1).mean()))
        for rule,valid,target,count,categories in rules:
            coef,se=fit_categories(x[valid],target[valid],cluster[valid],count)
            out[rule]=coef
            for j,c in enumerate(categories):
                for name,b,s in zip(names,coef[j],se[j]):
                    report.append(dict(training_end=y,rule=rule,action=c,term=name,coefficient=b,cluster_se=s,
                        n=int(valid.sum()),households=len(set(cluster[valid])),action_n=int((target[valid]==j).sum())))
        fit[y]=out
    pd.DataFrame(report).to_csv(output/'behavior_parameters.csv',index=False)
    pd.DataFrame(transfer_report).to_csv(output/'adjustment_distribution.csv',index=False)
    mc=members.loc[members.year.eq(2010)&members.age.between(16,79)].copy()
    mc['used']=mc.homefarmdays.where(mc.homefarmdays.between(0,365))
    mc['nonfarm']=mc[['homenonfarmdays','timeout']].where(lambda d:d.ge(0)&d.le(365)).max(axis=1)
    mc['stratum']=np.select([mc.nonfarm.gt(0),mc.nonfarm.eq(0)],['reported_nonfarm','reported_no_nonfarm'],default='unknown')
    cf_fields=['urban','qh1','qh2','qh3','qh4']
    cf_file=next((RAW.parents[1]/'CFPS/2010').rglob('cfps2010adult*.dta'))
    with pd.io.stata.StataReader(cf_file) as reader:
        labels=reader.variable_labels()
    cf=pd.read_stata(cf_file,columns=cf_fields,convert_categoricals=False)
    valid=cf[cf.urban.eq(0)&cf.qh1.between(1,12)&cf.qh2.between(1,31)&cf.qh3.gt(0)&cf.qh3.le(24)]
    days=valid.qh1*valid.qh2
    ref=days.groupby(valid.qh4).agg(n='count',median='median',upper_75=lambda s:s.quantile(.75))
    ref.to_csv(output/'labor_parameters.csv')
    # Only migrant time is populated in these local NFPS member years.
    # The upper quartile of CFPS reported days proxies a ceiling, not time availability measured in NFPS.
    mc['capacity']=np.minimum(np.where(mc.nonfarm.gt(0),ref.at[1,'upper_75'],days.quantile(.75)),
                               365.-mc.nonfarm.fillna(0))
    capacity=mc.groupby('hhid').capacity.mean()
    for field in cf_fields:
        keep=cf[field].notna()&cf[field].ge(0)
        mapping.append(dict(source='CFPS',file=cf_file.name,year=2010,field=field,question='',
            dictionary=labels[field],unit='see questionnaire label',object='adult',
            period='past year for qh1-qh3; now for qh4',key='none (group only)',role='labor reference only',
            valid=int(keep.sum()),missing=int((~keep).sum()),n=len(cf),missing_rate=float((~keep).mean()),
            missing_rule='Stata missing and negative codes'))
    liquid_proxy=np.nan
    if chfs:
        ch_fields=['b1001','d1101','d1105','k1101','e3006']
        with pd.io.stata.StataReader(chfs) as reader:
            labels=reader.variable_labels()
        ch=pd.read_stata(chfs,columns=ch_fields,convert_categoricals=False)
        for field in ch_fields:
            keep=ch[field].notna()&ch[field].ge(0)
            mapping.append(dict(source='CHFS',file=Path(chfs).name,year=2011,field=field,question='',
                dictionary=labels[field],unit='yuan' if field not in ('b1001','d1101') else 'classification',
                object='household',period='survey-date stock; prior-year agricultural participation', key='none (group only)',
                role='liquidity group reference; debt is not available credit',valid=int(keep.sum()),missing=int((~keep).sum()),
                n=len(ch),missing_rate=float((~keep).mean()),missing_rule='missing excluded; d1101=2 denotes no bank account'))
        dep=ch.d1105.where(ch.d1105.ge(0)).mask(ch.d1101.eq(2),0.)
        cash=ch.k1101.where(ch.k1101.ge(0))
        stocks=(dep+cash).where(ch.b1001.eq(1))
        cpi=pd.read_csv(ROOT/'config/experiments/national_real2024_cpi_20260909.csv')
        factor=(1+cpi.loc[cpi.year.between(2012,2024),'annual_change_percent']/100).prod()
        liquid_proxy=float(stocks.median()*factor)
        pd.DataFrame([dict(reference='2011 CHFS agricultural participants',joint=int(stocks.count()),
                            factor=factor,median_liquid_yuan2024=liquid_proxy)]).to_csv(output/'liquidity_parameters.csv',index=False)
    pd.DataFrame(mapping).to_csv(output/'field_mapping.csv',index=False)
    prod = pd.DataFrame(index=selected.index)
    out_prod = []
    portfolio_a = np.zeros(len(selected))
    portfolio_y = np.zeros(len(selected))
    linked_keys=set(estimate_crop_transitions(RAW,set(),True,2010,True).hhid)
    for crop in ('w','r','c','s','t','o'):
        a,q=f'sow_{crop}',f'prod_{crop}'
        obs=frame[frame.year.le(2010)&frame[a].gt(0)&frame[q].gt(0)&~frame.duplicated(['year','hhid'],False)].copy()
        obs['region']=obs.smgen.map(geo)
        obs=obs[obs.region.notna()]
        obs['y']=obs[q]/obs[a]
        old=obs[obs.year.eq(2009)&~obs.hhid.isin(selected.index)].groupby('region').y.median()
        med=obs[~obs.hhid.isin(selected.index)].groupby('region').y.median()
        hist=obs[obs.year.lt(2010)].assign(theta=lambda f:np.log(f.y/f.region.map(old))).set_index('hhid').theta
        hold=obs[obs.year.eq(2010)&obs.hhid.isin(hist.index)&obs.hhid.isin(linked_keys)&~obs.hhid.isin(selected.index)].copy()
        base=hold.region.map(old)
        prediction=base*np.exp(hold.hhid.map(hist).clip(-.5,.5)/3)
        rmse=lambda x:float(np.sqrt(np.mean(np.square(np.log(x/hold.y)))))
        old_rmse,new_rmse=rmse(base),rmse(prediction)
        accept=len(hold)>=30 and new_rmse<old_rmse
        obs['theta']=np.log(obs.y/obs.region.map(med))
        by=obs[obs.year.eq(2010)|obs.hhid.isin(linked_keys)].groupby('hhid').theta.agg(['size','mean'])
        by_theta=by['mean'].clip(-.5,.5) * by['size']/(by['size']+2)
        crop_y=selected.smgen.map(geo).map(med) * np.exp(selected.index.map(by_theta).fillna(0.) if accept else 0.)
        area=selected[a].fillna(0.).clip(lower=0.).to_numpy()
        portfolio_a+=area
        portfolio_y+=area*crop_y.fillna(float(med.median())).to_numpy()
        out_prod.append(dict(crop=crop,repeat_available=int(by['size'].gt(1).sum()),forecast_n=len(hold),log_rmse_region=old_rmse,
            log_rmse_shrunk=new_rmse,use_individual=accept,selected_observed=int(selected.index.isin(by.index).sum()),
            **{f'base_{z}':v for z,v in med.items()}))
    pd.DataFrame(out_prod).to_csv(output/'production_parameters.csv',index=False)
    prod['grain_yield']=np.divide(portfolio_y,portfolio_a,out=np.full(len(portfolio_a),np.nan),where=portfolio_a>0.)
    prod['labor']=selected.index.map(capacity)*selected.hhlabornum
    prod['initial_liquidity']=np.where(selected[['saving_1','saving_2']].notna().any(axis=1),np.nan,liquid_proxy)
    prod['workers']=selected.hhlabornum.where(selected.hhlabornum.ge(0)).fillna(0.)
    prod['flow']=selected.area_in1.gt(0).astype(int)-selected.area_de2.gt(0).astype(int)
    prod['other']=selected.sow_oother
    prod['actor']=selected.actor
    prod['sector_labor_proxy']=prod.labor.isna()
    prod['known_rights']=selected.area_e1.notna()
    pd.DataFrame([dict(n=len(selected),unknown_rights=int(selected.area_e1.isna().sum()),
        unknown_rights_operated_mu=selected.loc[selected.area_e1.isna(),'area_e'].sum(),
        reported_zero_rights=int(selected.area_e1.eq(0).sum()),no_member_proxy=int(prod.labor.isna().sum()),
        no_composition_yield=int(prod.grain_yield.isna().sum()))]).to_csv(output/'initial_coverage.csv',index=False)
    return fit,prod.set_index('actor')
