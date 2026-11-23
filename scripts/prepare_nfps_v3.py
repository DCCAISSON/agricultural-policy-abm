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
