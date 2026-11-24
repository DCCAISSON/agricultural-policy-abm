"""Aggregate continuity check for the 19 newly eligible NFPS candidate families."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.prepare_household_portfolio_reference import RAW, MANIFEST
from scripts.prepare_nfps_v3 import candidate_cohort


def audit_continuity(output: Path) -> pd.DataFrame:
    early=RAW.parent/'1986-2008household.csv'
    history,flags,roster={}, {}, {}
    for f,k,flag,cols in [(early,'hh_id','nh1a_13','nh4a2_01 nh4a2_03 nh4a2_18 nh4a2_08 nh4a2_14'),
                           (RAW,'hhid','samplast','sow_r prod_r labor_d_r fer_r mach_r')]:
        fields=cols.split()
        d=pd.read_csv(f,usecols=[k,'year',flag]+fields+(['id'] if f==RAW else []),dtype={k:str,'id':str})
        if f==RAW:
            current=set(d.iloc[candidate_cohort().source_row.to_numpy()-2][k])
            farmers=json.loads(Path(json.loads(MANIFEST.read_bytes())['jobs'][8]['cohort']).read_bytes())['farmers']
            base=d.iloc[[r['source_row']-2 for r in farmers]]
            cohort=set(base.loc[~d.duplicated(['year',k],keep=False).loc[base.index],k])
            d=d[d.id.eq(d[k])]
        else:d[k]=d[k].str.removesuffix('.0')
        d=d[d.year.between(2003,2010)&~d.duplicated(['year',k],keep=False)]
        for y,g in d.groupby('year'):
            flags[int(y)]=g.set_index(k)[flag].to_dict()
            history[int(y)]=set(g.loc[g[fields[:2]].gt(0).all(axis=1)&g[fields[2:]].ge(0).all(axis=1),k])
    candidates=sorted(k for k in cohort if sum(k in history[y] for y in (2009,2010))<2 and sum(k in h for h in history.values())>=2)
    assert len(candidates)==19
    for name,key,cols in [('2003-2008','hh_id','nh1b_1 nh1b_2 nh1b_3'),('2009-2015','hhid','relation sex age')]:
        cols=cols.split();d=pd.read_csv(RAW.parent/(name+'individual.csv'),usecols=['year',key]+cols,dtype={key:str})
        d[key]=d[key].str.removesuffix('.0')
        d=d[d[key].isin(candidates)&d.year.between(2003,2010)&d[cols[1]].isin([1,2])&d[cols[2]].between(0,110)]
        for (k,y),g in d.groupby([key,'year']):
            roster[k,int(y)]=list(g[cols].itertuples(index=False,name=None))
    cases=[]
    for i,k in enumerate(candidates,1):
        certain=True; uninterrupted=True; n=0; possible=0; observed=0
        for y in range(2010,2002,-1):
            if k in history[y]:
                n+=int(certain); possible+=int(uninterrupted); observed+=1
            if y==2003:break
            mark=flags[y].get(k)
            if mark==2:uninterrupted=False
            # Require two distinct relation/sex/age matches, including the household head.
            old=roster.get((k,y-1),[]); new=roster.get((k,y),[])
            matched=[]
            for rel,sex,age in old:
                hits=[v for v in new if v[:2]==(rel,sex) and abs(v[2]-age-1)<=1]
                if len(hits)==1 and sum(r[:2]==(rel,sex) and abs(hits[0][2]-r[2]-1)<=1 for r in old)==1:matched.append(rel)
            certain=certain and mark==1 and 1 in matched and len(matched)>=2
        state='追访及成员支持两期' if n>=2 else ('明确断链后不足两期' if possible<2 else '不能确认两期')
        cases.append([i,observed,n,state,k in current])
    cases=pd.DataFrame(cases,columns=['audit_case','valid_same_code_years','confirmed_chain_years','finding','in_current_cohort'])
    cases.to_csv(output/'rice_history_19.csv',index=False,encoding='utf-8-sig')
    return cases


if __name__=='__main__':
    import sys
    print(Counter(audit_continuity(Path(sys.argv[1])).finding))
