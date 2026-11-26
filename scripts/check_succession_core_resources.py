"""Condition on fixed opening rights; do not replay choices or evaluate policy."""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from core.succession_economic_core import offer, solve
from scripts.estimate_succession_activity_budgets import DICT_EARLY, EARLY, HOUSEHOLD, LINK, estimate
from scripts.prepare_succession_budget_inputs import prepare
from scripts.run_succession_annual_allocation import csv_write, json_write
from scripts.run_succession_empirical_budget import settings


def run(root,out):
    design,_,_ = settings()
    assets,_ = prepare(root,design)
    economy, evidence, _, source = estimate(root,2010)
    reference = next(a["hired_wage_p50_2024"] for a in evidence if a["crop"]=="rice")
    rows = []
    for region in assets:
        plots = defaultdict(list)
        for p in region.plots:
            plots[p.owner].append(offer(p,"none",0.))
        summary = defaultdict(lambda: defaultdict(float))
        for a in region.households:
            own = plots[a.id]
            mu = sum(p.area for p in own)
            size = "over_30" if mu>30 else "up_to_30"
            hire = replace(economy,hired_limit=a.labor,hired_wage=reference)
            for case,cash,days,tech in (("opening",a.cash,a.labor,economy),
                    ("cash_x2",a.cash*2,a.labor,economy),("family_days_x2",a.cash,a.labor*2,economy),
                    ("both_x2",a.cash*2,a.labor*2,economy),("bounded_hire",a.cash,a.labor,hire)):
                plan = solve(tech,own,cash,days,a.wage,a.job_days,a.market)
                d = summary[size,case]
                d["households"]+=1
                d["physical_mu"]+=mu
                d["planted_mu"]+=sum(n for choices in plan.area.values() for _,n in choices)
                d["grain_mu"]+=sum(n for choices in plan.area.values() for c,n in choices if c.grain)
                d["hired_days"]+=plan.hired
                d["hired_cash_2024"]+=plan.hired*tech.hired_wage
                d["cash_binding_households"]+=int(plan.cash_shadow>1e-8)
                d["labor_binding_households"]+=int(plan.time_shadow>1e-8)
                d["objective_2024"]+=plan.value
        rows.extend(dict(region=region.name,initial_scale=size,case=case,**d)
                    for (size,case),d in sorted(summary.items()))
    folder = root/HOUSEHOLD
    dictionary = pd.read_excel(folder/DICT_EARLY,header=None,usecols=[0,1])
    land = ["area_b","area_e","area_e1","area_in1","area_de2"]
    definitions = dictionary.loc[dictionary[0].isin(land)].set_index(0)[1]
    assert set(definitions.index)==set(land) and definitions.index.is_unique
    assert all(term in definitions.loc[key] for key,term in zip(land,("年初","年末","承包田","转包入","转包出"),strict=True))
    keys = ["year","hhid","id","smgen"]
    raw = pd.read_csv(folder/EARLY,usecols=keys+land,dtype={"hhid":"string","id":"string"})
    cohort = pd.DataFrame(json.loads((root/LINK).read_bytes()))
    known = raw.iloc[cohort.source_row.to_numpy()-2].hhid
    data = raw.loc[raw.year.between(2010,2015)&raw.hhid.notna()&raw.hhid.eq(raw.id)].copy()
    data = data.loc[~data.duplicated(["hhid","year"],keep=False)]
    data["scope"] = data.hhid.isin(known).map({True:"frozen_cohort",False:"noncohort"})
    counts = []
    for (scope,year),a in data.groupby(["scope","year"]):
        ok = a[["area_in1","area_de2"]].ge(0).all(axis=1)
        stock_ok = a[["area_b","area_e"]].ge(0).all(axis=1)
        b,c = a.loc[ok],a.loc[stock_ok]
        counts.append(dict(scope=scope,year=year,records=len(a),both_transfer_known=len(b),
                           positive_in=int(b.area_in1.gt(0).sum()),positive_out=int(b.area_de2.gt(0).sum()),
                           reported_in_mu=b.area_in1.sum(),reported_out_mu=b.area_de2.sum(),
                           stock_pair_known=len(c),opening_area_mu=c.area_b.sum(),closing_area_mu=c.area_e.sum()))
    out.mkdir(parents=True,exist_ok=False)
    csv_write(out/"opening_resource_sensitivity.csv",rows)
    csv_write(out/"nfps_reported_land_scope.csv",counts)
    csv_write(out/"land_dictionary_trace.csv",[dict(field=k,label=v) for k,v in definitions.items()])
    json_write(out/"receipt.json",dict(source_sha256=source,hired_wage_2024=reference,
             boundary="Same opening rights/CPI; one season, no market, no fiscal treatment, no RP forecast; "
             "hire limit = household opening labor; no inferred mechanization or external pool. "
             "Land rows aggregate only reporters with both valid fields, not new transaction counts."))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args = parser.parse_args()
    run(args.project,args.out)
