"""Audit detailed accounts and describe historical correspondence without refitting."""

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from math import fsum,isclose
from pathlib import Path

import pandas as pd

from core.regional_succession_budget import make_budget,public_state,worlds_at_start
from scripts.prepare_succession_budget_inputs import prepare
from scripts.run_succession_annual_allocation import csv_write,json_write,seeds_of
from scripts.run_succession_empirical_budget import choose,settings
from scripts.succession_annual_learning import state_key


def check_accounts(world, fiscal, fields: list[dict], actors: list[dict]) -> None:
    require = lambda a,b: isclose(a,b,rel_tol=1e-12,abs_tol=1e-7)
    assert len(fields)==len(world.plots)==len({f["plot"] for f in fields})
    assert {r["id"] for r in actors}==set(world.operators) and len(actors)==len(world.operators)
    assert require(fsum(f["physical_area"] for f in fields),world.land)
    assert require(fsum(f["planted_area"] for f in fields),fsum(a["operation_area"] for a in actors))
    assert require(fsum(f["support"] for f in fields),fsum(a["support"] for a in actors))
    assert require(fsum(f["support"] for f in fields),fiscal["spending"])
    assert require(fsum(f["commitment"] for f in fields),fiscal["committed"])
    assert require(fsum(f["harvest"] for f in fields if f["crop"]=="grain"),fiscal["grain_kg"])
    assert require(fsum(f["rent"] for f in fields),fsum(a["rent_paid"] for a in actors))
    assert require(fsum(f["rent"] for f in fields),fsum(a["rent_income"] for a in actors))
    for p,f in zip(sorted(world.plots,key=lambda x:x.id),fields,strict=True):
        assert p.id==f["plot"] and p.owner==f["owner"]
        if f["lease"]:
            u = f["assigned_operator"]
            assert p.tradable and p.owner!=u and world.operators[p.owner].market==world.operators[u].market
            assert f["ask"]<=f["rent"]<=f["bid"] and f["contract_start"]<=world.year<=f["contract_end"]
        if f["crop"]!="idle":
            assert f["operator"] in world.operators and f["harvest"]>0
        if f["support"]>0:
            assert f["eligible"] and f["crop"]=="grain"
    for a in actors:
        assert a["closing"]>=0 and a["post_rent_cash"]>=-1e-7
        assert require(a["opening"]+a["income"]+a["restart_capital"]-a["consumption"],a["closing"])
        assert a["farm_days"]+a["recovery_days"]+a["job_days"]<=a["labor_capacity"]+1e-7
        assert a["inputs"]+a["recovery_cost"]-a["pre_support"]<=a["liquidity"]+1e-7


def survey_panel(project: Path) -> pd.DataFrame:
    folder=project/"校准用数据源/全国农村固定观察点"
    link=project/"results/diagnostics/national_crop_choice_soft_response_20261123/PRIVATE_DO_NOT_SHARE/cohort600.json"
    reference=pd.DataFrame(json.loads(link.read_bytes()))
    cols=["year","hhid","id","smgen","area_e","sow_food","sow_cash","prod_food"]
    data=pd.read_csv(folder/"2009-2015household.csv",usecols=cols,dtype={"hhid":"string","id":"string"})
    key=data.iloc[reference.source_row.to_numpy()-2].reset_index(drop=True)[["hhid","smgen"]].copy()
    key["actor"]=[f"H{i:04}" for i in range(len(key))]
    after=data[data.year.between(2011,2015)&data.hhid.notna()&data.hhid.eq(data.id)].copy()
    after=after.loc[~after.duplicated(["year","hhid"],keep=False)]
    return after.merge(key,on=["hhid","smgen"],validate="many_to_one").set_index(["year","actor"])


def run(project: Path, trained: Path, out: Path) -> None:
    design=settings()[0]
    assets,_=prepare(project,design)
    budget=make_budget(assets,design)
    menu=budget.menu()
    frozen=json.loads((trained/"frozen.json").read_bytes())
    source=survey_panel(project)
    seeds=list(seeds_of(design["fixed_dev_seeds"]))[:4]
    out.mkdir(parents=True,exist_ok=False)
    matched, counts = [], defaultdict(int)
    # P0 correspondence is retrospective; Q replay checks bookkeeping only.
    for seed,policy,horizon in [*((s,"fixed_00",5) for s in seeds),(seeds[0],"q",8)]:
        worlds=worlds_at_start(seed,assets,design)
        for year in range(1,horizon+1):
            key=state_key(public_state(worlds,design,budget))
            action=choose(policy,key,frozen,False)[0]
            worked=defaultdict(lambda:[0.,0.,0.])
            for world,(instrument,cap) in zip(worlds,menu[action],strict=True):
                scenario=world.scenario
                world.scenario=replace(scenario,budget=cap)
                row,fields,agents,_=world.step(instrument,ledger=True)
                world.scenario=scenario
                check_accounts(world,row,fields,agents)
                counts["region_years"]+=1
                counts["actor_years"]+=len(agents)
                counts["lot_years"]+=len(fields)
                if policy=="q":
                    continue
                for f in fields:
                    if f["crop"]=="grain":
                        worked[f["operator"]][0]+=f["physical_area"]
                        worked[f["operator"]][2]+=f["harvest"]
                    elif f["crop"]=="cash":
                        worked[f["operator"]][1]+=f["physical_area"]
            if policy=="q":
                continue
            rows=source.loc[2010+year]
            comparable=rows[["sow_food","sow_cash"]].notna().all(axis=1)&rows[["sow_food","sow_cash"]].ge(0).all(axis=1)
            sample=rows.loc[comparable]
            kg=sample.prod_food.notna()&sample.prod_food.ge(0)
            model=[worked[i] for i in sample.index]
            g,c,kg_sim=zip(*model,strict=True)
            survey_total=sample.sow_food+sample.sow_cash
            matched.append(dict(seed=seed,year=2010+year,original_cohort=600,matched=len(rows),
                comparable_crop=len(sample),comparable_grain_kg=int(kg.sum()),
                survey_sown=int(survey_total.gt(0).sum()),sim_sown=sum(a+b>0 for a,b in zip(g,c)),
                survey_mixed=int((sample.sow_food.gt(0)&sample.sow_cash.gt(0)).sum()),
                sim_mixed=sum(a*b>0 for a,b in zip(g,c)),
                survey_grain_mu_seasons=sample.sow_food.sum(),sim_grain_physical_mu=fsum(g),
                survey_cash_mu_seasons=sample.sow_cash.sum(),sim_cash_physical_mu=fsum(c),
                survey_grain_kg=sample.loc[kg,"prod_food"].sum(),
                sim_grain_kg=fsum(v for v,ok in zip(kg_sim,kg) if ok)))
    csv_write(out/"historical_household_comparison.csv",matched)
    json_write(out/"receipt.json",dict(status="passed detailed structural accounts",**counts,
        empirical_status="retrospective same-cohort description only; annual mu-seasons vs modeled physical mu not a pass criterion",
        panel_rules="2011-15 same household and province, duplicate household/year excluded; crop missing distinct from zero",
        history="four prespecified first fixed development seeds, P0, first five years; no parameters fitted"))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("project","trained","out"):
        cli.add_argument(key,type=Path)
    args=cli.parse_args()
    run(args.project,args.trained,args.out)
