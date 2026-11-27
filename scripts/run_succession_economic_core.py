"""Recompute the economic-core diagnostics without reading or training policy tables."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import json
from math import fsum, isclose
from pathlib import Path
import platform

import pandas as pd
import scipy

from core.regional_succession_budget import make_budget, worlds_at_start
from core.succession_economic_core import LEGACY_ACTIVITIES
from scripts.check_succession_empirical_households import check_accounts, survey_panel
from scripts.estimate_succession_activity_budgets import estimate
from scripts.prepare_succession_budget_inputs import digest, prepare
from scripts.run_succession_annual_allocation import csv_write, json_write
from scripts.run_succession_empirical_budget import settings


LOCK = Path("docs/全国经营接续/经济内核/设计.json")


def _heterogeneous_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(world, row, fields, actors):
    if world.scenario.economics is None:
        check_accounts(world,row,fields,actors)
        return
    near = lambda a,b:isclose(a,b,rel_tol=1e-12,abs_tol=1e-7)
    assert len(fields)==len({f["plot"] for f in fields})==len(world.plots)
    assert near(fsum(f["physical_area"] for f in fields),world.land)
    assert near(fsum(f["support"] for f in fields),row["spending"])
    assert near(fsum(f["commitment"] for f in fields),row["committed"])
    assert row["spending"]<=row["committed"]+1e-7<=world.scenario.budget+2e-7
    assert near(fsum(f["grain_kg"] for f in fields),row["grain_kg"])
    assert near(fsum(f["planted_area"] for f in fields),fsum(a["operation_area"] for a in actors))
    for f,p in zip(fields,sorted(world.plots,key=lambda p:p.id),strict=True):
        assert f["plot"]==p.id and f["owner"]==p.owner
        assert near(f["planted_area"],fsum(x["area"] for x in f["crop_accounts"].values()))
        assert 0<=f["planted_area"]<=p.area+1e-7 and f["support"]<=f["commitment"]+1e-7
        if f["planted_area"]>0:
            assert f["operator"] in world.operators
        if f["lease"]:
            assert p.tradable and p.owner!=f["operator"] and f["ask"]<=f["rent"]<=f["bid"]
            assert world.operators[p.owner].market==world.operators[f["operator"]].market
            assert f["contract_start"]<=world.year<=f["contract_end"]
    for a in actors:
        assert a["closing"]>=0 and a["post_rent_cash"]>=-1e-7
        assert near(a["opening"]+a["restart_capital"]+a["income"]-a["consumption"],a["closing"])
        assert a["farm_days"]+a["job_days"]<=a["labor_capacity"]+1e-7
        assert a["inputs"]+a["hired_cost"]-a["pre_support"]<=a["liquidity"]+1e-7
        assert near(a["sales"]+a["retained_value"],a["gross_output_value"])
        assert near(a["support"],a["pre_support"]+a["post_support"])
        assert near(a["profit"],a["sales"]-a["inputs"]-a["hired_cost"]-a["rent_paid"]-a["fees"]-a["setup"])


def _path(args):
    arm, seed, economic, design, assets, scopes, years = args
    worlds = worlds_at_start(seed,assets,design)
    for world in worlds:
        world.scenario = replace(world.scenario,economics=economic,budget=0)
    annual, metrics, crops = [], [], []
    for year in years:
        worked = defaultdict(lambda:[0.,0.,0.,0.,0.])
        for region,world in zip(assets,worlds,strict=True):
            cash0 = fsum(world.cash.values())
            row,fields,actors,trading = world.step("none",ledger=True)
            audit(world,row,fields,actors)
            assert isclose(row["owner_opening_cash"],cash0,rel_tol=1e-12)
            annual.append(dict(arm=arm,seed=seed,year=year,region=region.name,
                               audited_actors=len(actors),audited_plots=len(fields),
                               contracts_new=sum(f["matched"] for f in fields),
                               posted_no_match=sum(f["posted"] and not f["matched"] for f in fields),
                               **row))
            for a in actors:
                worked[a["id"]][3] = a["operation_area"]
            sums = defaultdict(lambda:[0.,0.])
            for f in fields:
                who = f["operator"]
                if who is None:
                    continue
                if economic:
                    vals = f["grain_area"], f["cash_crop_area"], f["grain_kg"]
                    for crop,v in f["crop_accounts"].items():
                        sums[crop][0] += v["area"]
                        sums[crop][1] += v["kg"]
                else:
                    vals = (f["planted_area"],0.,f["harvest"]) if f["crop"]=="grain" else (
                        0.,f["planted_area"],0.) if f["crop"]=="cash" else (0.,0.,0.)
                for i,v in enumerate(vals):
                    worked[who][i] += v
                if f["lease"]:
                    worked[who][4] += f["physical_area"]
            for name,(area,mass) in sorted(sums.items()):
                crops.append(dict(arm=arm,seed=seed,year=year,region=region.name,crop=name,
                                  physical_mu=area,harvest_kg=mass))
        matched, households, survey = scopes[year]
        model = [worked[k] for k in households]
        g,c,kg,used,lease = zip(*model,strict=True)
        metrics.append(dict(arm=arm,seed=seed,year=year,matched=matched,comparable=len(households),
                            **survey,sim_sown=sum(x+y>1e-8 for x,y in zip(g,c,strict=True)),
                            sim_mixed=sum(x>1e-8 and y>1e-8 for x,y in zip(g,c,strict=True)),
                            sim_pure_grain=sum(x>1e-8 and y<=1e-8 for x,y in zip(g,c,strict=True)),
                            sim_pure_nongrain=sum(x<=1e-8 and y>1e-8 for x,y in zip(g,c,strict=True)),
                            sim_grain_physical_mu=fsum(g),sim_nongrain_physical_mu=fsum(c),
                            sim_report_scope_grain_kg=fsum(x for x,ok in zip(kg,survey["grain_kg_valid_mask"]) if ok),
                            sim_planted_physical_mu=fsum(used),
                            sim_planted_on_land_reporting_mu=fsum(x for x,ok in zip(used,survey["land_mask"]) if ok),
                            sim_leased_physical_mu=fsum(lease)))
        for key in ("grain_kg_valid_mask","land_mask"):
            del metrics[-1][key]
    return annual, metrics, crops


def run(project, out, workers):
    if out.exists():
        raise FileExistsError(out)
    lock = Path(__file__).resolve().parents[1]/LOCK
    registered = json.loads(lock.read_bytes())
    design, _, _ = settings()
    assets, signatures = prepare(project,design)
    named, parameters, mapping, inputs = estimate(project,registered["crop_fit_year"])
    panel = survey_panel(project)
    scopes = {}
    for yr in registered["years"]:
        rows = panel.loc[yr]
        sample = rows.loc[rows[["sow_food","sow_cash"]].ge(0).all(axis=1)]
        kg_valid = sample.prod_food.ge(0)
        actual_land = sample.area_e.ge(0)
        scopes[yr] = (len(rows),tuple(sample.index),dict(
            survey_sown=int((sample.sow_food+sample.sow_cash).gt(0).sum()),
            survey_mixed=int((sample.sow_food.gt(0)&sample.sow_cash.gt(0)).sum()),
            survey_pure_grain=int((sample.sow_food.gt(0)&sample.sow_cash.eq(0)).sum()),
            survey_pure_cash=int((sample.sow_food.eq(0)&sample.sow_cash.gt(0)).sum()),
            survey_grain_mu_seasons=float(sample.sow_food.sum()),
            survey_all_cash_mu_seasons=float(sample.sow_cash.sum()),
            survey_grain_kg_n=int(kg_valid.sum()),survey_land_n=int(actual_land.sum()),
            survey_all_grain_kg=float(sample.loc[kg_valid,"prod_food"].sum()),
            survey_operated_mu=float(sample.loc[actual_land,"area_e"].sum()),
            grain_kg_valid_mask=tuple(kg_valid),land_mask=tuple(actual_land)))
    arms = zip(registered["arms"],(None,LEGACY_ACTIVITIES,named),strict=True)
    jobs = [(name,seed,economic,design,assets,scopes,registered["years"])
            for name,economic in arms for seed in registered["seeds"]]
    out.mkdir(parents=True,exist_ok=False)
    paths = out/"path_checkpoints"
    paths.mkdir()
    results = []

    def collect(iterator):
        for (arm,seed,*_),result in zip(jobs,iterator,strict=True):
            _heterogeneous_csv(paths/f"{arm}_{seed}_region_year.csv",result[0])
            csv_write(paths/f"{arm}_{seed}_household_history.csv",result[1])
            csv_write(paths/f"{arm}_{seed}_crop_year.csv",result[2])
            results.append(result)
            print(f"Completed {arm} / {seed}",flush=True)

    if workers==1:
        collect(map(_path,jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(_path,jobs))
    _heterogeneous_csv(out/"activity_evidence.csv",parameters)
    csv_write(out/"dictionary_trace.csv",mapping)
    choices = [dict(name=a.name,is_grain=a.grain,yield_kg_mu=a.kg,value_2024_kg=a.unit_value,
                    inputs_2024_mu=a.inputs,labor_days_mu=a.days,sold_share=a.sold,
                    provinces=";".join(map(str,a.markets))) for a in named.activities]
    csv_write(out/"enabled_parameters.csv",choices)
    _heterogeneous_csv(out/"region_year.csv",[row for run in results for row in run[0]])
    for idx,name in enumerate(("household_history","crop_year"),start=1):
        csv_write(out/f"{name}.csv",[row for run in results for row in run[idx]])
    json_write(out/"receipt.json",dict(status="structural accounts checked; historical comparison not validation",
        python=platform.python_version(),scipy=scipy.__version__,pandas=pd.__version__,source_sha256=inputs,
        cohort=signatures,design_sha256=digest(lock),specifications=registered["arms"],seeds=registered["seeds"],
        episode_count=len(jobs),region_years=sum(len(r[0]) for r in results),
        household_years=sum(a["audited_actors"] for r in results for a in r[0]),
        plot_years=sum(a["audited_plots"] for r in results for a in r[0]),
        RL_trained=False,policy_ranked=False))


if __name__=="__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=1)
    args = ap.parse_args()
    run(args.project,args.out,args.workers)
