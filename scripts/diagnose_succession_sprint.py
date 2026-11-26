"""Read-only historical bridge and exact conditional crop-choice screen."""

import argparse
from collections import Counter, defaultdict
import json
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from core.land_succession import CROPS
from core.regional_succession_budget import worlds_at_start
from scripts.check_succession_empirical_households import check_accounts,survey_panel
from scripts.prepare_succession_budget_inputs import prepare
from scripts.run_succession_annual_allocation import csv_write,json_write,seeds_of
from scripts.run_succession_empirical_budget import DESIGN,settings
from scripts.verify_empirical_succession import metric

PROTOCOL=DESIGN.with_name("sprint_protocol.json")


def exact_screen(lots: list, cash: float, labor: float, wage: float, outside: float) -> tuple:
    """Optimize the actual P0 single-season objective, preserving indivisibility."""
    n=len(lots)
    if n<2:
        return "fewer_than_two_lots",None,False
    area=np.array([x["physical_area"] for x in lots])
    qc=np.array([x["quality"] for x in lots])
    gi,ci=CROPS["grain"],CROPS["cash"]
    c=area[:,None]*[gi[2],ci[2]]
    t=area[:,None]*[gi[3],ci[3]]
    different=~np.eye(n,dtype=bool)
    cash_ok=different & (c[:,0,None]+c[None,:,1]<=cash+1e-7)
    labor_ok=different & (t[:,0,None]+t[None,:,1]<=labor+1e-7)
    if not (cash_ok&labor_ok).any():
        return ("cash_and_labor" if not cash_ok.any() and not labor_ok.any() else
                "cash" if not cash_ok.any() else "labor" if not labor_ok.any() else "joint_resource"),None,False
    gross=area[:,None]*(qc[:,None]*[gi[0]*gi[1],ci[0]*ci[1]]-[gi[2],ci[2]])
    limits=lil_matrix((n+3,2*n+1))
    for i in range(n):
        limits[i,2*i:2*i+2]=1.
    limits[n,:2*n]=c.ravel()
    limits[n+1,:2*n]=t.ravel()
    limits[n+2,:2*n]=t.ravel()
    limits[n+2,-1]=-1.
    objective=np.r_[-gross.ravel(),wage]
    upper=np.r_[np.ones(2*n),labor]
    result=milp(objective,integrality=np.r_[np.ones(2*n),0],bounds=Bounds(0.,upper),
                constraints=LinearConstraint(limits.tocsr(),-np.inf,np.r_[np.ones(n),cash,labor,labor-min(labor,outside)]),
                options={"time_limit":2.0,"mip_rel_gap":1e-10})
    if not result.success:
        return "mixed_feasible_reference_not_optimal",None,False
    chosen=result.x[:2*n].reshape(n,2)>0.5
    value=float(gross[chosen].sum())-wage*max(0.,float(t[chosen].sum())-max(0.,labor-outside))
    mixed=bool(chosen.any(axis=0).all())
    return "mixed_feasible",value,mixed


def model_year(worlds, diagnostic: bool, comparable: set) -> tuple[dict, Counter]:
    accounts, funnel = {}, Counter()
    for world in worlds:
        initial=Counter()
        for p in world.plots:
            if p.id in world.operator:
                initial[world.operator[p.id]]+=p.area
        row,fields,actors,_=world.step("none",ledger=True)
        check_accounts(world,row,fields,actors)
        offered, plots=Counter(),defaultdict(list)
        rented_in,rented_out=Counter(),Counter()
        for f in fields:
            offered[f["owner"]]+=f["physical_area"]*f["released"]
            if f["assigned_operator"] is not None:
                plots[f["assigned_operator"]].append(f)
            if f["matched"]:
                rented_in[f["assigned_operator"]]+=f["physical_area"]
                rented_out[f["owner"]]+=f["physical_area"]
        for a in actors:
            k=a["id"]
            land=plots[k]
            g=fsum(f["physical_area"] for f in land if f["crop"]=="grain")
            c=fsum(f["physical_area"] for f in land if f["crop"]=="cash")
            accounts[k]=dict(open_mu=initial[k],operated=fsum(f["physical_area"] for f in land),
                grain=g,crop=c,planted=g+c,grain_kg=fsum(f["harvest"] for f in land if f["crop"]=="grain"),
                rented_in=rented_in[k],rented_out=rented_out[k],offer=offered[k],
                bid=a["quoted_plots"]>0,cash=a["liquidity"],days=a["labor_capacity"])
            if not diagnostic or k not in comparable:
                continue
            for tag,truth in (("comparable",True),("offers_land",offered[k]>0),("bids",a["quoted_plots"]>0),
                 ("rents_in",rented_in[k]>0),("operating",bool(land)),("sown",g+c>0),("mixed",g>1e-7 and c>1e-7)):
                funnel[tag]+=int(truth)
            if g>1e-7 and c>1e-7:
                continue
            op=world.operators[k]
            name,value,mixed=exact_screen(land,a["liquidity"],a["labor_capacity"],a["off_farm_wage"],op.job_days)
            funnel[name]+=1
            if value is not None:
                expected=505.*g+635.*c-a["foregone_job_income"]
                if value < expected-1e-5:
                    raise ValueError("Exact conditional objective below implemented feasible plan")
                funnel["implementation_has_gain"]+=int(value>expected+1e-5)
                funnel["exact_convert_to_mixed"]+=int(mixed and value>expected+1e-5)
                funnel["mixed_feasible_but_no_strict_mixed_improvement"]+=int(not (mixed and value>expected+1e-5))
    return accounts,funnel


def run(project: Path, out: Path, refine: bool = False) -> None:
    lock=json.loads(PROTOCOL.read_bytes())
    design=settings()[0]
    design["crop_refinement"]=refine
    assets,receipt=prepare(project,design)
    out.mkdir(parents=True,exist_ok=False)
    panel=survey_panel(project)
    data=project/"results/diagnostics"
    prefix=data/"national_abm_v41_final_economic_closure_20261124/PRIVATE_DO_NOT_SHARE"
    cohort=pd.DataFrame(json.loads((data/"national_crop_choice_soft_response_20261123/PRIVATE_DO_NOT_SHARE/cohort600.json").read_bytes()))
    keys={str(v):f"H{i:04}" for i,v in enumerate(cohort.cohort_agent_id)}
    basis=pd.read_csv(data/"national_empirical_succession_20261126/locked_301/PRIVATE_DO_NOT_SHARE/baseline.csv")
    carried={f"H{i:04}":dict(operated=r.area_e,open_mu=r.area_e,
                grain=r.sow_food,crop=r.sow_cash,planted=np.nan,grain_kg=r.prod_food,
                rented_in=np.nan,rented_out=np.nan) for i,r in basis.iterrows()}
    protocol=json.loads((DESIGN.parent/"../validation_protocol.json").read_bytes())
    thresholds=protocol["thresholds"]
    comparisons, screen, annual=[],[],[]
    for seed in seeds_of(lock["historical_seeds"]):
        worlds=worlds_at_start(seed,assets,design)
        archive=pd.read_csv(prefix/f"F1_land_bridge_{seed}.csv",dtype={"id":"string"})
        archive["actor"]=archive.id.map(keys)
        assert archive.actor.notna().all() and len(archive)==3000
        for year in range(2011,2016):
            observed=archive[archive.year.eq(year)].set_index("actor")
            linked=observed.loc[observed.index.intersection(panel.loc[year].index)].copy()
            linked["prod_food"]=panel.loc[year].loc[linked.index,"prod_food"]
            valid=linked.sow_food.ge(0)&linked.sow_cash.ge(0)
            diagnostic=not refine and seed in lock["screening_seeds"] and year in lock["screening_years"]
            simulated,funnel=model_year(worlds,diagnostic,set(linked.index[valid]))
            if diagnostic:
                screen.append(dict(seed=seed,year=year,**funnel))
            model_label="M2_search" if refine else "M1"
            for label, sim in (("F1_archived",linked),(model_label,pd.DataFrame.from_dict(simulated,orient="index")),
                               ("persistence_2010",pd.DataFrame.from_dict(carried,orient="index"))):
                survey=linked[["area_b","area_e","area_in1","area_de2","sow_food","sow_cash","sow_total","prod_food"]]
                h=survey.join(sim[["grain","crop","operated","rented_in","rented_out","grain_kg"]],how="inner")
                stat=metric(h,lock["qualified_crop_area_epsilon"],300)
                gates={k: bool(stat[k]<=v) for k,v in thresholds.items()}
                kg=h.prod_food.ge(0)&h.sow_food.ge(0)&h.sow_cash.ge(0)
                comparisons.append(dict(seed=seed,year=year,model=label,**stat,**{f"passes_{k}":v for k,v in gates.items()},
                    crop_units="physical_mu" if label==model_label else "mu_seasons",
                    five_compatible_pass=(all(gates.values()) and stat["coverage_status"]=="comparable" if label!=model_label else ""),
                    common_kg_n=int(kg.sum()),survey_kg=h.loc[kg,"prod_food"].sum(),model_kg=h.loc[kg,"grain_kg"].sum()))
            if seed==lock["historical_seeds"][0]:
                survey=linked.loc[valid]
                both_positive=survey.area_b.ge(0)&survey.area_e.gt(0)
                second=(survey.sow_food+survey.sow_cash)>survey.area_e+1e-7
                annual.append(dict(year=year,linked=len(linked),valid_crops=len(survey),both_stock=int(both_positive.sum()),
                    opening_mu=survey.loc[both_positive,"area_b"].sum(),closing_mu=survey.loc[both_positive,"area_e"].sum(),
                    verified_zero_plant=int((survey.sow_total.eq(0)&survey.sow_food.eq(0)&survey.sow_cash.eq(0)).sum()),
                    no_grain_or_cash=int((survey.sow_food.eq(0)&survey.sow_cash.eq(0)).sum()),
                    reported_mixed=int((survey.sow_food.gt(0)&survey.sow_cash.gt(0)).sum()),
                    with_more_grain_cash_sowing_than_physical=int((second&survey.area_e.gt(0)).sum()),
                    survey_grain_mu_seasons=survey.sow_food.sum(),survey_cash_mu_seasons=survey.sow_cash.sum(),
                    survey_grain_kg=survey.prod_food.where(survey.prod_food.ge(0)).sum()))
    for file, table in (("retrospective.csv",comparisons),("conditional_funnel.csv",screen),("survey_unit_bridge.csv",annual)):
        if not table:
            continue
        columns=dict.fromkeys(k for row in table for k in row)
        csv_write(out/file,[{k:r.get(k,0) for k in columns} for r in table])
    rate=fsum(r.get("exact_convert_to_mixed",0) for r in screen)/fsum(r["comparable"] for r in screen) if screen else None
    json_write(out/"receipt.json",dict(status="completed",converted_rate=rate,
        solver_candidate_eligible=rate>=0.02 if rate is not None else None,refinement=refine,
        rows=len(comparisons),screen_blocks=len(screen),inputs=receipt,
        family_validation="retrospective and non-independent",
        frozen_f1_pairs="archived seed labels and same households; F1 and M1 have different shock formulas",
        one_step="exact conditional solve on actual model-year posttrade state; survey has stocks but not aligned peak-day cash/labor to reconstruct a factual posttrade feasible set"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--refine",action="store_true")
    args=parser.parse_args()
    run(args.project,args.output,args.refine)
