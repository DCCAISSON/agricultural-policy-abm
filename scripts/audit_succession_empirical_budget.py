"""Rebuild the policy tables, audit executed annual budgets, and report paired outcomes."""

import argparse
from collections import Counter, defaultdict
from math import fsum, isclose
import json
from pathlib import Path
from statistics import mean

from core.land_succession import CROPS
from scripts.audit_succession_annual_allocation import gz_csv, require, state
from scripts.audit_succession_handoff import keyed,load_csv
from scripts.prepare_succession_budget_inputs import digest
from scripts.run_succession_annual_allocation import csv_write,json_write,seeds_of
from scripts.run_succession_empirical_budget import METRICS,fit_design,policy_step,policy_values,settings
from scripts.run_succession_shared_budget import interval
from scripts.succession_annual_learning import fit,state_key
from scripts.succession_sprint_support import select,support

OUTCOMES = (*METRICS,"reservation")


def refit(training: Path, frozen: dict, design: dict) -> None:
    actions = range(len(frozen["menu"]))
    dev = seeds_of(design["fixed_dev_seeds"])
    rows = keyed(load_csv(training/"fixed_development.csv"),("seed","action"))
    require(set(rows)=={(str(s),str(a)) for s in dev for a in actions},"Fixed development grid")
    means = [mean(float(rows[str(s),str(a)]["grain_kg"]) for s in dev) for a in actions]
    tails = {y+1:mean(float(rows[str(s),str(y)]["p0_tail"]) for s in dev)
             for y in range(design["main_years"])}
    require(means==frozen["fixed_scores"] and max(actions,key=means.__getitem__)==frozen["strong_fixed"],
            "Fixed comparator selection")
    samples = defaultdict(dict)
    for row in gz_csv(training/"openings.csv.gz"):
        seed,key,a = int(row["seed"]),state(row["opening"]),int(row["action"])
        nxt = state(row["successor"])
        require(len(key)==len(nxt)==6 and nxt[0]==key[0]-1 and key[-1]==nxt[-1],"Training public-state clock")
        require(a not in samples[seed,key],"Duplicate action at training opening")
        samples[seed,key][a]=(float(row["grain_kg"]),nxt,0.)
    recorded=gz_csv(training/"screen.csv.gz")
    screened=[(int(x["seed"]),int(x["year"]),state(x["state"]),int(x["explore_action"])) for x in recorded]
    training=set(seeds_of(design["training_seeds"]))
    supplement=set(seeds_of(design["supplement_seeds"]))
    screen1=[r for r in screened if r[0] in training]
    selected1=select(screen1)
    first=support(screen1,selected1,len(actions),design)
    require(first==frozen["initial_coverage"],"Initial coverage mismatch")
    seed_set=training if first["coverage"]>=design["coverage_threshold"] else training|supplement
    require(Counter((seed,y) for seed,y,_,_ in screened)==Counter(
        (s,y) for s in seed_set for y in range(1,design["main_years"]+1)),
        "Training seeds and distinct screened openings")
    require(all(k[0]==design["main_years"]-y+1 and a in actions for _,y,k,a in screened),
            "Screen clock or exploration action")
    selected=select(screened,selected1)
    picked={(int(x["seed"]),int(x["year"])) for x in recorded if x["selected"]=="1"}
    require(picked==selected and support(screened,selected,len(actions),design)==frozen["coverage"],
            "Targeted selection or common support mismatch")
    actual={(seed,design["main_years"]-k[0]+1):k for seed,k in samples}
    expected={(seed,y):k for seed,y,k,_ in screened if (seed,y) in selected}
    require(actual==expected and len(actual)==len(samples),"Opening state did not match selected parents")
    require(all(set(rows)==set(actions) for rows in samples.values()),"Action coverage at opening")
    replay=[(s,k,[v[a] for a in actions]) for (s,k),v in samples.items()]
    table,_=fit(replay,fit_design(design),tails)
    repeated,_=fit([x for x in replay if x[0]%4!=0],fit_design(design),tails)
    require(table==frozen["table"] and repeated==frozen["repeat_table"],
            "Frozen main or repeated policies not reproduced")
    require(support(screened,{(s,y) for s,y in selected if s%4!=0},len(actions),design)==frozen["repeat_coverage"],
            "Repeat coverage mismatch")


def audit(training: Path, evaluation: Path, out: Path) -> None:
    design,h,code=settings()
    file=training/"frozen.json"
    frozen=json.loads(file.read_bytes())
    rec=json.loads((evaluation/"receipt.json").read_bytes())
    require(frozen["design_sha256"]==h and frozen["code_sha256"]==code and digest(file)==rec["frozen_sha256"],
            "Experiment inputs or freeze differ")
    refit(training,frozen,design)
    seeds=seeds_of(design["evaluation_seeds"])
    years=range(1,design["main_years"]+1)
    regions=design["region_order"]
    menu=frozen["menu"]
    capacities={x["region"]:x for x in frozen["inputs"]["regions"]}
    common=max(sum(pair[1] for pair in action) for action in menu)
    threshold=design["grain_bin_fraction"]*CROPS["grain"][0]
    strong=f"fixed_{frozen['strong_fixed']:02}"
    all_strategies=[*(f"fixed_{i:02}" for i in range(len(menu))),"myopic","q","myopic_repeat","q_repeat"]
    stopping=[strong,"myopic","q"]
    modes=[*(("main",p) for p in all_strategies),*(("withdrawal",p) for p in stopping),
           ("pulse","central_then_equal"),("pulse","west_then_equal")]
    ledger={}
    for r in gz_csv(evaluation/"region_year.csv.gz"):
        key=(int(r["seed"]),r["mode"],r["strategy"],int(r["year"]),r["region"])
        require(key not in ledger,"Duplicate region/year")
        ledger[key]=dict(r,action=int(r["action"]),**{m:float(r[m]) for m in (*OUTCOMES,"physical_mu","rate_mu")})
    require(set(ledger)=={(s,mode,k,y,r) for s in seeds for mode,k in modes for y in years for r in regions}
            and len(ledger)==rec["region_years"] and len(seeds)==rec["paired_seeds"],"Evaluation grid")
    frequencies=Counter()
    aggregates={}
    for s in seeds:
        for mode,k in modes:
            before={r:capacities[r]["observed_initial_grain_kg"] for r in regions}
            for y in years:
                key=state_key((len(years)-y+1,*(int(before[r]>=threshold*capacities[r]["physical_mu"])
                                               for r in regions),common))
                stopped=mode=="withdrawal" and y>design["withdrawal_after"]
                action,reason=policy_step(k,key,frozen,mode,y,design)
                q_value,current=map(str,policy_values(k,key,frozen,action,reason))
                rows=[ledger[s,mode,k,y,r] for r in regions]
                require(fsum(x["reservation"] for x in rows)<=common and
                        fsum(x["committed"] for x in rows)<=common+1e-7,"Common fiscal envelope")
                frequencies[mode,k,y,action,reason]+=1
                for i,(r,x) in enumerate(zip(regions,rows,strict=True)):
                    policy,cap=menu[action][i]
                    require((x["state"],x["action"],x["reason"],x["policy"],x["reservation"])==
                            (key,action,reason,policy,cap),"Observation, freeze, or fiscal allocation")
                    require((x["chosen_q_kg"],x["chosen_current_kg"])==(q_value,current),"Frozen action values")
                    area=x["physical_mu"]
                    require(isclose(area,capacities[r]["physical_mu"],abs_tol=1e-8) and
                            isclose(x["grain_area"]+x["cash_crop_area"]+x["idle_area"],area,abs_tol=1e-7),
                            "Annual physical land balance")
                    expected_rate=min(design["nominal_rate_yuan_per_mu"],cap/area,CROPS["grain"][2])
                    require(isclose(x["rate_mu"],expected_rate,abs_tol=1e-10) and
                            0<=x["spending"]<=x["committed"]+1e-7<=cap+2e-7 and
                            isclose(x["owner_support"],x["spending"],abs_tol=1e-7)
                            and x["spending"]<=x["rate_mu"]*x["grain_area"]+1e-7,
                            "Regional cap, forecast rate, and actual grain-linked payment")
                    if not cap:
                        require(x["spending"]==x["committed"]==0.,"No-support years paid")
                    require(isclose(x["rent_total"],x["owner_rent_income"],abs_tol=1e-7)
                            and isclose(x["rent_total"],x["owner_rent_paid"],abs_tol=1e-7)
                            and isclose(x["owner_area"],x["grain_area"]+x["cash_crop_area"],abs_tol=1e-7)
                            and x["matched_area"] <= x["posted_area"]+1e-7
                            and isclose(x["leased_area"],x["held_area"]+x["matched_area"],abs_tol=1e-7),
                            "Land transfer, rent, or planted-area subaccount")
                    if y>1:
                        require(isclose(x["owner_opening_cash"],ledger[s,mode,k,y-1,r]["owner_closing_cash"],abs_tol=1e-7),
                                "Household cash does not carry")
                    if mode=="withdrawal" and not stopped:
                        prior=ledger[s,"main",k,y,r]
                        require(all(v==prior[k0] for k0,v in x.items() if k0!="mode"),"Pre-stop prefix is not identical")
                    before[r]=x["grain_kg"]
            for period,window in (("all",years),("years7_8",range(7,9))):
                for region in (*regions,"joint"):
                    use=regions if region=="joint" else (region,)
                    aggregates[s,mode,k,period,region]={
                        m:fsum(ledger[s,mode,k,y,r][m] for y in window for r in use) for m in OUTCOMES
                    }
    out.mkdir(parents=True,exist_ok=False)
    csv_write(out/"action_counts.csv",[dict(mode=m,strategy=k,year=y,action=a,reason=r,count=n)
        for (m,k,y,a,r),n in sorted(frequencies.items())])
    annual = []
    for mode,strategy in modes:
        for year in years:
            for region in (*regions,"joint"):
                group = regions if region=="joint" else (region,)
                annual.append(dict(mode=mode,strategy=strategy,year=year,region=region,n=len(seeds),
                    **{m:mean(fsum(ledger[s,mode,strategy,year,r][m] for r in group) for s in seeds)
                       for m in OUTCOMES}))
    csv_write(out/"annual_means.csv",annual)
    summarize(aggregates,seeds,regions,modes,strong,out)
    pulse=[]
    for window,span in (("first",[1]),("after_first",range(2,9)),("all",years)):
        for region in (*regions,"joint"):
            rlist=regions if region=="joint" else (region,)
            for metric in ("grain_kg","spending","retained_grain","continuity_area","matched_area","held_area",
                           "leased_idle","owner_opening_cash","owner_closing_cash"):
                values=[fsum(ledger[s,"pulse","central_then_equal",y,r][metric]-
                             ledger[s,"pulse","west_then_equal",y,r][metric] for r in rlist for y in span) for s in seeds]
                mid,lo,hi=interval(values,.05)
                pulse.append(dict(window=window,region=region,metric=metric,mean=mid,lower=lo,upper=hi,n=len(seeds)))
    csv_write(out/"allocation_pulse.csv",pulse)
    json_write(out/"receipt.json",dict(
        status="passed",region_years=len(ledger),paths=len(seeds)*len(modes),seed_count=len(seeds),
        coverage=frozen["coverage"],coverage_pass=frozen["coverage"]["coverage"]>=design["coverage_threshold"],
        same_information_refit=True,repeat_refit=True,initial_coverage=frozen["initial_coverage"],
        repeat_coverage=frozen["repeat_coverage"],common_budget=common,three_primary_outcomes_per_comparison=True,
        checks=["common initial state","fiscal mask and frozen choices","lagged grain only",
                "regional cash, rent, land and planting accounting","withdrawal matched prefix and zero payments"],
    ))


def summarize(values: dict, seeds: range, regions: list[str], modes: list[tuple[str,str]],
              strong: str, out: Path) -> None:
    means,pairs=[],[]
    for mode,k in modes:
        for window in ("all","years7_8"):
            for region in (*regions,"joint"):
                means.append(dict(mode=mode,strategy=k,window=window,region=region,n=len(seeds),
                    **{m:mean(values[s,mode,k,window,region][m] for s in seeds) for m in OUTCOMES}))
    for ref in (strong,"myopic"):
        for metric in OUTCOMES:
            for region in (*regions,"joint"):
                for window,mode in (("all","main"),("years7_8","withdrawal")):
                    primary=region=="joint" and ((window=="all" and metric in ("grain_kg","spending"))
                                                or (window=="years7_8" and metric=="grain_kg"))
                    alpha=.05/(6 if primary else 1)
                    vals=[values[s,mode,"q",window,region][metric]-values[s,mode,ref,window,region][metric]
                          for s in seeds]
                    center,lo,hi=interval(vals,alpha)
                    pairs.append(dict(mode=mode,focal="q",reference=ref,window=window,region=region,
                        metric=metric,n=len(seeds),mean=center,lower=lo,upper=hi,primary=primary,confidence=1-alpha))
    stop=[]
    for k in ("q","myopic",strong):
        period="years7_8"
        for region in (*regions,"joint"):
            for metric in ("grain_kg","spending","retained_grain","continuity_area","leased_idle","rent_total","held_area"):
                for mode,ref in (("main",k),("main","fixed_00")):
                    diffs=[values[s,"withdrawal",k,period,region][metric]-values[s,mode,ref,period,region][metric]
                           for s in seeds]
                    center,lo,hi=interval(diffs,.05)
                    stop.append(dict(strategy=k,reference="continued_same_policy" if ref==k else "always_p0",
                        region=region,metric=metric,n=len(seeds),mean=center,lower=lo,upper=hi))
    csv_write(out/"means.csv",means)
    csv_write(out/"paired.csv",pairs)
    csv_write(out/"withdrawal.csv",stop)
    stability=[]
    for q, m in (("q","myopic"),("q_repeat","myopic_repeat"),("q_repeat","q")):
        for metric in ("grain_kg","spending"):
            v=[values[s,"main",q,"all","joint"][metric]-values[s,"main",m,"all","joint"][metric] for s in seeds]
            mid,lo,hi=interval(v,.05)
            stability.append(dict(focal=q,reference=m,metric=metric,n=len(seeds),mean=mid,lower=lo,upper=hi))
    csv_write(out/"stability.csv",stability)
    candidates=[r for r in means if r["region"]=="joint" and r["mode"]=="main" and r["window"]=="all"]
    csv_write(out/"tested_frontier.csv",[
        dict(strategy=r["strategy"],grain_kg=r["grain_kg"],spending=r["spending"],
             not_dominated=not any(o["grain_kg"]>=r["grain_kg"] and o["spending"]<=r["spending"] and
                         (o["grain_kg"]>r["grain_kg"] or o["spending"]<r["spending"]) for o in candidates))
        for r in candidates])


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("training","evaluation","out"):
        cli.add_argument(key,type=Path)
    args=cli.parse_args()
    audit(args.training,args.evaluation,args.out)
