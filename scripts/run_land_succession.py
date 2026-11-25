"""Run the preregistered synthetic succession experiment and independent policy evaluation."""
import argparse
import csv
import gzip
import io
import json
import subprocess
import sys
from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import t

from agents.government_agent import GovernmentAgent
from core.land_succession import CROPS, POLICIES, TIMING_CONTROLS, Scenario, Succession

YEARS = 8
SEEDS = tuple(range(301,325))
TRAIN_SEEDS = tuple(range(10001,10385))
EVAL_SEEDS = tuple(range(20001,20025))


def observation(world):
    r=world.observed
    offered=max(1.,r["offered_area"])
    return (int(r["grain_kg"]<.55*world.land*CROPS["grain"][0]),
            int(r["unfilled_area"]/offered>.35),
            int(r["successor_grain"]/offered>=.3),
            int(r["spending"]>.7*world.scenario.budget))


def transparent(state):
    low_grain, unmet, strong_takeover, high_spend=state
    if low_grain:
        return 1 if unmet or not strong_takeover else 2
    return 0 if high_spend else 2 if strong_takeover else 1


def reward(world, result):
    return (result["grain_kg"]/(world.land*CROPS["grain"][0])
            +.18*result["successor_grain"]/world.land
            -.14*result["spending"]/world.scenario.budget)


def train(scenarios):
    # Reuse the frozen government Bellman update without initializing its old economy.
    gov=SimpleNamespace(q_table={},q_table_frozen=False,learning_rate=.14,discount_factor=.92)
    history=[]
    for episode,seed in enumerate(TRAIN_SEEDS):
        condition,scenario=scenarios[episode % len(scenarios)]
        world=Succession(seed,scenario)
        rng=np.random.default_rng([seed,283])
        epsilon=max(.08,.55*(1-episode/len(TRAIN_SEEDS)))
        for year in range(1,YEARS+1):
            state=observation(world)
            gov.last_state=state
            table=gov.q_table.setdefault(state,[0.]*len(POLICIES))
            action=int(rng.integers(len(POLICIES))) if rng.random()<epsilon else int(np.argmax(table))
            gov.last_action=action
            out=world.step(POLICIES[action])
            nxt=observation(world)
            score=reward(world,out)
            GovernmentAgent._update_q_table(gov,score,nxt,done=year==YEARS)
            history.append(dict(episode=episode+1,seed=seed,condition=condition,year=year,
                 state=str(state),next_state=str(nxt),action=POLICIES[action],reward=score,epsilon=epsilon,**out))
    return {k:tuple(v) for k,v in gov.q_table.items()},pd.DataFrame(history)


def simulate(seed, scenario, strategy, q, log, group, condition):
    world=Succession(seed,scenario)
    result=[]
    for year in range(1,YEARS+1):
        state=observation(world)
        if strategy=="rule":
            action=POLICIES[transparent(state)]
        elif strategy=="q":
            action=POLICIES[int(np.argmax(q[state])) if state in q else transparent(state)]
        else:
            action=strategy
        out,plots,actors,trades=world.step(action,ledger=True)
        key=dict(group=group,condition=condition,seed=seed,year=year,strategy=strategy,policy=action)
        for label,entries in (("plots",plots),("operators",actors),("transactions",trades)):
            log(label,[key|entry for entry in entries])
        result.append(key|dict(previous_state=str(state),unseen_state=strategy=="q" and state not in q,
                               reward=reward(world,out))|out)
    return result


def designs():
    baseline=Scenario(search_failure=.08,cash=1.,labor=1.)
    factorial={}
    for search,liquidity,labor in product((False,True),repeat=3):
        name=f'{"high" if search else "low"}_search__{"tight" if liquidity else "ample"}_cash__{"tight" if labor else "ample"}_labor'
        factorial[name]=replace(baseline,search_failure=.48 if search else .08,
                                cash=.56 if liquidity else 1.,labor=.60 if labor else 1.)
    middle=replace(baseline,search_failure=.25,cash=.73,labor=.77)
    changes={
        "middle":middle,
        "friction_off":replace(middle,search_failure=0.,fee=0.),
        "no_tenants":replace(middle,entrants=0),
        "cash_relief":replace(middle,cash=4.),
        "labor_relief":replace(middle,labor=3.),
        "no_feedback":replace(middle,feedback=False),
        "wage_low":replace(middle,outside_wage=.7),
        "wage_high":replace(middle,outside_wage=1.3),
        "rate_low":replace(middle,rate=60.),
        "rate_high":replace(middle,rate=120.),
    }
    # Entire seed range and all held-out parameter combinations are unseen in training.
    holdout={
        "unseen_middle":middle,
        "unseen_cash_labor":replace(middle,search_failure=.18,cash=.66,labor=.49),
        "unseen_search":replace(middle,search_failure=.66,cash=.88,labor=.92),
    }
    return factorial,changes,holdout


def summarize(frame, out):
    keys=["group","condition","strategy"]
    metrics=["grain_kg","grain_area","idle_area","successor_grain","continuity_area","operator_change",
             "offered_area","unfilled_area","leased_idle","leased_cash","leased_area","expanded","shrunk",
             "exits","tenant_profit","owner_income","spending","reward"]
    world=frame.groupby(keys+["seed"],as_index=False)[metrics].mean()
    world.to_csv(out/"world_means.csv",index=False)
    frame.groupby(keys+["year"])[metrics].mean().to_csv(out/"annual_means.csv")
    summary=world.groupby(keys)[metrics].agg(["mean","std",lambda s:s.quantile(.05),lambda s:s.quantile(.95)])
    summary.columns=["_".join(x).replace("_<lambda_0>","_p05").replace("_<lambda_1>","_p95") for x in summary.columns]
    summary.to_csv(out/"distribution.csv")
    contrasts=[]
    for (group,condition),rows in world.groupby(["group","condition"]):
        base=rows[rows.strategy.eq("none")].set_index("seed")
        for strategy,x in rows.groupby("strategy"):
            x=x.set_index("seed")
            for reference in (["none"] if group!="evaluation" else ["none","grain_post","takeover_pre","rule"]):
                if strategy==reference:
                    continue
                r=base if reference=="none" else rows[rows.strategy.eq(reference)].set_index("seed")
                assert set(r.index)==set(x.index)
                for metric in metrics:
                    delta=x[metric]-r[metric]
                    n=len(delta)
                    mean=delta.mean()
                    half=t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n)
                    contrasts.append(dict(group=group,condition=condition,strategy=strategy,reference=reference,
                        metric=metric,n=n,mean=mean,ci95_low=mean-half,ci95_high=mean+half,positive=int((delta>0).sum())))
    pd.DataFrame(contrasts).to_csv(out/"paired_effects.csv",index=False)
    frame.groupby(keys+["policy"]).size().rename("years").to_csv(out/"policy_choices.csv")
    diag=world[world.group.eq("diagnostic")].pivot(index="seed",columns=["condition","strategy"])
    mechanism=[]
    for m in metrics:
        x=diag[m]
        comparisons={
            "timing_only":x["middle","takeover_pre"]-x["middle","takeover_post"],
            "eligibility_only":x["middle","takeover_pre"]-x["middle","grain_pre"],
            "feedback_difference_in_differences":
                (x["middle","takeover_pre"]-x["middle","none"])
                -(x["no_feedback","takeover_pre"]-x["no_feedback","none"]),
        }
        for contrast,delta in comparisons.items():
            n=len(delta)
            avg=delta.mean()
            half=t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n)
            mechanism.append(dict(contrast=contrast,metric=m,n=n,mean=avg,ci95_low=avg-half,ci95_high=avg+half))
    pd.DataFrame(mechanism).to_csv(out/"mechanism_contrasts.csv",index=False)


def verify_ledgers(out, frame):
    key=["group","condition","seed","year","strategy","policy"]
    anchor=frame.set_index(key).sort_index()
    p=pd.read_csv(out/"plots.csv.gz",usecols=key+["plot","operator","crop","physical_area","planted_area","harvest","support"])
    assert not p.duplicated(key+["plot"]).any()
    assert p.loc[p.crop.ne("idle"),"operator"].notna().all()
    assert (p.planted_area<=p.physical_area).all()
    p["grain_mu"]=p.planted_area.where(p.crop.eq("grain"),0.)
    p["grain_yield"]=p.harvest.where(p.crop.eq("grain"),0.)
    pp=p.groupby(key)[["grain_mu","grain_yield","support"]].sum()
    np.testing.assert_allclose(pp,anchor[["grain_area","grain_kg","spending"]],rtol=1e-12,atol=1e-7)
    assert pp.index.equals(anchor.index)
    a=pd.read_csv(out/"operators.csv.gz",usecols=key+["id","opening","closing","income","consumption","rent_paid",
                                                       "rent_income","support","farm_days","job_days","labor_capacity"])
    assert not a.duplicated(key+["id"]).any() and a.closing.ge(0.).all()
    assert (a.farm_days+a.job_days<=a.labor_capacity+1e-7).all()
    np.testing.assert_allclose(a.opening+a.income-a.consumption,a.closing,rtol=1e-12,atol=1e-7)
    sums=a.groupby(key)[["rent_paid","rent_income","support"]].sum()
    np.testing.assert_allclose(sums.rent_paid,sums.rent_income,rtol=1e-12,atol=1e-7)
    np.testing.assert_allclose(sums.support,anchor.spending,rtol=1e-12,atol=1e-7)
    t=pd.read_csv(out/"transactions.csv.gz",usecols=key+["plot","area","buyer","rent"])
    assert not t.duplicated(key+["plot"]).any()
    offers=t.groupby(key).area.sum().reindex(anchor.index,fill_value=0.)
    matches=t[t.buyer.notna()].groupby(key)[["area","rent"]].sum().reindex(anchor.index,fill_value=0.)
    np.testing.assert_allclose(offers,anchor.offered_area)
    np.testing.assert_allclose(matches.area,anchor.leased_area)
    np.testing.assert_allclose(matches.rent,sums.rent_paid,rtol=1e-12,atol=1e-7)
    receipt=dict(years=len(frame),plots=len(p),operators=len(a),transactions=len(t),checks="passed")
    (out/"ledger_verification.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")


def run(out, *, ledgers):
    fact,changes,holdout=designs()
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    # Refuse formal RL before internal economic and held-out timing tests pass.
    subprocess.run([sys.executable,"-B","-m","pytest","-q","-p","no:cacheprovider",
                    "tests/test_land_succession.py",f"--junitxml={out/'gate.xml'}"],cwd=root,check=True)
    assert not set(TRAIN_SEEDS)&(set(SEEDS)|set(EVAL_SEEDS)) and not set(fact.values())&set(holdout.values())
    q,history=train(list(fact.items()))
    history.to_csv(out/"training_sar.csv",index=False)
    (out/"q_table.json").write_text(json.dumps({str(k):v for k,v in sorted(q.items())},indent=2),encoding="utf-8")
    with ExitStack() as stack:
        writers={}
        def log(name,rows):
            if not ledgers or not rows:
                return
            if name not in writers:
                raw=stack.enter_context((out/f"{name}.csv.gz").open("wb"))
                compressed=stack.enter_context(gzip.GzipFile(fileobj=raw,mode="wb",mtime=0))
                s=stack.enter_context(io.TextIOWrapper(compressed,encoding="utf-8",newline=""))
                writers[name]=csv.DictWriter(s,fieldnames=list(rows[0]))
                writers[name].writeheader()
            writers[name].writerows(rows)
        records=[]
        for group,scenarios,seeds,strategies in (
            ("factorial",fact,SEEDS,POLICIES),
            ("diagnostic",changes,SEEDS,POLICIES+TIMING_CONTROLS),
            ("evaluation",holdout,EVAL_SEEDS,POLICIES+("rule","q"))):
            for condition,scenario in scenarios.items():
                for seed,strategy in product(seeds,strategies):
                    records.extend(simulate(seed,scenario,strategy,q,log,group,condition))
            print(group,len(records),"year rows",flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(out/"annual.csv",index=False)
    summarize(frame,out)
    if ledgers:
        verify_ledgers(out,frame)
    manifest=dict(years=YEARS,experiment_seeds=SEEDS,train_seeds=TRAIN_SEEDS,evaluation_seeds=EVAL_SEEDS,
        scenario={g:{k:asdict(v) for k,v in cases.items()} for g,cases in
                 [("factorial",fact),("diagnostic",changes),("evaluation",holdout)]},
        versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),
        files={str(f.relative_to(root)):sha256(f.read_bytes()).hexdigest() for f in
               [root/"core/land_succession.py",Path(__file__),root/"tests/test_land_succession.py",
                root/"modules/land_clearing_house.py",root/"agents/government_agent.py"]})
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--summaries-only",action="store_true")
    args=parser.parse_args()
    run(args.output.resolve(),ledgers=not args.summaries_only)
