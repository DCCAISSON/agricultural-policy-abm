"""Train and evaluate frozen government policies on the NFPS600 economy."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
import platform
from statistics import mean

import numpy as np
import pandas as pd
import scipy

from core.land_succession import CROPS
from core.regional_succession_budget import fork_menu,make_budget,public_state,worlds_at_start
from scripts.prepare_succession_budget_inputs import digest,prepare
from scripts.run_succession_annual_allocation import csv_write,json_write,seeds_of
from scripts.succession_annual_learning import fit,state_key
from scripts.succession_sprint_support import select,support

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT/"docs/全国经营接续/统一共同预算/design.json"
PROTOCOL = DESIGN.with_name("sprint_protocol.json")
CHOICE = DESIGN.with_name("agricultural_freeze.json")
METRICS = (
    "grain_kg","grain_area","cash_crop_area","idle_area","retained_grain","continuity_area",
    "successor_grain","offered_area","posted_area","matched_area","unfilled_area","leased_area",
    "held_area","renewed_area","default_area","leased_idle","cash_unplanted","labor_unplanted",
    "owner_opening_cash","owner_closing_cash","owner_income","owner_profit","owner_nonfarm",
    "owner_rent_income","owner_rent_paid","owner_area","owner_support","rent_total","committed","spending",
)
CODE = (
    "core/land_succession.py","core/succession_area_market.py","core/succession_allocation.py",
    "core/regional_succession_budget.py","scripts/prepare_succession_budget_inputs.py",
    "scripts/run_succession_empirical_budget.py","scripts/succession_annual_learning.py",
    "scripts/run_succession_annual_allocation.py","scripts/succession_sprint_support.py",
)


def settings() -> tuple[dict, str, dict[str,str]]:
    data = DESIGN.read_bytes().replace(b"\r\n",b"\n")
    lock = PROTOCOL.read_bytes().replace(b"\r\n",b"\n")
    design = json.loads(data)
    design.update({k:v for k,v in json.loads(lock).items() if k.endswith("_seeds")})
    choice = CHOICE.read_bytes().replace(b"\r\n",b"\n") if CHOICE.exists() else b"{}"
    design.update(json.loads(choice))
    blocks = [set(seeds_of(design[k])) for k in ("fixed_dev_seeds","training_seeds","supplement_seeds","evaluation_seeds")]
    assert len(set.union(*blocks)) == sum(len(s) for s in blocks)
    return design,sha256(data+b"\n"+lock+b"\n"+choice).hexdigest(),{
        k:sha256((ROOT/k).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for k in CODE}


def init_worker(assets: tuple, design: dict) -> None:
    global ASSETS, DESIGN_DATA, BUDGET, MENU
    ASSETS, DESIGN_DATA = assets, design
    BUDGET = make_budget(assets,design)
    MENU = BUDGET.menu()


def fixed_local(seed: int) -> dict:
    """The same nine fixed paths per region generate all 21 combinations."""
    roots = worlds_at_start(seed,ASSETS,DESIGN_DATA)
    local = {}
    for i,root in enumerate(roots):
        for a in dict.fromkeys(move[i] for move in MENU):
            world = root.fork()
            local[i,a] = [BUDGET.local_step(world,a) for _ in range(DESIGN_DATA["main_years"])]
    return local


def fixed_worlds(seed: int) -> tuple[list[tuple], list[float]]:
    local = fixed_local(seed)
    totals = [(fsum(r["grain_kg"] for i,a in enumerate(move) for r in local[i,a]),
               fsum(r["spending"] for i,a in enumerate(move) for r in local[i,a])) for move in MENU]
    p0 = [fsum(r["grain_kg"] for i in range(len(ASSETS)) for r in local[i,MENU[0][i]][y:])
          for y in range(len(local[0,MENU[0][0]]))]
    return totals,p0


def explore_actions(seed: int) -> list[int]:
    return list(np.random.default_rng([seed,DESIGN_DATA["uniform_exploration_stream"]]).integers(
                    len(MENU),size=DESIGN_DATA["main_years"]))


def screen_world(seed: int) -> list[tuple]:
    worlds=worlds_at_start(seed,ASSETS,DESIGN_DATA)
    rows=[]
    for y,move in enumerate(explore_actions(seed),1):
        rows.append((seed,y,public_state(worlds,DESIGN_DATA,BUDGET),int(move)))
        BUDGET.step(worlds,MENU[move])
    return rows


def learn_world(job: tuple[int, dict]) -> list[tuple]:
    design = DESIGN_DATA
    seed, requested = job
    world = worlds_at_start(seed,ASSETS,design)
    openings = []
    for y,a in enumerate(explore_actions(seed),1):
        if y in requested:
            state = public_state(world,design,BUDGET)
            if state!=requested[y]:
                raise ValueError("Opening replay differs from screened state")
            branches,next_worlds = fork_menu(world,design,BUDGET,MENU)
            openings.append((seed,state,branches))
            world=next_worlds[a]
        else:
            BUDGET.step(world,MENU[a])
        if y==max(requested):
            break
    return openings


def fit_design(design: dict) -> dict:
    return dict(active_years=design["main_years"],min_openings_per_state_action=design["minimum_openings"],
                common_state_fraction_per_year=design["common_fraction"])


def train(project: Path, out: Path, workers: int) -> None:
    if not CHOICE.exists():
        raise ValueError("Freeze the agriculture candidate before training")
    design, fingerprint, code = settings()
    assets, inputs = prepare(project,design)
    out.mkdir(parents=True,exist_ok=False)
    json_write(out/"inputs.json",inputs)
    fixed_seeds = seeds_of(design["fixed_dev_seeds"])
    training = seeds_of(design["training_seeds"])
    print(f"Training input verified: 600 households, {len(assets)} regions, {make_budget(assets,design).common} yuan/year",flush=True)
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(assets,design)) as pool:
        print("Fixed comparator development",flush=True)
        fixed_samples = list(pool.map(fixed_worlds,fixed_seeds))
    menu = make_budget(assets,design).menu()
    scores = [mean(s[0][i][0] for s in fixed_samples) for i in range(len(menu))]
    strong = max(range(len(scores)),key=scores.__getitem__)
    tails = {y+1:mean(s[1][y] for s in fixed_samples) for y in range(design["main_years"])}
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(assets,design)) as pool:
        print("Screen parent development paths without counterfactual rewards",flush=True)
        screen=[r for block in pool.map(screen_world,training) for r in block]
        selected=select(screen)
        initial_support=support(screen,selected,len(menu),design)
        if initial_support["coverage"]<design["coverage_threshold"]:
            print(f"Initial coverage {initial_support['coverage']:.2%}; run the one registered supplemental block",flush=True)
            screen.extend(r for block in pool.map(screen_world,seeds_of(design["supplement_seeds"])) for r in block)
            selected=select(screen,selected)
        coverage=support(screen,selected,len(menu),design)
        requests={}
        for seed,y,key,_ in screen:
            if (seed,y) in selected:
                requests.setdefault(seed,{})[y]=key
        jobs=sorted(requests.items())
        opening_paths=[]
        print(f"Fork all actions at {len(selected)} selected openings spanning {len(jobs)} distinct parents",flush=True)
        for i,path in enumerate(pool.map(learn_world,jobs),1):
            opening_paths.extend(path)
            if i%24==0:
                print(f"completed {i}/{len(jobs)} opening replay parents",flush=True)
    table,_ = fit(opening_paths,fit_design(design),tails)
    repeated=[x for x in opening_paths if x[0]%4!=0]
    t_repeat,_=fit(repeated,fit_design(design),tails)
    s_repeat=support(screen,{(s,y) for s,y in selected if s%4!=0},len(menu),design)
    f = dict(design_sha256=fingerprint,code_sha256=code,inputs=inputs,menu=menu,strong_fixed=strong,
             fixed_scores=scores,pooled_p0=tails,table=table,coverage=coverage,initial_coverage=initial_support,
             repeat_table=t_repeat,repeat_coverage=s_repeat)
    json_write(out/"frozen.json",f)
    csv_write(out/"fixed_development.csv",[
        dict(seed=seed,action=i,grain_kg=grain,spending=cost,p0_tail=sample[1][i] if i<len(sample[1]) else "")
        for seed,sample in zip(fixed_seeds,fixed_samples) for i,(grain,cost) in enumerate(sample[0])
    ])
    csv_write(out/"openings.csv.gz",[
        dict(seed=seed,opening=state_key(state),action=i,grain_kg=r,successor=state_key(next_state))
        for seed,state,branches in opening_paths for i,(r,next_state,_) in enumerate(branches)
    ])
    csv_write(out/"screen.csv.gz",[dict(seed=s,year=y,state=state_key(k),explore_action=a,selected=int((s,y) in selected))
                                   for s,y,k,a in screen])
    json_write(out/"receipt.json",dict(
        frozen_sha256=digest(out/"frozen.json"),data=fingerprint,opening_visits=len(opening_paths),
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
    ))
    print(f"Frozen fixed action {strong}; common action support {coverage['coverage']:.3%}",flush=True)


def choose(strategy: str, key: str, frozen: dict, stopped: bool) -> tuple[int,str]:
    if stopped:
        return 0,"forced_stop"
    if strategy.startswith("fixed_"):
        return int(strategy.split("_")[1]),"fixed"
    what,_,repeat=strategy.partition("_")
    entry = frozen["repeat_table" if repeat else "table"].get(key)
    if entry is None:
        return 0,"unseen"
    if not entry["supported"]:
        return 0,"sparse"
    return entry[what],"supported"


def policy_step(strategy: str, key: str, frozen: dict, mode: str, year: int, design: dict) -> tuple[int,str]:
    if mode=="pulse":
        return ({"central_then_equal":5,"west_then_equal":9}[strategy],"pulse") if year==1 else (17,"common_followup")
    return choose(strategy,key,frozen,mode=="withdrawal" and year>design["withdrawal_after"])


def policy_values(strategy: str, key: str, frozen: dict, action: int, reason: str) -> tuple:
    if reason not in ("unseen","sparse","supported"):
        return "",""
    name="repeat_table" if strategy.endswith("_repeat") else "table"
    row=frozen[name].get(key)
    if row is None:
        return "",""
    return row["q_values"][action],row["current_grain"][action]


def path_rows(seed: int, strategy: str, mode: str, frozen: dict) -> list[dict]:
    worlds = worlds_at_start(seed,ASSETS,DESIGN_DATA)
    rows = []
    for y in range(1,DESIGN_DATA["main_years"]+1):
        key = state_key(public_state(worlds,DESIGN_DATA,BUDGET))
        action, reason = policy_step(strategy,key,frozen,mode,y,DESIGN_DATA)
        values=policy_values(strategy,key,frozen,action,reason)
        rows.extend(year_rows(seed,mode,strategy,y,key,action,reason,BUDGET.step(worlds,MENU[action]),values))
    return rows


def year_rows(seed: int, mode: str, strategy: str, y: int, key: str,
              action: int, reason: str, outputs: tuple[dict, ...], values: tuple=("", "")) -> list[dict]:
    return [dict(seed=seed,mode=mode,strategy=strategy,year=y,region=a.name,action=action,state=key,
                 reason=reason,chosen_q_kg=values[0],chosen_current_kg=values[1],
                 physical_mu=a.area,**{k:row[k] for k in ("policy","reservation","rate_mu",*METRICS)})
            for a,row in zip(ASSETS,outputs,strict=True)]


def eval_job(seed: int, frozen: dict) -> list[dict]:
    local = fixed_local(seed)
    fixed = []
    for action,move in enumerate(MENU):
        last = [r.grain_kg for r in ASSETS]
        for i in range(DESIGN_DATA["main_years"]):
            key = state_key((DESIGN_DATA["main_years"]-i,
                *(int(kg>=DESIGN_DATA["grain_bin_fraction"]*CROPS["grain"][0]*r.area)
                  for kg,r in zip(last,ASSETS,strict=True)),BUDGET.common))
            output = tuple(local[j,a][i] for j,a in enumerate(move))
            fixed.extend(year_rows(seed,"main",f"fixed_{action:02}",i+1,key,action,"fixed",output))
            last = [row["grain_kg"] for row in output]
    stops = [f"fixed_{frozen['strong_fixed']:02}","myopic","q"]
    return fixed + [r for mode,policies in (("main",("myopic","q","myopic_repeat","q_repeat")),("withdrawal",stops),
                                           ("pulse",("central_then_equal","west_then_equal")))
                    for p in policies for r in path_rows(seed,p,mode,frozen)]


def evaluate(project: Path, trained: Path, out: Path, workers: int) -> None:
    from itertools import repeat
    design, fingerprint, code = settings()
    assets, inputs = prepare(project,design)
    file = trained/"frozen.json"
    frozen = json.loads(file.read_bytes())
    receipt = json.loads((trained/"receipt.json").read_bytes())
    if (frozen["design_sha256"]!=fingerprint or frozen["code_sha256"]!=code
        or frozen["inputs"]!=inputs or digest(file)!=receipt["frozen_sha256"]):
        raise ValueError("Design, frozen code, or authorized input mismatch")
    out.mkdir(parents=True,exist_ok=False)
    seeds = seeds_of(design["evaluation_seeds"])
    rows = []
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(assets,design)) as pool:
        print("Begin independent evaluation with frozen tables",flush=True)
        for i,path in enumerate(pool.map(eval_job,seeds,repeat(frozen)),1):
            rows.extend(path)
            if i%8==0:
                print(f"completed {i}/{len(seeds)} paired evaluations",flush=True)
    csv_write(out/"region_year.csv.gz",rows)
    json_write(out/"receipt.json",dict(region_years=len(rows),paired_seeds=len(seeds),frozen_sha256=digest(file)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("train","evaluate"))
    cli.add_argument("project",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--trained",type=Path)
    cli.add_argument("--workers",type=int,default=1)
    args = cli.parse_args()
    if args.stage=="train":
        train(args.project,args.output,args.workers)
    elif args.trained:
        evaluate(args.project,args.trained,args.output,args.workers)
    else:
        cli.error("evaluate requires --trained")
