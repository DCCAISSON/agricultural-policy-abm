"""Paired single-year allocation experiment using two unmodified 7a economies."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from hashlib import sha1, sha256
import json
from math import fsum, sqrt
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t as student_t

from core.land_succession import AREA, CROPS, Scenario, Succession


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs/全国经营接续/共同预算设计.json"
INSTRUMENTS = ("grain_pre","takeover_pre")
OFF = ("none",0)
JointMove = tuple[tuple[str,int],tuple[str,int]]
METRICS = (
    "grain_kg","grain_area","retained_grain","continuity_area","successor_grain",
    "matched_area","leased_area","held_area","renewed_area","default_area","leased_idle",
    "owner_opening_cash","tenant_opening_cash","owner_closing_cash","tenant_closing_cash",
    "owner_support","tenant_support","rent_total","spending","committed","cash_unplanted",
    "labor_unplanted","exits",
)


@dataclass(frozen=True,slots=True)
class JointBudget:
    common: int
    caps: tuple[int,int]

    def feasible_moves(self) -> tuple[JointMove,...]:
        options=(OFF,*((policy,cap) for policy in INSTRUMENTS for cap in self.caps))
        return tuple((a,b) for a in options for b in options if a[1]+b[1]<=self.common)

    def advance(self, worlds: tuple[Succession,Succession], move: JointMove) -> tuple[dict,dict]:
        if len(worlds)!=2 or len(move)!=2 or worlds[0] is worlds[1] or worlds[0].year!=worlds[1].year:
            raise ValueError("Require two distinct regions at the same annual boundary")
        for policy,cap in move:
            if (policy=="none" and cap!=0) or (
                policy!="none" and (policy not in INSTRUMENTS or cap not in self.caps)):
                raise ValueError("Invalid instrument, tier, or inactive reservation")
        if sum(cap for _,cap in move)>self.common:
            raise ValueError("Reservations exceed the joint cap")
        # Both allocations are fixed before any agricultural decision; unused
        # reservations are not reassigned after seeing eligibility or harvest.
        result=[]
        for world,(policy,cap) in zip(worlds,move):
            original=world.scenario
            guarantee=min(original.rate*AREA,cap/len(world.plots),AREA*CROPS["grain"][2])/AREA
            world.scenario=replace(original,budget=cap)
            try:
                row=world.step(policy)
            finally:
                world.scenario=original
            if not 0<=row["spending"]<=row["committed"]+1e-7<=cap+2e-7:
                raise ValueError("Local spending or verified commitments violate the prior reservation")
            result.append(dict(row,policy=policy,reservation=cap,guarantee_mu=guarantee))
        if fsum(r["committed"] for r in result)>self.common+1e-7:
            raise ValueError("Annual commitments exceed common budget")
        return result[0],result[1]


def setup(design: dict) -> tuple[JointBudget,tuple[Scenario,Scenario]]:
    caps=tuple(design["positive_region_caps_yuan"])
    budget=JointBudget(design["budget_yuan"],caps)
    regions=tuple(Scenario(
        search_failure=r["search_failure"],cash=r["cash"],labor=r["labor"],
        contract_years=design["contract_years"],rate=design["nominal_rate_yuan_per_mu"],
    ) for r in design["regions"])
    if len(regions)!=2 or len(caps)!=2 or not (0<caps[0]<caps[1]<=budget.common and 2*caps[0]==budget.common):
        raise ValueError("Both low or one high must exhaust the same common ceiling")
    for r in regions:
        maxima=min(r.rate*AREA,AREA*CROPS["grain"][2])
        if min(caps[0]/(r.owners*2),maxima)>=min(caps[1]/(r.owners*2),maxima):
            raise ValueError("The nominal tiers collapse to the same effective guarantee")
    return budget,regions


def pulse_paths(seed: int, design: dict) -> list[dict]:
    budget,scenarios=setup(design)
    roots=tuple(Succession(2*seed+i,s) for i,s in enumerate(scenarios))
    low,high=budget.caps
    plans={"none":(OFF,OFF)}
    for k in INSTRUMENTS:
        plans.update({f"{k}_A":((k,high),OFF),f"{k}_B":(OFF,(k,high)),f"{k}_split":((k,low),(k,low))})
    rows=[]
    for arm,pulse in plans.items():
        worlds=tuple(r.fork() for r in roots)
        for year in range(1,design["horizon"]+1):
            action=pulse if year<=design["pulse_years"] else (OFF,OFF)
            for r,a in zip(design["regions"],budget.advance(worlds,action)):
                rows.append(dict(seed=seed,arm=arm,year=year,region=r["id"],
                                 policy=a["policy"],common_budget=budget.common,
                                 reservation=a["reservation"],guarantee_mu=a["guarantee_mu"],
                                 **{k:a[k] for k in METRICS}))
    return rows


def interval(values: list[float], alpha:float) -> tuple[float,float,float]:
    center=mean(values)
    radius=float(student_t.ppf(1.-alpha/2.,len(values)-1))*stdev(values)/sqrt(len(values))
    return center,center-radius,center+radius


def opposite_supported(first: tuple[float,float], cumulative: tuple[float,float]) -> bool:
    return (first[0]>0 and cumulative[1]<0) or (first[1]<0 and cumulative[0]>0)


def paired(rows: list[dict], design: dict) -> tuple[list[dict],list[dict]]:
    lookup={(r["seed"],r["arm"],r["year"],r["region"]):r for r in rows}
    seeds=sorted({r["seed"] for r in rows})
    out,decisions=[],[]
    for k in INSTRUMENTS:
        windows={"first_year":[1],"after_stop":range(2,design["horizon"]+1),
                 "cumulative":range(1,design["horizon"]+1)}
        focal={}
        for label,years in windows.items():
            for ref in ("B","none","split"):
                arm_ref="none" if ref=="none" else f"{k}_{ref}"
                for group,regions in {"total":["A","B"],"A":["A"],"B":["B"]}.items():
                    for metric in METRICS:
                        values=[fsum(lookup[s,f"{k}_A",y,r][metric]-lookup[s,arm_ref,y,r][metric]
                                     for y in years for r in regions) for s in seeds]
                        gate=(ref=="B" and group=="total" and metric=="grain_kg" and label!="after_stop")
                        alpha=design["gate_family_alpha"]/design["gate_family_size"] if gate else 0.05
                        center,lo,hi=interval(values,alpha)
                        if gate:
                            focal[label]=(lo,hi)
                        out.append(dict(instrument=k,focal=f"{k}_A",reference=arm_ref,region=group,window=label,
                                        metric=metric,n=len(seeds),mean=center,low=lo,high=hi,
                                        confidence=1-alpha,gate_metric=gate))
        decisions.append(dict(instrument=k,first_year_interval=focal["first_year"],
                              cumulative_interval=focal["cumulative"],
                              supported_reversal=opposite_supported(focal["first_year"],focal["cumulative"])))
    return out,decisions


def verify_base(design: dict) -> None:
    for name,expected in design["base_blobs_lf"].items():
        data=(ROOT/name).read_bytes().replace(b"\r\n",b"\n")
        header=f"blob {len(data)}\0".encode("ascii")
        if sha1(header+data).hexdigest()!=expected:
            raise ValueError(f"Unexpected agricultural base: {name}")


def run(output: Path, workers:int) -> None:
    source=DESIGN.read_bytes()
    design=json.loads(source)
    verify_base(design)
    setup(design)
    output.mkdir(parents=True,exist_ok=False)
    (output/"design.json").write_bytes(source)
    seeds=range(design["seeds_inclusive"][0],design["seeds_inclusive"][1]+1)
    if workers==1:
        rows=[r for seed in seeds for r in pulse_paths(seed,design)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows=[r for x in pool.map(pulse_paths,seeds,[design]*len(seeds),chunksize=4) for r in x]
    contrasts,decision=paired(rows,design)
    for name,table in [("region_year.csv",rows),("paired_effects.csv",contrasts)]:
        with (output/name).open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    receipt=dict(design_sha256=sha256(source).hexdigest(),paths=len(rows)//(2*design["horizon"]),
                 region_year_rows=len(rows),decisions=decision,
                 stage_2_eligible=any(r["supported_reversal"] for r in decision))
    (output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    run(args.output,args.workers)
