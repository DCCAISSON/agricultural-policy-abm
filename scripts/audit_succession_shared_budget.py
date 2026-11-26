"""Check the shared-budget release from its aggregate regional year ledger."""
import argparse
from collections import defaultdict
import csv
from hashlib import sha256
import json
from math import isclose
from pathlib import Path

from scripts.run_succession_shared_budget import INSTRUMENTS, METRICS, paired


def require(rule, message):
    if not rule:
        raise ValueError(message)


def audit(output: Path) -> dict:
    source=(output/"design.json").read_bytes()
    design=json.loads(source)
    receipt=json.loads((output/"receipt.json").read_bytes())
    require(sha256(source).hexdigest()==receipt["design_sha256"],"Wrong design fingerprint")
    arms=("none",*(f"{m}_{side}" for m in INSTRUMENTS for side in ("A","B","split")))
    regions=tuple(r["id"] for r in design["regions"])
    first,last=design["seeds_inclusive"]
    years=range(1,design["horizon"]+1)
    expected={(s,a,y,r) for s in range(first,last+1) for a in arms for y in years for r in regions}
    ledger,group={},defaultdict(list)
    with (output/"region_year.csv").open(newline="",encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            row=dict(raw,seed=int(raw["seed"]),year=int(raw["year"]))
            row.update({k:float(raw[k]) for k in (*METRICS,"reservation","common_budget","guarantee_mu")})
            key=tuple(row[k] for k in ("seed","arm","year","region"))
            require(key not in ledger,"Duplicate region/year")
            require(0<=row["spending"]<=row["committed"]+1e-7<=row["reservation"]+2e-7,
                    "Local payment exceeds reservation")
            if row["arm"]=="none" or row["year"]>design["pulse_years"]:
                require(row["policy"]=="none" and not any(row[k] for k in
                    ("reservation","spending","committed","guarantee_mu")),"Payment or promise persists after cutoff")
            if row["spending"]>0:
                plots=row["spending"]/(row["guarantee_mu"]*4.)
                require(isclose(plots,round(plots),abs_tol=1e-9),"Paid amount inconsistent with plot guarantee")
            ledger[key]=row
            group[key[:3]].append(row)
    require(set(ledger)==expected and len(ledger)==receipt["region_year_rows"],"Inconsistent regional path grid")
    for values in group.values():
        require(sum(r["reservation"] for r in values)<=design["budget_yuan"],"Joint reservation cap exceeded")
        require(all(r["common_budget"]==design["budget_yuan"] for r in values),"Inconsistent common budget")
    for (s,a,y,r),b in ledger.items():
        if y>1:
            previous=ledger[s,a,y-1,r]
            for operator in ("owner","tenant"):
                require(isclose(previous[f"{operator}_closing_cash"],b[f"{operator}_opening_cash"],
                    abs_tol=1e-8,rel_tol=1e-12),"Cash is not carried across the annual boundary")
        if a.endswith("_A") and r=="B" or a.endswith("_B") and r=="A":
            require(all(b[k]==ledger[s,"none",y,r][k] for k in METRICS),
                    "An unallocated region differs from its identical P0 path")
    new,decisions=paired(list(ledger.values()),design)
    with (output/"paired_effects.csv").open(newline="",encoding="utf-8") as file:
        frozen=list(csv.DictReader(file))
    require(len(frozen)==len(new),"Pair table length mismatch")
    for a,b in zip(frozen,new):
        require(all(a[k]==str(v) for k,v in b.items()),"Pair estimate does not reproduce")
    require(receipt["paths"]==len(ledger)//(len(years)*len(regions)),"Wrong path count")
    require(receipt["decisions"]==json.loads(json.dumps(decisions)) and receipt["stage_2_eligible"]==
            any(r["supported_reversal"] for r in decisions),"Gate result differs from paired intervals")
    return dict(audit="passed",paths=receipt["paths"],region_year_rows=len(ledger),paired_estimates=len(new),
                stage_2_eligible=receipt["stage_2_eligible"],
                checks=["grid","common cap","regional fiscal rows","stop cutoff","cash rollover",
                        "unchanged P0 in unfunded regions","paired estimates","training gate"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    print(json.dumps(audit(args.output),indent=2))
