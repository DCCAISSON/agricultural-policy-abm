"""Read the released single-region package without rerunning or editing its results."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8",newline="") as src:
        return list(csv.DictReader(src))


def audit(root: Path) -> dict:
    out=root/"results/synthetic_fair_comparison"
    design=json.loads((out/"design_before_evaluation.json").read_bytes())
    receipt=json.loads((out/"receipt.json").read_bytes())
    for folder,entries in ((root,design["code_sha256"]),(out,receipt["sha256"])):
        for name,expected in entries.items():
            if sha256((folder/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f"Source fingerprint mismatch: {name}")
    annual=read(out/"evaluation_annual.csv")
    cols=("condition","strategy","action","fallback")
    counted=Counter(tuple(r[k] for k in cols) for r in annual)
    detail=read(out/"evaluation_action_support.csv")
    if counted!={tuple(r[k] for k in cols):int(r["count"]) for r in detail}:
        raise ValueError("Released action frequency does not reproduce from annual records")
    scenarios=tuple(k for k in design["evaluation_scenarios"] if k!="middle_cash_reset")
    paths=defaultdict(dict)
    for r in read(out/"evaluation_paths.csv"):
        paths[r["condition"],r["strategy"]][int(r["seed"])]=(float(r["grain_kg"]),float(r["spending_yuan"]))
    metrics=("grain_kg","spending_yuan")
    strategies=sorted({s for c,s in paths if c=="middle"})
    coords={c:{s:tuple(mean(value[i] for value in paths[c,s].values()) for i in range(2))
               for s in strategies} for c in scenarios}
    coords["mean_of_designs"]={s:tuple(mean(coords[c][s][i] for c in scenarios) for i in range(2))
                               for s in strategies}
    dominance={}
    for c,cases in coords.items():
        q=cases["fitted_q"]
        dominance[c]=[s for s,(kg,cost) in cases.items() if kg>=q[0] and cost<=q[1] and (kg>q[0] or cost<q[1])]
    q=paths["middle","fitted_q"]
    p3=paths["middle","takeover_pre"]
    middle={}
    for i,k in enumerate(metrics):
        diffs=[p3[seed][i]-q[seed][i] for seed in sorted(q)]
        center=mean(diffs)
        radius=float(t.ppf(.975,len(diffs)-1))*stdev(diffs)/sqrt(len(diffs))
        middle[k]=dict(mean=center,low=center-radius,high=center+radius,n=len(diffs))
    early=defaultdict(list)
    for r in annual:
        if r["condition"] in scenarios and int(r["year"])<=design["treatment_stop_after_year"] and r["strategy"].endswith("_stop"):
            early[r["strategy"]].append(float(r["spending_yuan"]))
    prefix=design["treatment_stop_after_year"]
    main_q=[r for r in annual if r["strategy"]=="fitted_q" and r["condition"] in scenarios]
    return dict(verified_code=len(design["code_sha256"]),verified_result_files=len(receipt["sha256"]),
                action_counts_match=True,main_q_decisions=len(main_q),
                main_q_fallbacks=sum(bool(r["fallback"]) for r in main_q),
                mean_points={c:{s:dict(zip(metrics,v)) for s,v in cases.items()} for c,cases in coords.items()},
                strategies_dominating_q_at_mean=dominance,middle_fixed_p3_minus_q=middle,
                pre_stop_mean_expenditure={k:prefix*mean(v) for k,v in early.items()})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.package),indent=2))
