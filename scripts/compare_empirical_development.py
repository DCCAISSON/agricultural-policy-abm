"""Compare already-run B0 and development paths; never read a formal holdout."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from scripts.verify_empirical_succession import ROOT, metric


KEYS=["id","year"]
SURVEY="area_b area_e area_in1 area_de2 sow_food sow_cash sow_oother sow_total".split()
PREDICTED=["operated","grain","crop"]
METRICS=["participation_absolute_pp","mixed_absolute_pp","grain_share_quantile_mae",
         "grain_mu_times_relative_error","closing_physical_mu_relative_error"]
THRESHOLDS=[.10,.10,.15,.10,.10]
EPS=1e-7


def crop_state(grain: pd.Series, cash: pd.Series, other: pd.Series|None=None,
               total: pd.Series|None=None, eps:float=EPS) -> np.ndarray:
    known=grain.ge(0)&cash.ge(0)
    if other is None:
        other=pd.Series(0.,index=grain.index)
    if total is not None:
        other=other.mask(other.isna()&total.eq(0),0.)
    return np.select([~known,grain.gt(eps)&cash.gt(eps),grain.gt(eps),cash.gt(eps),
                      other.gt(eps),other.eq(0)],
                     ["unknown","mixed","grain","cash","other","unsown"],default="unknown_other")


def summary(frame: pd.DataFrame) -> dict:
    values=metric(frame,EPS,300)
    values["joint_pass"]=values["coverage_status"]=="comparable" and all(
        values[k]<=v for k,v in zip(METRICS,THRESHOLDS))
    crops=frame.sow_food.ge(0)&frame.sow_cash.ge(0)
    stock=frame.area_b.ge(0)&frame.area_e.ge(0)&frame.operated.notna()
    both=crops&stock&frame.area_e.gt(0)&frame.operated.gt(EPS)
    for prefix,g,c,a in [("obs","sow_food","sow_cash","area_e"),("sim","grain","crop","operated")]:
        s=frame[g]+frame[c]
        values[prefix+"_stock_real_zero"]=int(frame.loc[stock,a].le(EPS).sum())
        values[prefix+"_stock_zero_n"]=int(stock.sum())
        values[prefix+"_two_crop_common_stock_mu_times"]=s.loc[both].sum(min_count=1)
        values[prefix+"_common_stock_mu"]=frame.loc[both,a].sum(min_count=1)
        values[prefix+"_intensity_on_both"]=float((s.loc[both]/frame.loc[both,a]).median())
    values["common_stock_and_crop_n"]=int(both.sum())
    values["pre_endpoint_rescues"]=frame.get("pre_endpoint_rescue",pd.Series(dtype=float)).sum()
    values["post_endpoint_rescues"]=frame.get("post_endpoint_rescue",pd.Series(dtype=float)).sum()
    return values


def transitions(panel: pd.DataFrame, fields: list[str], mode: str, seed: int) -> list[dict]:
    rows=[]
    for origin_year in range(2010,2015):
        initial=panel.loc[panel.year.eq(origin_year)].set_index("id")
        later=panel.loc[panel.year.eq(origin_year+1)].set_index("id")
        both=initial.index.intersection(later.index)
        start=initial.loc[both]
        end=later.loc[both]
        for margin, s, e in [
            ("crop",start[fields[0]],end[fields[0]]),
            ("operated",start[fields[1]],end[fields[1]])]:
            if margin=="operated":
                valid=s.ge(0)&e.ge(0)
                before=np.where(s[valid]>EPS,"positive","zero")
                a,b=s[valid],e[valid]
                after=np.select([b.le(EPS),b.gt(a+EPS),b.lt(a-EPS)],["zero","expanded","contracted"],default="retained")
            else:
                valid=~s.str.startswith("unknown")&~e.str.startswith("unknown")
                before,after=s[valid],e[valid]
            counts=pd.crosstab(before,after)
            rows.extend(dict(variant=mode,seed=seed,year=origin_year+1,margin=margin,origin=origin,
                             destination=destination,n=int(counts.loc[origin,destination]),
                             paired_reporters=int(valid.sum()))
                for origin in counts.index for destination in counts.columns
                if counts.loc[origin,destination])
    return rows


def compare(project: Path, destination: Path, labeled_runs: list[str]) -> None:
    locked=json.loads((ROOT/"docs/全国经营接续/validation_protocol.json").read_bytes())
    old=project/"results/diagnostics/national_abm_v41_final_economic_closure_20261124/PRIVATE_DO_NOT_SHARE"
    rows,flows=[],[]
    seeds_by_mode={}
    for label in labeled_runs:
        mode,path=label.split("=",1)
        run=Path(path)
        receipt=json.loads((run/"inputs.json").read_bytes())
        if (receipt["source_sha256"],receipt["cohort_sha256"])!=(locked["authorized_household_sha256"],
                                                                locked["cohort_sha256"]):
            raise ValueError("Wrong sample or raw survey archive for paired comparison")
        settings=json.loads((run/"settings.json").read_bytes())
        if mode!=settings.get("development_variant","B0"):
            raise ValueError(f"Run settings do not match label {mode}")
        base=pd.read_csv(run/"PRIVATE_DO_NOT_SHARE/baseline.csv").rename(columns={"actor":"id"})
        base=base[KEYS+SURVEY].assign(operated=base.area_e,grain=base.sow_food,crop=base.sow_cash)
        for file in sorted((run/"PRIVATE_DO_NOT_SHARE").glob("*_both_*_accounts.csv")):
            seed=int(file.name.split("_")[0])
            matched=seeds_by_mode.setdefault(mode,set())
            if seed in matched:
                raise ValueError(f"Duplicate {mode}/{seed}")
            matched.add(seed)
            observed=pd.read_csv(old/f"F1_land_bridge_{seed}.csv")[KEYS+SURVEY]
            actors=pd.read_csv(file)
            actors=actors.loc[actors.kind.eq("household")]
            full=actors.merge(observed,on=KEYS,validate="1:1")
            panel=pd.concat([base,full],ignore_index=True)
            panel["obs_state"]=crop_state(panel.sow_food,panel.sow_cash,panel.sow_oother,panel.sow_total,eps=0.)
            panel["sim_state"]=crop_state(panel.grain,panel.crop)
            flows.extend(transitions(panel,["sim_state","operated"],mode,seed))
            common=panel.loc[panel.year.gt(2010)]
            for year,group in common.groupby("year"):
                rows.append(dict(variant=mode,seed=seed,year=year,diagnostic="recursion",**summary(group)))
                if "opencheck_land" in group:
                    opening=group.loc[group.opencheck_land.notna()].copy()
                    opening[PREDICTED]=opening[["opencheck_land","opencheck_grain","opencheck_crop"]].to_numpy()
                    v=summary(opening)
                    v["condition_opening_mu"]=float(opening.opencheck_land.sum())
                    for key in ["joint_pass","closing_physical_mu_relative_error","simulated_stock_close",
                                "sim_stock_p25","sim_stock_p50","sim_stock_p75",
                                "sim_stock_real_zero","sim_intensity_on_both"]:
                        v[key]=np.nan  # fixed opening scenario does not predict closing land
                    rows.append(dict(variant=mode,seed=seed,year=year,diagnostic="observed_opening_no_new_trade",**v))
            if mode=="B0":
                flows.extend(transitions(panel,["obs_state","area_e"],"survey",seed))
                for year,group in common.groupby("year"):
                    static=group.copy()
                    static[PREDICTED]=base.set_index("id").loc[group.id,PREDICTED].to_numpy()
                    rows.append(dict(variant="lag",seed=seed,year=year,diagnostic="recursion",**summary(static)))
                    prior=panel.loc[panel.year.eq(year-1),["id","area_e","sow_food","sow_cash"]]
                    one=group.drop(columns=PREDICTED).merge(prior.rename(columns=dict(zip(
                        ["area_e","sow_food","sow_cash"],PREDICTED))),on="id",validate="1:1")
                    one=one.loc[one[PREDICTED].ge(0).all(axis=1)]
                    rows.append(dict(variant="lag",seed=seed,year=year,diagnostic="observed_lag",**summary(one)))
    panel=pd.DataFrame(rows).sort_values(["diagnostic","variant","year","seed"])
    if not len(panel) or "B0" not in seeds_by_mode or not all(
        s<=seeds_by_mode["B0"] for s in seeds_by_mode.values()):
        raise ValueError("Each development seed must have a B0 seed")
    paired=[]
    decisions=[]
    recursive=panel.loc[panel.diagnostic.eq("recursion")]
    for mode,s in seeds_by_mode.items():
        if mode=="B0":
            continue
        a=recursive.loc[recursive.variant.eq(mode)].set_index(["year","seed"])
        b=recursive.loc[recursive.variant.eq("B0") & recursive.seed.isin(s)].set_index(["year","seed"])
        delta=a[METRICS]-b[METRICS]
        d=delta.groupby(level="year").mean()
        grain=METRICS[3]
        mix=METRICS[1]
        if mode=="I":
            target=d.loc[2011,grain]<=-.01 and not d.loc[[2014,2015],grain].gt(.025).any()
            other=[k for k in METRICS if k!=grain]
        elif mode=="C":
            target=d.loc[[2014,2015],mix].mean()<=-.01
            other=[k for k in METRICS if k!=mix]
        else:
            target=d.loc[2011,grain]<=-.01 and d.loc[[2014,2015],mix].mean()<=-.01
            other=[k for k in METRICS if k not in (grain,mix)]
        no_large_loss=not d[other].gt(.025).any().any()
        gate_noninferiority=a.joint_pass.sum()>=b.joint_pass.sum()
        decisions.append(dict(variant=mode,seeds=len(s),target_met=bool(target),
            no_large_other_gate_loss=bool(no_large_loss),joint_noninferiority=bool(gate_noninferiority),
            candidate_joint_count=int(a.joint_pass.sum()),b0_joint_count=int(b.joint_pass.sum()),
            eligible_for_next_development_stage=bool(target and no_large_loss and gate_noninferiority)))
        for year,h in delta.groupby(level="year"):
            n=len(h)
            for k in METRICS:
                m=h[k].mean()
                radius=float(t.ppf(.975,n-1)*h[k].std(ddof=1)/np.sqrt(n)) if n>1 else np.nan
                paired.append(dict(variant=mode,year=year,metric=k,n=n,mean_delta=m,low=m-radius,high=m+radius))
    destination.mkdir(parents=True,exist_ok=False)
    panel.to_csv(destination/"annual_development.csv",index=False)
    pd.DataFrame(flows).to_csv(destination/"development_transitions.csv",index=False)
    pd.DataFrame(paired).to_csv(destination/"paired_deltas.csv",index=False)
    (destination/"development_receipt.json").write_text(json.dumps(dict(comparisons=decisions,
        diagnostic_only=True,formal_holdout_read=False),indent=2),encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("project",type=Path)
    p.add_argument("destination",type=Path)
    p.add_argument("runs",nargs="+",help="B0=directory plus paired I= or C= runs")
    a=p.parse_args()
    compare(a.project,a.destination,a.runs)
