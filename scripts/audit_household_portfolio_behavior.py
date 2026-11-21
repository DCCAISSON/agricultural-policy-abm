"""Compare same-ID annual household crop areas before opening the policy comparison."""
import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "results/diagnostics"
OUT = ROOT / "national_household_portfolio"


def areas(path, year):
    folder = path / f"year_{year-2010:02d}"
    snap = json.loads((folder / "actual_plot_use.json").read_text(encoding="utf-8"))
    harvests = json.loads((folder / "actual_harvests.json").read_text(encoding="utf-8"))
    if snap["year"] != year or snap["reconciliation"]["unreconciled_operators"]:
        raise ValueError(f"Unreconciled or misdated harvest: {folder}")
    # Plot allocations are additive at operator level only. No crop is assigned to an unidentified plot.
    df = pd.DataFrame(snap["patches"]).groupby("operator_id")[[
        "allocated_grain_area_mu", "allocated_economic_crop_area_mu"]].sum()
    df.columns = ["grain", "cash"]
    return df.reindex(harvests, fill_value=0)


def shares(a):
    total = a.grain + a.cash
    return a.grain.div(total.where(total > 0))


def stats(a):
    s = shares(a)
    return {"n":len(a), "positive_n":int(s.notna().sum()),
            "both":int(((a.grain > 0) & (a.cash > 0)).sum()),
            "both_min_below_1e5th_mu":int(((a.min(axis=1)>0) & (a.min(axis=1)<1e-5)).sum()),
            "mixed_rate":((a.grain > 0) & (a.cash > 0)).mean(),
            **dict(zip(("p25","p50","p75"),s.quantile([.25,.5,.75])))}


def decision_counts(path, year, ids):
    counts=Counter()
    with (path/f"year_{year-2010:02d}/actor_trace.jsonl").open(encoding="utf-8") as stream:
        for row in map(json.loads,stream):
            if row["actor_id"] not in ids:
                continue
            options=row["alternatives"] or []
            if not options:
                counts["no_current_options"]+=1
                continue
            mixed,other=[],[]
            offered=False
            for i,opt in enumerate(options):
                account=opt.get("expected_account")
                if account is None:
                    raise ValueError("Household candidate areas not observed")
                g,a=account["grain_area_mu"],account["area_mu"]
                interior=1e-8<g<a-1e-8
                offered=offered or interior
                if interior and row["selected_index"]==i:
                    counts["selected_interior"]+=1
                if opt["account_feasible"] and opt["service_available"] is not False:
                    (mixed if interior else other).append(opt["score_yuan"])
            if not mixed:
                counts["no_feasible_interior"]+=1
                counts["interior_offered_but_infeasible" if offered else "interior_not_offered"]+=1
            elif not other:
                counts["only_interior_feasible"]+=1
            elif max(mixed)>max(other)+1e-7:
                counts["interior_scores_higher"]+=1
            elif max(other)>max(mixed)+1e-7:
                counts["endpoint_or_exit_scores_higher"]+=1
            else:
                counts["score_tie"]+=1
    return {k:counts[k] for k in ("no_current_options","no_feasible_interior","interior_not_offered",
                  "interior_offered_but_infeasible","only_interior_feasible",
                  "interior_scores_higher","endpoint_or_exit_scores_higher","score_tie","selected_interior")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("year", type=int, choices=[2011,2012])
    parser.add_argument("--old", nargs=2, required=True, type=Path, metavar=("SEED103","SEED109"))
    parser.add_argument("--new", nargs=2, type=Path, metavar=("SEED103","SEED109"))
    args = parser.parse_args()
    survey = pd.read_csv(OUT / "reference/reference_households.csv").rename(
        columns={"grain_area":"grain","cash_area":"cash"})
    def actual(year):
        frame = survey[survey.year.eq(year)].set_index("actor_id",verify_integrity=True)[["grain","cash"]]
        return frame.dropna().loc[lambda a: (a>=0).all(axis=1)]
    y=args.year
    obs,prior=actual(y),actual(y-1)
    records=[]
    for seed,old,new in zip((103,109),args.old,args.new or (None,None)):
        for rule,path in (("old",old),("portfolio",new)):
            if path is None:
                continue
            sim=areas(path,y).reindex(obs.index)
            if new is not None:
                receipt=json.loads((path.parent/"audit.json").read_text(encoding="utf-8"))
                if not receipt["ok"] or receipt["seed"]!=seed or receipt["rule"]!=("new" if rule=="portfolio" else "old"):
                    raise ValueError(f"Unverified experiment: {path}")
            if sim.isna().any().any():
                raise ValueError(f"Missing actual harvest account: {path}, {y}")
            s,o=shares(sim),shares(obs)
            p=shares(prior) if y==2011 else shares(areas(path,y-1))
            common=(s.notna() & o.notna())
            motions=pd.concat([o,shares(prior),s,p],axis=1,join="inner").dropna()
            obs_step=(motions.iloc[:,0]-motions.iloc[:,1]).abs().mean()
            sim_step=(motions.iloc[:,2]-motions.iloc[:,3]).abs().mean()
            O,S=stats(obs),stats(sim)
            row=dict(year=y,seed=seed,rule=rule,common_positive_n=int(common.sum()),
                     shared_change_n=len(motions),obs_abs_change=obs_step,sim_abs_change=sim_step)
            if new is not None:
                row["run"]=path.parent.name
            for label,values in (("obs",O),("sim",S),("common_obs",stats(obs[common])),("common_sim",stats(sim[common]))):
                row.update({f"{label}_{k}":v for k,v in values.items()})
            row.update(mixed_error=abs(O["mixed_rate"]-S["mixed_rate"]),
                       quantile_mae=sum(abs(O[k]-S[k]) for k in ("p25","p50","p75"))/3,
                       change_error=abs(obs_step-sim_step))
            row.update(decision_counts(path,y,obs.index))
            records.append(row)
        if new is None:
            continue
        a,b=records[-2:]
        b["pass"]=bool(b["mixed_error"]<=min(.10,a["mixed_error"]+1e-12)
                       and b["quantile_mae"]<=min(.15,a["quantile_mae"]+1e-12)
                       and b["shared_change_n"]>0 and b["change_error"]<=.15)
    result=OUT/(f"behavior_{y}.csv" if args.new else f"baseline_{y}.csv")
    table=pd.DataFrame(records)
    table.to_csv(result,index=False,encoding="utf-8-sig")
    print(table.reindex(columns=["year","seed","rule","obs_n","obs_both","sim_both","mixed_error","quantile_mae","change_error","pass"]).to_string(index=False))
    if y==2012 and args.new:
        dev=pd.read_csv(OUT/"behavior_2011.csv")
        assert dev[["seed","run","rule"]].equals(table[["seed","run","rule"]])
        checks=pd.concat([dev,table]).query("rule == 'portfolio'")
        assert len(checks)==4
        unlocked=checks["pass"].eq(True).all()
        gate=dict(unlocked=bool(unlocked), rule="PROTOCOL.md §5",
                  failed=checks.loc[~checks["pass"].eq(True),["year","seed"]].to_dict("records"),
                  hashes={f:sha256((OUT/f).read_bytes()).hexdigest() for f in
                          ("PROTOCOL.md","behavior_2011.csv","behavior_2012.csv","reference/reference_households.csv")})
        (OUT/"behavior_gate.json").write_text(json.dumps(gate,indent=2,ensure_ascii=False),encoding="utf-8")
        if not unlocked:
            from scripts.audit_national_succession_transmission import performance
            annual=[]
            for s,old,new in zip((103,109),args.old,args.new):
                for r,path in (("old",old),("portfolio",new)):
                    ids={z["id"] for z in json.loads((path/"initial_households.json").read_text(encoding="utf-8"))}
                    for t in (1,2,3):
                        with (path/f"year_{t:02d}/actors.jsonl").open(encoding="utf-8") as f:
                            households=[z for z in map(json.loads,f) if z["id"] in ids]
                        assert len(households)==len({z["id"] for z in households})==len(ids)==600
                        accounts=[z["income_account"] for z in households]
                        assert all(a["step"]==t for a in accounts)
                        def mean(k):
                            return sum(a[k] for a in accounts)/600
                        out=performance(path,t)
                        pay=mean("support_at_harvest_yuan")+mean("verified_support_received_yuan")
                        wages=mean("off_farm_income_yuan")+mean("labor_sale_income_yuan")
                        rent=mean("rent_received_yuan")-mean("rent_paid_yuan")
                        operating=mean("sales_yuan")-sum(mean(k) for k in
                                   ("production_cost_yuan","service_cost_yuan","conversion_cost_yuan"))
                        other=mean("common_income_yuan")
                        residual=out["h0_cash_yuan"]-sum((pay,wages,rent,operating,other))
                        assert abs(residual)<1e-6,(path,t,residual)
                        annual.append(dict(seed=s,rule=r,year=2010+t,**out,h0_operating=operating,
                                           h0_wages=wages,h0_net_rent=rent,h0_support=pay,h0_common=other,
                                           account_residual=residual))
            base=pd.DataFrame(annual)
            base.to_csv(OUT/"baseline_accounts.csv",index=False,encoding="utf-8-sig")
            # Empty policy effects mean not evaluated, never a zero effect.
            status=pd.DataFrame([dict(seed=s,rule=r,policy=p,
                 status="behavior_only" if p=="base" else "locked_not_evaluated",
                 incremental_grain_t=None,incremental_h0_income_yuan=None,fiscal_cost_difference_yuan=None)
                 for s in (103,109) for r in ("old","portfolio") for p in ("base","ordinary_post","scale80_post")])
            sums=base.groupby(["seed","rule"],as_index=False).agg(
                total_grain_kg=("grain_kg","sum"),mean_h0_net_yuan=("h0_cash_yuan","mean"),
                total_fiscal_yuan=("fiscal_cost_yuan","sum")).assign(policy="base")
            status.merge(sums,how="left",on=["seed","rule","policy"],validate="one_to_one").to_csv(
                OUT/"policy_comparison_status.csv",index=False,encoding="utf-8-sig")


if __name__=="__main__":
    main()
