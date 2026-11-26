"""Replay the locked historical evaluation, without publishing identifiable rows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCOPES = ("M0","R0","R3","F1")


def metric(h, eps, min_n):
    observed = h.loc[h.sow_food.ge(0)&h.sow_cash.ge(0)]
    g,c = observed.sow_food,observed.sow_cash
    sg,sc = observed.grain,observed.crop
    positive, simulated = g+c>0,sg+sc>eps
    q1 = (g/(g+c)).loc[positive].quantile([.25,.5,.75])
    q2 = (sg/(sg+sc)).loc[simulated].quantile([.25,.5,.75])
    stock = h.loc[h.area_b.ge(0)&h.area_e.gt(0)]
    joint = h.loc[h.area_in1.ge(0)&h.area_de2.ge(0)]
    ratio = lambda a,b: abs(a-b)/b if b>0 else np.nan
    u = dict(n=len(observed),stock_n=len(stock),flow_joint_n=len(joint),
        survey_sown=int(positive.sum()),sim_sown=int(simulated.sum()),
        survey_mixed=int((g.gt(0)&c.gt(0)).sum()),sim_mixed=int((sg.gt(eps)&sc.gt(eps)).sum()),
        survey_grain=g.sum(),sim_grain=sg.sum(),participation_absolute_pp=abs(positive.mean()-simulated.mean()),
        mixed_absolute_pp=abs((g.gt(0)&c.gt(0)).mean()-(sg.gt(eps)&sc.gt(eps)).mean()),
        grain_share_quantile_mae=(q1-q2).abs().mean(),grain_mu_times_relative_error=ratio(sg.sum(),g.sum()),
        observed_stock_open=stock.area_b.sum(min_count=1),observed_stock_close=stock.area_e.sum(min_count=1),
        simulated_stock_close=stock.operated.sum(min_count=1),
        closing_physical_mu_relative_error=ratio(stock.operated.sum(),stock.area_e.sum()),
        observed_joint_in=joint.area_in1.sum(min_count=1),observed_joint_out=joint.area_de2.sum(min_count=1),
        simulated_joint_in=joint.rented_in.sum(min_count=1),simulated_joint_out=joint.rented_out.sum(min_count=1))
    for q in (.25,.5,.75):
        u[f"survey_share_p{int(q*100)}"],u[f"sim_share_p{int(q*100)}"] = q1.loc[q],q2.loc[q]
        u[f"survey_stock_p{int(q*100)}"] = stock.area_e.quantile(q)
        u[f"sim_stock_p{int(q*100)}"] = stock.operated.quantile(q)
    u["survey_intensity_median"] = ((g+c)/observed.area_e.where(observed.area_e>0)).median()
    u["sim_intensity_median"] = ((sg+sc)/observed.operated.where(observed.operated>0)).median()
    u["coverage_status"] = "comparable" if min(u["n"],u["stock_n"])>=min_n else "insufficient"
    return u


def verify(project: Path, runs: list[Path], destination: Path):
    protocol_path=ROOT/"docs/全国经营接续/validation_protocol.json"
    protocol=json.loads(protocol_path.read_bytes())
    historical=protocol["historical"]
    thresholds=protocol["thresholds"]
    eps=protocol["definitions"]["simulation_sown_tolerance_mu_times"]
    d=project/"results/diagnostics"
    old=d/"national_abm_v41_final_economic_closure_20261124"
    frozen=pd.read_csv(old/"run_01/annual_behavior.csv")
    prior=pd.read_csv(d/"national_abm_v41_scale_closure_20261124/run_01/annual_behavior.csv")
    compared=["seed","year","n","obs_active","sim_active","obs_mixed","sim_mixed","obs_grain","sim_grain","share_mae"]
    q=prior[prior.spec.isin(SCOPES[:3]) & prior.seed.isin(historical["seeds"])].sort_values(["spec","seed","year"])
    p=frozen[frozen.spec.isin(SCOPES[:3]) & frozen.seed.isin(historical["seeds"])].sort_values(["spec","seed","year"])
    pd.testing.assert_frame_equal(p[["spec",*compared]].reset_index(drop=True),q[["spec",*compared]].reset_index(drop=True))
    frozen=frozen[frozen.seed.isin(historical["seeds"])].copy()
    historical_pass = (
        ((frozen.sim_active-frozen.obs_active).abs()/frozen.n<=.1)
        & ((frozen.sim_mixed-frozen.obs_mixed).abs()/frozen.n<=.1)
        & (frozen.share_mae<=.15)
        & ((frozen.sim_grain-frozen.obs_grain).abs()/frozen.obs_grain<=.1))
    frozen["four_recalculated"]=historical_pass
    if frozen[frozen.spec.eq("F1")].four_recalculated.sum()!=45:
        raise ValueError("Archived F1 tally does not reproduce")
    views=[]
    bands=[]
    files=[]
    economic_cols=["operated","cultivated","rented_in","rented_out","grain","crop","grain_kg","paid","closing"]
    for run in runs:
        spec=json.loads((run/"inputs.json").read_bytes())
        if spec["source_sha256"]!=protocol["authorized_household_sha256"] or spec["cohort_sha256"]!=protocol["cohort_sha256"]:
            raise ValueError("Frozen weighted initialization cohort or raw survey fingerprint changed")
        archived_fit=old/"PRIVATE_DO_NOT_SHARE/F1_301/behavior_parameters.csv"
        if (run/"behavior_parameters.csv").read_bytes()!=archived_fit.read_bytes():
            raise ValueError("Behavior coefficient artifact differs from the frozen F1")
        for ledger in sorted((run/"PRIVATE_DO_NOT_SHARE").glob("*_both_*_accounts.csv")):
            seed=int(ledger.name.split("_")[0])
            candidate=pd.read_csv(ledger)
            historical_bridge=pd.read_csv(old/f"PRIVATE_DO_NOT_SHARE/F1_land_bridge_{seed}.csv")
            hh=candidate[candidate.kind.eq("household")]
            link_cols="year id area_b area_e area_in1 area_de2 sow_food sow_cash sow_oother".split()
            counts=hh.merge(historical_bridge[link_cols],on=["year","id"],validate="1:1")
            lookup=historical_bridge.set_index(["year","id"]).sort_index()
            observed=counts.set_index(["year","id"]).sort_index()
            np.testing.assert_allclose(lookup[economic_cols],observed[economic_cols],rtol=1e-10,atol=1e-6)
            for year,group in counts.groupby("year"):
                value=metric(group,eps,protocol["definitions"]["min_pooled_annual_coverage"])
                passes={name:bool(value[name]<=limit) for name,limit in thresholds.items()}
                views.append(dict(seed=seed,year=year,**value,**{f"pass_{k}":v for k,v in passes.items()},
                    historical_four=all(list(passes.values())[:4]),historical_five=value["coverage_status"]=="comparable" and all(passes.values())))
                small=group.assign(band=pd.cut(group.base_area,[-np.inf,5,15,30,np.inf],right=False))
                for dimension in ("region","band"):
                    for label,sub in small.groupby(dimension,observed=True):
                        b=metric(sub,eps,protocol["definitions"]["min_descriptive_subgroup"])
                        if b["coverage_status"]=="insufficient":
                            b={k:b[k] for k in ("n","stock_n","flow_joint_n","coverage_status")}
                        bands.append(dict(seed=seed,year=year,dimension=dimension,stratum=str(label),**b))
            files.append(dict(seed=seed,actor_years=len(candidate),household_years=len(hh),max_grain_difference=float((lookup.grain-observed.grain).abs().max())))
    history=pd.DataFrame(views).sort_values(["seed","year"])
    expected={(y,s) for y in historical["years"] for s in historical["seeds"]}
    if set(zip(history.year,history.seed))!=expected or len(history)!=len(expected):
        raise ValueError("Historical seed-year coverage does not match the protocol")
    reference=frozen.loc[frozen.spec.eq("F1"),["year","seed","four_recalculated"]].sort_values(["seed","year"])
    np.testing.assert_array_equal(reference.four_recalculated,history.historical_four)
    conditions=d/"f1_validity_20261126"
    chunks=range(301,321,5)
    conditional=pd.concat([pd.read_csv(conditions/f"conditions_{s}.csv") for s in chunks])
    transition=pd.concat([pd.read_csv(conditions/f"transitions_{s}.csv") for s in chunks])
    receipts=pd.concat([pd.read_csv(conditions/f"receipts_{s}.csv") for s in chunks])
    if set(zip(receipts.year,receipts.seed))!=expected or len(receipts)!=len(expected):
        raise ValueError("Frozen conditional experiment coverage differs from replay")
    if not (receipts.households.eq(600).all() and receipts[["max_grain_diff","max_cash_diff"]].lt(1e-6).all().all()):
        raise ValueError("Frozen conditional baseline does not reproduce")
    succession=pd.concat([pd.read_csv(r/"succession_summary.csv") for r in runs]).sort_values(["seed","year"])
    if set(zip(succession.year,succession.seed))!=expected or len(succession)!=len(expected):
        raise ValueError("Missing or duplicated succession accounts")
    # Earlier replay shards left absent event classes blank rather than writing zero.
    event_mu=["retained_grain_mu","strict_synthetic_succession_mu","other_grain_recovery_mu",
        "leased_not_sown_mu","leased_non_grain_mu","unidentified_origin_or_use_mu","other_mu",
        "contracted_not_sown_mu","contracted_non_grain_mu"]
    succession[event_mu]=succession.reindex(columns=event_mu,fill_value=0).fillna(0)
    destination.mkdir(parents=True,exist_ok=False)
    frozen.groupby(["spec","year"])["four_recalculated"].agg(["sum","count"]).to_csv(destination/"historical_frozen_four.csv")
    history.to_csv(destination/"candidate_historical_gates.csv",index=False)
    pd.DataFrame(bands).to_csv(destination/"candidate_strata.csv",index=False)
    history.groupby("year")[["historical_four","historical_five","pass_closing_physical_mu_relative_error"]].agg(["sum","count"]).to_csv(destination/"candidate_annual_tally.csv")
    succession.to_csv(destination/"candidate_synthetic_succession.csv",index=False)
    conditional.groupby(["year","contrast","case"]).mean(numeric_only=True).drop(columns="seed").to_csv(destination/"f1_conditions_reused.csv")
    transition.groupby(["year","case","origin","destination"]).n.sum().div(len(historical["seeds"])).to_csv(
        destination/"f1_transitions_reused.csv",header=["mean_households_per_seed"])
    pd.DataFrame(files).to_csv(destination/"recursive_replay_checks.csv",index=False)
    (destination/"validation_receipt.json").write_text(json.dumps(dict(
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),historical_cells=len(history),
        historical_four=int(history.historical_four.sum()),historical_five=int(history.historical_five.sum()),
        independent_recursive="not_decidable_without_certified_untouched_sample",
        formal_policy_training="not_started", decision="C"),indent=2),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("destination",type=Path)
    parser.add_argument("runs",type=Path,nargs="+")
    args=parser.parse_args()
    verify(args.project,args.runs,args.destination)
