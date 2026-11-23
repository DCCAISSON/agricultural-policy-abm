"""Read the frozen 2011 baseline accounts; no model execution or estimation."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_household_portfolio_reference import MANIFEST, OUT as REF, RAW
from scripts.audit_household_portfolio_behavior import areas

BASE = REF.parent
DEST = BASE / "decision_audit"
PRIVATE = DEST / "PRIVATE_DO_NOT_SHARE"
PUBLIC = DEST / "SHAREABLE_AGGREGATES"
CROPS = ["sow_w", "sow_r", "sow_c", "sow_s", "sow_t", "sow_o"]
COST = ["production_cost_yuan", "service_cost_yuan", "rent_in_yuan", "management_cost_yuan", "conversion_cost_yuan"]
BENEFIT = ["sales_yuan", "support_yuan", "rent_out_yuan", "off_farm_yuan", "common_income_yuan"]
METRICS = COST + BENEFIT + ["cash_cost_yuan", "risk_penalty_yuan", "family_labor_days", "hired_labor_days",
                            "off_farm_days", "hired_labor_cost_yuan", "working_capital_available",
                            "capital_slack_yuan", "score_yuan"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def survey():
    job = read(MANIFEST)["jobs"][8]
    cohort = Path(job["cohort"]).read_bytes()
    assert hashlib.sha256(cohort).hexdigest() == job["cohort_sha256"]
    f = pd.DataFrame(json.loads(cohort)["farmers"])
    cols = ["year", "hhid", "id", "hhnumbers", "area_b", "area_e", "sow_total", "sow_food", "sow_cash", "sow_oother", *CROPS]
    raw = pd.read_csv(RAW, usecols=cols, dtype={"hhid": "string", "id": "string"}, keep_default_na=False, na_values=[""])
    raw["source_row"] = raw.index + 2
    origin = raw.iloc[f.source_row.to_numpy()-2].reset_index(drop=True)
    assert origin.year.eq(2010).all() and origin.hhnumbers.eq(f.family_size).all() and f.cohort_agent_id.nunique()==600
    origin = origin.assign(actor_id=f.cohort_agent_id)
    base_ids = raw.loc[raw.year.eq(2010), "hhid"]
    keys = origin.loc[~origin.hhid.isin(base_ids[base_ids.duplicated(False)]), ["hhid","id","actor_id"]]
    follow = raw.loc[raw.year.eq(2011)]
    s = keys.merge(follow.loc[~follow.hhid.duplicated(False)], on=["hhid","id"], validate="one_to_one").drop(columns=["hhid","id"])
    s = s.set_index("actor_id").sort_index()
    ref = pd.read_csv(REF/"reference_households.csv").query("year==2011").set_index("actor_id").sort_index()
    assert s.index.equals(ref.index) and len(s)==562
    np.testing.assert_allclose(s[["sow_food","sow_cash"]], ref[["grain_area","cash_area"]], equal_nan=True)
    assert s[["sow_food","sow_cash"]].notna().all().all()
    s["survey_class"] = np.select([s.sow_food.gt(0)&s.sow_cash.gt(0),s.sow_food.gt(0),s.sow_cash.gt(0)],
                                  ["mixed","grain_only","cash_only"],default="neither")
    # An aggregate zero is distinct from an unknown reason for that zero.
    s["survey_zero_evidence"] = np.select([s.survey_class.ne("neither"),s.sow_oother.gt(0),
        s.sow_total.gt(0),s.sow_total.eq(0)], ["not_neither","other_crop_positive","positive_total_unallocated",
        "zero_total_sowing"],default="total_missing")
    return s


def run():
    s = survey()
    obs, options, roles = [], [], []
    init = read(REF/"initial_grain_shares.json")
    for seed in (103,109):
        run = BASE/"runs"/f"s{seed}_new_base_20261121_220058_199520"
        receipt = read(run/"audit.json")
        assert receipt["ok"] and receipt["seed"]==seed and receipt["rule"]=="new"
        ep = run/"episode"
        assert (ep/"COMPLETED.json").is_file()
        yr = ep/"year_01"
        sim = areas(ep,2011).loc[s.index]
        decisions = {r["operator_id"]:r for r in read(yr/"decision_trace.json")}
        traces = {r["actor_id"]:r for r in map(json.loads,(yr/"actor_trace.jsonl").read_text(encoding="utf-8").splitlines())}
        balances = {r["id"]:r["preproduction_cash_account"] for r in map(json.loads,(yr/"actors.jsonl").read_text(encoding="utf-8").splitlines())}
        leases = pd.DataFrame(read(yr/"lease_transactions.json")).groupby("supplier_id").area_mu.sum()
        for actor,r in s.iterrows():
            d,t,b = decisions[actor],traces[actor],balances[actor]
            g,c = sim.loc[actor]
            action = t["selected_action"]
            assert t["year"]==d["year"]==2011 and action==d["action"] and b["step"]==1
            assert abs(g+c-b["cultivated_area_mu"])<1e-6
            result = "mixed" if min(g,c)>0 else "grain_only" if g>0 else "cash_only" if c>0 else "neither"
            zero = ("not_neither" if result!="neither" else "selected_exit" if action=="exit"
                    else "no_held_land_after_market" if b["held_area_mu"]==0 else "held_land_uncultivated")
            obs.append(dict(seed=seed,actor_id=actor,**r.to_dict(),sim_grain_mu=g,sim_cash_mu=c,sim_class=result,
                            selected_action=action,sim_zero_evidence=zero,held_mu=b["held_area_mu"],
                            supplied_lease_mu=leases.get(actor,0),opening_plan_mu=d["expand_cash_area_mu"]))
            if (r.survey_class,result)!=("mixed","cash_only"):
                continue
            prev=d["previous_grain_share"]
            assert action=="expand_cash" and init.get(actor)==prev
            area=d["expand_cash_area_mu"]
            candidates=[]
            for i,o in enumerate(t["alternatives"]):
                a=o["expected_account"]
                cost=sum(a[k] for k in COST)
                score=sum(a[k] for k in BENEFIT)-cost-a["risk_penalty_yuan"]
                assert abs(score-o["score_yuan"])<1e-6
                assert a["area_mu"]==0 or abs(a["area_mu"]-area)<1e-8
                why=o["constraint_reason"] or ""
                assert bool(why)==(not o["account_feasible"])
                entry=dict(seed=seed,actor_id=actor,action=o["action"],mode=o["mode"],selected=i==t["selected_index"],
                           feasible=o["account_feasible"] and o["service_available"],reason=why,service_available=o["service_available"],
                           prior_share=prev,share=(a["grain_area_mu"]/a["area_mu"] if a["area_mu"] else None),
                           plan_mu=a["area_mu"],inertia_yuan=d["inertia_yuan"],score_yuan=score,
                           hired_labor_available_days=a["hired_labor_available_days"])
                vals=dict(a,cash_cost_yuan=cost,capital_slack_yuan=a["working_capital_available"]-cost,score_yuan=score)
                # Divide exit's annual score by the same opening area for comparison, not its zero planted area.
                for k in METRICS:
                    entry[k+"_per_open_mu"]=vals[k]/area
                candidates.append(entry)
            cash=next(v for v in candidates if v["selected"])
            assert cash["action"]==action and cash["feasible"] and cash["share"]==0
            for v in candidates:
                v["score_delta_vs_cash_per_mu"]=(v["score_yuan"]-cash["score_yuan"])/area
            options.extend(candidates)
            planted=[v for v in candidates if v["share"] is not None]
            shares=sorted(set(v["share"] for v in planted))
            def higher(start, feasible=False):
                return min((v["share"] for v in planted if (not feasible or v["feasible"]) and v["share"]>start+1e-12),default=None)
            for role,share in [("cash",0),("keep",prev),("min_positive_offered",higher(0)),
                               ("min_positive_feasible",higher(0,True)),("next_above_prior",higher(prev) if prev is not None else None)]:
                matches=[] if share is None else [v for v in planted if abs(v["share"]-share)<1e-12]
                if matches:
                    v=max(matches,key=lambda x:(x["feasible"],x["service_available"],x["score_yuan"]))
                    status="feasible" if v["feasible"] else "offered_infeasible"
                    roles.append(dict(v,role=role,status=status))
                else:
                    roles.append(dict(seed=seed,actor_id=actor,role=role,status="not_available",prior_share=prev))
    households,opts,comp=map(pd.DataFrame,(obs,options,roles))
    assert households.groupby("seed").size().tolist()==[562,562]
    targets=households.query("survey_class=='mixed' and sim_class=='cash_only'")
    assert len(comp)==5*len(targets) and len(opts.query("selected"))==len(targets)
    quantities=[k+"_per_open_mu" for k in METRICS]+["share","score_delta_vs_cash_per_mu"]
    pairs=comp.query("status=='feasible'").merge(comp.query("role=='cash'")[["seed","actor_id",*quantities]],
             on=["seed","actor_id"],suffixes=("","_cash"),validate="many_to_one")
    for k in quantities:
        pairs["delta_"+k]=pairs[k]-pairs[k+"_cash"]
    grouped=pairs.groupby(["seed","role"])
    summary=grouped[quantities+["delta_"+k for k in quantities]].mean().join(grouped.size().rename("n")).reset_index()
    # Only sufficiently populated groups release means; no example records, identifiers or extrema.
    summary=summary[summary.n>=5]
    flags=comp.groupby(["seed","role","status"]).size().rename("n").reset_index()
    zero=households.groupby(["seed","survey_class","sim_class","sim_zero_evidence"]).size().rename("n").reset_index()
    survey_zero=s.query("survey_class=='neither'").groupby("survey_zero_evidence").size().rename("n").reset_index()
    constraints=opts.groupby(["seed","action","mode","feasible","reason","service_available"]).size().rename("n").reset_index()
    crop_coverage=[]
    for seed,group in targets.groupby("seed"):
        grain_sum=group[CROPS].sum(axis=1,min_count=len(CROPS))
        crop_coverage.append(dict(seed=seed,n=len(group),grain_detail_incomplete=int(grain_sum.isna().sum()),
            grain_detail_matches_total=int(np.isclose(grain_sum,group.sow_food,atol=1e-5,rtol=0).sum()),
            other_area_positive=int(group.sow_oother.gt(0).sum()),other_area_missing=int(group.sow_oother.isna().sum()),
            total_sown_gt_year_end=int((group.sow_total-group.area_e>1e-5).sum()),
            total_vs_year_end_comparable=int(group[["sow_total","area_e"]].notna().all(axis=1).sum())))
    output={PRIVATE/"household_crosswalk.csv":households,PRIVATE/"all_alternatives.csv":opts,
        PRIVATE/"role_accounts.csv":comp,PUBLIC/"mean_paired_accounts.csv":summary,
        PUBLIC/"availability_counts.csv":flags,PUBLIC/"choice_constraints.csv":constraints,
        PUBLIC/"survey_model_class_counts.csv":zero,PUBLIC/"survey_double_zero.csv":survey_zero,
        PUBLIC/"annual_crop_coverage.csv":pd.DataFrame(crop_coverage)}
    for p,df in output.items():
        p.parent.mkdir(parents=True,exist_ok=True)
        data=df.to_csv(index=False,lineterminator="\n").encode("utf-8-sig")
        if p.exists():
            if p.read_bytes()!=data:
                raise FileExistsError(p)
        else:
            with p.open("xb") as f:
                f.write(data)
    print(summary[["seed","role","n","score_delta_vs_cash_per_mu"]].to_string(index=False))


if __name__=="__main__":
    run()
