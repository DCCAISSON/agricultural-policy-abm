"""Read-only, first-decision replays and source-row comparisons for the one-mu switchers."""
from unittest.mock import patch

import numpy as np
import pandas as pd

from agents.farmer_agent import CropType
from scripts import audit_portfolio_one_mu as audit
from scripts.audit_national_crop_budget_basis_20261118 import DATA, LABELS, SALES, ROOT
from scripts.prepare_household_portfolio_reference import RAW

BASE = audit.BASE/"decision_audit/one_mu"
OUT = BASE/"labor_trace"


def trace(seed, chosen, survey):
    saved = []
    original = audit.organization_accounts

    def watch(f, actions):
        if str(f.unique_id) not in chosen:
            return original(f, actions)
        previews = []
        demand = f.model._get_farmer_service_demand

        def seen(preview):
            offer = demand(preview)
            previews.append((preview,offer))
            return offer

        with patch.object(f.model,"_get_farmer_service_demand",seen):
            options = original(f, actions)
        for item,(p,offer) in zip((i for i in options if i.account.area_mu>0),previews,strict=True):
            a = item.account
            if item.mode!="outsource" or not (a.action=="expand_cash" or abs(a.grain_area_mu-1)<1e-9):
                continue
            assert offer and item.service_available
            rates = p.model.config["production"]["cost_structure"]["input_requirements"]["labor_days"]
            rate = [p.estimate_production_labor_profile(c,a.area_mu)["labor_days_per_mu"]
                    for c in (CropType.GRAIN,CropType.CASH_CROP)]
            need = rate[0]*a.grain_area_mu+rate[1]*(a.area_mu-a.grain_area_mu)
            cap = p._get_family_labor_capacity_for_production()
            assert np.allclose([a.family_labor_days,a.hired_labor_days],
                               [min(need,cap),max(0,need-cap)])
            with patch.object(p,"labor_reduction_ratio",0):
                before = p.estimate_production_labor_profile(p.planned_crop,a.area_mu)["total_labor_need"]
            cfg = f.model.service_market._get_service_type_config(offer["service_type"])
            assert np.isclose(p.labor_reduction_ratio,cfg["coverage_ratio"]*.85*a.service_area_mu/a.area_mu)
            saved.append(dict(seed=seed,actor_id=str(f.unique_id),plan="cash" if not a.grain_area_mu else "one",
                area=a.area_mu,grain_mu=a.grain_area_mu,package=offer["service_type"].value,
                service_mu=a.service_area_mu,coverage=cfg["coverage_ratio"],signal=p.labor_reduction_ratio,
                fee=a.service_cost_yuan,grain_base=rates["grain_farmer"],cash_base=rates["cash_crop_farmer"],
                grain_days_mu=rate[0],cash_days_mu=rate[1],no_service_need=before,need=need,
                family_cap=cap,family=a.family_labor_days,hired=a.hired_labor_days,hired_cost=a.hired_labor_cost_yuan,
                wage=f.model.labor_market.current_agricultural_wage,off_farm_days=a.off_farm_days,risk=a.risk_penalty_yuan))
            assert np.isclose(a.hired_labor_cost_yuan,a.hired_labor_days*saved[-1]["wage"])
        return options

    with patch.object(audit,"organization_accounts",watch):
        audit.probe(seed,survey,{"booked":1.})
    return saved


def raw_2011(s,ids):
    grain = "w r c s t o".split()
    cols = ["year","sow_food","sow_cash",*[f"inc_cash_{i}" for i in range(1,11)]]
    cols += [f"{k}_{crop}" for crop in grain for k in ("sow","labor_d","labor_hd","prod","inc","mach")]
    raw = pd.read_csv(RAW,usecols=cols).iloc[s.loc[ids,"source_row"].to_numpy()-2].set_axis(ids)
    assert raw.year.eq(2011).all() and np.allclose(raw.sow_cash,s.loc[ids,"sow_cash"])
    dic = pd.read_excel(DATA/"household variables2009-2015.xlsx",header=None).set_index(0)[1]
    assert not dic.str.contains("出售|销售",na=False).any()
    records=[]
    for crop in grain:
        name=dic[f"sow_{crop}"].split("1.")[0]
        block=raw[[f"{k}_{crop}" for k in ("sow","labor_d","labor_hd","prod","inc","mach")]].copy()
        block.columns=["mu","days","hired_days","output_kg","gross_yuan","machine_fee"]
        records.append(block.assign(crop=name,actor_id=ids))
    for i in range(1,11):
        name=dic[f"inc_cash_{i}"].split("4.")[0]
        records.append(pd.DataFrame(dict(crop=name,actor_id=ids,gross_yuan=raw[f"inc_cash_{i}"].to_numpy())))
    return pd.concat(records,ignore_index=True)


def budgets():
    reg=pd.read_csv(ROOT/"results/national_crop_budget_basis_20261118/run_04/real2024_revision/record_selection_money2024.csv")
    dic=pd.read_excel(DATA/"household variables2016-2017.xlsx",header=None).set_index(0)[1]
    cols={}
    for crop,(q,m) in SALES.items():
        fields={k:dic.index[dic.eq(crop+LABELS[k])].item() for k in ("area","labor","hired_days","output","machinery")}
        assert "出售" in dic[q] and crop in dic[q] and "金额" in dic[m] and crop in dic[m]
        cols[crop]=dict(fields,sales_quantity=q,sales_amount=m)
    raw=pd.read_csv(DATA/"2016-2017household.csv",usecols={"year",*(v for fields in cols.values() for v in fields.values())})
    rows=[]
    for (year,crop),g in reg.loc[reg.eligible].groupby(["year","crop"]):
        f=cols[crop]
        src=raw.iloc[g.source_row.to_numpy()-2].rename(columns={v:k for k,v in f.items()})
        assert src.year.eq(year).all() and np.allclose(src[list(f)],g[list(f)],equal_nan=True)
        sales=src.dropna(subset=["output","sales_quantity","sales_amount"])
        def per(d,k):
            return d[k].sum()/d.area.sum() if len(d) else np.nan
        rows.append(dict(year=year,crop=crop,n=len(src),area_mu=src.area.sum(),days_mu=per(src,"labor"),
            hired_days_mu=per(src,"hired_days"),machine_fee_positive=int(src.machinery.gt(0).sum()),
            sales_complete=len(sales),sales_joint_mu=sales.area.sum(),output_kg_mu=per(sales,"output"),
            sold_kg_mu=per(sales,"sales_quantity"),sold_yuan_mu=per(sales,"sales_amount"),
            sold_above_output=int(sales.sales_quantity.gt(sales.output).sum())))
    return pd.DataFrame(rows)


def main():
    src=BASE/"PRIVATE_DO_NOT_SHARE"
    h=pd.read_csv(src/"one_mu_household.csv")
    c=pd.read_csv(src/"repriced_choices.csv").query("budget=='booked' and grid=='plus_one_mu' and grain_mu>0")
    target=h.merge(c[["seed","actor_id"]],validate="one_to_one")
    assert target.groupby("seed").size().tolist()==[13,21]
    s=audit.survey()
    prior=h.drop_duplicates("actor_id").set_index("actor_id")
    for k,col in (("field_cash_income","survey_field_cash_income"),("inc_cash_7","survey_vegetable_income"),
                  ("inc_cash_9","survey_fruit_income")):
        s[k]=prior[col].reindex(s.index)
    records=[]
    for seed,g in target.groupby("seed"):
        records.extend(trace(seed,set(g.actor_id),s))
    plans=pd.DataFrame(records)
    key=["seed","actor_id"]
    a,b=[plans.query("plan==@name").set_index(key).sort_index() for name in ("cash","one")]
    t=target.set_index(key).sort_index()
    assert a.index.equals(t.index) and b.index.equals(t.index)
    d=pd.DataFrame(dict(crop_change_without_service=a.no_service_need-b.no_service_need,
        difference_in_service_saving=(b.no_service_need-b.need)-(a.no_service_need-a.need),
        total_days_saved=a.need-b.need,family_days_saved=a.family-b.family,hired_days_saved=a.hired-b.hired,
        hired_yuan_saved=a.hired_cost-b.hired_cost,delta_fee=b.fee-a.fee,
        package_unchanged=a.package.eq(b.package),same_service_area=np.isclose(a.service_mu,b.service_mu),
        delta_offfarm_days=b.off_farm_days-a.off_farm_days,delta_risk=b.risk-a.risk),index=a.index)
    assert np.allclose(d.crop_change_without_service+d.difference_in_service_saving,d.total_days_saved)
    assert np.allclose(d.total_days_saved,d.family_days_saved+d.hired_days_saved)
    assert np.allclose(d.hired_days_saved,-t.delta_hired_days) and np.allclose(d.hired_yuan_saved,-t.delta_hired_cost)
    ids=t.index.get_level_values("actor_id").unique()
    raw=raw_2011(s,ids)
    tagged=target[key].merge(raw,on="actor_id",validate="many_to_many")
    agg=[]
    for (seed,crop),g in tagged.groupby(["seed","crop"]):
        lab=g[g.mu.gt(0) & g.days.notna()]
        prod=g[g.mu.gt(0) & g.output_kg.notna()]
        agg.append(dict(seed=seed,crop=crop,n=len(g),positive_area=g.mu.gt(0).sum(),
            labor_comparable=len(lab),days_mu=lab.days.sum()/lab.mu.sum() if len(lab) else None,
            output_comparable=len(prod),kg_mu=prod.output_kg.sum()/prod.mu.sum() if len(prod) else None,
            gross_positive=g.gross_yuan.gt(0).sum(),gross_reported=g.gross_yuan.count()))
    for name,df in [("PRIVATE_DO_NOT_SHARE/plan_accounts",plans),("PRIVATE_DO_NOT_SHARE/paired_labor",d.reset_index()),
                    ("PRIVATE_DO_NOT_SHARE/survey_crop_rows",raw),("median_labor",d.groupby("seed").median().reset_index()),
                    ("survey_2011",pd.DataFrame(agg)),("budget_year_crop",budgets())]:
        path=OUT/(name+".csv")
        path.parent.mkdir(exist_ok=True,parents=True)
        content=df.to_csv(index=False,lineterminator="\n").encode("utf-8-sig")
        if path.exists():
            assert content==path.read_bytes(),path
        else:
            with path.open("xb") as f:
                f.write(content)
    print("Unique households:",len(ids))
    print(plans.groupby(["seed","plan","package"]).size().to_string())
    print(d.groupby("seed").median().to_string())


def audit_capacity():
    """Static first-decision sensitivity. Do not feed these budgets back into the model."""
    from scripts.prepare_household_portfolio_reference import MANIFEST
    from pathlib import Path
    import json
    from hashlib import sha256
    from contextlib import nullcontext

    job=json.loads(MANIFEST.read_bytes())["jobs"][8]
    cohort=Path(job["cohort"]).read_bytes()
    assert sha256(cohort).hexdigest()==job["cohort_sha256"]
    farmers=pd.DataFrame(json.loads(cohort)["farmers"])
    keys=["hhid","RegionId","cmgen","smgen","hmgen"]
    types={k:"string" for k in keys}
    frame=pd.read_csv(RAW,usecols=keys+["year","hhlabornum","pop_labor","hhnumbers"],dtype=types)
    origin=frame.iloc[farmers.source_row.to_numpy()-2].reset_index(drop=True)
    assert origin.year.eq(2010).all() and origin.hhnumbers.eq(farmers.family_size).all()
    origin["actor_id"]=farmers.cohort_agent_id
    src=BASE/"PRIVATE_DO_NOT_SHARE"
    target=pd.read_csv(src/"repriced_choices.csv").query("grid=='plus_one_mu' and budget=='booked' and grain_mu>0")
    hh=origin[origin.actor_id.isin(target.actor_id)].copy()
    assert len(hh)==24 and not hh.duplicated(keys).any()
    members=pd.read_csv(RAW.parent/"2009-2015individual.csv",usecols=keys+["year","homefarmdays","homenonfarmdays","timeout","idind"],dtype=types)
    j=hh.merge(members[members.year.eq(2010)],on=keys+["year"],how="left",validate="one_to_many")
    print("NFPS 2010:", hh[["hhlabornum","pop_labor"]].describe().to_string(),j[["idind","homefarmdays","homenonfarmdays","timeout"]].count().to_dict())
    cf=next((RAW.parents[1]/"CFPS"/"2010").rglob("cfps2010adult_201906.dta"))
    cf=pd.read_stata(cf,columns=["urban","qh1","qh2","qh3","qh4"],convert_categoricals=False)
    cf=cf[cf.urban.eq(0)&cf.qh1.between(1,12)&cf.qh2.between(1,31)&cf.qh3.gt(0)&cf.qh3.le(24)].copy()
    cf["reported_days"]=cf.qh1*cf.qh2
    print("CFPS recalled annual agricultural work, not capacity:", cf.groupby("qh4").reported_days.agg(["size","median"]).to_dict("index"))
    nworkers=hh.set_index("actor_id").hhlabornum
    usable=nworkers.notna() & nworkers.ge(0) & nworkers.le(hh.set_index("actor_id").hhnumbers) & nworkers.mod(1).eq(0)
    s=audit.survey()
    prior=pd.read_csv(src/"one_mu_household.csv").drop_duplicates("actor_id").set_index("actor_id")
    for k,col in (("field_cash_income","survey_field_cash_income"),("inc_cash_7","survey_vegetable_income"),("inc_cash_9","survey_fruit_income")):
        s[k]=prior[col].reindex(s.index)
    saved=[]
    original=audit.organization_accounts
    for seed,g in target.groupby("seed"):
        ids=set(g.actor_id)
        def watch(f,actions):
            base=original(f,actions)
            actor=str(f.unique_id)
            if actor not in ids or "expand_cash" not in actions:
                return base
            area=next(i.account.area_mu for i in base if i.account.action=="expand_cash")
            shares=dict(f.household_candidate_grain_shares)
            one=next((k for k,v in shares.items() if abs(v*area-1)<1e-10),"portfolio_probe_one_mu")
            shares[one]=1/area
            menu=tuple(dict.fromkeys([*actions,one]))
            n=float(nworkers[actor])
            refs=cf.groupby("qh4").reported_days.median()
            for label,days,cap in (("frozen",365.,None),("workers_calendar",n*365.,None),
                ("cfps_with_paid_nonfarm",n*365.,n*refs[1]),("cfps_no_paid_nonfarm",n*365.,n*refs[0])):
                if label!="frozen" and not usable[actor]:
                    continue
                capacity=nullcontext() if cap is None else patch.object(f,"_get_family_labor_capacity_for_production",return_value=cap)
                with patch.object(f,"household_candidate_grain_shares",shares),patch.object(f,"total_labor_units",days),capacity:
                    opts=original(f,menu)
                    choice=audit.choose_action_income(f,audit.best_crop_accounts(opts))
                    a,b=[next(i.account for i in opts if i.mode=="outsource" and i.account.action==k) for k in ("expand_cash",one)]
                    saved.append(dict(seed=seed,actor_id=actor,case=label,workers=nworkers[actor],cap=f._get_family_labor_capacity_for_production(),
                        cash_hired=a.hired_labor_days,one_hired=b.hired_labor_days,hired_saved=a.hired_labor_days-b.hired_labor_days,
                        net_delta=b.net_income-a.net_income,both_feasible=a.feasible and b.feasible,positive=b.score>a.score,
                        one_chosen=choice==one,choice=choice))
            return base
        with patch.object(audit,"organization_accounts",watch):
            audit.probe(seed,s,{"booked":1.})
    res=pd.DataFrame(saved)
    assert len(res)==len(target)+3*target.actor_id.map(usable).sum() and not res.duplicated(["seed","actor_id","case"]).any()
    chk=res.query("case=='frozen'").merge(pd.read_csv(src/"one_mu_household.csv"),on=["actor_id","seed"],validate="one_to_one")
    assert chk.one_chosen.all() and np.allclose(chk.net_delta,chk.delta_net) and np.allclose(chk.hired_saved,-chk.delta_hired_days)
    matched=res[res.actor_id.map(usable)&res.case.eq("frozen")].assign(case="frozen_same_ids")
    summary=pd.concat([res,matched]).groupby(["seed","case"]).agg(n=("actor_id","size"),median_cap=("cap","median"),median_cash_hired=("cash_hired","median"),
        median_one_hired=("one_hired","median"),median_hired_saved=("hired_saved","median"),median_delta_yuan=("net_delta","median"),
        both_feasible=("both_feasible","sum"),positive=("positive","sum"),one_chosen=("one_chosen","sum")).reset_index()
    for path,df in [(OUT/"PRIVATE_DO_NOT_SHARE/capacity_choice_with_cfps.csv",res),(OUT/"capacity_comparison_with_cfps.csv",summary)]:
        data=df.to_csv(index=False,lineterminator="\n").encode("utf-8-sig")
        if path.exists():
            assert path.read_bytes()==data,path
        else:
            with path.open("xb") as stream:
                stream.write(data)
    print(summary.to_string(index=False))


def audit_v1_one_mu(runs):
    """Reprice just two plans at the first decision; never clear markets or settle a year."""
    import json
    from dataclasses import asdict
    from scripts import run_national_household_portfolio as runner

    old, legacy = runner.old, runner.legacy
    src = BASE / "PRIVATE_DO_NOT_SHARE"
    choices = pd.read_csv(src / "repriced_choices.csv").query(
        "grid=='plus_one_mu' and budget=='booked' and grain_mu>0")
    target = choices[["seed", "actor_id"]].merge(pd.read_csv(src/"one_mu_household.csv"),
                                                 on=["seed", "actor_id"], validate="one_to_one")
    assert target.groupby("seed").size().to_dict() == {103: 13, 109: 21} and target.actor_id.nunique() == 24
    rates, workers, ratios, ref = runner.v1_household_inputs()
    _, economics, budget, inputs = legacy.load_base()
    share_path = runner.ROOT/"reference/initial_grain_shares.json"
    counts, metrics, fingerprints = [], [], []
    for seed, run in zip((103, 109), runs, strict=True):
        ep = run/"episode"
        config_file, trace_file = ep/"effective_config.json", ep/"year_01/actor_trace.jsonl"
        design, cfg = old.read(run/"design.json"), old.read(config_file)
        rule = cfg["runtime_fixed_rule"]
        assert (design["seed"], design["rule"], design["policy"], cfg["scale"]["n_enterprises"]) == (seed,"v1","base",60)
        assert rule["national_v1"] is True and cfg["experiment"]["seed"] == seed
        assert rule["grain_kg_mu_by_region"] == rates and rule["household_sales_ratio"] == ratios
        assert inputs == design["inputs"] and json.loads(json.dumps(ref)) == old.read(run/"reference_counts.json")
        assert old.digest(share_path) == design["share_sha256"]
        group = target[target.seed.eq(seed)].set_index("actor_id")
        with trace_file.open(encoding="utf-8") as stream:
            archived = [r for r in map(json.loads, stream) if r["actor_id"] in group.index]
        assert len(archived) == len(group) and {r["actor_id"] for r in archived} == set(group.index)
        model = legacy.build_model(cfg, economics, seed)
        model.household_initial_grain_shares = old.read(share_path)
        for f in model.initial_household_cohort:
            n, wages, proxy = workers[str(f.unique_id)]
            f.set_v1_labor(n, wages)
            f.worker_count_source = "region_size_proxy" if proxy else "NFPS_2010_workers"
        model.config["policy"]["government_q_project"].update(cfg["policy"]["government_q_project"])
        assert model.config == cfg and legacy.fingerprint(runner.parcels(model)) == design["initial_parcels_sha256"]
        legacy.configure_government(model.government, {}, seed, False)
        model.government._choose_action_rl = lambda state, action_count=3: 0
        old.apply_annual_exogenous_path(model, 0)
        old.advance_program(model, 0, budget)
        class AtDecision(Exception):
            pass
        with patch.object(model, "_run_fixed_rule_decisions", side_effect=AtDecision):
            try:
                model.step()
            except AtDecision:
                pass
            else:
                raise RuntimeError("first decision was not reached")
        actors = {str(f.unique_id): f for f in model.initial_household_cohort}
        pair_rows = []
        checked = 0
        for record in archived:
            assert record["year"] == 2011 and record["step"] == model.step_count == 1
            key = record["actor_id"]
            f = actors[key]
            prev = model.household_initial_grain_shares.get(key)
            area = f.get_operated_land_area()
            assert area >= 1 and np.isclose(area,group.at[key,"opening_mu"], rtol=0, atol=1e-9)
            options = [r for r in record["alternatives"] if r["mode"] == "outsource"]
            # Recompute the recorded all-cash and nearest nonzero crop budget before adding one mu.
            cash = next(r for r in options if r["action"] == "expand_cash")
            neighbor = min((r for r in options if r["expected_account"]["grain_area_mu"] > 0),
                           key=lambda r: abs(r["expected_account"]["grain_area_mu"] - 1))
            one_key = "maintain_grain" if area == 1 else "portfolio_probe_one_mu"
            shares = {"expand_cash":0., neighbor["action"]:neighbor["expected_account"]["grain_area_mu"]/area,
                      one_key:1/area}
            with patch.object(f, "household_previous_grain_share", prev, create=True), \
                 patch.object(f, "household_candidate_grain_shares", shares, create=True):
                modes = audit.organization_accounts(f, tuple(shares))
            opts = {o.account.action:o for o in modes if o.mode == "outsource"}
            for r in (cash, neighbor):
                o = opts[r["action"]]
                assert r["service_available"] == o.service_available and r["account_feasible"] == o.account.feasible
                for k, v in asdict(o.account).items():
                    saved = r["expected_account"][k]
                    if isinstance(v, (int,float)):
                        assert np.isclose(v, saved, atol=1e-7, rtol=1e-12), (seed, k)
                    else:
                        assert v == saved, (seed, k)
                checked += 1
            a, b = opts["expand_cash"], opts[one_key]
            assert np.allclose([a.account.area_mu,b.account.area_mu,b.account.grain_area_mu],[area,area,1],
                               rtol=0,atol=1e-9)
            def values(o):
                p = o.account
                return dict(days=p.family_labor_days+p.hired_labor_days, family=p.family_labor_days,
                            hired=p.hired_labor_days, hired_cost=p.hired_labor_cost_yuan,
                            sales=p.sales_yuan, cash_need=p.working_capital_required,
                            cash_available=p.working_capital_available, service_fee=p.service_cost_yuan,
                            service_mu=p.service_area_mu, net=p.net_income)
            x,y = values(a),values(b)
            row = dict(quote_a=a.service_available, quote_b=b.service_available,
                       feasible_a=a.service_available and a.account.feasible,
                       feasible_b=b.service_available and b.account.feasible,
                       reason_a=a.account.constraint_reason, reason_b=b.account.constraint_reason)
            if a.service_available and b.service_available:
                for k in x:
                    row.update({k+"_a":x[k], k+"_b":y[k], k+"_d":y[k]-x[k]})
            pair_rows.append(row)
        pairs = pd.DataFrame(pair_rows)
        quoted = pairs.loc[pairs.quote_a & pairs.quote_b]
        feasible = pairs.loc[pairs.feasible_a & pairs.feasible_b]
        counts.append(dict(seed=seed, n=len(group), checked=checked,
            quoted_a=int(pairs.quote_a.sum()), quoted_b=int(pairs.quote_b.sum()), both_quoted=len(quoted),
            feasible_a=int(pairs.feasible_a.sum()), feasible_b=int(pairs.feasible_b.sum()), both_feasible=len(feasible),
            old_hired_saved=-group.delta_hired_days.median(), old_net_gain=group.delta_net.median(),
            old_positive=int(group.delta_net.gt(0).sum())))
        for label, rows in (("both_quoted", quoted), ("both_feasible",feasible)):
            if rows.empty:
                continue
            saved = -rows.hired_d
            metrics.append(dict(seed=seed, population=label, n=len(rows), net_positive=int(rows.net_d.gt(0).sum()),
                hired_saved_24_to_26=int(saved.between(24,26).sum()), min_hired_saved=saved.min(),
                max_hired_saved=saved.max(), **rows.filter(regex="_(a|b|d)$").select_dtypes("number").median().to_dict()))
        fingerprints.append(f"- {run.name}: effective_config.json {old.digest(config_file)}; "
                            f"year_01/actor_trace.jsonl {old.digest(trace_file)}")
    out = runner.ROOT.parent/"national_v1_20261122"
    command = ".\\.venv\\Scripts\\python.exe -B -X utf8 -m scripts.audit_one_mu_hired_days --v1 " + " ".join(
        '"'+str(p.resolve().relative_to(ROOT.resolve())).replace("\\","/")+'"' for p in runs)
    description = ("# V1 首年同户一亩粮预算复核\n\n2011 年首年，P0，60 家组织。旧复核 booked / plus_one_mu "
        "换粮的同一批家庭，种子间有重叠，共 24 户。两方案固定首年决策时经营地，面积逐户对旧账；初始地块指纹与各次运行一致。"
        "在土地竞租前停止，只调用原家庭预算与委托服务报价函数；并非模拟再次选择、实际雇工、成交报价单或收获结算。\n\n"
        "counts 表列示原名单、复核的归档账户数、两边报价及可行数；old 为旧版同组中位数或计数。"
        "budgets 表按双边有报价、双边有报价且现金和雇工均可行分别汇总；没有报价不按免费服务计。"
        "a 为全经，b 为同面积换 1 亩粮，d 为各户 b 减 a 后的中位数，不是两列中位数之差。"
        "days 为总工日，family/hired 为家庭及雇工日，hired_cost 为雇工支出，sales 为按出售比例计的预计销售额，"
        "cash_need 为预算需付现金，cash_available 为季初可动用现金，service_fee/mu 为服务费/服务亩数，"
        "net 为含非农收入、预期支持和租金的家庭预计净收入（未扣风险惩罚）；金额单位元。"
        "约25天计数预定为少雇 24 至 26 天。不可行报价的净额只是预算算术。\n\n"
        "输入指纹（SHA256）：\n"+"\n".join(fingerprints)+"\n\n根目录复算：\n\n```powershell\n"+command+"\n```\n")
    for name, content in (("one_mu_v1_counts.csv",pd.DataFrame(counts).to_csv(index=False,lineterminator="\n")),
                          ("one_mu_v1_budgets.csv",pd.DataFrame(metrics).to_csv(index=False,lineterminator="\n")),
                          ("一亩换粮预算复核.md",description)):
        path = out/name
        data = content.encode("utf-8-sig" if name.endswith(".csv") else "utf-8")
        if path.exists():
            assert path.read_bytes() == data, f"existing aggregate differs: {name}"
        else:
            with path.open("xb") as stream:
                stream.write(data)
    print(pd.DataFrame(counts).to_string(index=False))
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__=="__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", nargs=2, type=Path, metavar=("RUN103","RUN109"))
    args = parser.parse_args()
    if args.v1:
        audit_v1_one_mu(args.v1)
    else:
        main()
