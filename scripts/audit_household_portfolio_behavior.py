"""Compare same-ID annual household crop areas before opening the policy comparison."""
import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "results/diagnostics"
OUT = ROOT / "national_household_portfolio"
V1_OUT = ROOT / "national_v1_20261122"


def v1_run(path, seed, rule, old):
    def read(file):
        return json.loads(file.read_text(encoding="utf-8"))
    receipt = read(path.parent / "audit.json")
    design = read(path.parent / "design.json")
    done = read(path / "COMPLETED.json")
    if (receipt["ok"] is not True or (receipt["seed"], receipt["rule"]) != (seed, rule)
            or (design["seed"], design["rule"], design["policy"]) != (seed, rule, "base")
            or (done["seed"], done["years"], done["training"]) != (seed, 3, False)):
        raise ValueError(f"Not an audited three-year {rule} base run for seed {seed}: {path}")
    cfg = read(path / "effective_config.json")
    if cfg["scale"]["n_enterprises"] != 60 or bool(cfg["runtime_fixed_rule"].get("national_v1")) != (rule == "v1"):
        raise ValueError(f"Not the 60-organization main comparison: {path}")
    if rule == "v1" and Path(receipt["baseline"]).resolve() != old.resolve():
        raise ValueError(f"Different paired old episode: {path}")
    version = [design["source_hashes"], design["runner_sha256"], design["inputs"], design["share_sha256"]]
    return sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()


def v1_figure(table, result):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import matplotlib as mpl

    with mpl.rc_context({"font.family": "Microsoft YaHei"}):
        fig = Figure(figsize=(8.6, 3.4), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 3, sharey=True)
        labels = ["旧" if r == "old" else "V1" for r in table.rule]
        for ax, key, limit, title in zip(axes, ("mixed_error", "quantile_mae", "change_error"),
                (.10, .15, .15), ("粮经兼种率", "粮食份额三个分位数", "同户份额调整幅度")):
            bars = ax.bar(range(len(table)), table[key] * 100,
                          color=["#83939f" if r == "旧" else "#176397" for r in labels])
            ax.axhline(limit * 100, color="#994824", linestyle="--", linewidth=1)
            ax.bar_label(bars, fmt="%.1f", fontsize=9)
            ax.set_xticks(range(len(table)), [f"{r}\n{s}" for r, s in zip(labels, table.seed)])
            ax.set_title(title, fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("相对调查的绝对误差（百分点）")
        axes[0].set_ylim(0, max(18, 115 * table[["mixed_error", "quantile_mae", "change_error"]].max().max()))
        fig.suptitle(f"{int(table.year.iloc[0])} 年家庭种植行为核验（虚线为预定误差上限）")
        fig.savefig(result.with_suffix(".png"), dpi=200)


def v1_policy_status(stage, runs=(), version=None):
    # This audit consumes only base episodes, never an independently evaluated policy branch.
    rows = [dict(rule="v1", version=version, seed=s, policy=p, behavior_gate=stage,
                 base_episode=str(path) if p == "P0" and path else None,
                 evaluation=("base_behavior_only" if path and p == "P0" else "not_evaluated_here"),
                 incremental_grain_t=None, incremental_h0_income_yuan=None, fiscal_cost_difference_yuan=None)
            for s, path in zip((103, 109), runs or (None, None))
            for p in ("P0", "P1", "P2", "P3", "P4", "transparent")]
    pd.DataFrame(rows).to_csv(V1_OUT/"policy_comparison_status.csv", index=False, encoding="utf-8-sig")


def preplant_funds(old):
    from scripts.audit_productivity_sources import CHFS

    file = CHFS/"chfs2011_hh_20191120_version14.dta"
    cols = "hhid b1001 b2001 b3001 b3002 b3005 b3030 b3033 d1101 d1105 k1101".split()
    hh = pd.read_stata(file, columns=cols, convert_categoricals=False).set_index("hhid", verify_integrity=True)
    farm = hh.loc[hh.b1001.eq(1)]
    complete = farm.b3001.isin([1, 2]) & farm.b3030.isin([1, 2])
    joint = farm[complete]
    either = joint.b3001.eq(1) | joint.b3030.eq(1)
    cash = farm.k1101.where(farm.k1101.ge(0))
    deposit = farm.d1105.where(farm.d1101.eq(1) & farm.d1105.ge(0)).mask(farm.d1101.eq(2), 0)
    def summary(values):
        valid = values.dropna()
        return f"{len(valid):,} 户；中位数 {valid.median():,.2f} 元；零值 {valid.eq(0).sum():,} 户"

    lines = ["# 播前资金：家庭流动资产、经营借款与资金缺口", "",
        "## 2011 年 CHFS：观测到的是既有借款，不是新增授信", "",
        f"原户表 {len(hh):,} 户；按 B1001=1 取上年从事农业的 {len(farm):,} 户。"
        f"全表 B1001 缺答 {hh.b1001.isna().sum()} 户，不列入农业户。以下为等户权样本汇总，并非全国加权比例。"
        "B3001 与 B3030 问调查时农业或工商业生产经营借款；其中兼营家庭不能把全部借款归给农业。", "",
        "| 观测项目 | 分子 / 有效分母（户）或金额 |", "|---|---:|",
        f"| 有经营银行贷款（B3001） | {farm.b3001.eq(1).sum()} / {farm.b3001.isin([1,2]).sum()} |",
        f"| 有其他渠道经营借款（B3030） | {farm.b3030.eq(1).sum()} / {farm.b3030.isin([1,2]).sum()} |",
        f"| 两项均答：任一渠道 / 两项均答 | {either.sum()} / {len(joint)} |",
        f"| 两项均答：两种渠道都有 / 两项均答 | {(joint.b3001.eq(1)&joint.b3030.eq(1)).sum()} / {len(joint)} |",
        f"| 仅从事农业（B2001=2），任一渠道 / 两项均答 | {(either & joint.b2001.eq(2)).sum()} / {joint.b2001.eq(2).sum()} |",
        f"| 银行借款户的经营贷款未还余额（B3005） | {summary(farm.b3005.where(farm.b3001.eq(1)&farm.b3005.ge(0)))} |",
        f"| 其他借款户的经营借款未还余额（B3033） | {summary(farm.b3033.where(farm.b3030.eq(1)&farm.b3033.ge(0)))} |",
        f"| 家庭持有现金（K1101） | {summary(cash)} |",
        f"| 家庭现金加活期存款（K1101、D1101、D1105） | {summary(cash+deposit)} |", "",
        "金额为调查当时的元。只有明确无活期账户者的活期余额计零；金额未知、负码和区间回答不填点值。"
        "负债余额只在确认借款且有有效金额的户内汇总，零余额与未借、未答分开。银行和其他借款可重叠。", "",
        f"明确无银行经营贷款的 {farm.b3001.eq(2).sum()} 户中，"
        + "、".join(f"{name} {int((farm.b3001.eq(2)&farm.b3002.eq(k)).sum())} 户" for k,name in
                     enumerate(('不需要','需要但未申请','申请被拒','曾借且已还清'),1))
        + "；其余原因缺答。没有贷款不等于没有需求，更不等于能按需借到。"
        "问卷没有同时给出基期播前尚可提用的额度、到账时间和用途限制。已有贷款可能已进入持有现金，再把贷款余额加入家庭流动资产会重复计数。", "",
        "## 固定 600 户：冻结资产与播前约束", ""]
    manifest = ROOT/"national_formal_revision_20261118/run_01/prepared_seed17_release1/preparation_manifest.json"
    cohort = Path(json.loads(manifest.read_text(encoding="utf-8"))["jobs"][8]["cohort"])
    frozen = pd.DataFrame(json.loads(cohort.read_text(encoding="utf-8"))["farmers"]).set_index("cohort_agent_id", verify_integrity=True)
    basis = pd.DataFrame(frozen.economic_basis_20261021.tolist(), index=frozen.index)
    available = (frozen.initial_working_capital_yuan-frozen.household_cash_reserve_yuan).clip(lower=0)
    if (len(frozen) != 600 or not basis.alpha.eq(1).all() or not basis.reserve_months.eq(3).all()
            or not frozen.initial_working_capital_yuan.eq(frozen.cash_plus_deposits_yuan).all()
            or not (basis.annual_reported_cash_candidate_yuan2024/4-frozen.household_cash_reserve_yuan).abs().lt(1e-7).all()
            or not (available-basis.working_funds_yuan2024).abs().lt(1e-7).all()):
        raise ValueError("Unexpected frozen cohort")
    lines += ["固定观察点 2010 年末存量用于 2011 年开局，按 2024 年不变价元报告。"
              "不从 CHFS 给任何模型户匹配贷款，不把两个调查的名义金额直接相比。每个初始家庭计一次，两个种子不是 1,200 户。",
              f"冻结流动资金：{summary(frozen.initial_working_capital_yuan)}；家庭生活留存：{summary(frozen.household_cash_reserve_yuan)}；"
              f"扣除留存后的可用资金：{summary(available)}。"
              "期初按现金加存款全额可动用（α=1），以三个月家庭现金消费为留存目标，可用额为两者之差与零的较大者，留存只扣一次。"
              f"冻结记录中，{basis.financial_imputed_fields.ne('').sum()}/600 户有存款分项插补标记，"
              f"{basis.cash_status.ne('reported_unchanged').sum()}/600 户的消费现金口径不是原样实报。本次不改既有插补。"
              "该冻结表没有可用的逐户未提款授信额度，不据此断言真实家庭没有负债。", "",
              "以下是冻结旧基础路径的 2011 年播前账，不是 V1 运行结果。固定该户原计划、投入报价、租约及生活留存，"
              "所需资金等于原全额计划报价加应付而未预留的租金。比较零新增借款与每户最多可即时借入 1,000 或 5,000 元两种有限上限，"
              "实际只借补缺所需额；两档均是 2024 年不变价假设，不是 CHFS 估出的额度。只计算资金缺口，不重新选方案或清算市场。", "",
              "| 种子 | 可借上限（元/户） | 播前账（户） | 仍有缺口（户） | 恰好补足缺口（户） | 假设新增短债合计（元） | 剩余缺口合计（元） |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, path in zip((103,109), old):
        v1_run(path, seed, "old", path)
        initial = pd.DataFrame(json.loads((path/"initial_households.json").read_text(encoding="utf-8"))).set_index("id", verify_integrity=True)
        pair = [("frozen_initial_liquidity","initial_working_capital_yuan"), ("frozen_household_reserve","household_cash_reserve_yuan")]
        if (set(initial.index)!=set(frozen.index) or any(not initial[a].eq(frozen.loc[initial.index,b]).all() for a,b in pair)
                or not (initial.working_capital-available.loc[initial.index]).abs().lt(1e-7).all()):
            raise ValueError("Initial household liquidity differs from job 8")
        with (path/"year_01/actors.jsonl").open(encoding="utf-8") as stream:
            a = pd.DataFrame([r["preproduction_cash_account"] for r in map(json.loads,stream) if r["id"] in initial.index]).set_index("operator_id",verify_integrity=True)
        fields = ["full_plan_required_yuan","ordinary_rent_due_yuan","initial_rent_due_yuan","rent_reserved_yuan","opening_available_yuan"]
        if (set(a.index)!=set(initial.index) or not a.step.eq(1).all() or not a.status.eq("settled").all()
                or a[fields].isna().any().any()):
            raise ValueError("Incomplete preplant accounts")
        need = a.full_plan_required_yuan + a.ordinary_rent_due_yuan + a.initial_rent_due_yuan - a.rent_reserved_yuan
        gap = (need-a.opening_available_yuan).clip(lower=0)
        for limit in (0,1000,5000):
            loan = gap.clip(upper=limit)
            remain = gap-loan
            lines.append(f"| {seed} | {limit:,} | {len(a)} | {remain.gt(1e-8).sum()} | "
                         f"{(gap.gt(1e-8)&remain.le(1e-8)).sum()} | {loan.sum():,.2f} | {remain.sum():,.2f} |")
    lines += ["", "借入时现金与短债等额增加，净资产不因提款增加；保留生活留存，未将未来销售、工资或补贴提前入账。"
              "此处只问既定计划能否筹足款项。已因缺钱而没有入选的方案不在全额计划账内；未推算借款利息、还款、实收或政策效果。"
              "因此，无新增可动用信贷是检验依靠自有资金能否完成计划的基准假设，不是由外部样本估计的零信贷事实。", "",
              "## 来源与复算", "", f"- CHFS：`{file.name}`，第 B1、B2、B3、D1、K1 节；同目录资料包 2011 年家庭金融调查问卷。"
              f"原始户表 SHA-256：`{sha256(file.read_bytes()).hexdigest()}`。",
              f"- 600 户：冻结 job 8 队列 `{cohort.name}`，SHA-256：`{sha256(cohort.read_bytes()).hexdigest()}`。"
              "两条旧路径的 initial_households.json 核对期初，year_01/actors.jsonl 的 preproduction_cash_account 核对播前计划。",
              "- 对应计算：agents/farmer_agent.py 的 balance_sheet_snapshot；core/fixed_rule_runtime.py 的 prepare_preproduction_cash。", "",
              "在开发项目根目录运行（只读本地授权数据，不运行模型）：", "", "```powershell",
              "python -B -X utf8 -m scripts.audit_household_portfolio_behavior --funds --old " +
              " ".join('"'+p.resolve().relative_to(ROOT.parents[1]).as_posix()+'"' for p in old), "```", ""]
    V1_OUT.mkdir(parents=True, exist_ok=True)
    (V1_OUT/"播前资金.md").write_text("\n".join(lines),encoding="utf-8")


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
    parser.add_argument("year", nargs="?", type=int, choices=[2011,2012])
    parser.add_argument("--old", nargs=2, required=True, type=Path, metavar=("SEED103","SEED109"))
    parser.add_argument("--new", nargs=2, type=Path, metavar=("SEED103","SEED109"))
    parser.add_argument("--v1", action="store_true", help="audit V1 base episodes; without --new, plot audited old controls only")
    parser.add_argument("--funds", action="store_true", help="summarize 2011 CHFS and frozen opening cash accounts only")
    args = parser.parse_args()
    if args.funds:
        if args.year or args.v1 or args.new:
            parser.error("--funds uses only --old SEED103 SEED109")
        preplant_funds(args.old)
        return
    if args.year is None:
        parser.error("year is required for the behavior audit")
    dest = V1_OUT if args.v1 else OUT
    candidate = "v1" if args.v1 else "portfolio"
    if args.v1:
        dest.mkdir(parents=True, exist_ok=True)
    survey = pd.read_csv(OUT / "reference/reference_households.csv").rename(
        columns={"grain_area":"grain","cash_area":"cash"})
    def actual(year):
        frame = survey[survey.year.eq(year)].set_index("actor_id",verify_integrity=True)[["grain","cash"]]
        return frame.dropna().loc[lambda a: (a>=0).all(axis=1)]
    y=args.year
    obs,prior=actual(y),actual(y-1)
    records=[]
    for seed,old,new in zip((103,109),args.old,args.new or (None,None)):
        for rule,path in (("old",old),(candidate,new)):
            if path is None:
                continue
            if args.v1:
                version = v1_run(path, seed, rule, old)
            elif new is not None:
                receipt=json.loads((path.parent/"audit.json").read_text(encoding="utf-8"))
                if not receipt["ok"] or receipt["seed"]!=seed or receipt["rule"]!=("new" if rule=="portfolio" else "old"):
                    raise ValueError(f"Unverified experiment: {path}")
            sim=areas(path,y).reindex(obs.index)
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
            if new is not None or args.v1:
                row["run"]=path.parent.name
            if args.v1:
                row.update(episode=str(path.resolve()), version=version)
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
    result=dest/(f"behavior_{y}.csv" if args.new else f"baseline_{y}.csv")
    table=pd.DataFrame(records)
    if args.v1:
        if table.groupby("rule").version.nunique().ne(1).any():
            raise ValueError("The two seeds do not share the same source, runner and input version")
        if args.new and y == 2012:
            dev = pd.read_csv(dest/"behavior_2011.csv")
            saved = json.loads((dest/"behavior_gate.json").read_text(encoding="utf-8"))
            cols = ["seed", "rule", "episode", "version"]
            if (not dev.year.eq(2011).all() or not dev[cols].equals(table[cols])
                    or any(sha256(path.read_bytes()).hexdigest() != saved["development_hashes"][path.name]
                           for path in (dest/"behavior_2011.csv", OUT/"PROTOCOL.md",
                                        OUT/"reference/reference_households.csv", Path(__file__)))):
                raise ValueError("Development and holdout must use the same old/V1 episodes and versions")
    table.to_csv(result,index=False,encoding="utf-8-sig")
    if args.v1:
        if args.new and y == 2011:
            development = {p.name:sha256(p.read_bytes()).hexdigest() for p in
                           (result, OUT/"PROTOCOL.md", OUT/"reference/reference_households.csv", Path(__file__))}
            (dest/"behavior_gate.json").write_text(json.dumps(dict(rule="v1", unlocked=False,
                status="holdout_pending", development_hashes=development), indent=2), encoding="utf-8")
            v1_policy_status("holdout_pending", args.new, version)
        elif not args.new and not (dest/"behavior_gate.json").exists():
            v1_policy_status("v1_behavior_not_verified")
        v1_figure(table, result)
    print(table.reindex(columns=["year","seed","rule","obs_n","obs_both","sim_both","mixed_error","quantile_mae","change_error","pass"]).to_string(index=False))
    if y==2012 and args.new:
        dev=pd.read_csv(dest/"behavior_2011.csv")
        assert dev[["seed","run","rule"]].equals(table[["seed","run","rule"]])
        checks=pd.concat([dev,table]).loc[lambda frame: frame.rule.eq(candidate)]
        assert len(checks)==4
        unlocked=checks["pass"].eq(True).all()
        gate=dict(unlocked=bool(unlocked), rule="PROTOCOL.md §5",
                  failed=checks.loc[~checks["pass"].eq(True),["year","seed"]].to_dict("records"),
                  hashes={f:sha256(((dest if f.startswith("behavior_") else OUT)/f).read_bytes()).hexdigest() for f in
                          ("PROTOCOL.md","behavior_2011.csv","behavior_2012.csv","reference/reference_households.csv")})
        if args.v1:
            gate.update(rule="v1", protocol=gate["rule"], version=checks.version.iloc[0],
                        development_hashes=saved["development_hashes"],
                        episodes=table[["seed","rule","episode"]].to_dict("records"))
            v1_policy_status("passed" if unlocked else "failed", args.new, version)
        (dest/"behavior_gate.json").write_text(json.dumps(gate,indent=2,ensure_ascii=False),encoding="utf-8")
        if not unlocked:
            from scripts.audit_national_succession_transmission import performance
            annual=[]
            for s,old,new in zip((103,109),args.old,args.new):
                for r,path in (("old",old),(candidate,new)):
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
            base.to_csv(dest/"baseline_accounts.csv",index=False,encoding="utf-8-sig")
            if args.v1:
                return
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


def audit_area_receipts(folder, crop_mode=None):
    """Read back private accounts. Only group totals and counts are exportable."""
    import hashlib
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import json
    root=Path(folder)
    priv=root/'PRIVATE_DO_NOT_SHARE'
    b=pd.read_csv(priv/'baseline.csv').set_index('actor')
    cases=pd.read_csv(root/'fingerprints.csv').case
    receipts=[];exits=[];rosters={}; hashes={}
    for name in cases:
        data=pd.read_csv(priv/(name+'_accounts.csv'))
        contracts=pd.read_csv(priv/(name+'_contracts.csv'))
        orders=pd.read_csv(priv/(name+'_orders.csv'))
        gov=json.loads((priv/(name+'_government.json')).read_bytes())
        ros=pd.read_csv(priv/(name+'_organizations.csv'))
        rosters[name]=ros
        hashes[name]=[hashlib.sha256((priv/(name+s)).read_bytes()).hexdigest() for s in
                      ('_accounts.csv','_contracts.csv','_orders.csv','_government.json','_organizations.csv')]
        assert (contracts.start==contracts.end).all() and not data.duplicated(['year','id']).any()
        last=None
        for year,g in data.groupby('year'):
            c=contracts[contracts.start==year];o=orders[orders.year==year]
            assert (c.ask<=c.rent+1e-7).all() and (c.rent<=c.bid+1e-7).all()
            assert c.groupby('region').rent.nunique().max()==1
            marginal=(c.groupby('region').ask.max()+c.groupby('region').bid.min())/2
            assert np.allclose(c.groupby('region').rent.first(),marginal)
            obs=g.set_index('id')
            assert np.allclose(obs.rented_out,c.groupby('seller').area.sum().reindex(obs.index,fill_value=0))
            assert np.allclose(obs.rented_in,c.groupby('buyer').area.sum().reindex(obs.index,fill_value=0))
            assert (obs.rented_out<=obs.rights+1e-6).all()
            assert np.allclose(obs.operated,obs.base_area-obs.rented_out+obs.rented_in)
            assert np.allclose(obs.operated,obs.idle+obs.cultivated)
            assert np.allclose(obs.net,obs.nonfarm+obs.surplus+obs.paid)
            if last is not None:
                assert np.allclose(last.closing.reindex(obs.index),obs.opening)
            s=gov[year-2011]
            assert not s['training'] and s['action']=='P0_rights_cultivated'
            if year>2011: assert s['state']==gov[year-2012]['next_state']
            assert abs(sum(x['grain_kg'] for x in s['next_state'].values())-g.grain_kg.sum())<1e-5
            assert abs(s['reward']-sum(st['grain_kg']-s['state'][z]['grain_kg'] for z,st in s['next_state'].items()))<1e-5
            oq=o.merge(obs.kind.rename('actor_kind'),left_on='id',right_index=True)
            for k,r in g.groupby('kind'):
                oo=oq[oq.actor_kind==k]; cc=c[c.buyer.isin(r.id)]
                receipts.append(dict(case=name,year=year,kind=k,sellers=(oo.side=='sell').sum(),
                    buyers=(oo.side=='buy').sum(),offered_mu=r.offer.sum(),demanded_mu=r.demand.sum(),
                    highest_bid=oo.loc[oo.side=='buy','price'].max(),contracts=len(cc),acquired_mu=cc.area.sum(),
                    paid_rent=cc.area.dot(cc.rent),positive_harvest=(r.grain_kg>1e-7).sum(),
                    household_or_org_sales=r.sales.sum(),operating_surplus=r.surplus.sum(),subsidy=r.paid.sum(),
                    positive_funds_partial=((r.working_funds>0)&(r.cash_shadow>1e-7)&(r.cultivated>1e-7)&(r.cultivated<r.base_area-1e-7)).sum(),
                    partial_lease_and_cultivate=((r.rented_out>1e-7)&(r.cultivated>1e-7)).sum(),
                    net_income=r.net.sum(),exited=r.exited.sum()))
            fam=obs[obs.kind=='household']
            was=(b.sow_food+b.sow_cash>0 if last is None else last.cultivated>1e-7)
            drop=fam[(fam.cultivated<=1e-7)&was.reindex(fam.index)]
            moved=c[c.seller.isin(drop.index)]
            exits.append(dict(case=name,year=year,stopped_cultivating=len(drop),opening_endowment=drop.base_area.sum(),
                to_households=moved.loc[moved.buyer.isin(fam.index),'area'].sum(),
                to_organizations=moved.loc[~moved.buyer.isin(fam.index),'area'].sum(),
                retained_unused=drop.operated.sum(),expired_prior_leases=(0 if last is None else last.loc[drop.index,'rented_in'].sum())))
            assert np.isclose(drop.base_area.sum()+drop.rented_in.sum(),drop.operated.sum()+moved.area.sum())
            last=obs
    if crop_mode!='baseline':
        assert hashes['103_both_60_1.0']==hashes['103_repeat_60_1.0']
    for seed in (103,109):
        if crop_mode:
            name=f'{seed}_both_60_1.0'
            old=ROOT/'national_area_market_01/final/PRIVATE_DO_NOT_SHARE'
            for j,s in enumerate(('_accounts.csv','_contracts.csv','_orders.csv','_government.json','_organizations.csv')):
                if crop_mode=='baseline' or j==4:
                    assert hashes[name][j]==hashlib.sha256((old/(name+s)).read_bytes()).hexdigest()
            continue
        def roster(s): return rosters[f'{seed}_'+s]
        full=roster('both_60_1.0')
        for structure in ('family','company','coop'):
            assert full.drop(columns='activated').equals(roster(structure+'_60_1.0').drop(columns='activated'))
        def totals(r): return r.groupby(['kind','region'])[['capital','capacity','fixed','management']].sum()
        for n in (30,90): assert np.allclose(totals(full),totals(roster(f'both_{n}_1.0')))
        for mult in (.5,1.5):
            part=roster(f'both_60_{mult}')
            assert np.allclose(full.capital,part.capital) and np.allclose(full.capacity*mult,part.capacity)
    pd.DataFrame(receipts).to_csv(root/'market_and_income.csv',index=False)
    pd.DataFrame(exits).to_csv(root/'succession.csv',index=False)
    from scripts.prepare_household_portfolio_reference import RAW,MANIFEST
    from core.extract_microdata_features import REGION_MAPPING
    region={p:r for r,ps in REGION_MAPPING.items() for p in ps}
    src=pd.read_csv(RAW,usecols=['year','hhid','id','smgen','sow_food','sow_cash','area_e','prod_food'],
                    dtype={'hhid':'string','id':'string'})
    def wave(y):
        v=src[src.year==y].dropna(subset=['hhid','id'])
        return v[~v.hhid.duplicated(False)]
    p=wave(2009).merge(wave(2010),on=['hhid','id'],validate='1:1',suffixes=('09','10'))
    shares=[p['sow_food'+y]/(p['sow_food'+y]+p['sow_cash'+y]).where(lambda v:v>0) for y in ['09','10']]
    delta=(shares[1]-shares[0]).abs().dropna()
    pd.DataFrame([dict(years='2009-2010',matched=len(p),positive_both=len(delta),
        median_absolute_change=delta.median(),mean_absolute_change=delta.mean(),
        identified_adjustment_cost=False)]).to_csv(root/'historical_shares.csv',index=False)
    job=json.loads(MANIFEST.read_bytes())['jobs'][8]
    cohort=pd.DataFrame(json.loads(Path(job['cohort']).read_bytes())['farmers'])
    key=src.iloc[cohort.source_row.to_numpy()-2][['hhid','id']].copy()
    key['actor']=cohort.cohort_agent_id.to_numpy()
    key=key.merge(wave(2010)[['hhid','id']],on=['hhid','id'],validate='m:1')
    regional=[]
    for seed in (103,109):
        structure='both' if crop_mode else 'family'
        sim=pd.read_csv(priv/f'{seed}_{structure}_60_1.0_accounts.csv')
        for yr in range(2011,2016):
            compare=key.merge(wave(yr),on=['hhid','id'],validate='1:1').merge(
                sim[sim.year==yr],left_on='actor',right_on='id',validate='1:1')
            for z,g in compare.groupby(compare.smgen.map(region)):
                y=g[g.prod_food.notna()&g.sow_food.notna()]
                a=g[g.area_e.notna()]
                regional.append(dict(seed=seed,year=yr,region=z,production_n=len(y),land_n=len(a),
                            observed_grain_kg=y.prod_food.sum(),model_grain_kg=y.grain_kg.sum(),
                            observed_grain_mu=y.sow_food.sum(),model_grain_mu=y.grain.sum(),
                            observed_operated_mu=a.area_e.sum(),model_operated_mu=a.operated.sum()))
    pd.DataFrame(regional).to_csv(root/'regional_matched_validation.csv',index=False)
    print(len(cases),"five-year paths verified; repeated accounts/orders/contracts/cohort/government identical")


def crop_parameter_sources(parent):
    from scripts.prepare_household_portfolio_reference import RAW
    from scripts.run_national_household_portfolio import v1_household_inputs,legacy
    rates,workers,ratios,refs=v1_household_inputs()
    out=Path(parent)/'run_01'
    dictionary=pd.read_excel(RAW.parent/'household variables2009-2015.xlsx',header=None).set_index(0)
    names=['inc_cash_'+str(i) for i in range(1,11)]
    d=pd.read_csv(RAW,usecols=['year','sow_cash']+names)
    rows=[]
    for year,g in d.groupby('year'):
        for col in names:
            rows.append(dict(year=year,field=col,question=dictionary.loc[col,1],rows=len(g),
                             valid=g[col].ge(0).sum(),positive=g[col].gt(0).sum(),
                             positive_with_group_area=(g[col].gt(0)&g.sow_cash.gt(0)).sum(),
                             complete_ten_with_area=(g[names].ge(0).all(axis=1)&g.sow_cash.gt(0)).sum()))
    pd.DataFrame(rows).to_csv(out/'nfps_income_coverage.csv',index=False)
    bb=pd.read_csv(ROOT.parents[1]/'results/national_crop_budget_basis_20261118/run_04/real2024_revision/record_selection_money2024.csv')
    detail=[]
    for crop,g in bb[bb.eligible].groupby('crop'):
        for col in ['sales_amount','income','seed','fertilizer','pesticide','machinery','irrigation','hired_cost','labor','hired_days']:
            obs=g[g[col].ge(0)&g.area.gt(0)]
            money=col not in ['labor','hired_days']
            detail.append(dict(crop=crop,field=col,n=len(obs),denominator_mu=obs.area.sum(),
                unit='2024 yuan/sown mu' if money else 'days/sown mu',
                value=(obs[col]*(obs.to_2024 if money else 1)).sum()/obs.area.sum()))
    pd.DataFrame(detail).to_csv(out/'crop_budget_sources.csv',index=False)
    cfg,*_=legacy.load_base()
    req=cfg['production']['cost_structure']['input_requirements']
    params=[('conditional crop share','estimated','2009-2010 NFPS non-cohort transitions; positive next-year sowing','probability and share'),
            ('mixed share grid',5,'equal-width interior bins; target is conditional sample mean','bins'),
            ('grain price',cfg['environment']['market']['base_prices']['grain']*2,'baseline price converted from jin','yuan/kg'),
            ('grain sold fraction',ratios['grain'],'rice sales quantity over output, comparable budget records','fraction'),
            ('cash crop revenue',json.loads((Path(parent)/'PRIVATE_DO_NOT_SHARE/baseline/inputs.json').read_bytes())['cash_sale'],'vegetable and fruit sale amounts / sown area; no second sales discount','2024 yuan/sown mu'),
            ('hired wage',cfg['labor_market']['base_agricultural_wage'],'unchanged baseline','yuan/day'),
            ('available labor','120 / 180 / 176','NFPS laborers x CFPS group proxy: paid / no paid / unknown','days/worker/year'),
            ('P0',cfg['fiscal_system']['p3_support']['preserved_projects'][0]['rate'],'verified cultivated physical rights; paid after harvest','yuan/mu'),
            ('budget',cfg['fiscal_system']['p3_support']['base_budget_yuan'],'fixed annual cap','yuan'),
            ('intensity','household specific','2010 annual two-crop sowings / year-end area; regional median if not positive','sown mu/physical mu')]
    params.extend((region+'_grain',v,'2010 NFPS regional median','kg/sown mu') for region,v in rates.items())
    for key,unit in [('labor_days','days/sown mu'),('other_cost_per_mu','2024 yuan/sown mu')]:
        params.extend((crop+'_'+key,req[key][crop],'effective baseline crop budget; no second machinery charge',unit) for crop in ['grain','cash_crop'])
    pd.DataFrame(params,columns=['parameter','value','basis','unit']).to_csv(out/'parameter_provenance.csv',index=False)


def audit_crop_mix(parent):
    import numpy as np
    from scripts.prepare_household_portfolio_reference import RAW, MANIFEST
    from scripts.run_national_household_portfolio import v1_household_inputs, legacy
    from core.fixed_rule_runtime import continuous_farm_plan
    from types import SimpleNamespace
    parent=Path(parent)
    out=parent/'run_01'; out.mkdir(exist_ok=True)
    crop_parameter_sources(parent)
    private=parent/'PRIVATE_DO_NOT_SHARE'
    rates,workers,ratios,refs=v1_household_inputs()
    cfg,*_=legacy.load_base()
    req=cfg['production']['cost_structure']['input_requirements']
    K,L=([req[key][crop] for crop in ('grain','cash_crop')] for key in ('other_cost_per_mu','labor_days'))
    wage=cfg['labor_market']['base_agricultural_wage']
    rate=cfg['fiscal_system']['p3_support']['preserved_projects'][0]['rate']
    cols=['year','hhid','id','sow_food','sow_cash','area_e','prod_food','sow_total']
    src=pd.read_csv(RAW,usecols=cols,dtype={'hhid':'string','id':'string'})
    cohort=pd.DataFrame(json.loads(Path(json.loads(MANIFEST.read_bytes())['jobs'][8]['cohort']).read_bytes())['farmers'])
    keys=src.iloc[cohort.source_row.to_numpy()-2][['hhid','id']].copy()
    keys['actor']=cohort.cohort_agent_id.to_numpy()
    unique=src.dropna(subset=['hhid','id'])
    unique=unique[~unique.duplicated(['year','hhid'],keep=False)]
    keys=keys.merge(unique[unique.year.eq(2010)][['hhid','id']],validate='m:1')
    survey=unique.merge(keys,on=['hhid','id'],validate='m:1').dropna(subset=['sow_food','sow_cash'])
    def share(g,c): return g/(g+c).where(g+c>1e-7)
    comp=[]; dist=[]; production=[]; income=[]; market=[]; budgets=[]; changes=[]
    store={}
    for version in ('baseline','candidate'):
        root=private/version; audit_area_receipts(root,version)
        pi=root/'PRIVATE_DO_NOT_SHARE'
        inputs=json.loads((root/'inputs.json').read_bytes())
        base=pd.read_csv(pi/'baseline.csv').set_index('actor')
        market.append(pd.read_csv(root/'market_and_income.csv').assign(version=version))
        for seed in (103,109):
            name=f'{seed}_both_60_1.0'
            a=pd.read_csv(pi/(name+'_accounts.csv')).query("kind=='household'").copy()
            assert len(a)==3000 and not a.duplicated(['id','year']).any()
            a['actor']=a.id
            a=a.join(base[['region','intensity','region_land','held_reserve','sow_food','sow_cash']].rename(
                columns=dict(region='macro',sow_food='initial_g',sow_cash='initial_c')),on='actor')
            a['capacity']=[workers[str(k)][0]*{True:120,False:180,None:176}[workers[str(k)][1]] for k in a.id]
            a['family_days']=a.grain*L[0]+a.crop*L[1]-a.hired_days
            a['gsale']=a.macro.map(rates)*cfg['environment']['market']['base_prices']['grain']*2*ratios['grain']
            cs=inputs['cash_sale']
            weather={}
            for yr,z in set(zip(a.year,a.region)):
                rng=np.random.default_rng([seed,int(yr),int(z)])
                u=rng.random(); weather[yr,z]=1-rng.uniform(.05,.10) if u<.05 else (1+rng.uniform(.0275,.055) if u<.13 else 1.)
            shock=np.array([weather[y,z] for y,z in zip(a.year,a.region)])
            assert np.allclose(a.grain_kg,a.grain*a.macro.map(rates)*shock)
            assert np.allclose(a.sales,(a.grain*a.gsale+a.crop*cs)*shock)
            assert np.allclose(a.grain*K[0]+a.crop*K[1]+a.hired_days*wage,a.input_hire_cost)
            assert (a.family_days>=-1e-6).all() and (a.family_days<=a.capacity+1e-6).all()
            assert (a.input_hire_cost<=(a.opening-a.held_reserve-a.rent_paid).clip(lower=0)+1e-5).all()
            assert np.allclose(a.cultivated*a.intensity,a.grain+a.crop)
            assert np.allclose(a.closing,a.opening+a.net-a.consumption)
            if version=='candidate':
                assert np.allclose(a.weather_factor,shock)
                for k,g in a.sort_values('year').groupby('id'):
                    h=share(g.grain,g.crop).shift().to_numpy()
                    h[0]=share(g.initial_g,g.initial_c).iloc[0]
                    assert np.allclose(h,g.prior_share,equal_nan=True)
                planted=a.grain+a.crop>1e-7
                hit=planted&a.target_share.notna()
                assert np.allclose(a.grain[hit]/(a.grain+a.crop)[hit],a.target_share[hit])
            else:
                for r in a.itertuples():
                    agent=SimpleNamespace(sowing_intensity=r.intensity,family_days=r.capacity,area_limit=r.region_land)
                    funds=max(0,r.opening-r.held_reserve-r.rent_paid)
                    args=(r.gsale,cs,*K,*L,wage,rate)
                    prior=r.initial_g/(r.initial_g+r.initial_c) if r.initial_g+r.initial_c>0 else None
                    for option,t in [('free',None),('grain',1.),('cash',0.),('mix',prior if prior and 0<prior<1 else .5),('stop',None)]:
                        agent.crop_mix_target=t
                        p=continuous_farm_plan(agent,r.operated if option!='stop' else 0.,r.rights-r.rented_out,funds,0.,args,False)
                        g,c,h=p['grain'],p['crop'],p['hired']
                        if option=='free': assert np.isclose(g,r.grain) and np.isclose(c,r.crop)
                        budgets.append(dict(id=r.id,seed=seed,year=r.year,option=option,target=t,
                            operated=r.operated,grain=g,crop=c,grain_unit_sales=r.gsale,cash_unit_sales=cs,
                            expected_sales=g*r.gsale+c*cs,inputs=g*K[0]+c*K[1],family_days=g*L[0]+c*L[1]-h,
                            hire_cost=h*wage,hired_days=h,rent_in=r.rent_paid,rent_out=r.rent_received,
                            nonfarm=r.nonfarm,working_funds=funds,family_capacity=r.capacity,
                            preplant_cost=p['cost'],support=rate*p['eligible'],expected_net=p['expected']+r.rent_received-r.rent_paid+r.nonfarm,
                            cash_shadow=p['cash_shadow'],land_shadow=p['land_value'],labor_shadow=p['labor_shadow']))
            store[version,seed]=a.set_index(['id','year'])
            prev=survey[survey.year.eq(2010)].set_index('actor')
            prev=pd.DataFrame(dict(obs=share(prev.sow_food,prev.sow_cash),sim=share(prev.sow_food,prev.sow_cash)))
            checks=pd.read_csv(root/'behavior.csv').query('case==@name').set_index('year')
            for yr,group in a.groupby('year'):
                v=survey[survey.year.eq(yr)].merge(group,on='actor',validate='1:1').set_index('actor')
                cur=pd.DataFrame(dict(obs=share(v.sow_food,v.sow_cash),sim=share(v.grain,v.crop)))
                joint=cur.join(prev,lsuffix='1',rsuffix='0',how='inner').dropna()
                delta=(joint.obs1-joint.obs0).abs().mean(); simulated=(joint.sim1-joint.sim0).abs().mean()
                z=checks.loc[yr].to_dict()
                assert len(v)==z['matched']
                comp.append(dict(**z,version=version,seed=seed,year=yr,adjustment_n=len(joint),survey_change=delta,
                                 sim_change=simulated,adjustment_error=abs(delta-simulated)))
                prev=cur
                for who,g,c,area in [('survey',v.sow_food,v.sow_cash,v.area_e),('model',v.grain,v.crop,v.operated)]:
                    s=share(g,c); n=len(v)
                    dist.append(dict(version=version,seed=seed,year=yr,source=who,matched=n,planted=s.count(),
                        participation=s.count()/n,positive_grain=(g>1e-7).mean(),mixed=((g>1e-7)&(c>1e-7)).sum(),
                        grain_only=((g>1e-7)&(c<=1e-7)).sum(),cash_only=((c>1e-7)&(g<=1e-7)).sum(),
                        no_crops=s.isna().sum(),grain_mu=g.sum(),cash_mu=c.sum(),operated_mu=area.sum(),
                        operated_p25=area.quantile(.25),operated_median=area.median(),operated_p75=area.quantile(.75),
                        share_p25=s.quantile(.25),share_median=s.median(),share_p75=s.quantile(.75),
                        intensity_median=((g+c)/area.where(area>0)).median()))
                production.append(dict(version=version,seed=seed,year=yr,matched=len(v),
                    observed_kg=v.prod_food.sum(min_count=1),output_reporting=v.prod_food.count(),
                    simulated_kg_on_reporting=v.loc[v.prod_food.notna(),'grain_kg'].sum(),simulated_kg=v.grain_kg.sum(),
                    model_kg_per_mu=v.grain_kg.sum()/v.grain.sum(),survey_kg_per_mu=z['survey_kg_per_mu']))
                income.append(dict(version=version,seed=seed,year=yr,matched=len(v),
                    **v[['net','nonfarm','paid','surplus','sales','rent_received','rent_paid','input_hire_cost','hired_days','family_days','rented_in','rented_out','idle']].sum().to_dict(),
                    planning_cash_binding=(v.cash_shadow>1e-7).mean(),family_labor_exhausted=((v.family_days>=v.capacity-1e-7)&v.capacity.gt(0)).mean()))
    cmp=pd.DataFrame(comp)
    old=cmp[cmp.version=='baseline'].set_index(['seed','year'])
    cmp['not_both_worse']=[not(r.mixed_error>old.loc[(r.seed,r.year),'mixed_error'] and r.share_mae>old.loc[(r.seed,r.year),'share_mae']) for r in cmp.itertuples()]
    cmp['all_gates']=cmp.passed & cmp.not_both_worse & (cmp.adjustment_error<=.15)
    cmp.to_csv(out/'behavior_before_after.csv',index=False)
    for seed in (103,109):
        a,b=store['baseline',seed],store['candidate',seed]
        for yr in range(2011,2016):
            x,y=(z.xs(yr,level='year') for z in (a,b))
            X=x.grain+x.crop; Y=y.grain+y.crop
            both=X.gt(1e-7)&Y.gt(1e-7); sx=x.grain[both]/X[both]; sy=y.grain[both]/Y[both]
            entry=y.grain[~both].sum()-x.grain[~both].sum()
            shares=((sy-sx)*(X[both]+Y[both])/2).sum()
            sizes=((Y[both]-X[both])*(sx+sy)/2).sum()
            change=y.grain.sum()-x.grain.sum()
            assert np.isclose(change,entry+shares+sizes)
            changes.append(dict(seed=seed,year=yr,total_change=change,entry_minus_exit=entry,same_household_share=shares,
                same_household_total_sowing=sizes,grain_positive_both=((x.grain>1e-7)&(y.grain>1e-7)).sum(),
                higher_grain_lower_cash=((y.grain>x.grain+1e-7)&(y.crop<x.crop-1e-7)).sum(),
                grain_on_leased_mu_lower=np.maximum(0,y.grain-(y.operated-y.rented_in)*y.intensity).sum(),
                grain_on_leased_mu_upper=np.minimum(y.grain,y.rented_in*y.intensity).sum()))
    pd.DataFrame(changes).to_csv(out/'grain_area_decomposition.csv',index=False)
    budget=pd.DataFrame(budgets)
    budget.to_csv(private/'alternative_accounts.csv',index=False)
    budget.drop(columns=['id']).groupby(['seed','year','option']).mean().to_csv(out/'income_decomposition.csv')
    pd.DataFrame(income).to_csv(out/'household_behavior_summary.csv',index=False)
    pd.DataFrame(production).to_csv(out/'production_summary.csv',index=False)
    pd.DataFrame(dist).to_csv(out/'crop_mix_distribution.csv',index=False)
    pd.concat(market).to_csv(out/'market_account.csv',index=False)
    t=pd.read_csv(private/'candidate/transition_estimates.csv')
    t['next']=t.category.map(lambda c:'cash' if c==0 else ('grain' if c==6 else 'mixed'))
    t=t.groupby(['prior','next'],as_index=False).agg(n=('n','first'),count=('count','sum'))
    t['p']=t['count']/t.n; t['cluster_se']=np.sqrt(t.p*(1-t.p)/(t.n-1))
    t.to_csv(out/'transition_estimates.csv',index=False)
    d=cmp[(cmp.version=='candidate') & (cmp.seed==103)]
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(12,3.6))
    for axis,obs,sim,label in zip(ax,['survey_mixed','survey_grain','survey_share_median'],['sim_mixed','sim_grain','sim_share_median'],['Mixed households','Grain sown (mu)','Median grain share']):
        axis.plot(d.year,d[obs],label='Survey',marker='o')
        for ver in ['baseline','candidate']:
            w=cmp[(cmp.version==ver)&(cmp.seed==103)]
            axis.plot(w.year,w[sim],label=ver,marker='.')
        axis.set(title=label,xticks=d.year)
    ax[0].legend();fig.tight_layout()
    (out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/behavior.png',dpi=150);plt.close(fig)


def audit_soft_response(folder):
    import hashlib
    import matplotlib.pyplot as plt
    import numpy as np
    from scripts.prepare_nfps_v3 import candidate_cohort,clean_area,RAW,MANIFEST
    from scripts.run_national_household_portfolio import v1_household_inputs
    root=Path(folder);out=root/'run_01';pr=root/'PRIVATE_DO_NOT_SHARE'
    old=pd.DataFrame(json.loads(Path(json.loads(MANIFEST.read_bytes())['jobs'][8]['cohort']).read_bytes())['farmers'])
    new=candidate_cohort(); prices,_,ratios,_=v1_household_inputs()
    raw=pd.read_csv(RAW,usecols='id hhid year sow_food sow_cash sow_total sow_oother area_e area_b area_de area_de2 area_in area_in1'.split(),dtype={'id':str,'hhid':str})
    clean_area(raw)
    params=pd.read_csv(ROOT/'national_crop_mix_repair_20261123/run_01/parameter_provenance.csv').set_index('parameter').value
    kg,kc,lg,lc,w,rate=[float(params[k]) for k in ['grain_other_cost_per_mu','cash_crop_other_cost_per_mu','grain_labor_days','cash_crop_labor_days','hired wage','P0']]
    unique=raw[~raw.duplicated(['year','hhid'],keep=False)].dropna(subset=['id','hhid'])
    ids2010=unique[unique.year.eq(2010)][['id','hhid']]
    configs={'A':pr/'A','Bmid':pr/'B','Bhigh':pr/'Bhigh','C':pr/'equal_consumption/C',
             **{v:pr/'equal_consumption'/f for v,f in [('Dmid','D'),('Dlow','Dlow'),('Dhigh','Dhigh')]}}
    scores={}
    for m in ['Dlow','Dmid','Dhigh']:
        df=pd.read_csv(configs[m]/'behavior.csv').query("year<=2012 and not case.str.contains('repeat')",engine='python')
        assert len(df)==4
        scores[m]=(df.mixed_error+df.share_mae).mean()
    chosen=min(scores,key=scores.get)
    pd.DataFrame([dict(mode=m,loss=v,chosen=m==chosen) for m,v in scores.items()]).to_csv(out/'calibration.csv',index=False)
    if chosen=='Dlow':configs['Blow']=pr/'Blow'
    configs.update(Bmob=pr/'mobility/B',Dmob=pr/'mobility/D')
    formal={'A':'A','B':'Bmob','C':'C','D':'Dmob'}
    def hashfile(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    from scripts.prepare_nfps_v3 import WEIGHTS,ROOT as project
    job=json.loads(MANIFEST.read_bytes())['jobs'][8]
    sources=[RAW, RAW.parent/'2009-2015individual.csv', RAW.parent/'household variables2009-2015.xlsx',
             RAW.parent/'2009-2017农户调查问卷.txt', WEIGHTS, MANIFEST, Path(job['cohort']),
             project/'config/experiments/national_real2024_cpi_20260909.csv',
             project/'results/national_crop_budget_basis_20261118/run_04/real2024_revision/record_selection_money2024.csv']
    fingerprints={p.name:hashfile(p) for p in sources}
    assert fingerprints[sources[6].name]==job['cohort_sha256']
    spec=json.loads((configs['Dmob']/'inputs.json').read_bytes())
    assert fingerprints[RAW.name]==spec['source_sha256'] and fingerprints[sources[-1].name]==spec['references']['budget_sha256']
    pd.Series(fingerprints,name='sha256').rename_axis('source').to_csv(out/'source_manifest.csv')
    provenance=pd.read_csv(ROOT/'national_crop_mix_repair_20261123/run_01/parameter_provenance.csv')
    provenance=provenance[~provenance.parameter.isin(['conditional crop share','mixed share grid'])]
    penalty_rows=[('reference share','conditional draw','NFPS prior status, share and region; past transitions through min(year-1,2012)','fraction'),
                  ('adjustment anchor',pd.read_csv(configs['Dmid']/'fit.csv').penalty.iloc[0],'median of four regional differences in sale less nonlabor cost; not a measured cost','2024 yuan/sown mu'),
                  ('adjustment multipliers','0.5,1,1.5','structural sensitivities; 2011-2012 calibration only','ratio'),
                  ('household adjustment multiplier','1.5 - 0.5 * switch proxy','2009-2010 observed status switch, otherwise regional switch frequency; no cash outflow','ratio')]
    provenance=pd.concat([provenance,pd.DataFrame(penalty_rows,columns=provenance.columns)])
    provenance['basis_type']='structural assumption'
    calculated=['grain sold fraction','cash crop revenue','intensity','available labor','adjustment anchor',
                'grain_labor_days','cash_crop_labor_days','grain_other_cost_per_mu','cash_crop_other_cost_per_mu']
    provenance.loc[provenance.parameter.isin(calculated)|provenance.parameter.str.endswith('_grain'),'basis_type']='calculated from source, with proxies'
    provenance.loc[provenance.parameter.eq('reference share'),'basis_type']='empirical conditional distribution'
    provenance.loc[provenance.parameter.eq('household adjustment multiplier'),'basis_type']='assumption with observed switch or regional proxy'
    provenance.to_csv(out/'parameter_provenance.csv',index=False)
    summary=[]; annual=[];receipts=[];factor=[];tables={};sens=[];fields=[];bridge=[]
    ref=ROOT/'national_crop_mix_data_comparability_20261123/PRIVATE_DO_NOT_SHARE/v2/PRIVATE_DO_NOT_SHARE'
    for m,path in configs.items():
        roster=new if m.startswith(('C','D')) else old
        ids=raw.iloc[roster.source_row.to_numpy()-2][['id','hhid']].assign(actor=roster.cohort_agent_id.to_numpy()).merge(ids2010,validate='m:1')
        survey=unique.merge(ids,on=['id','hhid'],validate='m:1')
        base=pd.read_csv(path/'PRIVATE_DO_NOT_SHARE/baseline.csv').set_index('actor')
        spec=json.loads((path/'inputs.json').read_bytes())
        if m.startswith(('C','D')):
            assert np.allclose(base.consumption,roster.economic_basis_20261021.str['annual_reported_cash_candidate_yuan2024'])
            assert np.allclose(base.area_e,raw.iloc[roster.source_row.to_numpy()-2].area_e)
            # Correct the provenance-only identifier in receipts produced before this metadata fix.
            sig=hashlib.sha256(roster.to_json(orient='records',force_ascii=False).encode()).hexdigest()
            if spec['cohort_sha256']!=sig:
                assert spec['cohort_sha256']==json.loads(MANIFEST.read_bytes())['jobs'][8]['cohort_sha256']
                spec['cohort_sha256']=sig
                (path/'inputs.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf8')
        audit_area_receipts(path,'candidate')
        private=path/'PRIVATE_DO_NOT_SHARE'
        for f in private.glob('*.csv'):
            if m=='A': assert hashfile(f)==hashfile(ref/f.name)
        for suffix in ('accounts.csv','organizations.csv','government.json','contracts.csv','orders.csv'):
            a=private/f'103_both_60_1.0_{suffix}'; repeat=private/f'103_repeat_60_1.0_{suffix}'
            assert hashfile(a)==hashfile(repeat)
            if m=='A':assert all(hashfile(f)==hashfile(ref/f.name) for f in private.glob('*_'+suffix))
            receipts.append(dict(mode=m,artifact=suffix,seed=103,sha256=hashfile(a),repeat_identical=True))
        beh=pd.read_csv(path/'behavior.csv').set_index(['case','year'])
        for seed in (103,109):
            case=f'{seed}_both_60_1.0'; hh=pd.read_csv(private/(case+'_accounts.csv')).query("kind=='household'").copy()
            assert len(hh)==3000 and not hh.duplicated(['id','year']).any()
            b=base.reindex(hh.id)
            assert np.allclose(hh.cultivated*b.intensity.to_numpy(),hh.grain+hh.crop)
            shocks={}
            for y,z in set(zip(hh.year,hh.region)):
                rng=np.random.default_rng([seed,int(y),int(z)]);u=rng.random()
                shocks[y,z]=1-rng.uniform(.05,.10) if u<.05 else (1+rng.uniform(.0275,.055) if u<.13 else 1.)
            assert np.allclose([shocks[y,z] for y,z in zip(hh.year,hh.region)],hh.weather_factor)
            assert np.allclose(hh.grain_kg,hh.grain*b.region.map(prices).to_numpy()*hh.weather_factor)
            assert np.allclose(hh.sales,(hh.grain*b.region.map(prices).to_numpy()*float(params['grain price'])*ratios['grain']+hh.crop*spec['cash_sale'])*hh.weather_factor)
            assert np.allclose(hh.input_hire_cost,hh.grain*kg+hh.crop*kc+hh.hired_days*w)
            assert np.allclose(hh.family_used+hh.hired_days,hh.grain*lg+hh.crop*lc)
            assert (hh.input_hire_cost<=hh.preplant_funds+1e-5).all() and (hh.family_used<=hh.family_capacity+1e-6).all()
            assert np.allclose(hh.claim,rate*np.minimum(hh.rights-hh.rented_out,hh.cultivated)) and np.allclose(hh.closing,hh.opening+hh.net-hh.consumption)
            assert hashfile(private/(case+'_organizations.csv'))==hashfile(ref/(case+'_organizations.csv'))
            hh['share']=hh.grain/(hh.grain+hh.crop).where(hh.grain+hh.crop>1e-7)
            tables[m,seed]=hh.set_index(['year','id']).sort_index()
            label=next((k for k,v in formal.items() if v==m),'sensitivity')
            prev=survey[survey.year.eq(2010)].set_index('actor')
            prev=pd.DataFrame(dict(obs=prev.sow_food/(prev.sow_food+prev.sow_cash).replace(0,np.nan)))
            prev['sim']=prev.obs
            for y,z in hh.groupby('year'):
                t=survey[survey.year.eq(y)].merge(z.rename(columns={'id':'actor'}),on='actor',validate='1:1').set_index('actor')
                t=t.dropna(subset=['sow_food','sow_cash'])
                ob=t.sow_food/(t.sow_food+t.sow_cash).replace(0,np.nan);cur=pd.DataFrame(dict(obs=ob,sim=t.share))
                joint=cur.join(prev,lsuffix='1',rsuffix='0',how='inner').dropna(); prev=cur
                diff=abs((joint.obs1-joint.obs0).abs().mean()-(joint.sim1-joint.sim0).abs().mean())
                r=beh.loc[case,y].to_dict(); assert len(t)==r['matched']
                known=t[['area_b','area_e','area_in','area_de','area_de2']].ge(0).all(axis=1)&(t.area_b+t.area_in-t.area_de-t.area_e).abs().lt(1e-4)&t.area_b.gt(0)&t.area_in.eq(0)&t.area_de2.le(t.area_de)
                summary.append(dict(mode=m,model=label,seed=seed,year=y,**r,adjustment_n=len(joint),adjustment_error=diff,
                    pass_three=bool(r['passed'] and diff<=.15),survey_grain_only=(t.sow_food.gt(0)&t.sow_cash.eq(0)).sum(),
                    survey_cash_only=(t.sow_cash.gt(0)&t.sow_food.eq(0)).sum(),sim_grain_only=(t.grain.gt(1e-7)&t.crop.le(1e-7)).sum(),
                    sim_cash_only=(t.crop.gt(1e-7)&t.grain.le(1e-7)).sum(),survey_other_only=(ob.isna()&t.sow_oother.gt(0)).sum(),
                    survey_all_planted=t.sow_total.gt(0).sum(),survey_all_observed=t.sow_total.count(),survey_other_mu=t.sow_oother.sum(),
                    survey_all_sown_intensity=(t.sow_total/t.area_e.where(t.area_e.gt(0))).median(),survey_all_sown_mu=t.sow_total.sum(),
                    survey_cash_mu=t.sow_cash.sum(),sim_cash_mu=t.crop.sum(),family_used=t.family_used.sum(),hired_days=t.hired_days.sum(),
                    net_income=t.net.sum(),model_two_crop_idle=t.cultivated.le(1e-7).sum(),survey_operated_to_zero=(t.area_b.gt(0)&t.area_e.eq(0)).sum(),
                    survey_area_de_observed=t.area_de.count(),survey_area_de_reported_mu=t.area_de.sum(min_count=1),
                    survey_known_begin_end=(t.area_b.notna()&t.area_e.notna()).sum(),full_leases_identifiable=known.sum(),
                    full_leases_observed=(known&t.area_e.eq(0)&(t.area_b-t.area_de2).abs().lt(1e-4)).sum(),
                    survey_operated_q25=t.area_e.quantile(.25),survey_operated_q75=t.area_e.quantile(.75),
                    sim_operated_q25=t.operated.quantile(.25),sim_operated_q75=t.operated.quantile(.75)))
                planted=z.share.notna()&z.target_share.notna()
                sens.append(dict(mode=m,model=label,seed=seed,year=y,positive_with_history=planted.sum(),
                       above=(z.share[planted]>z.target_share[planted]+1e-7).sum(),below=(z.share[planted]<z.target_share[planted]-1e-7).sum(),
                       reference_median=z.target_share[planted].median(),realized_median=z.share[planted].median(),
                       pure_cash_below_reference=((z.share<1e-7)&(z.target_share>1e-7)).sum()))
        annual.append(pd.read_csv(path/'structures.csv').query("not case.str.contains('repeat')",engine='python').assign(mode=m,model=label))
        if (path/'fit.csv').exists():fields.append(pd.read_csv(path/'fit.csv').assign(mode=m))
    for seed in (103,109):
        for y in range(2011,2016):
            gs={k:tables[v,seed].loc[y] for k,v in formal.items()}
            for metric in ['grain','grain_kg','crop','net','rented_in','hired_days','paid']:
                a,b,c,d=[gs[k][metric].sum() for k in 'ABCD']
                factor.append(dict(seed=seed,year=y,metric=metric,A=a,B=b,C=c,D=d,rule=b-a,cohort=c-a,interaction=d-b-c+a,total=d-a))
            for left,right in [('A','B'),('C','D')]:
                a,b=gs[left],gs[right];sa=a.grain+a.crop;sb=b.grain+b.crop;both=sa.gt(1e-7)&sb.gt(1e-7)
                fa=a.share[both];fb=b.share[both]
                entry=b.grain[~both].sum()-a.grain[~both].sum();mix=((fb-fa)*(sb[both]+sa[both])/2).sum();size=((fb+fa)*(sb[both]-sa[both])/2).sum()
                assert np.isclose(b.grain.sum()-a.grain.sum(),entry+mix+size)
                bridge.append(dict(seed=seed,year=y,comparison=right+'-'+left,entry_exit=entry,within_share=mix,sown_scale=size,grain_change=entry+mix+size))
    # Hold actual leases, cash, labor, weather and the reference fixed; do not run a policy counterfactual.
    from core.fixed_rule_runtime import continuous_farm_plan,crop_history_group,continuous_crop_target
    from scripts.run_national_household_portfolio import estimate_crop_transitions
    from types import SimpleNamespace
    loc=configs[formal['D']];base=pd.read_csv(loc/'PRIVATE_DO_NOT_SHARE/baseline.csv').set_index('actor')
    mobility=pd.read_csv(loc/'PRIVATE_DO_NOT_SHARE/historical_mobility.csv').set_index('actor')
    assert len(mobility)==600 and mobility.index.is_unique
    mobility.assign(group=np.where(mobility.observed,np.where(mobility.switch_proxy.eq(1),'changed','stable'),'regional_proxy')).groupby('group').penalty.agg(['count','min','mean','max']).to_csv(out/'adjustment_costs.csv')
    rev=json.loads((loc/'inputs.json').read_bytes())['cash_sale']
    response=[];roll=[]
    pd_high=float(pd.read_csv(configs[chosen]/'fit.csv').penalty.iloc[0])
    cohorts=set(raw.iloc[np.r_[new.source_row,old.source_row]-2].hhid.dropna())
    from core.extract_microdata_features import REGION_MAPPING
    geo={p:z for z,ps in REGION_MAPPING.items() for p in ps}
    donors=pd.concat([estimate_crop_transitions(RAW,cohorts,True,y,True) for y in (2010,2011,2012)])
    histories={(z,k):(g.prior_share.to_numpy(),g.share.to_numpy()) for (z,k),g in donors.groupby([donors.smgen.map(geo),'prior'])}
    keys=raw.iloc[new.source_row.to_numpy()-2][['id','hhid']].assign(actor=new.cohort_agent_id.to_numpy()).merge(ids2010,validate='m:1')
    s=unique.merge(keys,on=['id','hhid'],validate='m:1').set_index(['year','actor']).sort_index()
    for seed in (103,109):
        for y in (2013,2014,2015):
            frame=tables[formal['D'],seed].loc[y]
            z=s.loc[y].join(s.loc[y-1][['sow_food','sow_cash']],rsuffix='_prior',how='inner').join(frame[['choice_u','target_share']]).dropna(subset=['sow_food','sow_cash','sow_food_prior','sow_cash_prior','target_share'])
            z=z[z.sow_food+z.sow_cash>0]
            pred=[]
            for i,r in z.iterrows():
                g,c=r.sow_food_prior,r.sow_cash_prior
                k=crop_history_group(g,c);k='mixed' if k.startswith('mixed') else k
                h=histories.get((base.at[i,'region'],k))
                pred.append(continuous_crop_target({k:h},k,g/(g+c) if g+c else None,r.choice_u) if h else np.nan)
            actual=z.sow_food/(z.sow_food+z.sow_cash); keep=np.isfinite(pred)
            for kind,point in [('actual_lag',np.asarray(pred)),('simulated_lag',z.target_share.to_numpy())]:
                roll.append(dict(seed=seed,year=y,train_end=2012,history=kind,n=keep.sum(),target_mae=np.mean(abs(point[keep]-actual[keep])),
                                 quartile_mae=np.abs(np.quantile(point[keep],[.25,.5,.75])-actual[keep].quantile([.25,.5,.75])).mean()))
        for mode in [chosen,formal['D']]:
            z=tables[mode,seed].loc[2015]; changes=[]
            for r in z.itertuples():
                k=mobility.at[r.Index,'penalty'] if mode==formal['D'] else pd_high
                b=base.loc[r.Index]; actor=SimpleNamespace(sowing_intensity=b.intensity,area_limit=b.region_land,
                    family_days=r.family_capacity,crop_adjustment_yuan=k,crop_mix_target=r.target_share if pd.notna(r.target_share) else None)
                income=prices[b.region]*float(params['grain price'])*ratios['grain']
                outcomes=[continuous_farm_plan(actor,r.operated,r.rights-r.rented_out,r.preplant_funds,0.,
                    (income*pg,rev*pc,kg,kc,lg,lc,w,rate),False) for pg,pc in [(1,1),(1.1,1),(1,1.1)]]
                assert np.isclose(outcomes[0]['grain'],r.grain) and np.isclose(outcomes[0]['crop'],r.crop)
                changes.append([p['grain'] for p in outcomes])
            a,b,c=np.array(changes).T
            response.append(dict(mode=mode,seed=seed,year=2015,families=len(a),grain_price_up_more_grain=(b>a+1e-7).sum(),grain_price_up_delta_mu=(b-a).sum(),
                                 cash_sale_up_less_grain=(c<a-1e-7).sum(),cash_sale_up_delta_mu=(c-a).sum()))
    pd.DataFrame(roll).to_csv(out/'one_step_history.csv',index=False)
    pd.DataFrame(response).to_csv(out/'conditional_price_response.csv',index=False)
    for name,rows in [('household_behavior',summary),('factorial',factor),('target_response',sens),('repeat_receipts',receipts),('grain_bridge',bridge)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    pd.concat(annual).to_csv(out/'production_and_land.csv',index=False)
    pd.concat(fields).to_csv(out/'fitted_reference.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7));df=pd.DataFrame(summary).query("model!='sensitivity'")
    for tag,g in df.groupby('model'):
        g=g.groupby('year')[['share_mae','sim_grain','survey_grain']].mean()
        axes[0].plot(g.index,g.share_mae,marker='o',label=tag)
        axes[1].plot(g.index,g.sim_grain-g.survey_grain,marker='o',label=tag)
    for ax,label,ref in zip(axes,['Grain-share quartile MAE','Grain area minus matched survey (mu)'],[.15,0]):
        ax.axhline(ref,linestyle=':',color='black');ax.set(ylabel=label,xticks=range(2011,2016));ax.legend()
    fig.tight_layout();fig.savefig(out/'factorial.png',dpi=160);plt.close(fig)


if __name__=="__main__" and __import__('sys').argv[1:2]==['soft-response']:
    import sys
    audit_soft_response(sys.argv[2])
elif __name__=="__main__" and __import__('sys').argv[1:2]==['crop-mix']:
    import sys
    audit_crop_mix(sys.argv[2])
elif __name__=="__main__" and __import__('sys').argv[1:2]==['area']:
    import sys
    audit_area_receipts(sys.argv[2])
elif __name__=="__main__":
    main()
