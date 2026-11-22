"""Reprice first-year household plans; stop before choices, bids or harvest."""
from dataclasses import asdict, replace
import json
import math
from unittest.mock import patch

import pandas as pd

from agents.government_agent import get_farmer_net_income
from core.fixed_rule_runtime import choose_action_income
from core.p3_organization import best_crop_accounts, organization_accounts
from scripts.audit_portfolio_2011_decisions import BASE, REF, areas, read, survey
from scripts.prepare_household_portfolio_reference import RAW
from scripts.run_national_household_portfolio import ARCHIVE, OLD, legacy, old


def probe(seed, survey_rows, factors):
    run = BASE / "runs" / f"s{seed}_new_base_20261121_220058_199520"
    ep = run / "episode"
    actual = areas(ep, 2011).loc[survey_rows.index]
    target = set(actual.index[survey_rows.survey_class.eq("mixed") & actual.iloc[:, 0].eq(0)
                              & actual.iloc[:, 1].gt(0)])
    assert len(target) == {103: 192, 109: 198}[seed]
    trace = {r["actor_id"]: r for r in map(json.loads, (ep/"year_01/actor_trace.jsonl").read_text(
        encoding="utf-8").splitlines()) if r["actor_id"] in target}
    cfg, economics, budget, inputs = legacy.load_base()
    assert inputs == read(ARCHIVE / "design.json")["frozen_inputs"]
    cfg["runtime_fixed_rule"]["household_crop_portfolio"] = True
    model = legacy.build_model(cfg, economics, seed)
    model.household_initial_grain_shares = read(REF/"initial_grain_shares.json")
    model.config["policy"]["government_q_project"].update(
        episode_end_step=3, uncapped_income_reward=True,
        grain_reference_kg=read(OLD/"training.json")["grain_reference_kg"],
        budget_area_menu=math.fsum(p.area for p in model.land_patches),
        h0_income_reference_yuan=math.fsum(get_farmer_net_income(f) for f in model.initial_household_cohort)/600)
    assert model.config == read(ep/"effective_config.json")
    legacy.configure_government(model.government, {}, seed, False)
    model.government._choose_action_rl = lambda state, action_count=3: 0
    old.apply_annual_exogenous_path(model, 0)
    old.advance_program(model, 0, budget)
    rows, modes, contrasts = [], [], []

    class Collected(Exception):
        pass

    def inspect(f, actions):
        original = organization_accounts(f, actions)
        actor = str(f.unique_id)
        if actor not in target:
            return original
        hist = trace[actor]
        for item, record in zip(original, hist["alternatives"], strict=True):
            assert (asdict(item.account), item.mode, item.service_available) == (
                record["expected_account"], record["mode"], record["service_available"]), (seed, actor)
        cash = original[hist["selected_index"]].account
        assert cash.action == choose_action_income(f, best_crop_accounts(original)) == "expand_cash"
        area = cash.area_mu
        shares = f.household_candidate_grain_shares
        row = dict(seed=seed, actor_id=actor, opening_mu=area, survey_grain_mu=survey_rows.at[actor,"sow_food"],
                   survey_cash_mu=survey_rows.at[actor,"sow_cash"], season_known=False,
                   min_grain_option_mu=min(v*area for v in shares.values() if v>0),
                   land_enough=area>=1, checked_original_accounts=len(original),
                   survey_field_cash_income=survey_rows.at[actor,"field_cash_income"],
                   survey_vegetable_income=survey_rows.at[actor,"inc_cash_7"],
                   survey_fruit_income=survey_rows.at[actor,"inc_cash_9"])
        extras = ()
        choice = None
        if area >= 1:
            choice = next((k for k,v in shares.items() if abs(v*area-1)<1e-10), None)
            if choice is None:
                choice = "portfolio_probe_one_mu"
                with patch.object(f,"household_candidate_grain_shares",dict(shares, **{choice:1/area})):
                    extras = organization_accounts(f,(choice,))
            row["new_point"] = bool(extras)
        cash_price = f._get_output_price_per_kg("cash_crop")
        grain_price = f._get_output_price_per_kg("crop_price")
        menu = original + extras
        for i in menu:
            a = i.account
            assert math.isclose(a.sales_yuan, a.grain_output_kg*grain_price+a.cash_output_kg*cash_price,abs_tol=1e-7)
            if a.action != choice:
                continue
            assert abs(a.grain_area_mu-1)<1e-9 and a.area_mu==area
            modes.append(dict(seed=seed,actor_id=actor,mode=i.mode,service_available=i.service_available,
                feasible=a.feasible,reason=a.constraint_reason,**asdict(a),
                family_days_available=f._get_family_labor_capacity_for_production(),
                delta_net=a.net_income-cash.net_income,delta_score=a.score-cash.score))
        feasible = [i for i in menu if i.account.action==choice and i.account.feasible and i.service_available]
        if feasible:
            best = max(feasible,key=lambda i:i.account.score)
            a = best.account
            lost = cash_price*(cash.cash_output_kg-a.cash_output_kg)
            row.update(mode=best.mode, cash_mode=original[hist["selected_index"]].mode,
                delta_net=a.net_income-cash.net_income,delta_score=a.score-cash.score,
                delta_cash_sales=-lost,delta_grain_sales=a.grain_output_kg*grain_price,
                delta_cost=a.cash_cost-cash.cash_cost,delta_support=a.support_yuan-cash.support_yuan,
                delta_offfarm=a.off_farm_yuan-cash.off_farm_yuan,delta_risk=a.risk_penalty_yuan-cash.risk_penalty_yuan,
                one_mu_cash_need=a.cash_cost,cash_available=a.working_capital_available,
                one_mu_family_days=a.family_labor_days,family_days_available=f._get_family_labor_capacity_for_production(),
                one_mu_hired_days=a.hired_labor_days,hired_days_available=a.hired_labor_available_days,
                delta_hired_cost=a.hired_labor_cost_yuan-cash.hired_labor_cost_yuan,
                delta_hired_days=a.hired_labor_days-cash.hired_labor_days,
                break_even_sales_cut_at_original_modes=(cash.score-a.score)/lost if lost>0 else None)
            assert a.family_labor_days<=row["family_days_available"]+1e-8
            assert math.isclose(row["delta_net"],row["delta_cash_sales"]+row["delta_grain_sales"]-
                row["delta_cost"]+row["delta_support"]+row["delta_offfarm"],abs_tol=1e-7)
        row["one_mu_feasible"] = bool(feasible)
        for label,factor in factors.items():
            for grid,options in (("original",original),("plus_one_mu",menu)):
                # Only expected receipts change; do not feed these accounts into the simulation.
                repriced = tuple(replace(i,account=replace(i.account,sales_yuan=i.account.sales_yuan+
                                 (factor-1)*i.account.cash_output_kg*cash_price)) for i in options)
                winner = choose_action_income(f,best_crop_accounts(repriced))
                if label=="booked" and grid=="original":
                    assert winner==cash.action
                contrasts.append(dict(seed=seed,actor_id=actor,budget=label,grid=grid,
                    factor=factor,action=winner,grain_mu=next(i.account.grain_area_mu for i in repriced if i.account.action==winner)))
        rows.append(row)
        if len(rows)==len(target):
            raise Collected
        return original

    with patch("core.p3_organization.organization_accounts",inspect):
        try:
            model.step()
        except Collected:
            pass
        else:
            raise AssertionError("decision capture incomplete")
    assert {r["actor_id"] for r in rows} == target
    return rows,modes,contrasts


def main():
    s = survey()
    fields = [f"inc_cash_{n}" for n in range(1,11)]
    raw = pd.read_csv(RAW,usecols=["year",*fields]).iloc[s.source_row.to_numpy()-2].set_axis(s.index)
    assert raw.year.eq(2011).all()
    s = s.join(raw[fields],validate="one_to_one")
    # Questionnaire B7-B14 are field crops; B15-B16 are garden crops, not part of this numerator.
    s["field_cash_income"] = s[fields[:8]].sum(axis=1,min_count=8)
    budgets = pd.read_csv(BASE/"专家审阅包/07_作物预算来源复核.csv").set_index("crop")
    factors = {"booked":1.0}
    for label,crop in (("vegetable","蔬菜"),("fruit","水果")):
        b = budgets.loc[crop]
        factors[label] = b.paired_sales_yuan2024_per_sown_mu/b.paired_value_yuan2024_per_sown_mu
    results = [probe(seed,s,factors) for seed in (103,109)]
    h,m,c = [pd.DataFrame(a+b) for a,b in zip(*results,strict=True)]
    assert not h.duplicated(["seed","actor_id"]).any()
    dest = BASE/"decision_audit/one_mu"
    g = h.groupby("seed")
    summary = g.agg(target=("actor_id","size"),area_ge_one=("land_enough","sum"),
        feasible=("one_mu_feasible","sum"),old_min_above_one=("min_grain_option_mu",lambda x:(x>1+1e-9).sum()),
        net_positive=("delta_net",lambda x:(x>0).sum()),median_delta_net=("delta_net","median"),
        median_delta_score=("delta_score","median"),median_cash_sales_lost=("delta_cash_sales","median"),
        median_grain_sales_added=("delta_grain_sales","median"),median_cost_delta=("delta_cost","median"),
        median_break_even_sales_cut=("break_even_sales_cut_at_original_modes","median"),
        checked_accounts=("checked_original_accounts","sum")).reset_index()
    counts = c.groupby(["seed","grid","budget","factor"]).agg(n=("actor_id","size"),
        cash_only=("action",lambda a:(a=="expand_cash").sum()),positive_grain=("grain_mu",lambda x:(x>0).sum()),
        exit=("action",lambda a:(a=="exit").sum())).reset_index()
    details = h[h.one_mu_feasible].groupby(["seed",h.delta_net.gt(0).rename("positive")]).agg(
        n=("actor_id","size"),both_outsourced=("cash_mode",lambda a:((a=="outsource") &
        h.loc[a.index,"mode"].eq("outsource")).sum()),median_change_hired_days=("delta_hired_days","median"),
        median_change_hired_cost=("delta_hired_cost","median"),median_delta_cost=("delta_cost","median")).reset_index()
    survey_check = g.agg(target=("actor_id","size"),complete_field_income=("survey_field_cash_income","count"),
        vegetable_positive=("survey_vegetable_income",lambda a:(a>0).sum()),
        fruit_positive=("survey_fruit_income",lambda a:(a>0).sum()),
        vegetable_reported=("survey_vegetable_income","count"),fruit_reported=("survey_fruit_income","count")).reset_index()
    for name,df in (("PRIVATE_DO_NOT_SHARE/one_mu_household.csv",h),("PRIVATE_DO_NOT_SHARE/one_mu_modes.csv",m),
                    ("income_coverage.csv",survey_check),("cost_decomposition.csv",details),
                    ("PRIVATE_DO_NOT_SHARE/repriced_choices.csv",c),("summary.csv",summary),("contrast.csv",counts)):
        path = dest/name
        path.parent.mkdir(parents=True,exist_ok=True)
        content = df.to_csv(index=False,lineterminator="\n").encode("utf-8-sig")
        if path.exists():
            assert path.read_bytes()==content,path
        else:
            with path.open("xb") as stream:
                stream.write(content)
    print(summary.to_string(index=False),counts.to_string(index=False),sep="\n")


if __name__=="__main__":
    main()
