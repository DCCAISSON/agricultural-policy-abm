"""Check the configured budget sample against the original survey, without imputing sales."""
import argparse
import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.audit_national_crop_budget_basis_20261118 import CROPS, DATA, LABELS, SALES

ROOT = Path(__file__).resolve().parents[1]


def main():
    b = pd.read_csv(ROOT / "results/national_crop_budget_basis_20261118/run_04/real2024_revision/record_selection_money2024.csv")
    b = b.loc[b.eligible].copy()
    labels = pd.read_excel(DATA / "household variables2016-2017.xlsx", header=None)
    fields = {}
    for crop in CROPS:
        codes = {k: labels.loc[labels[1].eq(crop + LABELS[k]), 0].item() for k in ("area", "income", "labor")}
        codes["sales_amount"] = SALES[crop][1]
        assert labels.loc[labels[0].eq(codes["sales_amount"]), 1].item() == "出售" + crop + "金额"
        fields[crop] = codes
    src = pd.read_csv(DATA / "2016-2017household.csv", usecols={"year", *(s for d in fields.values() for s in d.values())})
    out = []
    for crop, g in b.groupby("crop"):
        src_g = src.iloc[g.source_row.to_numpy() - 2]
        assert np.array_equal(src_g.year, g.year)
        for k, col in fields[crop].items():
            factor = g.to_2024.to_numpy() if k == "income" else 1
            assert np.allclose(src_g[col].to_numpy() * factor, g[k], equal_nan=True)
        v = g.loc[g.sales_amount.notna()]
        area = v.area.sum()
        assert area > 0
        out.append({"crop": crop, "budget_rows": len(g), "rows_with_reported_sales": len(v),
                    "budget_sown_mu": g.area.sum(), "paired_sown_mu": area,
                    "all_value_yuan2024_per_sown_mu": g.income.sum()/g.area.sum(),
                    "all_nonlabor_high_yuan2024_per_sown_mu": g.nonlabor_high.sum()/g.area.sum(),
                    "all_labor_days_per_sown_mu": g.labor.sum()/g.area.sum(),
                    "paired_value_yuan2024_per_sown_mu": v.income.sum()/area,
                    "paired_sales_yuan2024_per_sown_mu": (v.sales_amount*v.to_2024).sum()/area})
    assert len(out) == 3 and sum(r["budget_rows"] for r in out) == len(b)
    dest = ROOT / "results/diagnostics/national_household_portfolio/专家审阅包/07_作物预算来源复核.csv"
    pd.DataFrame(out).to_csv(dest, index=False, encoding="utf-8-sig")


def fixed_menu_choice(options, decision, price, delta):
    """Reprice only recorded cash sales; preserve V1 feasibility, ordering and ties."""
    previous = decision["previous_grain_share"]
    shares = {"maintain_grain": 1., "expand_cash": 0., "exit": None,
              "portfolio_keep": previous, "portfolio_adjusted": decision["adjusted_candidate_grain_share"]}
    shares.update({f"portfolio_target_{n}": n / 10 for n in range(1, 10)})
    best = {}
    for i, o in enumerate(options):
        if not o["account_feasible"] or not o["service_available"]:
            continue
        action = o["action"]
        score = o["score_yuan"] + o["expected_account"]["cash_output_kg"] * price * delta
        if action not in best or score > best[action][0]:
            best[action] = score, i
    targets = [k for k in best if k != "portfolio_adjusted"]
    keep = next((k for k in targets if previous is not None and shares[k] == previous), None)
    target = max(targets, key=lambda k: (best[k][0], k == keep))
    if previous is None or target == "exit":
        return best[target][1], best[target][1]
    goal, rate = shares[target], decision["household_crop_adjustment_rate"]
    step = goal if rate == 1 else previous + rate * (goal - previous)
    step_action = next((o["action"] for o in options if shares[o["action"]] == step), None)
    if step_action is None:
        return best[target][1], None
    choices = [k for k in (keep, step_action, "exit") if k in best]
    return best[target][1], best[max(choices, key=lambda k:best[k][0])][1]


def v1_sales():
    dest = ROOT/"results/diagnostics/national_v1_20261122"
    def read(p):
        return json.loads(p.read_text(encoding="utf-8"))
    gate = read(dest/"behavior_gate.json")
    runs = {r["seed"]:Path(r["episode"]) for r in gate["episodes"] if r["rule"] == "v1"}
    names = {103:"s103_v1_base_n60_20261123_001322_237093", 109:"s109_v1_base_n60_20261123_121347_546011"}
    if {s:p.parent.name for s,p in runs.items()} != names or gate["rule"] != "v1":
        raise ValueError("Expected the two specified V1 episodes recorded by the behavior audit")

    budget = ROOT/"results/national_crop_budget_basis_20261118/run_04/real2024_revision/record_selection_money2024.csv"
    b = pd.read_csv(budget)
    comp = b[b.eligible & b.output.gt(0) & b.sales_quantity.between(0,b.output)]
    volumes = comp.groupby("crop")[["output","sales_quantity"]].sum()
    ratios = volumes.sales_quantity / volumes.output
    cash = volumes.loc[["蔬菜","水果"]].sum()
    observed = []
    for (year,crop), group in comp.groupby(["year","crop"]):
        pairs = group.loc[group.sales_quantity.gt(0) & group.sales_amount.gt(0)]
        observed.append(dict(year=year,crop=crop,comparable=len(group),priced=len(pairs),
            kg=pairs.sales_quantity.sum(),yuan2024=(pairs.sales_amount*pairs.to_2024).sum()))
    summary, prices = [], []
    cohorts, reversals = {}, {}
    for seed,path in runs.items():
        cfg = read(path/"effective_config.json")
        ref = read(path.parent/"reference_counts.json")
        assert ref["budget_sha256"] == sha256(budget.read_bytes()).hexdigest()
        assert ref["sales_comparable"] == comp.groupby("crop").size().to_dict()
        assert cfg["runtime_fixed_rule"]["national_v1"] is True and cfg["scale"]["n_enterprises"] == 60
        frac = cfg["runtime_fixed_rule"]["household_sales_ratio"]
        rates = {"低":ratios["蔬菜"], "原":frac["cash_crop"], "高":ratios["水果"]}
        assert math.isclose(frac["grain"], ratios["稻谷"]) and math.isclose(rates["原"],cash.sales_quantity/cash.output)
        initial = {x["id"] for x in read(path/"initial_households.json")}
        assert len(initial)==600
        cohorts[seed] = initial
        reversals[seed] = {k:set() for k in rates}
        for t in (1,2,3):
            year = 2010+t
            y = path/f"year_{t:02d}"
            base = cfg["environment"]["market"]["base_prices"]
            actual = read(y/"market_outcomes.json")
            assert read(y/"price_inputs.json")["base_crop_prices"] == base
            pg, pc = 2*base["grain"], 2*base["cash_crop"]
            assert (pg, pc) == (2*actual["crop_price"], 2*actual["cash_crop"])
            decisions = {z["operator_id"]:z for z in read(y/"decision_trace.json") if z["operator_type"]=="household"}
            seen = set()
            counts = {k:Counter() for k in rates}
            nopts = 0
            maxerr = 0.
            with (y/"actor_trace.jsonl").open(encoding="utf-8") as stream:
                for z in map(json.loads,stream):
                    key = z["actor_id"]
                    if key not in decisions:
                        continue
                    assert key not in seen and z["year"] == year
                    seen.add(key)
                    d, opts = decisions[key], z["alternatives"]
                    for o in opts:
                        a = o["expected_account"]
                        expected = a["grain_output_kg"]*frac["grain"]*pg + a["cash_output_kg"]*frac["cash_crop"]*pc
                        err = abs(expected-a["sales_yuan"])
                        assert err < 1e-6, (seed,year,"sales",err)
                        maxerr = max(maxerr,err)
                    nopts += len(opts)
                    target, selected = fixed_menu_choice(opts, d, pc, 0.)
                    assert opts[target]["action"] == d["target_action"] and selected == z["selected_index"]
                    for label, ratio in rates.items():
                        a, s = fixed_menu_choice(opts, d, pc, ratio-rates["原"])
                        c = counts[label]
                        c["target_mode_or_action"] += a != target
                        if a != target:
                            reversals[seed][label].add(key)
                        c["target_action"] += opts[a]["action"] != opts[target]["action"]
                        c["unpriced_step"] += s is None
                        c["implemented_change"] += s is not None and s != selected
            assert seen == set(decisions)
            harvests = read(y/"actual_harvests.json")
            nh = 0
            harvest_error = 0.
            with (y/"actors.jsonl").open(encoding="utf-8") as stream:
                for row in map(json.loads,stream):
                    if row["id"] not in initial or row["id"] not in harvests:
                        continue
                    h, a = harvests[row["id"]], row["income_account"]
                    assert h["step"] == a["step"] == t
                    g,c = h["grain_physical_kg"]*frac["grain"], h["cash_output_model_units"]*frac["cash_crop"]
                    assert abs(g-a["grain_sold_kg"])<1e-6 and abs(c-a["cash_crop_sold_kg"])<1e-6
                    err = abs(g*pg+c*pc-a["sales_yuan"])
                    assert err < 1e-6
                    harvest_error = max(harvest_error,err)
                    nh += 1
            prices.append(dict(seed=seed,year=year,grain_price=pg,cash_price=pc,grain_ratio=frac["grain"],
                cash_ratio=rates["原"],options=nopts,max_expected_error=maxerr,initial_harvests=nh,max_harvest_error=harvest_error))
            for label,c in counts.items():
                summary.append(dict(seed=seed,year=year,scenario=label,cash_ratio=rates[label],decisions=len(seen),
                    initial_decisions=len(initial & seen),**c))
    assert cohorts[103] == cohorts[109] and all(r["decisions"]==r["initial_decisions"] for r in summary)
    unique = [dict(scenario=k,seed103=len(reversals[103][k]),seed109=len(reversals[109][k]),
                   either_seed=len(reversals[103][k]|reversals[109][k]),both_seeds=len(reversals[103][k]&reversals[109][k]),
                   cohort_n=len(cohorts[103])) for k in rates]
    for name,rows in (("经作销售_决策敏感性",summary),("经作销售_反转家庭数",unique),
                      ("经作销售_价格核验",prices),("经作销售_调查成交单价",observed)):
        frame = pd.DataFrame(rows)
        if rows is observed:
            frame["yuan2024_per_kg"] = frame.yuan2024/frame.kg
        frame.to_csv(dest/(name+".csv"),index=False,encoding="utf-8-sig")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v1", action="store_true", help="read the two accepted national V1 ledgers without running the model")
    if p.parse_args().v1:
        v1_sales()
    else:
        main()
