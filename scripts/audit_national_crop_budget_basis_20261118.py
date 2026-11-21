"""Same-row NFPS crop budgets; diagnostics only, no imputation or calibration."""
from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "校准用数据源/全国农村固定观察点"
OUT = ROOT / "results/diagnostics/national_crop_budget_basis_20261118/run_01"
CROPS = ("稻谷", "蔬菜", "水果")
LABELS = {
    "area": "1.播种面积(亩)", "harvest_area": "2.实际收获面积(亩)",
    "output": "3.总产量(千克)", "income": "4.总收入(元)", "total_cost": "5.总费用(元)",
    "seed": "6.种子种苗费用(元)", "manure": "7.农家肥折价(元)",
    "fertilizer": "8.化肥费用(元)", "film": "9.农膜费用(元)",
    "pesticide": "10.农药费用(元)", "irrigation": "11.水电及灌溉费用(元)",
    "other_material": "12.其他材料费(元)", "animal": "13.畜力费(元)",
    "machinery": "14.机械作业费用(元)", "depreciation_repair": "15.固定资产折旧及修理费(元)",
    "small_tools": "16.小农具购置修理费(元)", "rent": "16.土地租赁费用(元)",
    "other": "17.其他间接费用(元)", "material_total": "10.物质费用合计(元)",
    "labor": "18.投工量(日)", "hired_days": "19其中：雇工(日)",
    "hired_cost": "20雇工费用(元)", "tax": "10.支付税金(元)",
    "byproduct": "21.副产品价值（元）",
}
# The thirteen monetary lines on questionnaire PDF pp.14-15; no duplicate seed sublines.
COMPONENTS = ("seed", "manure", "fertilizer", "film", "pesticide", "irrigation",
              "animal", "machinery", "depreciation_repair", "small_tools", "rent", "other", "hired_cost")
BASIC = ("area", "output", "income", "total_cost", "labor")
SALES = {"稻谷": ("e7", "e8"), "蔬菜": ("e33", "e34"), "水果": ("e25", "e26")}


def derive_budget(raw):
    b = raw.copy(deep=True)
    b["basic_complete"] = b[list(BASIC)].notna().all(axis=1)
    b["basic_positive_area"] = b.basic_complete & b.area.gt(0)
    b["components_complete"] = b[list(COMPONENTS)].notna().all(axis=1)
    b["components_sum"] = b[list(COMPONENTS)].sum(axis=1, min_count=len(COMPONENTS))
    b["component_residual"] = b.total_cost - b.components_sum
    b["total_matches_components"] = b.component_residual.abs().le(1e-8)
    b["reported_margin"] = b.income - b.total_cost
    b["component_margin"] = b.income - b.components_sum
    b["sales_minus_reported_cost"] = b.sales_amount - b.total_cost
    b["family_days"] = b.labor - b.hired_days
    b["labor_consistent"] = b.labor.ge(0) & b.hired_days.ge(0) & b.family_days.ge(0)
    b["family_joint"] = b.basic_positive_area & b.labor_consistent
    b["wage_reference_valid"] = b.hired_days.gt(0) & b.hired_cost.ge(0) & b.labor_consistent
    b["own_hired_wage_reference"] = (b.hired_cost / b.hired_days.where(b.hired_days.gt(0))).where(b.wage_reference_valid)
    b["family_charge_at_own_hired_wage"] = b.family_days * b.own_hired_wage_reference
    b["margin_at_own_hired_wage_conditional"] = b.reported_margin - b.family_charge_at_own_hired_wage
    b["wage_joint"] = b.basic_positive_area & b.wage_reference_valid
    b["listed_expenses_excluding_manure_depr_repair"] = b.components_sum - b.manure - b.depreciation_repair
    # Conditional accounting range, not an identified cash cost: repair is inseparable from depreciation.
    b["cash_cost_low_conditional"] = b.total_cost - b.manure - b.depreciation_repair
    b["cash_cost_high_conditional"] = b.total_cost - b.manure
    b["cash_range_joint"] = (b.basic_positive_area & b[["manure", "depreciation_repair"]].notna().all(axis=1)
        & b.manure.ge(0) & b.depreciation_repair.ge(0) & b.cash_cost_low_conditional.ge(0))
    b["cash_margin_low_conditional"] = b.income - b.cash_cost_high_conditional
    b["cash_margin_high_conditional"] = b.income - b.cash_cost_low_conditional
    b["reported_margin_before_rent"] = b.reported_margin + b.rent
    b["reported_margin_before_hired_cost"] = b.reported_margin + b.hired_cost
    b["negative_raw_field"] = b[list(LABELS)].lt(0).any(axis=1)
    b["hired_days_exceed_labor"] = b.hired_days.gt(b.labor)
    b["positive_hired_cost_zero_days"] = b.hired_cost.gt(0) & b.hired_days.eq(0)
    for field in ("output", "income", "total_cost", "reported_margin", "rent", "hired_cost", "family_days",
                  "family_charge_at_own_hired_wage", "margin_at_own_hired_wage_conditional", "component_margin",
                  "cash_margin_low_conditional", "cash_margin_high_conditional", "reported_margin_before_rent"):
        b[field + "_per_mu"] = b[field] / b.area.where(b.area.gt(0))
    b["family_wage_break_even_conditional"] = b.reported_margin / b.family_days.where(b.family_days.gt(0))
    assert len(b) == len(raw)
    assert b.loc[~b.components_complete, "components_sum"].isna().all()
    assert np.allclose(b.reported_margin.dropna(), (b.income - b.total_cost).dropna())
    return b


def pair_budget(left, right):
    assert left.index.equals(right.index)
    p = left[["source_row", "year", "sm", "cm", "hm", "duplicate_household_year"]].copy()
    p["crop_left"], p["crop_right"] = left.crop.iloc[0], right.crop.iloc[0]
    p["joint_basic"] = left.basic_positive_area & right.basic_positive_area
    p["joint_family"] = p.joint_basic & left.family_joint & right.family_joint
    p["joint_wage_reference"] = p.joint_basic & left.wage_joint & right.wage_joint
    for field in ("income", "total_cost", "reported_margin", "family_days"):
        p["left_" + field + "_per_mu"] = left[field + "_per_mu"]
        p["right_" + field + "_per_mu"] = right[field + "_per_mu"]
        p["delta_" + field + "_per_mu"] = right[field + "_per_mu"] - left[field + "_per_mu"]
    p["common_wage_switch_threshold"] = (p.delta_reported_margin_per_mu /
        p.delta_family_days_per_mu.where(p.delta_family_days_per_mu.ne(0))).where(p.joint_family)
    p["common_wage_zero_slope"] = p.joint_family & p.delta_family_days_per_mu.eq(0)
    joint = p.loc[p.joint_basic]
    assert np.allclose(joint.delta_reported_margin_per_mu,
                       joint.delta_income_per_mu - joint.delta_total_cost_per_mu, atol=1e-7)
    return p


def stats(values, prefix):
    return {prefix + "_" + name: getattr(values, name)() for name in ("mean", "median", "min", "max")}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    csv = DATA / "2016-2017household.csv"
    dictionary = DATA / "household variables2016-2017.xlsx"
    questionnaire = DATA / "2009-2017农户调查问卷.pdf"
    agri = next((ROOT / "docs/reference_project").glob("*AgriPoliS*/Happe_2006_Agent_based_analysis.pdf"))
    regmas = ROOT / "docs/reference_project/RegMAS/RegMAS_2010_MPRA.pdf"
    sources = [csv, dictionary, questionnaire, agri, regmas, Path(__file__),
               ROOT / "tests/test_audit_national_crop_budget_basis_20261118.py"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    labels = pd.read_excel(dictionary, header=None)
    mapping, trace = {}, []
    for crop in CROPS:
        mapping[crop] = {}
        for name, suffix in LABELS.items():
            match = labels.loc[labels[1].eq(crop + suffix)]
            assert len(match) == 1, (crop, name)
            field = match.iloc[0, 0]
            mapping[crop][name] = field
            trace.append(dict(crop=crop, role=name, field=field, label=crop + suffix,
                              dictionary_row=int(match.index[0]) + 1, dictionary=str(dictionary)))
        for name, field in zip(("sales_quantity", "sales_amount"), SALES[crop]):
            mapping[crop][name] = field
            match = labels.loc[labels[0].eq(field)]
            trace.append(dict(crop=crop, role=name, field=field, label=match.iloc[0, 1],
                              dictionary_row=int(match.index[0]) + 1, dictionary=str(dictionary)))
    keys = ["year", "sm", "cm", "hm"]
    columns = list(dict.fromkeys(keys + [v for m in mapping.values() for v in m.values()]))
    data = pd.read_csv(csv, usecols=columns, low_memory=False)
    n = len(data)
    assert not data[keys].isna().any().any()
    data["source_row"] = np.arange(n) + 2
    data["duplicate_household_year"] = data.duplicated(keys, keep=False)
    budgets, coverage, summaries, reconciliations = {}, [], [], []
    for crop, m in mapping.items():
        raw = data[keys + ["source_row", "duplicate_household_year"] + list(m.values())].rename(columns={v:k for k,v in m.items()})
        raw["crop"] = crop
        b = derive_budget(raw)
        budgets[crop] = b
        for year, g in b.groupby("year", dropna=False):
            for role in LABELS:
                v = g[role]
                coverage.append(dict(year=year, crop=crop, role=role, field=m[role], raw_n=len(g),
                    missing=int(v.isna().sum()), zero=int(v.eq(0).sum()), negative=int(v.lt(0).sum()), positive=int(v.gt(0).sum())))
            for scope in ("basic_positive_area", "family_joint", "wage_joint", "cash_range_joint"):
                h = g.loc[g[scope]]
                row = dict(year=year, crop=crop, scope=scope, raw_year_n=len(g), joint_record_n=len(h),
                    distinct_household_year_n=len(h[keys].drop_duplicates()),
                    ambiguous_key_record_n=int(h.duplicate_household_year.sum()),
                    negative_raw_record_n=int(h.negative_raw_field.sum()))
                fields = ["income_per_mu", "total_cost_per_mu", "reported_margin_per_mu"]
                if scope in ("family_joint", "wage_joint"):
                    fields += ["family_days_per_mu"]
                if scope == "wage_joint":
                    fields += ["own_hired_wage_reference", "family_charge_at_own_hired_wage_per_mu", "margin_at_own_hired_wage_conditional_per_mu"]
                    row["proxy_margin_negative_n"] = int(h.margin_at_own_hired_wage_conditional.lt(0).sum())
                if scope == "cash_range_joint":
                    fields += ["cash_margin_low_conditional_per_mu", "cash_margin_high_conditional_per_mu"]
                for field in fields:
                    assert h[field].notna().all()
                    row.update(stats(h[field], field))
                summaries.append(row)
            complete = g.total_cost.notna() & g.components_complete
            for area_scope, mask in (("all_complete_including_zero_area", complete),
                                     ("positive_area_complete", complete & g.area.gt(0))):
                h = g.loc[mask]
                for hypothesis, expense in (("sum_13_questionnaire_cost_lines", h.components_sum),
                        ("sum_13_excluding_manure", h.components_sum - h.manure),
                        ("sum_13_excluding_hired_cost", h.components_sum - h.hired_cost)):
                    residual = h.total_cost - expense
                    reconciliations.append(dict(year=year, crop=crop, area_scope=area_scope,
                        hypothesis=hypothesis, complete_records=len(h), matches=int(residual.abs().le(1e-8).sum()),
                        **stats(residual, "total_minus_components")))
    pairs, pair_summaries = [], []
    for left, right in combinations(CROPS, 2):
        p = pair_budget(budgets[left], budgets[right])
        pairs.append(p)
        for year, g in p.groupby("year", dropna=False):
            for scope in ("joint_basic", "joint_family", "joint_wage_reference"):
                h = g.loc[g[scope]]
                row = dict(year=year, crop_left=left, crop_right=right, scope=scope,
                    raw_year_n=len(g), joint_record_n=len(h), distinct_household_year_n=len(h[keys].drop_duplicates()),
                    ambiguous_key_record_n=int(h.duplicate_household_year.sum()),
                    right_reported_margin_higher_n=int(h.delta_reported_margin_per_mu.gt(0).sum()))
                for field in ("delta_income_per_mu", "delta_total_cost_per_mu", "delta_reported_margin_per_mu"):
                    row.update(stats(h[field], field))
                if scope != "joint_basic":
                    row.update(stats(h.delta_family_days_per_mu, "delta_family_days_per_mu"))
                    row["nonzero_labor_slope_n"] = int(h.common_wage_switch_threshold.notna().sum())
                pair_summaries.append(row)
    triple = data[keys + ["source_row", "duplicate_household_year"]].copy()
    triple["all_three_basic"] = pd.concat([b.basic_positive_area for b in budgets.values()], axis=1).all(axis=1)
    triple["all_three_family"] = pd.concat([b.family_joint for b in budgets.values()], axis=1).all(axis=1)
    identity_summary = []
    for year, g in triple.groupby("year", dropna=False):
        identity_summary.append(dict(year=year, raw_rows=len(g),
            distinct_household_years=len(g[keys].drop_duplicates()), duplicate_key_rows=int(g.duplicate_household_year.sum()),
            triple_basic_n=int(g.all_three_basic.sum()), triple_family_n=int(g.all_three_family.sum())))
    for p in sources:
        assert hashes[str(p)] == hashlib.sha256(p.read_bytes()).hexdigest(), f"Source changed: {p}"
    OUT.mkdir(parents=True, exist_ok=False)
    outputs = {"field_trace": pd.DataFrame(trace), "field_coverage": pd.DataFrame(coverage),
        "joint_crop_budgets": pd.concat(budgets.values(), ignore_index=True),
        "joint_budget_summary": pd.DataFrame(summaries), "cost_reconciliation": pd.DataFrame(reconciliations),
        "same_household_crop_pairs": pd.concat(pairs, ignore_index=True),
        "same_household_pair_summary": pd.DataFrame(pair_summaries), "household_identity_summary": pd.DataFrame(identity_summary)}
    for name, frame in outputs.items():
        frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    import fitz
    excerpts = []
    for path, pages in ((questionnaire, (14, 15)), (agri, (4, 5, 8)), (regmas, (10, 11, 12, 13, 21))):
        with fitz.open(path) as doc:
            for page in pages:
                excerpts.append(f"SOURCE {path}\nPDF_PAGE {page}\n{doc[page-1].get_text()}\n")
            if path == questionnaire:
                for page in pages:
                    doc[page-1].get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).save(OUT / f"questionnaire_page_{page}.png")
    (OUT / "reference_excerpts.txt").write_text("\n".join(excerpts), encoding="utf-8")
    verification = dict(raw_rows=n, budget_rows=len(outputs["joint_crop_budgets"]), pair_rows=len(outputs["same_household_crop_pairs"]),
        rows_deleted=0, missing_filled=0, winsorized=0, model_runs=0, calibrated_parameters=0,
        identity_key=keys, comparison="same original CSV row; no cross-row merge or deduplication",
        source_hashes=hashes, output_rows={k:len(v) for k,v in outputs.items()},
        claim="conditional_same_observation_budget_diagnostics_not_cash_cost_identification_or_calibration")
    (OUT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out":str(OUT), "raw_rows":n, "identity":identity_summary}, ensure_ascii=False))


if __name__ == "__main__":
    run()
