"""Link raw columns to questionnaire labels, and estimate only supported budgets."""
from dataclasses import replace
from math import prod
from pathlib import Path
import json

import pandas as pd

from core.succession_economic_core import Activity, EconomicCore
from scripts.prepare_succession_budget_inputs import digest


HOUSEHOLD = Path("校准用数据源/全国农村固定观察点")
LINK = Path("results/diagnostics/national_crop_choice_soft_response_20261123/PRIVATE_DO_NOT_SHARE/cohort600.json")
CPI = Path("config/experiments/national_real2024_cpi_20260909.csv")
EARLY = Path("2009-2015household.csv")
LATE = Path("2016-2017household.csv")
DICT_EARLY = Path("household variables2009-2015.xlsx")
DICT_LATE = Path("household variables2016-2017.xlsx")
FORM = Path("2009-2017农户调查问卷.txt")
GRAINS = (("wheat","小麦","w"), ("rice","稻谷","r"), ("maize","玉米","c"))
COST = {
    "seed":"种子种苗费用(元)", "fer":"化肥费用(元)", "film":"农膜费用(元)",
    "pes":"农药费用(元)", "weir":"水电及灌溉费用(元)", "anim":"畜力费(元)",
    "mach":"机械作业费用(元)", "srep":"小农具购置修理费(元)",
}
MEASUREMENT = {
    "sow":"播种面积(亩)", "prod":"总产量(千克)", "inc":"总收入(元)",
    "labor_d":"投工量(日)", "labor_hd":"雇工(日)", "labor_hf":"雇工费用(元)", **COST,
}


def _match(labels, key, prefix, suffix, records, source):
    s = labels.loc[labels[0].eq(key),1]
    if len(s)!=1 or not str(s.iloc[0]).startswith(prefix) or suffix not in str(s.iloc[0]):
        raise ValueError(f"Label not verified in {source}: {key}: {prefix} / {suffix}")
    records.append(dict(source=str(source), column=key, label=str(s.iloc[0]), row=int(s.index[0])+1))
    return key


def _measures(rows, costs, factor, year, name, sold=1.):
    required = ["area","output","gross","labor",*costs]
    valid = rows[required].notna().all(axis=1) & rows[required].ge(0).all(axis=1) & rows.area.gt(0)
    # Explicitly exclude nonpositive area; reported output and receipts zero
    # on a positive area are retained. No expense missing is set to zero.
    x = rows.loc[valid].copy()
    total_area = x.area.sum()
    if x.empty or x.output.sum()<=0 or x.gross.sum()<=0 or x.labor.sum()<=0:
        raise ValueError(f"Not enough complete same-row activity budgets for {name}")
    x["factor"] = x.year.map(factor)
    # Presence is identified independently of completeness of cash expenses.
    present = rows.area.gt(0) & rows.output.gt(0)
    markets = tuple(int(v) for v in sorted(rows.loc[present,"market"].unique()))
    a = Activity(name, name!="vegetables", float(x.output.sum()/total_area),
                 float((x.gross*x.factor).sum()/x.output.sum()),
                 float((x[costs].sum(axis=1)*x.factor).sum()/total_area),
                 float(x.labor.sum()/total_area), sold, markets)
    hired = x.hired_days.gt(0) & x.hired_cost.gt(0) & x.hired_days.le(x.labor)
    detail = dict(crop=name, source_years=year, donor_rows=len(rows), positive_area=int(rows.area.gt(0).sum()),
                  complete_rows=len(x), sample_mu=float(total_area), supported_provinces=len(markets),
                  full_budget_provinces=x.market.nunique(), positive_output_rows=int(present.sum()),
                  yield_all_positive_reports_kg_mu=float(rows.loc[present,"output"].sum()/rows.loc[present,"area"].sum()),
                  yield_kg_mu=a.kg, gross_unit_value_2024=a.unit_value, inputs_2024_mu=a.inputs,
                  total_days_mu=a.days, marketed_share=a.sold, hire_complete_positive=int(hired.sum()),
                  hired_wage_p50_2024=float((x.loc[hired,"hired_cost"]*x.loc[hired,"factor"]
                                            /x.loc[hired,"hired_days"]).median()) if hired.any() else None)
    return a, detail


def estimate(project, fit_year):
    d = project/HOUSEHOLD
    cpi = pd.read_csv(project/CPI).set_index("year").annual_change_percent
    factor = {year:prod(1+cpi.loc[y]/100 for y in range(year+1,2025)) for year in (fit_year,2016,2017)}
    q = (d/FORM).read_text(encoding="utf-8")
    if not all(x in q for x in ("小麦 稻谷 玉米", "蔬菜", "出售蔬菜金额", "投工量")):
        raise ValueError("Expected NH3/NH5 questionnaire items not found")
    labels = pd.read_excel(d/DICT_EARLY,header=None,usecols=[0,1])
    fields = []
    for _, cn, prefix in GRAINS:
        for kind, suffix in MEASUREMENT.items():
            fields.append(_match(labels,f"{kind}_{prefix}",cn,suffix,[],DICT_EARLY))
    key = ["year","hhid","id","smgen"]
    raw = pd.read_csv(d/EARLY,usecols=key+fields,dtype={"hhid":"string","id":"string"},encoding="utf-8")
    link = pd.DataFrame(json.loads((project/LINK).read_bytes()))
    observed_ids = raw.iloc[link.source_row.to_numpy()-2].hhid
    source = raw.loc[raw.year.eq(fit_year) & raw.hhid.notna() & raw.hhid.eq(raw.id)
                     & raw.smgen.notna() & ~raw.hhid.isin(observed_ids)]
    source = source.loc[~source.duplicated(["hhid"],keep=False)]
    mapping, parameters, activities = [], [], []
    for en, cn, prefix in GRAINS:
        columns = {k:_match(labels,f"{k}_{prefix}",cn,v,mapping,DICT_EARLY) for k,v in MEASUREMENT.items()}
        rename = {"sow":"area","prod":"output","inc":"gross","labor_d":"labor",
                  "labor_hd":"hired_days","labor_hf":"hired_cost"}
        rows = source[["year","smgen",*columns.values()]].rename(
            columns={v:rename.get(k,k) for k,v in columns.items()}|{"smgen":"market"})
        activity, details = _measures(rows,list(COST),factor,str(fit_year),en)
        details.update(marketed_basis="all-sale structural ceiling: sale fields unavailable in early dictionary",
                       unit_value_basis="reported gross valuation divided by output; not observed selling price")
        activities.append(activity)
        parameters.append(details)
    labels_late = pd.read_excel(d/DICT_LATE,header=None,usecols=[0,1])
    late = dict(area=("d50","蔬菜","播种面积(亩)"), output=("d86","蔬菜","总产量(千克)"),
                gross=("d106","蔬菜","总收入(元)"), labor=("d472","蔬菜","投工量(日)"),
                hired_days=("d489","蔬菜","雇工(日)"),hired_cost=("d506","蔬菜","雇工费用(元)"),
                sales_qty=("e33","十一","出售蔬菜数量"),sales_amount=("e34","出售","出售蔬菜金额"))
    late.update({k:(v,"蔬菜",COST[k]) for k,v in zip(COST,("d139","d237","d270","d303","d320",
                "d354","d371","d405"),strict=True)})
    columns = {_match(labels_late,*entry,mapping,DICT_LATE):k for k,entry in late.items()}
    veg = pd.read_csv(d/LATE,usecols=["year","sm","cm","hm",*columns]).rename(columns=columns|{"sm":"market"})
    keys = ["year","market","cm","hm"]
    veg = veg.loc[veg[keys].notna().all(axis=1)&~veg.duplicated(keys,keep=False)]
    joint_sale = (veg.area.gt(0)&veg.output.gt(0)&veg.sales_qty.ge(0)&veg.sales_qty.le(veg.output)
                  &veg.sales_amount.ge(0)&veg.sales_qty.gt(0).eq(veg.sales_amount.gt(0)))
    x = veg.loc[joint_sale].copy()
    fraction = float(x.sales_qty.sum()/x.output.sum())
    observed_price = float((x.sales_amount*x.year.map(factor)).sum()/x.sales_qty.sum())
    activity, details = _measures(veg,list(COST),factor,"2016-2017","vegetables",sold=fraction)
    # Keep the gross-value budget and market realization separate. Equal
    # per-kg value for retained and sold vegetables is an explicit assumption
    # for the objective; the actual sales ratio uses the marketing subsample.
    details.update(marketed_basis="later-year named vegetable quantity sold / output, valid same row",
                   unit_value_basis="reported gross valuation divided by output; not observed selling price",
                   marketing_rows=len(x), price_2024_yuan_per_kg=observed_price)
    activities.append(replace(activity, unit_value=observed_price))
    parameters.append(details)
    inputs = [HOUSEHOLD/f for f in (EARLY,LATE,DICT_EARLY,DICT_LATE,FORM)]+[LINK,CPI]
    return EconomicCore(tuple(activities)), parameters, mapping, {str(p):digest(project/p) for p in inputs}
