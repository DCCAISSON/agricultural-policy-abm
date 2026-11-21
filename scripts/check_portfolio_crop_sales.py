"""Check the configured budget sample against the original survey, without imputing sales."""
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


if __name__ == "__main__":
    main()
