"""Build the NFPS sowing-area reference for the frozen job-8 households."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "results/diagnostics"
OUT = DIAG / "national_household_portfolio/reference"
RAW = ROOT / "校准用数据源/全国农村固定观察点/2009-2015household.csv"
MANIFEST = DIAG / "national_formal_revision_20261118/run_01/prepared_seed17_release1/preparation_manifest.json"


def prepare():
    job = json.loads(MANIFEST.read_text(encoding="utf-8"))["jobs"][8]
    cohort = Path(job["cohort"]).read_bytes()
    assert hashlib.sha256(cohort).hexdigest() == job["cohort_sha256"]
    farmers = pd.DataFrame(json.loads(cohort)["farmers"])
    cols = ["year", "hhid", "id", "hhnumbers", "sow_food", "sow_cash", "sow_total", "area_e"]
    raw = pd.read_csv(RAW, usecols=cols, dtype={"hhid": "string", "id": "string"},
                      keep_default_na=False, na_values=[""])
    origin = raw.iloc[farmers.source_row.to_numpy() - 2].reset_index(drop=True).copy()
    assert len(origin) == origin.hhid.nunique() == farmers.source_row.nunique() == farmers.cohort_agent_id.nunique() == 600
    assert origin.year.eq(2010).all() and origin.hhnumbers.eq(farmers.family_size).all()
    origin["actor_id"] = farmers.cohort_agent_id
    base = raw.loc[raw.year.eq(2010), "hhid"]
    ambiguous = base[base.duplicated(keep=False)]
    keys = origin.loc[~origin.hhid.isin(ambiguous), ["actor_id", "hhid", "id"]].dropna()
    references = []
    for year in (2010, 2011, 2012):
        if year == 2010:
            rows = origin
        else:
            current = raw.loc[raw.year.eq(year)].dropna(subset=["hhid", "id"])
            current = current.loc[~current.hhid.duplicated(keep=False)]
            rows = keys.merge(current, on=["hhid", "id"], validate="one_to_one")
        rows = rows.rename(columns={"sow_food": "grain_area", "sow_cash": "cash_area"}).copy()
        g, c = rows.grain_area, rows.cash_area
        for a in (g, c):
            assert (np.isfinite(a.dropna()) & a.dropna().ge(0)).all()
        rows["grain_share"] = g / (g + c).where((g + c).gt(0))
        physical = rows.area_e.notna() & rows.sow_total.notna()
        print(year, "matched, ambiguous_base, other_unmatched", len(rows),
              0 if year == 2010 else 600-len(keys), 0 if year == 2010 else len(keys)-len(rows),
              "grain_only, cash_only, both, neither, missing",
              *((g.gt(0) & c.eq(0)).sum(), (g.eq(0) & c.gt(0)).sum(),
                (g.gt(0) & c.gt(0)).sum(), (g.eq(0) & c.eq(0)).sum(), (g.isna() | c.isna()).sum()),
              "missing_grain, missing_cash", g.isna().sum(), c.isna().sum(),
              "share_valid", rows.grain_share.count(), "sown_gt_yearend, comparable",
              (physical & (rows.sow_total - rows.area_e).gt(0.00001)).sum(), physical.sum())
        references.append(rows[["year", "actor_id", "grain_area", "cash_area", "grain_share"]])
    ref = pd.concat(references, ignore_index=True).sort_values(["year", "actor_id"])
    assert not ref.duplicated(["year", "actor_id"]).any()
    valid = ref.grain_share.dropna()
    assert valid.between(0, 1).all()
    shares = ref.loc[ref.year.eq(2010)].set_index("actor_id").grain_share.dropna().to_dict()
    files = {"initial_grain_shares.json": json.dumps(shares, indent=2, allow_nan=False) + "\n",
             "reference_households.csv": ref.to_csv(index=False, lineterminator="\n")}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = OUT / name
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise FileExistsError(f"Refusing to replace changed output: {path}")
    for name, content in files.items():
        path = OUT / name
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="") as f:
                f.write(content)


if __name__ == "__main__":
    prepare()
