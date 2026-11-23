import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_national_crop_budget_basis_20261118 import COMPONENTS, LABELS, OUT, derive_budget, pair_budget


def fixture():
    d = pd.DataFrame({name:[0., 0., 0., 0.] for name in LABELS})
    d["area"] = [2., 1., 0., 1.]
    d["income"] = [1000., -100., 0., 100.]
    d["output"] = [100., 10., 0., 10.]
    d["seed"] = [100., np.nan, 0., 10.]
    d["manure"] = [20., 0., 0., 0.]
    d["depreciation_repair"] = [30., 0., 0., 0.]
    d["rent"] = [40., 0., 0., 0.]
    d["hired_cost"] = [60., 0., 0., 10.]
    d["total_cost"] = [250., 50., 0., 20.]
    d["labor"] = [10., 2., 0., 1.]
    d["hired_days"] = [2., np.nan, 0., 2.]
    d["sales_amount"] = [500., 0., 0., 0.]
    d["sales_quantity"] = [50., 0., 0., 0.]
    d["year"], d["sm"], d["cm"], d["hm"] = 2016, 1, 1, [1,2,3,4]
    d["source_row"] = [2,3,4,5]
    d["crop"], d["duplicate_household_year"] = "left", False
    return d


def test_no_double_deduction_or_missing_fill():
    d = fixture()
    original = d.copy(deep=True)
    b = derive_budget(d)
    assert b.loc[0,"components_sum"] == 250
    assert b.loc[0,"reported_margin"] == 750
    assert b.loc[0,"family_days"] == 8
    assert b.loc[0,"family_charge_at_own_hired_wage"] == 240
    assert b.loc[0,"margin_at_own_hired_wage_conditional"] == 510
    assert b.loc[0,"reported_margin_before_rent"] == 790
    assert b.loc[0,"cash_cost_low_conditional"] == 200
    assert b.loc[0,"cash_cost_high_conditional"] == 230
    assert pd.isna(b.loc[1,"components_sum"])
    assert pd.isna(b.loc[1,"family_days"])
    assert b.loc[1,"reported_margin"] == -150
    assert pd.isna(b.loc[2,"reported_margin_per_mu"])
    assert b.loc[3,"family_days"] == -1
    assert not b.loc[3,"labor_consistent"]
    assert pd.isna(b.loc[3,"own_hired_wage_reference"])
    pd.testing.assert_frame_equal(d, original)


def test_same_record_pair_identity_and_joint_decomposition():
    d = fixture()
    left = derive_budget(d)
    d["crop"] = "right"
    d["income"] += 200
    d["total_cost"] += 50
    d["labor"] += 2
    right = derive_budget(d)
    p = pair_budget(left, right)
    assert p.loc[0,"delta_reported_margin_per_mu"] == 75
    assert p.loc[0,"delta_income_per_mu"] == 100
    assert p.loc[0,"delta_total_cost_per_mu"] == 25
    assert p.loc[0,"delta_family_days_per_mu"] == 1
    assert p.loc[0,"common_wage_switch_threshold"] == 75
    assert not p.loc[1,"joint_family"]
    assert len(p) == len(d)


def test_delivered_rows_and_source_integrity():
    manifest = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert manifest["raw_rows"] == 12252
    assert manifest["budget_rows"] == manifest["pair_rows"] == 36756
    for path, sha in manifest["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    rows = pd.read_csv(OUT / "joint_crop_budgets.csv")
    assert not rows.duplicated(["source_row", "crop"]).any()
    assert rows.loc[~rows.components_complete,"components_sum"].isna().all()
    assert rows.loc[rows.hired_days.isna(),"family_days"].isna().all()
    cov = pd.read_csv(OUT / "field_coverage.csv")
    assert (cov.missing + cov.zero + cov.negative + cov.positive == cov.raw_n).all()
