"""Check early NFPS source fields and linkage without opening a candidate holdout."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import prepare_household_portfolio_reference as ref
from scripts import prepare_nfps_v3 as prep
from scripts.verify_empirical_succession import ROOT


FIELDS="area_b area_e area_b1 area_e1 area_in area_in1 area_de area_de2 sow_total sow_food sow_cash sow_oother".split()


def audit(project: Path, output: Path) -> None:
    protocol=json.loads((ROOT/"docs/全国经营接续/validation_protocol.json").read_bytes())
    raw=project/ref.RAW.relative_to(ref.ROOT)
    with raw.open("rb") as file:
        fingerprint=hashlib.file_digest(file,"sha256").hexdigest()
    if fingerprint!=protocol["authorized_household_sha256"]:
        raise ValueError("NFPS source fingerprint changed")
    w=project/prep.WEIGHTS.relative_to(prep.ROOT)
    manifest=(project/"results/diagnostics/national_formal_revision_20261118/run_01/"
              "prepared_seed17_release1/preparation_manifest.json")
    with patch.multiple(prep,RAW=raw,WEIGHTS=w,MANIFEST=manifest):
        cohort=prep.candidate_cohort()
    cols=["year","hhid","id","smgen",*FIELDS]
    frame=pd.read_csv(raw,usecols=cols,dtype={"hhid":"string","id":"string"})
    original=frame.iloc[cohort.source_row.to_numpy()-2][["hhid","id","smgen"]]
    dictionary=pd.read_excel(raw.parent/"household variables2009-2015.xlsx",header=None).set_index(0)
    survey=(raw.parent/"2009-2017农户调查问卷.txt").read_text(encoding="utf-8-sig")
    dictionary_titles={"area_in1":("转包入","NH2-7"),"area_de2":("转包出","NH2-10")}
    for k,(meaning,question) in dictionary_titles.items():
        assert meaning in dictionary.loc[k,1] and any(
            meaning in line and question in line and "亩" in line for line in survey.splitlines())
    normalized=frame.copy()
    prep.clean_area(normalized)
    fields,flow=[],[]
    for year in range(2010,2016):
        yraw=frame.loc[frame.year.eq(year)&frame.hhid.isin(original.hhid)]
        yclean=normalized.loc[yraw.index]
        duplicates=set(yraw.loc[yraw.hhid.duplicated(False),"hhid"])
        joint=original.merge(yclean.loc[~yclean.hhid.isin(duplicates)],on=["hhid","id","smgen"],validate="1:1")
        open_close=joint.area_b.ge(0)&joint.area_e.ge(0)
        valid_crops=joint.sow_food.ge(0)&joint.sow_cash.ge(0)
        joint_flows=joint.area_in1.ge(0)&joint.area_de2.ge(0)
        flow.append(dict(year=year,fixed_cohort=len(original),
            absent_panel_key=len(set(original.hhid)-set(yraw.hhid)),duplicate_panel_key=len(duplicates),
            identity_or_province_unconfirmed=len(set(yraw.hhid)-duplicates-set(joint.hhid)),
            linked=len(joint),crop_joint=int(valid_crops.sum()),stock_nonnegative=int(open_close.sum()),
            stock_end_positive=int((open_close&joint.area_e.gt(0)).sum()),
            stock_end_zero=int((open_close&joint.area_e.eq(0)).sum()),
            flow_dual_reports=int(joint_flows.sum())))
        for k in FIELDS:
            s=joint[k]
            fields.append(dict(year=year,field=k,dictionary_label=dictionary.loc[k,1],unit="mu_times" if
                k.startswith("sow_") else "physical_mu",source_records=len(yraw),linked=len(joint),
                raw_zero_before_key_filter=int(yraw[k].eq(0).sum()),
                raw_blank_before_key_filter=int(yraw[k].isna().sum()),
                negative_codes_linked=int(s.lt(0).sum()),measured_zero_linked=int(s.eq(0).sum()),
                missing_or_corrected_linked=int(s.isna().sum()),positive_linked=int(s.gt(0).sum()),
                confirmed_field_errors_removed=int((yraw[k].notna()&yclean[k].isna()).sum())))
    output.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(flow).to_csv(output/"cohort_flow.csv",index=False)
    pd.DataFrame(fields).to_csv(output/"field_manifest.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("project",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    audit(a.project,a.output)
