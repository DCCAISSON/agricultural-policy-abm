"""Historical household succession replay; policy learning is gated outside this run."""
import argparse
import csv
import gzip
import hashlib
import io
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np

from core.empirical_succession import SyntheticLandLedger
from core.v4_household import DIRECTIONS
from scripts import prepare_household_portfolio_reference as reference
from scripts import prepare_nfps_v3 as preparation
from scripts import run_national_household_portfolio as portfolio


F1_COEFFICIENTS_SHA = "548e79936010a9f5607e77f5813b307f551e483ca4580e304adb0f9f296f6ae0"


def development_intensity(frame: pd.DataFrame, excluded: set[str], regions: dict) -> tuple[dict,pd.DataFrame]:
    """Predict combined annual sown intensity, not the number of seasons."""
    early=frame.loc[frame.year.isin([2009,2010]) & ~frame.hhid.isin(excluded)].copy()
    early=early.loc[early.area_b.ge(0)&early.area_e.gt(0)&early.sow_food.ge(0)&early.sow_cash.ge(0)]
    early=early.assign(macro=early.smgen.map(regions),band=np.digitize(early.area_b,[5.,15.,30.]),
                       intensity=(early.sow_food+early.sow_cash)/early.area_e)
    early=early.loc[early.intensity.gt(0)&np.isfinite(early.intensity)&early.macro.notna()]
    groups=[("pooled",None,early)]
    for region in sorted(set(regions.values())):
        local=early.loc[early.macro.eq(region)]
        for band in range(4):
            cell=local.loc[local.band.eq(band)]
            groups.append((region,band,cell if len(cell)>=30 else local if len(local)>=30 else early))
    profile,receipt={},[]
    for scope,band,donors in groups:
        value=float(donors.intensity.median())
        if not np.isfinite(value) or value<=0:
            raise ValueError("No comparable early noncohort sowing intensity")
        profile[scope,band]=value
        receipt.append(dict(scope=scope,band=band,n=len(donors),n_households=donors.hhid.nunique(),
                            first_year=int(donors.year.min()),last_year=int(donors.year.max()),intensity=value))
    return profile,pd.DataFrame(receipt)


def use_frozen_fit(fitted, current, locked):
    """Replay the published fit; re-estimation is not numerical serialization."""
    keys = ["training_end","rule","action","term"]
    data = ["n","households","action_n"]
    pd.testing.assert_frame_equal(current[keys+data],locked[keys+data])
    coefficients = locked.set_index(keys)["coefficient"]
    for year, rule, action in locked[keys[:3]].drop_duplicates().itertuples(index=False,name=None):
        values = coefficients.loc[year,rule,action].to_numpy()
        if rule.startswith("direction_"):
            category = ("cash","mixed","grain").index(rule.removeprefix("direction_"))
            fitted[year]["direction_by_prior"][category,DIRECTIONS.index(action)] = values
        else:
            positive = action in ("positive","sowing_two")
            fitted[year][rule][int(positive)] = values


def run(input_root: Path, output: Path, seeds: list[int], variant: str="B0", one_step: bool=False) -> None:
    """Read authorized inputs without copying into the checkout or changing their values."""
    if any(os.environ.get(k) != "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")):
        raise ValueError("Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before Python starts, as in the F1 runner")
    diagram = input_root / "results/diagnostics"
    base = diagram / "national_formal_revision_20261118/run_01"
    manifest = base / "prepared_seed17_release1/preparation_manifest.json"
    raw = input_root / reference.RAW.relative_to(reference.ROOT)
    coefficient_file = (diagram/"national_abm_v41_final_economic_closure_20261124"/
                        "PRIVATE_DO_NOT_SHARE/F1_301/behavior_parameters.csv")
    if hashlib.sha256(coefficient_file.read_bytes()).hexdigest()!=F1_COEFFICIENTS_SHA:
        raise ValueError("F1 archived behavior coefficients do not match the recorded fingerprint")
    locked = pd.read_csv(coefficient_file,float_precision="round_trip")
    prepare_inputs = preparation.prepare_v4_inputs
    summary = []
    ledger = None

    def prepare_with_frozen_fit(cohort, dest, *args, **kwargs):
        fitted, profiles = prepare_inputs(cohort,dest,*args,**kwargs)
        parameters = dest/"behavior_parameters.csv"
        use_frozen_fit(fitted,pd.read_csv(parameters),locked)
        parameters.rename(dest/"refit_for_comparison_only.csv")
        shutil.copyfile(coefficient_file,parameters)
        if "I" in variant:
            from core.extract_microdata_features import REGION_MAPPING
            columns="year hhid id smgen area_b area_e sow_food sow_cash".split()
            panel=pd.read_csv(raw,usecols=columns,dtype={"hhid":"string","id":"string"})
            selected=set(panel.iloc[cohort.source_row.to_numpy()-2].hhid)
            panel=panel.loc[panel.hhid.notna()&panel.hhid.eq(panel.id)&
                            ~panel.duplicated(["year","hhid"],keep=False)]
            regions={p:z for z,provinces in REGION_MAPPING.items() for p in provinces}
            profile, receipt=development_intensity(panel,selected,regions)
            receipt.to_csv(dest/"intensity_development_fit.csv",index=False)
            for model in fitted.values():
                model["development_intensity"]=profile
        return fitted,profiles

    def observe(seed, year, baseline, model, private):
        nonlocal ledger
        if year == 2011:
            ledger = SyntheticLandLedger(baseline.to_dict("records"))
        records, aggregates = ledger.advance(year, model.area_last[0],
            model.area_economy["contracts"], model.area_external_last, model.area_economy["budget"])
        file = private/f"synthetic_units_{seed}.csv.gz"
        with file.open("ab") as binary, gzip.GzipFile(
            fileobj=binary,mode="wb",mtime=0,filename=""
        ) as compressor, io.TextIOWrapper(compressor,encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(records[0]))
            if year == 2011:
                writer.writeheader()
            writer.writerows(records)
        summary.append({"seed":seed,"year":year,**aggregates})

    settings = dict(mode="V4",seeds=seeds,repeat=False,scalar=False,space=True,rights=False,
                    organizations=60,capital=1.,incentive=1.,budget="frozen",ablation=None,
                    repair=True,scale="F1",grain=0.,cash=0.,wage=0.,rent=0.,cash_funds=0.,stage=4)
    if variant!="B0":
        settings["development_variant"]=variant
    if one_step:
        settings["development_opening_check"]=True
    with (
        patch.multiple(reference,RAW=raw,MANIFEST=manifest),
        patch.multiple(preparation,RAW=raw,MANIFEST=manifest,
                       WEIGHTS=input_root/preparation.WEIGHTS.relative_to(preparation.ROOT)),
        patch.object(preparation,"prepare_v4_inputs",prepare_with_frozen_fit),
        patch.object(portfolio.old,"ROOT",input_root),
        patch.object(portfolio.legacy,"BASE",base),
    ):
        portfolio.run_area_structures(output,"D",settings=settings,observer=observe)
    pd.DataFrame(summary).to_csv(output/"succession_summary.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--seeds",nargs="+",type=int,default=list(range(301,321)))
    parser.add_argument("--variant",choices=("B0","I","C","IC"),default="B0")
    parser.add_argument("--one-step",action="store_true")
    args = parser.parse_args()
    run(args.input_root.resolve(),args.output.resolve(),args.seeds,args.variant,args.one_step)
