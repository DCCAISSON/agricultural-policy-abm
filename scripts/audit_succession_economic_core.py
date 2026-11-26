"""Rebuild the report and check the aggregate release against a clean replay."""
import argparse
from pathlib import Path
import json

import pandas as pd

from scripts.prepare_succession_budget_inputs import digest
from scripts.run_succession_annual_allocation import json_write


def compare(first,second):
    a = {p.name:digest(p) for p in first.iterdir() if p.is_file()}
    b = {p.name:digest(p) for p in second.iterdir() if p.is_file()}
    if a!=b:
        raise ValueError(f"Different clean replay outputs: {sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))}")
    return a


def audit(main,replay,resources,replayed_resources,out):
    files = compare(main,replay)
    resource_files = compare(resources,replayed_resources)
    counts = json.loads((main/"receipt.json").read_bytes())
    history = pd.read_csv(main/"household_history.csv")
    regions = pd.read_csv(main/"region_year.csv")
    keys = ["arm","seed","year"]
    assert not history.duplicated(keys).any() and not regions.duplicated([*keys,"region"]).any()
    assert (regions.groupby(keys).size()==4).all() and len(history)==60 and len(regions)==counts["region_years"]==240
    assert counts["episode_count"]==12 and counts["household_years"]==36_000 and counts["plot_years"]==132_600
    assert not counts["RL_trained"] and not counts["policy_ranked"]
    assert regions[["spending","committed","budget"]].eq(0).all().all()
    assert (regions.groupby(keys).audited_actors.sum()==600).all()
    assert history.groupby("year")[[c for c in history if c=="comparable" or c.startswith("survey_")]].nunique().eq(1).all().all()
    assert history.sim_sown.eq(history.sim_pure_grain+history.sim_pure_nongrain+history.sim_mixed).all()
    assert (history.sim_sown<=history.comparable).all() and (history.survey_sown<=history.comparable).all()
    metrics = ["comparable","survey_sown","sim_sown","survey_mixed","sim_mixed",
               "survey_operated_mu","sim_planted_on_land_reporting_mu","survey_grain_mu_seasons",
               "sim_grain_physical_mu","survey_all_grain_kg","sim_report_scope_grain_kg"]
    summary = history.groupby(["year","arm"],sort=False)[metrics].mean().reset_index()
    assert not out.exists()
    out.mkdir(parents=True,exist_ok=False)
    summary.to_csv(out/"annual_means.csv",index=False,lineterminator="\n")
    json_write(out/"verification.json",dict(path_rows=len(history),regional_annual_rows=len(regions),
        matching_main_artifacts=files,matching_resource_artifacts=resource_files,RL_trained=False,policy_ranked=False,
        scope="same seed/program deterministic check, not independent empirical validation"))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("main","replay","resources","replayed-resources","out"):
        parser.add_argument(f"--{arg}",required=True,type=Path)
    a = parser.parse_args()
    audit(a.main,a.replay,a.resources,a.replayed_resources,a.out)
