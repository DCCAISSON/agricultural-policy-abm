"""Bind the frozen F1 survey cohort to the active 7a economy without exporting it."""

from dataclasses import dataclass
import json
from math import ceil, fsum
from pathlib import Path

import numpy as np
import pandas as pd

from core.land_succession import Operator, Plot


@dataclass(frozen=True, slots=True)
class RegionAssets:
    name: str
    households: tuple[Operator, ...]
    plots: tuple[Plot, ...]
    grain_kg: float

    @property
    def area(self) -> float:
        return fsum(p.area for p in self.plots)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        from hashlib import file_digest
        return file_digest(stream,"sha256").hexdigest()


def split_lots(owner: str, area: float, right: float, grain: float, cash: float, total: float) -> list[Plot]:
    """Preserve physical area and permission, not hypothetical survey parcel IDs."""
    denom = max(total, grain+cash)
    shares = ((grain/denom, cash/denom) if denom>0 else (0.,0.))
    plots = []
    for land, tradable in ((right,True), (area-right,False)):
        for crop, fraction in zip(("grain","cash","idle"), (*shares,max(0.,1.-sum(shares))), strict=True):
            size = land*fraction
            if size <= 1e-10:
                continue
            n = ceil(size/4.)
            for _ in range(n):
                plots.append(Plot(f"{owner}P{len(plots):03}", owner, 1., crop, size/n, tradable))
    return plots


def prepare(project: Path, design: dict) -> tuple[tuple[RegionAssets, ...], dict]:
    diagnosis = project/"results/diagnostics"
    baseline = diagnosis/"national_empirical_succession_20261126/locked_301/PRIVATE_DO_NOT_SHARE/baseline.csv"
    link = diagnosis/"national_crop_choice_soft_response_20261123/PRIVATE_DO_NOT_SHARE/cohort600.json"
    folder = project/"校准用数据源/全国农村固定观察点"
    raw_path = folder/"2009-2015household.csv"
    signatures = {
        "household_baseline_sha256":digest(baseline), "cohort_link_sha256":digest(link),
        "nfps_2009_2015_sha256":digest(raw_path),
    }
    if any(signatures[k] != design[k] for k in signatures):
        raise ValueError("Input is not the frozen household/cohort source")
    labels = pd.read_excel(folder/"household variables2009-2015.xlsx",header=None,usecols=[0,1])
    fields = {"hhlabornum","area_e","area_e1","sow_total","sow_food","sow_cash","prod_food",
              "inc_wage_o","inc_wage_l","saving_1","saving_2"}
    if not fields.issubset(set(labels[0])):
        raise ValueError("Frozen data dictionary does not identify all required fields")
    b = pd.read_csv(baseline,dtype={"actor":"string"})
    cohort = pd.DataFrame(json.loads(link.read_bytes()))
    columns = ["hhid","id","year","smgen","hhnumbers","hhlabornum",*sorted(fields-{"hhlabornum"})]
    raw = pd.read_csv(raw_path,usecols=columns,dtype={"hhid":"string","id":"string"})
    sel = raw.iloc[cohort.source_row.to_numpy()-2].reset_index(drop=True)
    if not (len(b)==len(cohort)==len(sel)==600 and cohort.source_row.is_unique and b.actor.is_unique
            and sel.hhid.is_unique and sel.hhid.notna().all() and sel.hhid.eq(sel.id).all()
            and sel.year.eq(2010).all()
            and b.actor.eq(cohort.cohort_agent_id.astype("string")).all()
            and b.region.eq(cohort.region).all() and sel.hhnumbers.eq(cohort.family_size).all()):
        raise ValueError("Frozen cohort identity/year does not reconcile")
    for col in ("smgen","hhnumbers",*sorted(fields-{"hhlabornum"})):
        if not np.allclose(b[col],sel[col],rtol=1e-12,atol=1e-9,equal_nan=True):
            raise ValueError(f"Frozen baseline differs from NFPS in {col}")
    province_region = b[["smgen","region"]].drop_duplicates().set_index("smgen").region
    if not province_region.index.is_unique:
        raise ValueError("Frozen province mapping is ambiguous")
    pre = raw.loc[raw.year.eq(2010),["smgen","hhnumbers","hhlabornum"]].copy()
    pre["region"] = pre.smgen.map(province_region)
    valid = pre.hhlabornum.between(0,pre.hhnumbers) & pre.hhlabornum.mod(1).eq(0)
    proxy = pre[valid].groupby(["region","hhnumbers"]).hhlabornum.quantile(.5,interpolation="lower")
    count_proxy = 0
    assets, coverage = [], []
    for name in design["region_order"]:
        actors, parcels, output = [], [], 0.
        sample = b[b.region.eq(name)]
        for row in sample.itertuples():
            obs = sel.iloc[row.Index]
            n = obs.hhlabornum
            if not 0<=n<=row.hhnumbers or n%1:
                n = proxy.loc[name,row.hhnumbers]
                count_proxy += 1
            if obs.inc_wage_l>0 or obs.inc_wage_o>0:
                annual = 120.
            elif obs.inc_wage_l==obs.inc_wage_o==0:
                annual = 180.
            else:
                annual = 176.
            labor = n*annual/4.
            off_days = n*(365.-annual)
            wage = row.wage_income/off_days if off_days>0 else 0.
            background = row.wage_income*.75 if off_days>0 else row.wage_income
            owner = f"H{row.Index:04}"
            liquid = max(0.,row.opening_cash-row.held_reserve)
            actors.append(Operator(owner,False,liquid,labor,wage,off_days/4.,
                                   int(row.smgen),row.consumption,background))
            parcels.extend(split_lots(owner,row.area_e,row.verified_right,row.sow_food,row.sow_cash,row.sow_total))
            output += row.prod_food
        region = RegionAssets(name,tuple(actors),tuple(parcels),output)
        if not np.isclose(region.area,sample.area_e.sum(),rtol=0,atol=1e-7) or not np.isclose(
            fsum(p.area for p in parcels if p.tradable),sample.verified_right.sum(),rtol=0,atol=1e-7
        ):
            raise ValueError("Synthetic lots change household-region area or verified rights")
        assets.append(region)
        coverage.append(dict(region=name,households=len(actors),provinces=sample.smgen.nunique(),
                             physical_mu=region.area,verified_mu=fsum(p.area for p in parcels if p.tradable),
                             synthetic_lots=len(parcels),observed_initial_grain_kg=output,
                             under_reserve=int((sample.opening_cash<sample.held_reserve).sum()),
                             missing_right=int(sample.area_e1.isna().sum()),
                             proxy_savings=int((sample.saving_1_proxy|sample.saving_2_proxy).sum()),
                             proxy_wages=int((sample.inc_wage_l_proxy|sample.inc_wage_o_proxy).sum())))
    if sum(x["households"] for x in coverage)!=600:
        raise ValueError("At least one region was lost")
    return tuple(assets),dict(**signatures,family_labor_count_proxies=count_proxy,regions=coverage,
                              dictionary_sha256=digest(folder/"household variables2009-2015.xlsx"))
