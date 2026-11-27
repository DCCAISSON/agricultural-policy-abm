"""Strictly pre-2011 (2009-2010 NFPS + CFPS 2010 + CHFS 2011) calibration for Mesa Agri-ABM V2.

Key upgrades over v1:
1. Named grain crop portfolios (wheat, rice, maize, soybean, other_grain) with empirical regional yields,
   farmgate prices, non-machinery material inputs, and observed custom machinery service fees (`mach_*`).
2. Direct household-level 2010 subsistence staple demand (`foodc`, 596/600 observed) for MP-MAS
   in-kind food consumption vs. marketed grain cash revenue accounting.
3. Observed 2009-2010 baseline fertility/direct subsidy rate (`inc_govern_3/4/5`) per region.
4. Three differentiated synthetic scale organization specs (Family Farm, Cooperative, Agri-Company)
   with zero arbitrary yield advantage and strictly segregated non-survey accounting.
"""

from dataclasses import dataclass
from hashlib import file_digest
import json
from math import ceil, fsum
from pathlib import Path

import numpy as np
import pandas as pd

from mesa_agri_abm.config import V2Design

NAMED_GRAINS: tuple[str, ...] = ("wheat", "rice", "maize", "soybean", "other_grain")
NFPS_GRAIN_TAG_MAP: dict[str, tuple[str, ...]] = {
    "wheat": ("w",),
    "rice": ("r",),
    "maize": ("c",),
    "soybean": ("s",),
    "other_grain": ("t", "o"),
}


@dataclass(frozen=True, slots=True)
class NamedGrainTech:
    crop: str
    yield_kg_sown_mu: float
    price_yuan_kg: float
    non_mach_input_yuan_sown_mu: float
    mach_service_yuan_sown_mu: float
    labor_days_sown_mu: float


@dataclass(frozen=True, slots=True)
class RegionTechV2:
    region: str
    named_grains: dict[str, NamedGrainTech]
    default_grain_weights: dict[str, float]
    grain_yield_kg_sown_mu: float
    grain_price_yuan_kg: float
    grain_non_mach_input_yuan_sown_mu: float
    grain_mach_service_yuan_sown_mu: float
    grain_labor_days_sown_mu: float
    cash_rev_yuan_sown_mu: float
    cash_non_mach_input_yuan_sown_mu: float
    cash_mach_service_yuan_sown_mu: float
    cash_labor_days_sown_mu: float
    hired_wage_yuan_day: float
    ask_rent_yuan_phys_mu: float
    fertility_base_yuan_phys_mu: float
    max_mci: float
    baseline_grain_share: float
    baseline_margin_gap: float
    turnover_pool_share: float
    scale_pool_share: float
    village_net_inflow_rate: float


@dataclass(frozen=True, slots=True)
class ActorBlueprint:
    id: str
    region: str
    province: int
    submarket: int
    actor_category: str           # "survey_household" | "family_farm" | "cooperative" | "agri_company"
    is_survey_household: bool
    hh_members: float
    workers: float
    ag_days_year: float
    slack_ag_days_year: float
    offfarm_wage_day: float
    offfarm_annual_base: float
    savings_stock: float
    working_capital_init: float
    consumption_cash_base: float
    subsistence_grain_kg: float   # MP-MAS annual staple requirement from 2010 NFPS `foodc`
    mci: float
    grain_weights: tuple[tuple[str, float], ...]
    grain_yield_kg_sown_mu: float
    grain_price_yuan_kg: float
    grain_non_mach_yuan_sown_mu: float
    grain_mach_yuan_sown_mu: float
    grain_labor_days_sown_mu: float
    ref_grain_share: float
    initial_cultivating: bool
    own_physical_mu: float
    verified_tradable_mu: float
    admin_overhead_yuan: float
    service_coord_saving: float


@dataclass(frozen=True, slots=True)
class ParcelBlueprint:
    id: str
    offer_group_id: str           # Contract offer group so row-splitting never multiplies setup fees
    owner: str
    region: str
    province: int
    submarket: int
    area: float
    tradable: bool
    quality: float
    initial_crop: str
    grid_x: int                   # Synthetic inspection coordinate (explicitly non-GIS)
    grid_y: int
    is_synthetic_coord: bool = True


@dataclass(frozen=True, slots=True)
class CalibratedRegionV2:
    name: str
    tech: RegionTechV2
    households: tuple[ActorBlueprint, ...]
    organizations: tuple[ActorBlueprint, ...]
    parcels: tuple[ParcelBlueprint, ...]
    initial_grain_kg: float

    @property
    def physical_area(self) -> float:
        return fsum(p.area for p in self.parcels)

    @property
    def verified_tradable_area(self) -> float:
        return fsum(p.area for p in self.parcels if p.tradable)


def sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return file_digest(fh, "sha256").hexdigest()


def resolve_project_data_root(hint: Path | None = None) -> Path:
    start = (hint or Path(__file__)).resolve()
    for cand in (start, *start.parents):
        if (cand / "校准用数据源").exists() and (cand / "results/diagnostics").exists():
            return cand
    raise FileNotFoundError(f"Could not locate project data root above {start}")


def _split_parcels_v2(
    owner: str,
    region: str,
    province: int,
    submarket: int,
    area_e: float,
    verified_right: float,
    sow_food: float,
    sow_cash: float,
    sow_total: float,
    quality: float,
    start_idx: int,
    grid_cols: int = 28,
) -> list[ParcelBlueprint]:
    denom = max(sow_total, sow_food + sow_cash)
    g_sh, c_sh = (sow_food / denom, sow_cash / denom) if denom > 0 else (0.0, 0.0)
    i_sh = max(0.0, 1.0 - g_sh - c_sh)
    out: list[ParcelBlueprint] = []
    offer_id = f"{owner}_OFFER"
    for bucket_area, tradable in ((verified_right, True), (max(0.0, area_e - verified_right), False)):
        if bucket_area <= 1e-10:
            continue
        for crop, frac in (("grain", g_sh), ("cash", c_sh), ("idle", i_sh)):
            sz = bucket_area * frac
            if sz <= 1e-10:
                continue
            n_lots = max(1, ceil(sz / 6.0))
            for _ in range(n_lots):
                seq = start_idx + len(out)
                out.append(
                    ParcelBlueprint(
                        id=f"{owner}P{len(out):03d}",
                        offer_group_id=offer_id if tradable else f"{owner}_NONTRAD",
                        owner=owner,
                        region=region,
                        province=province,
                        submarket=submarket,
                        area=sz / n_lots,
                        tradable=tradable,
                        quality=quality,
                        initial_crop=crop,
                        grid_x=seq % grid_cols,
                        grid_y=seq // grid_cols,
                        is_synthetic_coord=True,
                    )
                )
    return out


def load_v2_calibrated_regions(project_root: Path | None, design: V2Design) -> tuple[tuple[CalibratedRegionV2, ...], dict]:
    """Calibrate named crops, subsistence `foodc`, machinery fees, and 3 org types strictly on <=2010 data."""
    project_root = resolve_project_data_root(project_root)
    diag_dir = project_root / "results/diagnostics"
    baseline_path = diag_dir / "national_empirical_succession_20261126/locked_301/PRIVATE_DO_NOT_SHARE/baseline.csv"
    cohort_path = diag_dir / "national_crop_choice_soft_response_20261123/PRIVATE_DO_NOT_SHARE/cohort600.json"
    nfps_dir = project_root / "校准用数据源/全国农村固定观察点"
    raw_path = nfps_dir / "2009-2015household.csv"
    dict_path = nfps_dir / "household variables2009-2015.xlsx"

    sigs = {
        "household_baseline_sha256": sha256_file(baseline_path),
        "cohort_link_sha256": sha256_file(cohort_path),
        "nfps_2009_2015_sha256": sha256_file(raw_path),
        "dictionary_sha256": sha256_file(dict_path),
    }
    if (
        sigs["household_baseline_sha256"] != design.household_baseline_sha256
        or sigs["cohort_link_sha256"] != design.cohort_link_sha256
        or sigs["nfps_2009_2015_sha256"] != design.nfps_2009_2015_sha256
    ):
        raise ValueError("Authorized source SHA-256 mismatch against V2Design")

    b = pd.read_csv(baseline_path, dtype={"actor": "string"})
    cohort = pd.DataFrame(json.loads(cohort_path.read_bytes()))
    if len(b) != 600 or len(cohort) != 600:
        raise ValueError("Cohort size must be exactly 600")

    grain_tags = ("w", "r", "c", "s", "t", "o")
    non_mach_prefixes = ("seed_", "fer_", "film_", "pes_", "weir_", "anim_")
    base_cols = [
        "year", "hhid", "id", "smgen", "hhnumbers", "hhlabornum",
        "area_b", "area_e", "area_e1", "area_in1", "area_de2",
        "sow_total", "sow_food", "prod_food", "sow_cash",
        "inc_wage_l", "inc_wage_o", "saving_1", "saving_2", "inc_rent",
        "inc_govern_3", "inc_govern_4", "inc_govern_5", "foodc", "foodcbuy",
    ]
    crop_cols = [
        f"{p}{t}"
        for t in grain_tags
        for p in ("sow_", "prod_", "inc_", "labor_d_", "labor_hd_", "labor_hf_", "mach_", *non_mach_prefixes)
    ]
    cash_cols = [f"inc_cash_{i}" for i in range(1, 11)]
    use_cols = sorted(set(base_cols + crop_cols + cash_cols))

    raw_all = pd.read_csv(raw_path, usecols=use_cols, dtype={"hhid": "string", "id": "string"}, low_memory=False)
    sel_2010 = raw_all.iloc[cohort["source_row"].to_numpy() - 2].reset_index(drop=True)
    if not (sel_2010["year"].eq(2010).all() and sel_2010["hhid"].eq(sel_2010["id"]).all()):
        raise ValueError("2010 cohort linkage failed reconciliation")

    # HARD GUARDRAIL: Strictly retain 2009-2010 ONLY and discard post-2010 rows immediately!
    df0910 = raw_all.loc[raw_all["year"].isin([2009, 2010])].copy()
    del raw_all
    assert int(df0910["year"].max()) <= 2010, "Post-2010 leakage strictly forbidden in V2 calibration"

    prov_to_region = b[["smgen", "region"]].drop_duplicates().set_index("smgen")["region"].to_dict()
    df0910 = df0910.loc[df0910["smgen"].isin(prov_to_region)].copy()
    df0910["region"] = df0910["smgen"].map(prov_to_region)

    cpi_mult = np.where(df0910["year"].eq(2010), design.cpi_2010_to_2024, design.cpi_2009_to_2024)
    df0910["cpi"] = cpi_mult
    df0910["g_sow"] = df0910[[f"sow_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1)
    df0910["g_prod"] = df0910[[f"prod_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1)
    df0910["g_rev"] = df0910[[f"inc_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1) * cpi_mult
    df0910["g_non_mach"] = sum(df0910[[f"{p}{t}" for t in grain_tags]].fillna(0.0).sum(axis=1) for p in non_mach_prefixes) * cpi_mult
    df0910["g_mach"] = df0910[[f"mach_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1) * cpi_mult
    df0910["g_lab"] = df0910[[f"labor_d_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1)
    df0910["g_hd"] = df0910[[f"labor_hd_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1)
    df0910["g_hf"] = df0910[[f"labor_hf_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1) * cpi_mult
    df0910["c_rev"] = df0910[cash_cols].fillna(0.0).sum(axis=1) * cpi_mult
    df0910["sub3"] = (
        df0910["inc_govern_3"].fillna(0.0)
        + df0910["inc_govern_4"].fillna(0.0)
        + df0910["inc_govern_5"].fillna(0.0)
    ) * cpi_mult

    # Per-capita foodc fallback for the 4/600 households missing foodc in 2010
    valid_fc = sel_2010.loc[(sel_2010["foodc"].fillna(0.0) > 0) & (sel_2010["hhnumbers"].fillna(0.0) > 0)]
    per_capita_foodc_med = float((valid_fc["foodc"] / valid_fc["hhnumbers"]).median())

    df10_lab = df0910.loc[df0910["year"].eq(2010), ["region", "hhnumbers", "hhlabornum"]]
    valid_lab = df10_lab["hhlabornum"].between(0, df10_lab["hhnumbers"]) & df10_lab["hhlabornum"].mod(1).eq(0)
    worker_proxy = df10_lab.loc[valid_lab].groupby(["region", "hhnumbers"])["hhlabornum"].quantile(0.5, interpolation="lower")

    prov_counts = b.groupby("smgen").size().to_dict()
    reg_thin_code = {"东部": 901, "中部": 902, "西部": 903, "东北": 904}

    regions_out: list[CalibratedRegionV2] = []
    region_summaries: list[dict] = []

    for reg_idx, reg_name in enumerate(design.region_order):
        g = df0910.loc[df0910["region"].eq(reg_name)]
        sub_b = b.loc[b["region"].eq(reg_name)]

        vg = g.loc[(g["g_sow"] >= 1.0) & (g["g_prod"] > 0) & (g["g_rev"] > 0)]
        donor_yield = float((vg["g_prod"] / vg["g_sow"]).median())
        cohort_pos_g = sub_b.loc[(sub_b["sow_food"] >= 1.0) & (sub_b["prod_food"] > 0)]
        cohort_yield_med = float((cohort_pos_g["prod_food"] / cohort_pos_g["sow_food"]).median())
        y_reg = 0.5 * donor_yield + 0.5 * cohort_yield_med
        p_reg = float((vg["g_rev"] / vg["g_prod"]).median())

        vg_nm = vg.loc[vg["g_non_mach"] > 0]
        c_non_mach_reg = float((vg_nm["g_non_mach"] / vg_nm["g_sow"]).median())
        vg_mach = vg.loc[vg["g_mach"] > 0]
        c_mach_reg = float((vg_mach["g_mach"] / vg_mach["g_sow"]).median())
        vg_lab = vg.loc[vg["g_lab"] > 0]
        l_grain_reg = float((vg_lab["g_lab"] / vg_lab["g_sow"]).median())

        # Named grain crop sub-technologies calibrated on 2009-2010 regional observations
        named_techs: dict[str, NamedGrainTech] = {}
        reg_crop_areas: dict[str, float] = {}
        for cname, tags in NFPS_GRAIN_TAG_MAP.items():
            s_c = sum(g[f"sow_{t}"].fillna(0.0) for t in tags)
            p_c = sum(g[f"prod_{t}"].fillna(0.0) for t in tags)
            r_c = sum(g[f"inc_{t}"].fillna(0.0) for t in tags) * g["cpi"]
            nm_c = sum(g[f"{pref}{t}"].fillna(0.0) for t in tags for pref in non_mach_prefixes) * g["cpi"]
            mc_c = sum(g[f"mach_{t}"].fillna(0.0) for t in tags) * g["cpi"]
            lb_c = sum(g[f"labor_d_{t}"].fillna(0.0) for t in tags)
            reg_crop_areas[cname] = float(sum(sel_2010.loc[sub_b.index, f"sow_{t}"].fillna(0.0).sum() for t in tags))

            valid_c = (s_c >= 0.5) & (p_c > 0) & (r_c > 0)
            if int(valid_c.sum()) >= 40:
                yc = float((p_c[valid_c] / s_c[valid_c]).median())
                pc = float((r_c[valid_c] / p_c[valid_c]).median())
                nm_med = float((nm_c[valid_c & (nm_c > 0)] / s_c[valid_c & (nm_c > 0)]).median()) if (valid_c & (nm_c > 0)).sum() >= 25 else c_non_mach_reg
                mc_med = float((mc_c[valid_c & (mc_c > 0)] / s_c[valid_c & (mc_c > 0)]).median()) if (valid_c & (mc_c > 0)).sum() >= 25 else c_mach_reg
                lb_med = float((lb_c[valid_c & (lb_c > 0)] / s_c[valid_c & (lb_c > 0)]).median()) if (valid_c & (lb_c > 0)).sum() >= 25 else l_grain_reg
            else:
                yc, pc, nm_med, mc_med, lb_med = y_reg, p_reg, c_non_mach_reg, c_mach_reg, l_grain_reg

            named_techs[cname] = NamedGrainTech(
                crop=cname,
                yield_kg_sown_mu=round(yc, 3),
                price_yuan_kg=round(pc, 4),
                non_mach_input_yuan_sown_mu=round(nm_med, 2),
                mach_service_yuan_sown_mu=round(mc_med, 2),
                labor_days_sown_mu=round(lb_med, 2),
            )

        tot_reg_c_area = max(1e-9, fsum(reg_crop_areas.values()))
        default_weights = {k: v / tot_reg_c_area for k, v in reg_crop_areas.items()}

        vh = g.loc[(g["g_hd"] > 0) & (g["g_hf"] > 0)]
        w_hire = float((vh["g_hf"] / vh["g_hd"]).median())

        vc = g.loc[(g["sow_cash"] >= 0.5) & (g["c_rev"] > 0)]
        r_cash = float((vc["c_rev"] / vc["sow_cash"]).median())
        c_cash_total = design.cash_crop_input_cost_share * r_cash
        c_cash_mach = round(min(0.45 * c_cash_total, design.cash_crop_mach_share_of_grain * c_mach_reg), 2)
        c_cash_non_mach = round(max(20.0, c_cash_total - c_cash_mach), 2)
        l_cash = round(design.cash_crop_labor_ratio * l_grain_reg, 2)

        vsub = g.loc[(g["g_sow"] >= 1.0) & (g["sub3"] > 0)]
        fert_base = float((vsub["sub3"] / vsub["g_sow"]).median())

        grain_net_before_lab = y_reg * p_reg - (c_non_mach_reg + c_mach_reg)
        ask_rent = float(np.clip(0.30 * grain_net_before_lab, 160.0, 290.0))

        mci_series = (sub_b["sow_total"] / sub_b["area_e"].replace(0, np.nan)).dropna()
        max_mci = 1.15 if reg_name == "东北" else float(np.clip(mci_series.quantile(0.95), 1.75, 2.75))

        tot_sow_reg = float((sub_b["sow_food"] + sub_b["sow_cash"]).sum())
        base_g_share = float(sub_b["sow_food"].sum() / max(1.0, tot_sow_reg))

        shadow_w0 = design.slack_labor_reservation_ratio * w_hire
        pi_g0 = y_reg * p_reg - (c_non_mach_reg + c_mach_reg) - shadow_w0 * l_grain_reg
        pi_c0 = r_cash - (c_cash_non_mach + c_cash_mach) - shadow_w0 * l_cash
        margin_gap = pi_c0 - pi_g0

        rent_out_share = float((g["area_de2"].fillna(0.0) > 0).mean())
        rent_in_share = float((g["area_in1"].fillna(0.0) > 0).mean())
        turnover_share = float(np.clip(rent_out_share + rent_in_share, 0.14, 0.30))
        thresh_mu = design.scale_threshold_for(reg_name)
        scale_land_share = float(sub_b.loc[sub_b["area_e"] >= thresh_mu, "area_e"].sum() / max(1.0, sub_b["area_e"].sum()))
        scale_pool_share = float(np.clip(max(turnover_share, scale_land_share + 0.12), 0.18, 0.45))

        v_inflow = {"东北": 0.045, "东部": 0.015, "中部": 0.022, "西部": 0.028}[reg_name]

        tech = RegionTechV2(
            region=reg_name,
            named_grains=named_techs,
            default_grain_weights=default_weights,
            grain_yield_kg_sown_mu=round(y_reg, 3),
            grain_price_yuan_kg=round(p_reg, 4),
            grain_non_mach_input_yuan_sown_mu=round(c_non_mach_reg, 2),
            grain_mach_service_yuan_sown_mu=round(c_mach_reg, 2),
            grain_labor_days_sown_mu=round(l_grain_reg, 2),
            cash_rev_yuan_sown_mu=round(r_cash, 2),
            cash_non_mach_input_yuan_sown_mu=c_cash_non_mach,
            cash_mach_service_yuan_sown_mu=c_cash_mach,
            cash_labor_days_sown_mu=l_cash,
            hired_wage_yuan_day=round(w_hire, 2),
            ask_rent_yuan_phys_mu=round(ask_rent, 2),
            fertility_base_yuan_phys_mu=round(fert_base, 2),
            max_mci=round(max_mci, 3),
            baseline_grain_share=round(base_g_share, 4),
            baseline_margin_gap=round(margin_gap, 3),
            turnover_pool_share=round(turnover_share, 4),
            scale_pool_share=round(scale_pool_share, 4),
            village_net_inflow_rate=v_inflow,
        )

        hh_list: list[ActorBlueprint] = []
        parcel_list: list[ParcelBlueprint] = []

        for row in sub_b.itertuples():
            obs = sel_2010.iloc[row.Index]
            hh_num = max(1.0, float(row.hhnumbers))
            n_work = float(obs.hhlabornum) if pd.notna(obs.hhlabornum) else -1.0
            if not (0.0 <= n_work <= hh_num and n_work % 1.0 == 0.0):
                n_work = float(worker_proxy.loc[(reg_name, row.hhnumbers)])

            has_offfarm = (pd.notna(obs.inc_wage_l) and obs.inc_wage_l > 0) or (
                pd.notna(obs.inc_wage_o) and obs.inc_wage_o > 0
            )
            if has_offfarm:
                ag_days_per_worker = 120.0
                slack_share = 0.72
            elif obs.inc_wage_l == 0 and obs.inc_wage_o == 0:
                ag_days_per_worker = 180.0
                slack_share = 0.92
            else:
                ag_days_per_worker = 160.0
                slack_share = 0.82

            tot_ag_days = max(15.0, n_work * ag_days_per_worker) * design.family_labor_scale
            slack_days = tot_ag_days * slack_share

            wage_yr = float(row.wage_income)
            offfarm_days_implied = max(30.0, n_work * (310.0 - ag_days_per_worker))
            if wage_yr > 0:
                w_opp = wage_yr / offfarm_days_implied
            else:
                w_opp = round(0.30 * w_hire, 2)

            sav_stock = max(0.0, float(row.opening_cash) - float(getattr(row, "held_reserve", 0.0)))
            sow_tot_obs = max(float(row.sow_total), float(row.sow_food) + float(row.sow_cash))
            base_crop_need = (
                float(row.sow_food) * (c_non_mach_reg + c_mach_reg)
                + float(row.sow_cash) * (c_cash_non_mach + c_cash_mach)
            )
            base_lab_need = float(row.sow_food) * l_grain_reg + float(row.sow_cash) * l_cash
            obs_2010_upfront = base_crop_need + design.upfront_hired_cash_share * w_hire * max(0.0, base_lab_need - tot_ag_days)
            wc_raw = max(
                850.0 if (float(row.sow_food) + float(row.sow_cash)) > 0 else 200.0,
                0.15 * sav_stock,
                0.95 * obs_2010_upfront,
            )
            wc_init = min(wc_raw, max(2_800.0, 1.85 * obs_2010_upfront + 1_800.0)) * design.working_cap_scale

            # Household-specific named grain weights in 2010
            hh_c_areas: dict[str, float] = {}
            for cname, tags in NFPS_GRAIN_TAG_MAP.items():
                hh_c_areas[cname] = fsum(max(0.0, float(getattr(obs, f"sow_{t}", 0.0) or 0.0)) for t in tags)
            hh_c_tot = fsum(hh_c_areas.values())
            if hh_c_tot > 1e-6:
                g_weights = {k: v / hh_c_tot for k, v in hh_c_areas.items()}
            else:
                g_weights = default_weights.copy()

            # Weighted named-crop yield/price/costshrunk softly with regional anchor to avoid single-year noise
            y_named = fsum(g_weights[k] * named_techs[k].yield_kg_sown_mu for k in NAMED_GRAINS)
            p_named = fsum(g_weights[k] * named_techs[k].price_yuan_kg for k in NAMED_GRAINS)
            nm_named = fsum(g_weights[k] * named_techs[k].non_mach_input_yuan_sown_mu for k in NAMED_GRAINS)
            mc_named = fsum(g_weights[k] * named_techs[k].mach_service_yuan_sown_mu for k in NAMED_GRAINS)
            lb_named = fsum(g_weights[k] * named_techs[k].labor_days_sown_mu for k in NAMED_GRAINS)

            if float(row.sow_food) >= 1.0 and float(row.prod_food) > 0:
                hh_y_obs = float(row.prod_food) / float(row.sow_food)
                y_hh = float(np.clip(0.50 * y_reg + 0.25 * y_named + 0.25 * hh_y_obs, 0.65 * y_reg, 1.45 * y_reg))
            else:
                y_hh = float(0.65 * y_reg + 0.35 * y_named)

            p_hh = round(0.50 * p_reg + 0.50 * p_named, 4)
            nm_hh = round(0.50 * c_non_mach_reg + 0.50 * nm_named, 2)
            mc_hh = round(0.50 * c_mach_reg + 0.50 * mc_named, 2)
            lb_hh = round(0.50 * l_grain_reg + 0.50 * lb_named, 2)

            # Subsistence grain requirement from 2010 observed `foodc` (kg/year)
            fc_val = float(obs.foodc) if pd.notna(obs.foodc) and float(obs.foodc) > 0 else (per_capita_foodc_med * hh_num)
            subsistence_kg = float(np.clip(fc_val, 120.0, 1_800.0))

            if float(row.area_e) > 0.2 and sow_tot_obs > 0:
                mci_hh = float(np.clip(sow_tot_obs / float(row.area_e), 1.0, max_mci))
            else:
                mci_hh = 1.0 if reg_name == "东北" else min(1.35, max_mci)

            denom_gc = float(row.sow_food) + float(row.sow_cash)
            ref_sh = float(row.sow_food) / denom_gc if denom_gc > 0 else base_g_share

            prov = int(row.smgen)
            submkt = prov if prov_counts.get(prov, 0) >= design.thin_province_min_households else reg_thin_code[reg_name]
            aid = f"H{row.Index:04d}"

            hh_list.append(
                ActorBlueprint(
                    id=aid,
                    region=reg_name,
                    province=prov,
                    submarket=submkt,
                    actor_category="survey_household",
                    is_survey_household=True,
                    hh_members=hh_num,
                    workers=n_work,
                    ag_days_year=round(tot_ag_days, 2),
                    slack_ag_days_year=round(slack_days, 2),
                    offfarm_wage_day=round(w_opp, 2),
                    offfarm_annual_base=round(wage_yr, 2),
                    savings_stock=round(sav_stock, 2),
                    working_capital_init=round(wc_init, 2),
                    consumption_cash_base=round(float(row.consumption), 2),
                    subsistence_grain_kg=round(subsistence_kg, 2),
                    mci=round(mci_hh, 3),
                    grain_weights=tuple((k, round(g_weights[k], 4)) for k in NAMED_GRAINS),
                    grain_yield_kg_sown_mu=round(y_hh, 3),
                    grain_price_yuan_kg=p_hh,
                    grain_non_mach_yuan_sown_mu=nm_hh,
                    grain_mach_yuan_sown_mu=mc_hh,
                    grain_labor_days_sown_mu=lb_hh,
                    ref_grain_share=round(ref_sh, 4),
                    initial_cultivating=(sow_tot_obs > 0),
                    own_physical_mu=round(float(row.area_e), 4),
                    verified_tradable_mu=round(float(row.verified_right), 4),
                    admin_overhead_yuan=0.0,
                    service_coord_saving=0.0,
                )
            )
            parcel_list.extend(
                _split_parcels_v2(
                    owner=aid,
                    region=reg_name,
                    province=prov,
                    submarket=submkt,
                    area_e=float(row.area_e),
                    verified_right=float(row.verified_right),
                    sow_food=float(row.sow_food),
                    sow_cash=float(row.sow_cash),
                    sow_total=float(row.sow_total),
                    quality=1.0,
                    start_idx=len(parcel_list),
                )
            )

        # Differentiated Scale Organizations (Family Farm / Cooperative / Agri-Company; zero arbitrary yield bonus!)
        org_list: list[ActorBlueprint] = []
        if design.enable_scale_organizations:
            submarkets = sorted({h.submarket for h in hh_list})
            wc_p85 = max(20_000.0, float(np.percentile([h.working_capital_init for h in hh_list], 85)))
            mci_org = 1.05 if reg_name == "东北" else min(1.65, max_mci)
            w_tup = tuple((k, round(default_weights[k], 4)) for k in NAMED_GRAINS)

            for sm_idx, sm in enumerate(submarkets):
                prov_rep = next(h.province for h in hh_list if h.submarket == sm)
                sm_trad_mu = fsum(p.area for p in parcel_list if p.submarket == sm and p.tradable)

                # 1) Commercial Family Farm in every active submarket (2 family workers + hired labor)
                org_list.append(
                    ActorBlueprint(
                        id=f"FF_{reg_idx}_{sm}_{0}",
                        region=reg_name,
                        province=prov_rep,
                        submarket=sm,
                        actor_category="family_farm",
                        is_survey_household=False,
                        hh_members=3.0,
                        workers=2.0,
                        ag_days_year=340.0 * design.family_labor_scale,
                        slack_ag_days_year=240.0 * design.family_labor_scale,
                        offfarm_wage_day=round(0.45 * w_hire, 2),
                        offfarm_annual_base=0.0,
                        savings_stock=round(2.5 * wc_p85, 2),
                        working_capital_init=round(2.2 * wc_p85 * design.working_cap_scale, 2),
                        consumption_cash_base=14_000.0,
                        subsistence_grain_kg=0.0,
                        mci=round(mci_org, 3),
                        grain_weights=w_tup,
                        grain_yield_kg_sown_mu=round(y_reg, 3),
                        grain_price_yuan_kg=round(p_reg, 4),
                        grain_non_mach_yuan_sown_mu=round(c_non_mach_reg, 2),
                        grain_mach_yuan_sown_mu=round(c_mach_reg, 2),
                        grain_labor_days_sown_mu=round(l_grain_reg, 2),
                        ref_grain_share=round(max(0.75, base_g_share), 4),
                        initial_cultivating=True,
                        own_physical_mu=0.0,
                        verified_tradable_mu=0.0,
                        admin_overhead_yuan=0.0,
                        service_coord_saving=0.0,
                    )
                )

                # 2) Agricultural Cooperative in submarkets with >= 90 mu tradable land (0 unpaid family labor, AgriPoliS bulk input/service economy)
                if sm_trad_mu >= 90.0:
                    org_list.append(
                        ActorBlueprint(
                            id=f"COOP_{reg_idx}_{sm}_{0}",
                            region=reg_name,
                            province=prov_rep,
                            submarket=sm,
                            actor_category="cooperative",
                            is_survey_household=False,
                            hh_members=0.0,
                            workers=0.0,
                            ag_days_year=0.0,
                            slack_ag_days_year=0.0,
                            offfarm_wage_day=round(w_hire, 2),
                            offfarm_annual_base=0.0,
                            savings_stock=round(3.4 * wc_p85, 2),
                            working_capital_init=round(3.0 * wc_p85 * design.working_cap_scale, 2),
                            consumption_cash_base=0.0,
                            subsistence_grain_kg=0.0,
                            mci=round(mci_org, 3),
                            grain_weights=w_tup,
                            grain_yield_kg_sown_mu=round(y_reg, 3),
                            grain_price_yuan_kg=round(p_reg, 4),
                            grain_non_mach_yuan_sown_mu=round(0.86 * c_non_mach_reg, 2),
                            grain_mach_yuan_sown_mu=round(0.84 * c_mach_reg, 2),
                            grain_labor_days_sown_mu=round(0.75 * l_grain_reg, 2),
                            ref_grain_share=round(max(0.78, base_g_share), 4),
                            initial_cultivating=True,
                            own_physical_mu=0.0,
                            verified_tradable_mu=0.0,
                            admin_overhead_yuan=design.coop_admin_overhead_yuan,
                            service_coord_saving=design.coop_contiguous_service_saving,
                        )
                    )

                # 3) Corporate Agri-Company in large grain submarkets >= 200 mu tradable land (AgriPoliS corporate farm: bulk input/mechanization scale economy + paid labor + overhead)
                if sm_trad_mu >= 200.0:
                    org_list.append(
                        ActorBlueprint(
                            id=f"CORP_{reg_idx}_{sm}_{0}",
                            region=reg_name,
                            province=prov_rep,
                            submarket=sm,
                            actor_category="agri_company",
                            is_survey_household=False,
                            hh_members=0.0,
                            workers=0.0,
                            ag_days_year=0.0,
                            slack_ag_days_year=0.0,
                            offfarm_wage_day=round(w_hire, 2),
                            offfarm_annual_base=0.0,
                            savings_stock=round(4.6 * wc_p85, 2),
                            working_capital_init=round(4.0 * wc_p85 * design.working_cap_scale, 2),
                            consumption_cash_base=0.0,
                            subsistence_grain_kg=0.0,
                            mci=round(mci_org, 3),
                            grain_weights=w_tup,
                            grain_yield_kg_sown_mu=round(y_reg, 3),
                            grain_price_yuan_kg=round(p_reg, 4),
                            grain_non_mach_yuan_sown_mu=round(0.80 * c_non_mach_reg, 2),
                            grain_mach_yuan_sown_mu=round(0.78 * c_mach_reg, 2),
                            grain_labor_days_sown_mu=round(0.68 * l_grain_reg, 2),
                            ref_grain_share=round(max(0.72, base_g_share), 4),
                            initial_cultivating=True,
                            own_physical_mu=0.0,
                            verified_tradable_mu=0.0,
                            admin_overhead_yuan=design.company_admin_overhead_yuan,
                            service_coord_saving=0.08,
                        )
                    )

        init_grain_kg = float(sub_b["prod_food"].fillna(0.0).sum())
        cal_reg = CalibratedRegionV2(
            name=reg_name,
            tech=tech,
            households=tuple(hh_list),
            organizations=tuple(org_list),
            parcels=tuple(parcel_list),
            initial_grain_kg=init_grain_kg,
        )
        regions_out.append(cal_reg)
        region_summaries.append({
            "region": reg_name,
            "n_households": len(hh_list),
            "n_family_farms": sum(1 for o in org_list if o.actor_category == "family_farm"),
            "n_cooperatives": sum(1 for o in org_list if o.actor_category == "cooperative"),
            "n_agri_companies": sum(1 for o in org_list if o.actor_category == "agri_company"),
            "n_parcels": len(parcel_list),
            "physical_area_mu": round(cal_reg.physical_area, 4),
            "verified_tradable_mu": round(cal_reg.verified_tradable_area, 4),
            "fertility_base_yuan_mu": tech.fertility_base_yuan_phys_mu,
            "grain_mach_yuan_sown_mu": tech.grain_mach_service_yuan_sown_mu,
            "grain_non_mach_yuan_sown_mu": tech.grain_non_mach_input_yuan_sown_mu,
            "grain_yield_kg_sown_mu": tech.grain_yield_kg_sown_mu,
            "grain_price_yuan_kg": tech.grain_price_yuan_kg,
            "cash_rev_yuan_sown_mu": tech.cash_rev_yuan_sown_mu,
            "hired_wage_yuan_day": tech.hired_wage_yuan_day,
            "ask_rent_yuan_phys_mu": tech.ask_rent_yuan_phys_mu,
        })

    tot_phys = fsum(r.physical_area for r in regions_out)
    assert abs(tot_phys - 5277.5) < 1e-6, f"Physical area conservation violated: {tot_phys}"

    meta = {
        "design_id": design.id,
        "signatures": sigs,
        "calibration_window": "2009-2010 NFPS + CFPS 2010 + CHFS 2011 ONLY (strictly <= 2010)",
        "total_households": sum(len(r.households) for r in regions_out),
        "total_organizations": sum(len(r.organizations) for r in regions_out),
        "total_physical_mu": round(tot_phys, 4),
        "total_tradable_mu": round(fsum(r.verified_tradable_area for r in regions_out), 4),
        "region_summaries": region_summaries,
    }
    return tuple(regions_out), meta
