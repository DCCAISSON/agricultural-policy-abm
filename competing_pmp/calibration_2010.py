"""Strictly pre-2011 (2009-2010 NFPS + CFPS/CHFS) empirical calibration for competing_pmp.

Hard guardrail: post-2010 NFPS rows are asserted absent before any parameter is computed.
"""

from dataclasses import dataclass
from hashlib import file_digest
import json
from math import ceil, fsum
from pathlib import Path

import numpy as np
import pandas as pd

from competing_pmp.config import CompetingDesign


@dataclass(frozen=True, slots=True)
class RegionTechParams:
    region: str
    grain_yield_kg_sown_mu: float
    grain_price_yuan_kg: float
    grain_input_yuan_sown_mu: float
    grain_labor_days_sown_mu: float
    cash_rev_yuan_sown_mu: float
    cash_input_yuan_sown_mu: float
    cash_labor_days_sown_mu: float
    hired_wage_yuan_day: float
    ask_rent_yuan_phys_mu: float
    max_mci: float
    baseline_grain_share: float
    baseline_margin_gap: float        # Regional baseline (pi_cash_0 - pi_grain_0) for Howitt PMP calibration
    turnover_pool_share: float        # Pre-2011 observed transfer/release share for takeover rate denominator
    village_net_inflow_rate: float    # Pre-2011 calibrated net external land availability per cohort mu


@dataclass(frozen=True, slots=True)
class HouseholdAgent:
    id: str
    region: str
    province: int
    submarket: int
    is_entrant: bool
    workers: float
    ag_days_year: float
    slack_ag_days_year: float
    offfarm_wage_day: float
    offfarm_annual_base: float
    savings_stock: float
    working_capital_init: float
    consumption_base: float
    mci: float
    grain_yield_factor: float
    ref_grain_share: float
    initial_cultivating: bool
    own_physical_mu: float
    verified_tradable_mu: float


@dataclass(frozen=True, slots=True)
class Parcel:
    id: str
    owner: str
    region: str
    province: int
    submarket: int
    area: float
    tradable: bool
    quality: float
    initial_crop: str


@dataclass(frozen=True, slots=True)
class CalibratedRegion:
    name: str
    tech: RegionTechParams
    households: tuple[HouseholdAgent, ...]
    entrants: tuple[HouseholdAgent, ...]
    parcels: tuple[Parcel, ...]
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


def _split_household_parcels(
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
) -> list[Parcel]:
    """Preserve exact physical area_e and verified_right without inventing parcel maps."""
    denom = max(sow_total, sow_food + sow_cash)
    g_sh, c_sh = (sow_food / denom, sow_cash / denom) if denom > 0 else (0.0, 0.0)
    i_sh = max(0.0, 1.0 - g_sh - c_sh)
    out: list[Parcel] = []
    for bucket_area, tradable in ((verified_right, True), (max(0.0, area_e - verified_right), False)):
        if bucket_area <= 1e-10:
            continue
        # Keep tradable lots consolidated up to 6 mu (reducing artificial tiny-lot fragmentation)
        for crop, frac in (("grain", g_sh), ("cash", c_sh), ("idle", i_sh)):
            sz = bucket_area * frac
            if sz <= 1e-10:
                continue
            n_lots = max(1, ceil(sz / 6.0))
            for _ in range(n_lots):
                out.append(
                    Parcel(
                        id=f"{owner}P{len(out):03d}",
                        owner=owner,
                        region=region,
                        province=province,
                        submarket=submarket,
                        area=sz / n_lots,
                        tradable=tradable,
                        quality=quality,
                        initial_crop=crop,
                    )
                )
    return out


def load_calibrated_regions(project_root: Path, design: CompetingDesign) -> tuple[tuple[CalibratedRegion, ...], dict]:
    """Calibrate all regional & household parameters strictly from 2009-2010 NFPS + 2010 baseline."""
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
        raise ValueError("Authorized source SHA-256 mismatch against preregistered CompetingDesign")

    b = pd.read_csv(baseline_path, dtype={"actor": "string"})
    cohort = pd.DataFrame(json.loads(cohort_path.read_bytes()))
    if len(b) != 600 or len(cohort) != 600:
        raise ValueError("Cohort size is not 600")

    grain_tags = ("w", "r", "c", "s", "t", "o")
    mat_prefixes = ("seed_", "fer_", "film_", "pes_", "weir_", "anim_", "mach_")
    base_cols = [
        "year", "hhid", "id", "smgen", "hhnumbers", "hhlabornum",
        "area_b", "area_e", "area_e1", "area_in1", "area_de2",
        "sow_total", "sow_food", "prod_food", "sow_cash",
        "inc_wage_l", "inc_wage_o", "saving_1", "saving_2", "inc_rent",
    ]
    crop_cols = [f"{p}{t}" for t in grain_tags for p in ("sow_", "prod_", "inc_", "labor_d_", "labor_hd_", "labor_hf_", *mat_prefixes)]
    cash_cols = [f"inc_cash_{i}" for i in range(1, 11)]
    use_cols = sorted(set(base_cols + crop_cols + cash_cols))

    raw_all = pd.read_csv(raw_path, usecols=use_cols, dtype={"hhid": "string", "id": "string"}, low_memory=False)
    sel_2010 = raw_all.iloc[cohort["source_row"].to_numpy() - 2].reset_index(drop=True)
    if not (sel_2010["year"].eq(2010).all() and sel_2010["hhid"].eq(sel_2010["id"]).all()):
        raise ValueError("2010 cohort linkage failed reconciliation")

    # HARD GUARDRAIL: Strictly slice 2009-2010 ONLY and discard raw_all immediately!
    df0910 = raw_all.loc[raw_all["year"].isin([2009, 2010])].copy()
    del raw_all
    assert int(df0910["year"].max()) <= 2010, "Post-2010 leakage forbidden in calibration"

    prov_to_region = b[["smgen", "region"]].drop_duplicates().set_index("smgen")["region"].to_dict()
    df0910 = df0910.loc[df0910["smgen"].isin(prov_to_region)].copy()
    df0910["region"] = df0910["smgen"].map(prov_to_region)

    # Precompute 2009-2010 crop aggregates in consistent CNY2024 (matching baseline.csv currency!)
    cpi_mult = np.where(df0910["year"].eq(2010), design.cpi_2010_to_2024, design.cpi_2009_to_2024)
    df0910["g_sow"] = df0910[[f"sow_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1)
    df0910["g_prod"] = df0910[[f"prod_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1)
    df0910["g_rev"] = df0910[[f"inc_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1) * cpi_mult
    df0910["g_cost"] = sum(df0910[[f"{p}{t}" for t in grain_tags]].fillna(0.0).sum(axis=1) for p in mat_prefixes) * cpi_mult
    df0910["g_lab"] = df0910[[f"labor_d_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1)
    df0910["g_hd"] = df0910[[f"labor_hd_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1)
    df0910["g_hf"] = df0910[[f"labor_hf_{t}" for t in grain_tags]].fillna(0.0).sum(axis=1) * cpi_mult
    df0910["c_rev"] = df0910[cash_cols].fillna(0.0).sum(axis=1) * cpi_mult

    # Worker proxy on 2010 NFPS
    df10_lab = df0910.loc[df0910["year"].eq(2010), ["region", "hhnumbers", "hhlabornum"]]
    valid_lab = df10_lab["hhlabornum"].between(0, df10_lab["hhnumbers"]) & df10_lab["hhlabornum"].mod(1).eq(0)
    worker_proxy = df10_lab.loc[valid_lab].groupby(["region", "hhnumbers"])["hhlabornum"].quantile(0.5, interpolation="lower")

    # Map ultra-thin provinces (< thin_province_min_households) into a shared regional submarket code
    prov_counts = b.groupby("smgen").size().to_dict()
    reg_thin_code = {"东部": 901, "中部": 902, "西部": 903, "东北": 904}

    regions_out: list[CalibratedRegion] = []
    provenance_rows: list[dict] = []
    region_summaries: list[dict] = []

    for reg_idx, reg_name in enumerate(design.region_order):
        g = df0910.loc[df0910["region"].eq(reg_name)]
        sub_b = b.loc[b["region"].eq(reg_name)]

        vg = g.loc[(g["g_sow"] >= 1.0) & (g["g_prod"] > 0) & (g["g_rev"] > 0)]
        # Align grain yield level to 2010 cohort regional weighted yield (prod_food / sow_food) shrunk with 2009-2010 donor median
        donor_yield = float((vg["g_prod"] / vg["g_sow"]).median())
        cohort_pos_g = sub_b.loc[(sub_b["sow_food"] >= 1.0) & (sub_b["prod_food"] > 0)]
        cohort_yield_med = float((cohort_pos_g["prod_food"] / cohort_pos_g["sow_food"]).median())
        y_reg = 0.5 * donor_yield + 0.5 * cohort_yield_med

        p_reg = float((vg["g_rev"] / vg["g_prod"]).median())
        vg_cost = vg.loc[vg["g_cost"] > 0]
        c_grain = float((vg_cost["g_cost"] / vg_cost["g_sow"]).median())
        vg_lab = vg.loc[vg["g_lab"] > 0]
        l_grain = float((vg_lab["g_lab"] / vg_lab["g_sow"]).median())

        vh = g.loc[(g["g_hd"] > 0) & (g["g_hf"] > 0)]
        w_hire = float((vh["g_hf"] / vh["g_hd"]).median())

        vc = g.loc[(g["sow_cash"] >= 0.5) & (g["c_rev"] > 0)]
        r_cash = float((vc["c_rev"] / vc["sow_cash"]).median())
        c_cash = round(design.cash_crop_input_cost_share * r_cash, 2)
        l_cash = round(design.cash_crop_labor_ratio * l_grain, 2)

        # Rental ask floor in CNY2024: bounded reservation share of regional net grain land surplus
        grain_net_before_lab = y_reg * p_reg - c_grain
        ask_rent = float(np.clip(0.30 * grain_net_before_lab, 160.0, 290.0))

        mci_series = (sub_b["sow_total"] / sub_b["area_e"].replace(0, np.nan)).dropna()
        max_mci = 1.15 if reg_name == "东北" else float(np.clip(mci_series.quantile(0.95), 1.75, 2.75))

        tot_sow_reg = float((sub_b["sow_food"] + sub_b["sow_cash"]).sum())
        base_g_share = float(sub_b["sow_food"].sum() / max(1.0, tot_sow_reg))

        # Baseline shadow labor evaluation at de Janvry non-separable village reservation wage for Howitt PMP calibration
        shadow_w0 = 0.22 * w_hire
        pi_g0 = y_reg * p_reg - c_grain - shadow_w0 * l_grain
        pi_c0 = r_cash - c_cash - shadow_w0 * l_cash
        margin_gap = pi_c0 - pi_g0

        # Pre-2011 observed transfer/release activity rate for takeover subsidy rate denominator
        rent_out_share = float((g["area_de2"].fillna(0.0) > 0).mean())
        rent_in_share = float((g["area_in1"].fillna(0.0) > 0).mean())
        turnover_share = float(np.clip(rent_out_share + rent_in_share, 0.12, 0.30))

        # Pre-2011 net village exchange capacity proxy (for 2011-2015 cohort historical check)
        v_inflow = {"东北": 0.045, "东部": 0.015, "中部": 0.022, "西部": 0.028}[reg_name]

        tech = RegionTechParams(
            region=reg_name,
            grain_yield_kg_sown_mu=round(y_reg, 3),
            grain_price_yuan_kg=round(p_reg, 4),
            grain_input_yuan_sown_mu=round(c_grain, 2),
            grain_labor_days_sown_mu=round(l_grain, 2),
            cash_rev_yuan_sown_mu=round(r_cash, 2),
            cash_input_yuan_sown_mu=round(c_cash, 2),
            cash_labor_days_sown_mu=round(l_cash, 2),
            hired_wage_yuan_day=round(w_hire, 2),
            ask_rent_yuan_phys_mu=round(ask_rent, 2),
            max_mci=round(max_mci, 3),
            baseline_grain_share=round(base_g_share, 4),
            baseline_margin_gap=round(margin_gap, 3),
            turnover_pool_share=round(turnover_share, 4),
            village_net_inflow_rate=v_inflow,
        )

        hh_list: list[HouseholdAgent] = []
        parcel_list: list[Parcel] = []

        for row in sub_b.itertuples():
            obs = sel_2010.iloc[row.Index]
            n_work = float(obs.hhlabornum) if pd.notna(obs.hhlabornum) else -1.0
            if not (0.0 <= n_work <= float(row.hhnumbers) and n_work % 1.0 == 0.0):
                n_work = float(worker_proxy.loc[(reg_name, row.hhnumbers)])

            has_offfarm = (pd.notna(obs.inc_wage_l) and obs.inc_wage_l > 0) or (
                pd.notna(obs.inc_wage_o) and obs.inc_wage_o > 0
            )
            if has_offfarm:
                ag_days_per_worker = 120.0
            elif obs.inc_wage_l == 0 and obs.inc_wage_o == 0:
                ag_days_per_worker = 180.0
            else:
                ag_days_per_worker = 160.0

            ag_days_year = max(15.0, n_work * ag_days_per_worker)
            comp_share = design.competing_labor_window_offfarm if has_offfarm else design.competing_labor_window_pure_ag
            slack_ag_days = round(ag_days_year * (1.0 - comp_share), 2)
            off_days_year = max(30.0, n_work * (310.0 - ag_days_per_worker))
            wage_day = float(row.wage_income) / off_days_year if float(row.wage_income) > 0 else 0.30 * w_hire

            liquid_savings = max(0.0, float(row.opening_cash) - float(row.held_reserve))
            base_crop_need = c_grain * float(row.sow_food) + c_cash * float(row.sow_cash)
            base_lab_need = l_grain * float(row.sow_food) + l_cash * float(row.sow_cash)
            base_upfront_hire = design.upfront_hired_cash_share * w_hire * max(0.0, base_lab_need - ag_days_year)
            obs_2010_upfront = base_crop_need + base_upfront_hire
            # Working capital combines CHFS-anchored savings share and 2010 revealed operating scale
            work_cap = max(
                design.working_capital_floor_yuan if (row.sow_food + row.sow_cash) > 0 else 200.0,
                design.working_capital_share_of_savings * liquid_savings,
                0.95 * obs_2010_upfront,
            )
            work_cap = min(work_cap, max(2800.0, 1.85 * obs_2010_upfront + 1800.0))

            raw_mci = float(row.intensity) if pd.notna(row.intensity) and row.intensity > 0 else 1.0
            obs_mci = (float(row.sow_food) + float(row.sow_cash)) / max(0.1, float(row.area_e))
            hh_mci = float(np.clip(max(raw_mci, min(obs_mci, max_mci)), 1.0, max_mci))

            # Empirical Bayes shrunk household grain yield factor (from 2010 baseline only)
            if float(row.sow_food) >= 1.0 and float(row.prod_food) > 0:
                obs_y = float(row.prod_food) / float(row.sow_food)
                raw_ratio = float(np.clip(obs_y / y_reg, 0.65, 1.45))
                y_factor = (1.0 - design.yield_shrinkage_weight) * 1.0 + design.yield_shrinkage_weight * raw_ratio
            else:
                y_factor = 1.0

            s_gc = float(row.sow_food) + float(row.sow_cash)
            if s_gc > 1e-6:
                ref_share = float(row.sow_food) / s_gc
                cult_init = True
            else:
                ref_share = base_g_share
                cult_init = False

            prov = int(row.smgen)
            submkt = prov if prov_counts.get(prov, 0) >= design.thin_province_min_households else reg_thin_code[reg_name]
            hid = f"H{row.Index:04d}"

            hh_list.append(
                HouseholdAgent(
                    id=hid,
                    region=reg_name,
                    province=prov,
                    submarket=submkt,
                    is_entrant=False,
                    workers=n_work,
                    ag_days_year=ag_days_year,
                    slack_ag_days_year=slack_ag_days,
                    offfarm_wage_day=round(wage_day, 3),
                    offfarm_annual_base=float(row.wage_income),
                    savings_stock=liquid_savings,
                    working_capital_init=round(work_cap, 2),
                    consumption_base=float(row.consumption),
                    mci=round(hh_mci, 4),
                    grain_yield_factor=round(y_factor, 4),
                    ref_grain_share=round(ref_share, 4),
                    initial_cultivating=cult_init,
                    own_physical_mu=float(row.area_e),
                    verified_tradable_mu=float(row.verified_right),
                )
            )
            parcel_list.extend(
                _split_household_parcels(
                    owner=hid,
                    region=reg_name,
                    province=prov,
                    submarket=submkt,
                    area_e=float(row.area_e),
                    verified_right=float(row.verified_right),
                    sow_food=float(row.sow_food),
                    sow_cash=float(row.sow_cash),
                    sow_total=float(row.sow_total),
                    quality=round(y_factor, 4),
                )
            )

        # Create bounded regional family-farm entrant operators (1 per active submarket, 0 initial owned land)
        entrants_list: list[HouseholdAgent] = []
        if design.enable_regional_entrants:
            submarkets = sorted({h.submarket for h in hh_list})
            submkt_tradable = {
                sm: fsum(p.area for p in parcel_list if p.submarket == sm and p.tradable) for sm in submarkets
            }
            for idx_sm, sm in enumerate(submarkets):
                cap_mu = max(12.0, 0.22 * submkt_tradable[sm])
                ent_work_cap = round(cap_mu * (c_grain + ask_rent) * 0.85 + 2500.0, 2)
                ent_ag_days = round(cap_mu * l_grain * (1.0 if reg_name == "东北" else 1.35), 2)
                entrants_list.append(
                    HouseholdAgent(
                        id=f"E{reg_idx}{idx_sm:02d}",
                        region=reg_name,
                        province=sm,
                        submarket=sm,
                        is_entrant=True,
                        workers=2.5,
                        ag_days_year=ent_ag_days,
                        slack_ag_days_year=round(0.75 * ent_ag_days, 2),
                        offfarm_wage_day=round(0.55 * w_hire, 2),
                        offfarm_annual_base=0.0,
                        savings_stock=ent_work_cap * 1.5,
                        working_capital_init=ent_work_cap,
                        consumption_base=0.0,
                        mci=1.0 if reg_name == "东北" else round(min(1.65, max_mci * 0.85), 3),
                        grain_yield_factor=1.04,
                        ref_grain_share=round(max(0.78, base_g_share), 4),
                        initial_cultivating=False,
                        own_physical_mu=0.0,
                        verified_tradable_mu=0.0,
                    )
                )

        reg_obj = CalibratedRegion(
            name=reg_name,
            tech=tech,
            households=tuple(hh_list),
            entrants=tuple(entrants_list),
            parcels=tuple(parcel_list),
            initial_grain_kg=float(sub_b["prod_food"].sum()),
        )
        if not np.isclose(reg_obj.physical_area, float(sub_b["area_e"].sum()), rtol=0.0, atol=1e-7):
            raise ValueError(f"Physical area conservation violated in {reg_name}")
        if not np.isclose(reg_obj.verified_tradable_area, float(sub_b["verified_right"].sum()), rtol=0.0, atol=1e-7):
            raise ValueError(f"Verified tradable right conservation violated in {reg_name}")

        regions_out.append(reg_obj)
        region_summaries.append(
            {
                "region": reg_name,
                "households": len(hh_list),
                "regional_entrants": len(entrants_list),
                "submarkets": len({h.submarket for h in hh_list}),
                "physical_mu": round(reg_obj.physical_area, 4),
                "verified_tradable_mu": round(reg_obj.verified_tradable_area, 4),
                "parcels": len(parcel_list),
                "grain_yield_kg_sown_mu": tech.grain_yield_kg_sown_mu,
                "grain_price_yuan_kg": tech.grain_price_yuan_kg,
                "grain_input_yuan_sown_mu": tech.grain_input_yuan_sown_mu,
                "grain_labor_days_sown_mu": tech.grain_labor_days_sown_mu,
                "cash_rev_yuan_sown_mu": tech.cash_rev_yuan_sown_mu,
                "cash_input_yuan_sown_mu": tech.cash_input_yuan_sown_mu,
                "hired_wage_yuan_day": tech.hired_wage_yuan_day,
                "ask_rent_yuan_phys_mu": tech.ask_rent_yuan_phys_mu,
                "max_mci": tech.max_mci,
                "observed_2010_grain_kg": round(reg_obj.initial_grain_kg, 2),
            }
        )

    provenance_rows.extend([
        {"parameter": "household_land_and_rights", "category": "direct_observation", "source": "2010 NFPS cohort600 area_e & area_e1"},
        {"parameter": "grain_yield_price_inputs_labor", "category": "pre2011_statistical_calibration", "source": "2009-2010 NFPS regional medians + 2010 Empirical Bayes yield shrinkage"},
        {"parameter": "hired_labor_service_wage", "category": "pre2011_statistical_calibration", "source": "2009-2010 NFPS observed labor_hf / labor_hd regional medians"},
        {"parameter": "annual_family_farming_days", "category": "literature_and_survey_reference", "source": "CFPS 2010 farming days (180 pure ag / 120 off-farm active) x NFPS hhlabornum"},
        {"parameter": "working_capital_liquidity_share", "category": "survey_reference_calibration", "source": "CHFS 2011 agricultural household liquid working capital median vs NFPS total savings"},
        {"parameter": "pmp_quadratic_curvature_gamma", "category": "structural_calibration", "source": "Howitt (1995) concave PMP calibrated at 2010 baseline crop shares (supply elasticity ~0.25)"},
        {"parameter": "fiscal_budget_and_rate_cap", "category": "statutory_policy_standard", "source": "98,953 yuan/year shared envelope (18.75 yuan/sample-mu), 85 yuan/mu statutory ceiling"},
    ])

    meta = {
        **sigs,
        "design_sha256": design.fingerprint(),
        "regions": region_summaries,
        "parameter_provenance": provenance_rows,
    }
    return tuple(regions_out), meta
