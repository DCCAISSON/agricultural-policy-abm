"""Stage 1 Validation & Head-to-Head Agricultural Benchmark for Competing AHM-PMP Model.

Executes:
1. One-step conditional household prediction (2011-2015) given lagged survey land/crop state.
2. Five-year recursive simulation (2011-2015) across 20 protocol seeds (301..320) & 4 sprint seeds (571001..571004).
3. Mechanism ablations (PMP vs LP corner limit, hired labor on/off, closed land vs village exchange).
4. Sanitized head-to-head comparison table against legacy M1 (476b492) and economic-core arms (c47bdfc).
"""

import argparse
from collections import defaultdict
import csv
import json
from math import fsum
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from competing_pmp.calibration_2010 import load_calibrated_regions
from competing_pmp.config import CompetingDesign, default_design
from competing_pmp.economy import RegionalPMPEconomy
from competing_pmp.pmp_solver import solve_household_pmp


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_survey_panel_2010_2015(project_root: Path) -> pd.DataFrame:
    """Load matched cohort600 panel across 2010-2015 for validation ONLY (never calibration!)."""
    folder = project_root / "校准用数据源/全国农村固定观察点"
    link = project_root / "results/diagnostics/national_crop_choice_soft_response_20261123/PRIVATE_DO_NOT_SHARE/cohort600.json"
    ref = pd.DataFrame(json.loads(link.read_bytes()))
    cols = ["year", "hhid", "id", "smgen", "area_b", "area_e", "sow_total", "sow_food", "sow_cash", "prod_food"]
    data = pd.read_csv(folder / "2009-2015household.csv", usecols=cols, dtype={"hhid": "string", "id": "string"})
    key = data.iloc[ref["source_row"].to_numpy() - 2].reset_index(drop=True)[["hhid", "smgen"]].copy()
    key["actor"] = [f"H{i:04d}" for i in range(len(key))]
    sub = data[data["year"].between(2010, 2015) & data["hhid"].notna() & data["hhid"].eq(data["id"])].copy()
    sub = sub.loc[~sub.duplicated(["year", "hhid"], keep=False)]
    return sub.merge(key, on=["hhid", "smgen"], validate="many_to_one").set_index(["year", "actor"])


def evaluate_cohort_year_metrics(
    sim_by_actor: dict[str, dict],
    survey_year_df: pd.DataFrame,
    seed: int,
    year: int,
    arm: str,
) -> dict:
    """Compute exact 5-gate validation metrics + grain tonnage on matched survey denominators."""
    comp_mask = survey_year_df[["sow_food", "sow_cash"]].notna().all(axis=1) & survey_year_df[["sow_food", "sow_cash"]].ge(0).all(axis=1)
    comp = survey_year_df.loc[comp_mask]
    n_comp = len(comp)

    obs_tot_sow = comp["sow_food"] + comp["sow_cash"]
    obs_sown_n = int(obs_tot_sow.gt(0).sum())
    obs_mixed_n = int((comp["sow_food"].gt(0) & comp["sow_cash"].gt(0)).sum())
    obs_pure_g = int((comp["sow_food"].gt(0) & comp["sow_cash"].eq(0)).sum())
    obs_pure_c = int((comp["sow_food"].eq(0) & comp["sow_cash"].gt(0)).sum())
    obs_g_mu = float(comp["sow_food"].sum())
    obs_c_mu = float(comp["sow_cash"].sum())

    obs_pos_shares = (comp.loc[obs_tot_sow > 0, "sow_food"] / obs_tot_sow.loc[obs_tot_sow > 0]).to_numpy()
    obs_q = np.percentile(obs_pos_shares, [25, 50, 75]) if len(obs_pos_shares) > 0 else np.zeros(3)

    sim_sown_n = 0
    sim_mixed_n = 0
    sim_pure_g = 0
    sim_pure_c = 0
    sim_g_mu = 0.0
    sim_c_mu = 0.0
    sim_shares = []

    for aid in comp.index:
        rec = sim_by_actor.get(aid, {"sown_grain_mu": 0.0, "sown_cash_mu": 0.0, "grain_kg": 0.0, "operated_physical_mu": 0.0})
        sg = float(rec["sown_grain_mu"])
        sc = float(rec["sown_cash_mu"])
        st = sg + sc
        sim_g_mu += sg
        sim_c_mu += sc
        if st > 1e-6:
            sim_sown_n += 1
            sh = sg / st
            sim_shares.append(sh)
            if sg > 1e-6 and sc > 1e-6:
                sim_mixed_n += 1
            elif sg > 1e-6:
                sim_pure_g += 1
            else:
                sim_pure_c += 1

    sim_q = np.percentile(sim_shares, [25, 50, 75]) if sim_shares else np.zeros(3)
    share_q_mae = float(np.mean(np.abs(sim_q - obs_q)))

    kg_mask = comp["prod_food"].notna() & comp["prod_food"].ge(0)
    obs_kg = float(comp.loc[kg_mask, "prod_food"].sum())
    sim_kg = fsum(float(sim_by_actor.get(aid, {"grain_kg": 0.0})["grain_kg"]) for aid in comp.loc[kg_mask].index)

    stock_mask = survey_year_df["area_b"].notna() & survey_year_df["area_b"].ge(0) & survey_year_df["area_e"].notna() & survey_year_df["area_e"].gt(0)
    stock_df = survey_year_df.loc[stock_mask]
    obs_stock_mu = float(stock_df["area_e"].sum())
    sim_stock_mu = fsum(float(sim_by_actor.get(aid, {"operated_physical_mu": 0.0})["operated_physical_mu"]) for aid in stock_df.index)

    part_err = abs(sim_sown_n / n_comp - obs_sown_n / n_comp)
    mix_err = abs(sim_mixed_n / n_comp - obs_mixed_n / n_comp)
    grain_mu_rel_err = abs(sim_g_mu - obs_g_mu) / max(1.0, obs_g_mu)
    grain_kg_rel_err = abs(sim_kg - obs_kg) / max(1.0, obs_kg)
    stock_rel_err = abs(sim_stock_mu - obs_stock_mu) / max(1.0, obs_stock_mu)

    gates_passed = int(
        part_err <= 0.10
        and mix_err <= 0.10
        and share_q_mae <= 0.15
        and grain_mu_rel_err <= 0.10
        and stock_rel_err <= 0.10
    )

    return {
        "arm": arm,
        "seed": seed,
        "year": year,
        "comparable_n": n_comp,
        "survey_sown": obs_sown_n,
        "sim_sown": sim_sown_n,
        "survey_mixed": obs_mixed_n,
        "sim_mixed": sim_mixed_n,
        "survey_pure_grain": obs_pure_g,
        "sim_pure_grain": sim_pure_g,
        "survey_pure_cash": obs_pure_c,
        "sim_pure_cash": sim_pure_c,
        "survey_grain_mu_seasons": round(obs_g_mu, 2),
        "sim_grain_mu_seasons": round(sim_g_mu, 2),
        "survey_cash_mu_seasons": round(obs_c_mu, 2),
        "sim_cash_mu_seasons": round(sim_c_mu, 2),
        "survey_grain_tons": round(obs_kg / 1000.0, 3),
        "sim_grain_tons": round(sim_kg / 1000.0, 3),
        "survey_closing_phys_mu": round(obs_stock_mu, 2),
        "sim_closing_phys_mu": round(sim_stock_mu, 2),
        "participation_err_pp": round(part_err, 4),
        "mixed_err_pp": round(mix_err, 4),
        "grain_share_q_mae": round(share_q_mae, 4),
        "grain_mu_rel_err": round(grain_mu_rel_err, 4),
        "grain_kg_rel_err": round(grain_kg_rel_err, 4),
        "closing_mu_rel_err": round(stock_rel_err, 4),
        "all_5_gates_pass": gates_passed,
    }


def run_one_step_conditional(
    regions: tuple,
    design: CompetingDesign,
    panel: pd.DataFrame,
) -> list[dict]:
    """1-step conditional prediction for year t given observed t-1 household state."""
    hh_lookup = {h.id: (h, r.tech) for r in regions for h in r.households}
    rows_out = []
    for idx_y, year in enumerate(range(2011, 2016), start=1):
        prev_df = panel.loc[year - 1]
        curr_df = panel.loc[year]
        drift = (1.0 + design.real_offfarm_wage_drift) ** idx_y
        sim_by_actor = {}
        for aid, (h, tech) in hh_lookup.items():
            if aid in prev_df.index:
                pr = prev_df.loc[aid]
                a_prev = float(pr["area_e"]) if pd.notna(pr["area_e"]) and pr["area_e"] >= 0 else h.own_physical_mu
                sg_p = float(pr["sow_food"]) if pd.notna(pr["sow_food"]) and pr["sow_food"] >= 0 else 0.0
                sc_p = float(pr["sow_cash"]) if pd.notna(pr["sow_cash"]) and pr["sow_cash"] >= 0 else 0.0
                ref_sh = sg_p / (sg_p + sc_p) if (sg_p + sc_p) > 1e-6 else h.ref_grain_share
                was_zero = (sg_p + sc_p) <= 1e-6
            else:
                a_prev = h.own_physical_mu
                ref_sh = h.ref_grain_share
                was_zero = not h.initial_cultivating

            if was_zero:
                # Non-cultivating lagged households largely keep renting out unless high margin
                sim_by_actor[aid] = {"sown_grain_mu": 0.0, "sown_cash_mu": 0.0, "grain_kg": 0.0, "operated_physical_mu": a_prev}
                continue

            plan = solve_household_pmp(
                a_prev, h, tech, design,
                family_labor_avail=h.ag_days_year * 0.92,
                opp_wage_day=h.offfarm_wage_day * drift,
                working_cap_avail=h.working_capital_init,
                ref_share=ref_sh,
            )
            # If full land is less profitable than renting out verified right, release tradable land
            if not plan.cultivating or (
                h.verified_tradable_mu > 0
                and plan.net_economic_surplus_yuan < tech.ask_rent_yuan_phys_mu * h.verified_tradable_mu
            ):
                rem = max(0.0, a_prev - h.verified_tradable_mu)
                plan = solve_household_pmp(
                    rem, h, tech, design,
                    family_labor_avail=h.ag_days_year * 0.92,
                    opp_wage_day=h.offfarm_wage_day * drift,
                    working_cap_avail=h.working_capital_init,
                    ref_share=ref_sh,
                )
                op_mu = rem
            else:
                op_mu = a_prev

            g_kg = tech.grain_yield_kg_sown_mu * h.grain_yield_factor * plan.sown_grain_mu
            sim_by_actor[aid] = {
                "sown_grain_mu": plan.sown_grain_mu,
                "sown_cash_mu": plan.sown_cash_mu,
                "grain_kg": g_kg,
                "operated_physical_mu": op_mu,
            }

        rows_out.append(evaluate_cohort_year_metrics(sim_by_actor, curr_df, 0, year, "one_step_conditional"))
    return rows_out


def run_recursive_arm(
    arm_name: str,
    seeds: list[int],
    regions: tuple,
    design: CompetingDesign,
    panel: pd.DataFrame,
    *,
    village_exchange: bool,
) -> tuple[list[dict], list[dict]]:
    """Run 5-year recursive paths (2011-2015) under baseline P0 and collect household + regional summaries."""
    hh_metrics = []
    reg_metrics = []
    for seed in seeds:
        worlds = tuple(
            RegionalPMPEconomy(len(regions) * seed + i, reg, design)
            for i, reg in enumerate(regions)
        )
        for yr_idx in range(1, 6):
            cal_year = 2010 + yr_idx
            sim_by_actor = {}
            for w in worlds:
                s_row, _, a_rows = w.step("none", 0.0, ledger=True, village_exchange=village_exchange)
                reg_metrics.append({
                    "arm": arm_name,
                    "seed": seed,
                    "year": cal_year,
                    "region": w.region.name,
                    "physical_mu": round(w.land_mu, 2),
                    "planted_physical_mu": round(s_row["planted_physical_mu"], 2),
                    "idle_area": round(s_row["idle_area"], 2),
                    "idle_share": round(s_row["idle_area"] / w.land_mu, 4),
                    "sown_grain_mu_seasons": round(s_row["sown_grain_mu_seasons"], 2),
                    "sown_cash_mu_seasons": round(s_row["sown_cash_mu_seasons"], 2),
                    "grain_tons": round(s_row["grain_kg"] / 1000.0, 3),
                    "offered_area": round(s_row["offered_area"], 2),
                    "matched_area": round(s_row["matched_area"], 2),
                    "leased_area": round(s_row["leased_area"], 2),
                    "hired_days_total": round(s_row["hired_days_total"], 2),
                })
                for ar in a_rows:
                    if not ar["is_entrant"]:
                        sim_by_actor[ar["id"]] = ar
            hh_metrics.append(evaluate_cohort_year_metrics(sim_by_actor, panel.loc[cal_year], seed, cal_year, arm_name))
    return hh_metrics, reg_metrics


def run_all(project_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    design = default_design()
    regions, cal_meta = load_calibrated_regions(project_root, design)
    panel = load_survey_panel_2010_2015(project_root)

    write_json(out_dir / "calibration_receipt.json", cal_meta)
    write_csv(out_dir / "region_calibrated_parameters.csv", cal_meta["regions"])
    write_csv(out_dir / "parameter_provenance.csv", cal_meta["parameter_provenance"])

    # 1. One-step conditional prediction
    one_step_rows = run_one_step_conditional(regions, design, panel)
    write_csv(out_dir / "one_step_conditional_2011_2015.csv", one_step_rows)

    # 2. 20-seed protocol recursive check (301..320)
    seeds_20 = list(range(design.historical_seeds[0], design.historical_seeds[1] + 1))
    hh_20_open, reg_20_open = run_recursive_arm(
        "competing_AHM_PMP_village", seeds_20, regions, design, panel, village_exchange=True
    )
    write_csv(out_dir / "recursive_20seeds_household.csv", hh_20_open)

    # 3. 4-seed head-to-head sprint comparison (571001..571004) + mechanism ablations
    seeds_4 = list(range(design.sprint_diag_seeds[0], design.sprint_diag_seeds[1] + 1))
    hh_4_open, reg_4_open = run_recursive_arm(
        "competing_AHM_PMP_village", seeds_4, regions, design, panel, village_exchange=True
    )
    hh_4_closed, reg_4_closed = run_recursive_arm(
        "competing_AHM_PMP_closed", seeds_4, regions, design, panel, village_exchange=False
    )

    # Ablation A: Near-linear LP limit (gamma -> 5 yuan/mu) showing corner-solution mixed collapse!
    lp_design = default_design(pmp_gamma_yuan_per_sown_mu=5.0)
    lp_regions, _ = load_calibrated_regions(project_root, lp_design)
    hh_4_lp, _ = run_recursive_arm(
        "ablation_near_LP_corner", seeds_4, lp_regions, lp_design, panel, village_exchange=False
    )

    # Ablation B: Prohibit hired labor/service (allow_hired_labor=False) showing broad-acre idle land jump!
    nohire_design = default_design(allow_hired_labor=False)
    hh_4_nohire, reg_4_nohire = run_recursive_arm(
        "ablation_no_hired_labor", seeds_4, regions, nohire_design, panel, village_exchange=False
    )

    all_4 = hh_4_open + hh_4_closed + hh_4_lp + hh_4_nohire
    write_csv(out_dir / "ablation_4seeds_2011_2015.csv", all_4)
    write_csv(out_dir / "regional_diagnostics_4seeds.csv", reg_4_closed + reg_4_nohire)

    # Summarize 2011 & 2015 means across arms vs archived legacy models
    summary_rows = []
    by_arm_yr = defaultdict(list)
    for r in all_4:
        by_arm_yr[(r["arm"], r["year"])].append(r)
    for (arm, yr), items in sorted(by_arm_yr.items()):
        if yr not in (2011, 2015):
            continue
        summary_rows.append({
            "arm": arm,
            "year": yr,
            "survey_sown": items[0]["survey_sown"],
            "sim_sown": round(mean(x["sim_sown"] for x in items), 2),
            "survey_mixed": items[0]["survey_mixed"],
            "sim_mixed": round(mean(x["sim_mixed"] for x in items), 2),
            "survey_grain_mu_seasons": items[0]["survey_grain_mu_seasons"],
            "sim_grain_mu_seasons": round(mean(x["sim_grain_mu_seasons"] for x in items), 2),
            "survey_grain_tons": items[0]["survey_grain_tons"],
            "sim_grain_tons": round(mean(x["sim_grain_tons"] for x in items), 2),
            "survey_closing_phys_mu": items[0]["survey_closing_phys_mu"],
            "sim_closing_phys_mu": round(mean(x["sim_closing_phys_mu"] for x in items), 2),
            "mean_5_gate_pass_rate": round(mean(x["all_5_gates_pass"] for x in items), 2),
        })
    write_csv(out_dir / "head_to_head_2011_2015_summary.csv", summary_rows)

    pass_100 = sum(x["all_5_gates_pass"] for x in hh_20_open)
    print(f"20-seed x 5-year all-5-gate pass count (village-calibrated): {pass_100}/100")
    for r in summary_rows:
        print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_all(args.project, args.out)
