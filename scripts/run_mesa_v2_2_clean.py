"""Full Zero-Water V2.2 Experiment Suite (`mesa_agri_abm`, 48 Held-Out Seeds).

Preserves frozen V2.0 outputs (`docs/Mesa农业经营接续V2重构/结果/`) untouched and writes clean V2.2
results to `docs/Mesa农业经营接续V2重构/结果_V2_2净版/`:
  - Exp 0: Stage 1 (2011-2015) Conditional & 5-Year Recursive Survey Cohort Validation + Ablations
  - Exp 1: Fixed-Contract vs. Endogenous Market Liquidity Decomposition & Cash Fungibility
  - Exp 2: Symmetric Rent Price-Area Decomposition, Common-Plot Unit Rent Diff & Equal-Spending Efficiency
  - Exp 3: Village Collective Coordination & Dual-Track vs. Uniform Lease Duration Comparison
  - Exp 4: Endogenous Interprovincial Order-Grain Arbitrage vs. Verified Grain-Area Cap (with Bilateral Conservation)
  - Exp 5: Clean 6-Way Government Policy Benchmark on 48 Independent Holdout Seeds (`574001-574048`)
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import fsum, isclose
import os
from pathlib import Path
import time

import numpy as np

from mesa_agri_abm import (
    AgriSuccessionMesaModel,
    V2Design,
    clean_v2_2_design,
    default_v2_design,
    fit_v2_government_fqi,
    load_v2_calibrated_regions,
    run_policy_episode,
    select_dev_fixed_policies,
    symmetric_rent_decomposition,
)
from mesa_agri_abm.calibration import ADJACENT_SURPLUS_PROVINCES, PROVINCE_NAMES, resolve_project_data_root
from mesa_agri_abm.pmp_production import SubsidySchedule, solve_actor_pmp_v2
from scripts.run_mesa_v2_experiments import (
    T_CRIT_DF31,
    T_CRIT_DF47,
    _episode_worker,
    _exp1_all_inst_seed_worker,
    eval_cohort_metrics,
    load_survey_panel_2010_2015,
    paired_ci,
    write_csv,
    write_json,
)

N_WORKERS = min(8, max(2, (os.cpu_count() or 4) - 2))


def _stage1_worker(args: tuple) -> list[dict]:
    sd, regs, design, arm_label, v_exch = args
    m = AgriSuccessionMesaModel(sd, regs, design, policy_mode="p0")
    out = []
    for st in range(1, 6):
        cy = 2010 + st
        m.step(explicit_action_idx=0, collect_micro_ledger=True, village_exchange=v_exch)
        sim_a = {
            ar["actor"]: ar
            for r in m.regions
            for ar in r.last_actor_rows
            if ar["is_survey_household"]
        }
        ne_idle = next(r.observed["idle_rate"] for r in m.regions if r.name == "东北")
        out.append({"year": cy, "seed": sd, "arm": arm_label, "sim_actors": sim_a, "ne_idle": ne_idle})
    return out


def run_exp0_stage1_v2_2(project_root: Path, out_dir: Path, d20: V2Design, d22: V2Design) -> tuple[list[dict], list[dict]]:
    panel = load_survey_panel_2010_2015(project_root)
    regs20, _ = load_v2_calibrated_regions(project_root, d20)
    regs22, meta22 = load_v2_calibrated_regions(project_root, d22)
    write_json(out_dir / "v2_2_calibration_metadata.json", meta22)

    # 1. One-step conditional (2011-2015) on clean_v2_2_design
    hh_map, tech_map = {}, {}
    for r in regs22:
        for h in r.households:
            hh_map[h.id] = h
            tech_map[h.id] = r.tech
    no_sub = SubsidySchedule("none", 0.0, 50.0, 250.0, True, 0.0)
    one_step_rows = []
    for yr in range(2011, 2016):
        prev_df = panel.loc[yr - 1]
        cur_df = panel.loc[yr]
        drift = (1.0 + d22.real_offfarm_wage_drift) ** (yr - 2010)
        sim_map = {}
        for aid, h in hh_map.items():
            if aid in cur_df.index and float(cur_df.loc[aid, "area_e"]) >= 0:
                op_mu = float(cur_df.loc[aid, "area_e"])
            elif aid in prev_df.index and float(prev_df.loc[aid, "area_e"]) >= 0:
                op_mu = float(prev_df.loc[aid, "area_e"])
            else:
                op_mu = h.own_physical_mu
            ref_sh = h.ref_grain_share
            if aid in prev_df.index:
                ps_g = float(prev_df.loc[aid, "sow_food"]) if prev_df.loc[aid, "sow_food"] >= 0 else 0.0
                ps_c = float(prev_df.loc[aid, "sow_cash"]) if prev_df.loc[aid, "sow_cash"] >= 0 else 0.0
                if ps_g + ps_c > 1e-6:
                    ref_sh = ps_g / (ps_g + ps_c)
            pl = solve_actor_pmp_v2(
                op_mu, 0.0, h, tech_map[aid], d22, no_sub,
                family_labor_avail=h.ag_days_year,
                opp_wage_day=h.offfarm_wage_day * drift,
                working_cap_avail=h.working_capital_init + h.own_physical_mu * tech_map[aid].fertility_base_yuan_phys_mu,
                ref_share=ref_sh,
            )
            sim_map[aid] = {
                "sown_grain_mu": pl.sown_grain_mu,
                "sown_cash_mu": pl.sown_cash_mu,
                "grain_kg": pl.expected_grain_kg,
                "operated_physical_mu": op_mu,
            }
        m = eval_cohort_metrics(sim_map, cur_df, 0, yr, "v2_2_one_step")
        one_step_rows.append(m)
    write_csv(out_dir / "stage1_one_step_v2_2.csv", one_step_rows)

    # 2. Five-year recursive validation (20 seeds for V2.0 & V2.2; 8 seeds for structural ablations)
    seeds_20 = list(range(301, 321))
    specs = [
        ("v2_0_baseline", regs20, d20, seeds_20, True),
        ("v2_2_clean_full", regs22, d22, seeds_20, True),
        ("ablation_no_restart_cost", regs22, d22.with_overrides(restart_cultivation_fixed_days=0.0, restart_cultivation_fixed_yuan=0.0), seeds_20[:8], True),
        ("ablation_no_rent_foodc_offset", regs22, d22.with_overrides(rent_offsets_subsistence_wedge=False, dynamic_expected_rent_update=False), seeds_20[:8], True),
        ("ablation_near_lp_gamma5", regs22, d22.with_overrides(pmp_gamma_yuan_per_sown_mu=5.0), seeds_20[:8], True),
        ("ablation_no_machinery_service", regs22, d22.with_overrides(enable_machinery_service=False, allow_hired_labor=False), seeds_20[:8], True),
    ]

    yearly_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for arm_name, regs, d_spec, s_list, v_exch in specs:
            by_yr: dict[int, list[dict]] = defaultdict(list)
            ne_2015: list[float] = []
            for seed_list in pool.map(_stage1_worker, [(sd, regs, d_spec, arm_name, v_exch) for sd in s_list]):
                for item in seed_list:
                    cy = item["year"]
                    by_yr[cy].append(eval_cohort_metrics(item["sim_actors"], panel.loc[cy], item["seed"], cy, arm_name))
                    if cy == 2015:
                        ne_2015.append(item["ne_idle"])
            for cy, m_list in sorted(by_yr.items()):
                g_m = float(np.mean([x["sim_grain_tons"] for x in m_list]))
                obs_g = m_list[0]["survey_grain_tons"]
                yearly_rows.append({
                    "arm": arm_name,
                    "n_seeds": len(s_list),
                    "year": cy,
                    "comparable_n": m_list[0]["comparable_n"],
                    "survey_sown": m_list[0]["survey_sown"],
                    "sim_sown_mean": round(float(np.mean([x["sim_sown"] for x in m_list])), 2),
                    "participation_err_pp": round(float(np.mean([x["participation_err_pp"] for x in m_list])) * 100.0, 2),
                    "survey_mixed": m_list[0]["survey_mixed"],
                    "sim_mixed_mean": round(float(np.mean([x["sim_mixed"] for x in m_list])), 2),
                    "mixed_err_pp": round(float(np.mean([x["mixed_err_pp"] for x in m_list])) * 100.0, 2),
                    "survey_grain_tons": obs_g,
                    "sim_grain_tons_mean": round(g_m, 2),
                    "grain_tons_rel_err_pct": round(abs(g_m - obs_g) / max(1.0, obs_g) * 100.0, 2),
                    "grain_share_q_mae_pct": round(float(np.mean([x["grain_share_q_mae"] for x in m_list])) * 100.0, 2),
                    "ne_idle_rate_2015_pct": round(float(np.mean(ne_2015)) * 100.0, 2) if cy == 2015 else None,
                })
    write_csv(out_dir / "stage1_cohort_validation_2011_2015.csv", yearly_rows)
    return one_step_rows, yearly_rows


def run_exp1_and_exp2_v2_2(regions, out_dir: Path, d22: V2Design, eval_seeds: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    # 1. Fixed-Contract vs. Endogenous Market Two-Step Decomposition
    inst_actions = [
        ("producer_pre", 1),
        ("producer_post", 2),
        ("scale_takeover_pre", 3),
        ("scale_takeover_post", 4),
        ("service_trusteeship", 5),
    ]
    d_ban = d22.with_overrides(cash_subsidy_fungible_in_bidding=False)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        seed_maps = list(pool.map(_exp1_all_inst_seed_worker, [(sd, regions, d22, d_ban, inst_actions) for sd in eval_seeds]))

    exp1_rows = []
    for inst_name, _ in inst_actions:
        res_list = [sm[inst_name] for sm in seed_maps]
        fc_m, fc_lo, fc_hi = paired_ci([x[0] for x in res_list], T_CRIT_DF47)
        re_m, re_lo, re_hi = paired_ci([x[1] for x in res_list], T_CRIT_DF47)
        tot_m, tot_lo, tot_hi = paired_ci([x[2] for x in res_list], T_CRIT_DF47)
        exp1_rows.append({
            "instrument": inst_name,
            "fixed_contract_grain_gain_tons": round(fc_m, 3),
            "fixed_contract_ci95": f"[{fc_lo:.2f}, {fc_hi:.2f}]",
            "market_reallocation_grain_gain_tons": round(re_m, 3),
            "market_reallocation_ci95": f"[{re_lo:.2f}, {re_hi:.2f}]",
            "total_endogenous_grain_gain_tons": round(tot_m, 3),
            "total_endogenous_ci95": f"[{tot_lo:.2f}, {tot_hi:.2f}]",
            "endogenous_spending_wan": round(float(np.mean([x[4] for x in res_list])), 3),
            "endogenous_fungible_rent_diff_wan": round(float(np.mean([x[6] for x in res_list])), 4),
            "legacy_rent_ban_rent_diff_wan": round(float(np.mean([x[7] for x in res_list])), 4),
            "strict_succession_diff_mu_yr": round(float(np.mean([x[10] for x in res_list])), 2),
        })
    write_csv(out_dir / "exp1_fixed_contract_liquidity_v2_2.csv", exp1_rows)

    # 2. Symmetric Rent Price-Area Decomposition + Common-Plot Diff + Equal Spending
    arms = [
        ("p0", "p0", 0),
        ("producer_pre", "uniform_fixed", 1),
        ("producer_post", "uniform_fixed", 2),
        ("scale_takeover_pre", "uniform_fixed", 3),
        ("scale_takeover_post", "uniform_fixed", 4),
        ("service_trusteeship", "uniform_fixed", 5),
        ("order_farming_capped", "uniform_fixed", 28),
    ]
    ep_tasks = [(sd, regions, d22, mode, lbl, a_idx, None, False) for lbl, mode, a_idx in arms for sd in eval_seeds]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        ep_map = {(ep["policy_label"], ep["seed"]): ep for ep in pool.map(_episode_worker, ep_tasks)}

    exp2_rows = []
    for lbl, _, _ in arms:
        g_tot, g_diff_l, sp_tot, eff_l = [], [], [], []
        tot_r_d, price_e, area_e, comm_u_d, leased_l, unit_r_l = [], [], [], [], [], []
        for sd in eval_seeds:
            ep, ep0 = ep_map[(lbl, sd)], ep_map[("p0", sd)]
            mj, mj0 = ep["main_joint"], ep0["main_joint"]
            ny = len(mj)
            gt = fsum(r["grain_kg"] for r in mj) / 1000.0
            g0 = fsum(r["grain_kg"] for r in mj0) / 1000.0
            sp = fsum(r["spending"] for r in mj) / 10_000.0
            g_tot.append(gt); g_diff_l.append(gt - g0); sp_tot.append(sp)
            if sp > 1e-4:
                eff_l.append(((gt - g0) * 1000.0) / (sp * 100.0))
            leased_l.append(fsum(r["leased_area_mu"] for r in mj) / ny)
            unit_r_l.append(float(np.mean([r["unit_rent_yuan_mu"] for r in mj])))

            pe_sd, ae_sd, td_sd, cu_l = 0.0, 0.0, 0.0, []
            for k in range(ny):
                dec = symmetric_rent_decomposition(
                    mj[k]["leased_area_mu"], mj[k]["total_rent_yuan"],
                    mj0[k]["leased_area_mu"], mj0[k]["total_rent_yuan"],
                    ep["parcel_rents_by_year"][k + 1], ep0["parcel_rents_by_year"][k + 1],
                )
                pe_sd += dec["price_effect_yuan"]; ae_sd += dec["area_effect_yuan"]
                td_sd += dec["total_rent_diff_yuan"]; cu_l.append(dec["common_unit_rent_diff_yuan_mu"])
            tot_r_d.append(td_sd / 10_000.0); price_e.append(pe_sd / 10_000.0)
            area_e.append(ae_sd / 10_000.0); comm_u_d.append(float(np.mean(cu_l)))

        gm, glo, ghi = paired_ci(g_diff_l, T_CRIT_DF47)
        sp_m = float(np.mean(sp_tot))
        em, elo, ehi = paired_ci(eff_l, T_CRIT_DF47) if eff_l else (0.0, 0.0, 0.0)
        pe_m = float(np.mean(price_e))
        exp2_rows.append({
            "policy": lbl,
            "grain_8yr_tons": round(float(np.mean(g_tot)), 2),
            "grain_gain_vs_p0_tons": round(gm, 2),
            "grain_gain_ci95": f"[{glo:.2f}, {ghi:.2f}]",
            "spending_8yr_wan": round(sp_m, 3),
            "kg_per_100yuan_spent": round((gm * 1000.0) / (sp_m * 100.0), 2) if sp_m > 1e-4 else 0.0,
            "kg_per_100yuan_ci95": f"[{elo:.2f}, {ehi:.2f}]" if eff_l else "—",
            "leased_area_mu_yr": round(float(np.mean(leased_l)), 2),
            "unit_rent_yuan_mu": round(float(np.mean(unit_r_l)), 2),
            "total_rent_diff_wan": round(float(np.mean(tot_r_d)), 3),
            "rent_price_effect_wan": round(pe_m, 3),
            "rent_area_effect_wan": round(float(np.mean(area_e)), 3),
            "pure_price_capitalization_pct": round((pe_m / sp_m) * 100.0, 2) if sp_m > 1e-4 else 0.0,
            "common_unit_rent_diff_yuan_mu": round(float(np.mean(comm_u_d)), 2),
        })
    write_csv(out_dir / "exp2_rent_capitalization_v2_2.csv", exp2_rows)

    # 3. Equal-Spending Comparison matched to scale_takeover_pre realized outlay
    target_wan = next(r["spending_8yr_wan"] for r in exp2_rows if r["policy"] == "scale_takeover_pre")
    iso_budget = int(round((target_wan * 10_000.0) / d22.main_years))
    d_iso = d22.with_overrides(annual_common_budget_yuan=iso_budget)
    iso_specs = [("producer_pre", 1, d_iso), ("scale_takeover_pre", 3, d22), ("service_trusteeship", 5, d_iso), ("order_farming_capped", 28, d_iso)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        iso_tasks = [(sd, regions, d_use, "uniform_fixed", lbl, a_idx, None, False) for lbl, a_idx, d_use in iso_specs for sd in eval_seeds]
        iso_map = {(ep["policy_label"], ep["seed"]): ep for ep in pool.map(_episode_worker, iso_tasks)}

    iso_rows = []
    for lbl, _, _ in iso_specs:
        gd_l, sp_l, eff_l, pe_l, cu_l = [], [], [], [], []
        for sd in eval_seeds:
            ep, ep0 = iso_map[(lbl, sd)], ep_map[("p0", sd)]
            gd = (fsum(r["grain_kg"] for r in ep["main_joint"]) - fsum(r["grain_kg"] for r in ep0["main_joint"])) / 1000.0
            sp = fsum(r["spending"] for r in ep["main_joint"]) / 10_000.0
            gd_l.append(gd); sp_l.append(sp); eff_l.append((gd * 1000.0) / max(1e-4, sp * 100.0))
            pe_sd, cu_sd = 0.0, []
            for k in range(d22.main_years):
                dec = symmetric_rent_decomposition(
                    ep["main_joint"][k]["leased_area_mu"], ep["main_joint"][k]["total_rent_yuan"],
                    ep0["main_joint"][k]["leased_area_mu"], ep0["main_joint"][k]["total_rent_yuan"],
                    ep["parcel_rents_by_year"][k + 1], ep0["parcel_rents_by_year"][k + 1],
                )
                pe_sd += dec["price_effect_yuan"]; cu_sd.append(dec["common_unit_rent_diff_yuan_mu"])
            pe_l.append(pe_sd / 10_000.0); cu_l.append(float(np.mean(cu_sd)))
        gm, glo, ghi = paired_ci(gd_l, T_CRIT_DF47)
        em, elo, ehi = paired_ci(eff_l, T_CRIT_DF47)
        sp_m = float(np.mean(sp_l))
        iso_rows.append({
            "policy": lbl,
            "realized_8yr_spending_wan": round(sp_m, 3),
            "grain_gain_vs_p0_tons": round(gm, 2),
            "grain_gain_ci95": f"[{glo:.2f}, {ghi:.2f}]",
            "kg_per_100yuan_spent": round(em, 2),
            "kg_per_100yuan_ci95": f"[{elo:.2f}, {ehi:.2f}]",
            "pure_price_capitalization_pct": round(float(np.mean(pe_l)) / max(1e-4, sp_m) * 100.0, 2),
            "common_unit_rent_diff_yuan_mu": round(float(np.mean(cu_l)), 2),
        })
    write_csv(out_dir / "exp2_equal_spending_v2_2.csv", iso_rows)
    return exp1_rows, exp2_rows, iso_rows


def run_exp3_village_and_leases_v2_2(regions, out_dir: Path, d22: V2Design, eval_seeds: list[int]) -> list[dict]:
    d_1yr = d22.with_overrides(enable_heterogeneous_lease_terms=False, lease_duration_years=1)
    d_3yr = d22.with_overrides(enable_heterogeneous_lease_terms=False, lease_duration_years=3)
    d_vill = d22.with_overrides(enable_village_coordination=True)

    arms = [
        ("p0_dual", "1/3/4年双轨", d22, "p0", 0),
        ("p0_1yr", "统一1年短租", d_1yr, "p0", 0),
        ("p0_3yr", "统一3年长租", d_3yr, "p0", 0),
        ("scale_alone_dual", "1/3/4年双轨", d22, "uniform_fixed", 3),
        ("scale_plus_village", "1/3/4年双轨", d_vill, "uniform_fixed", 3),
        ("service_alone_dual", "1/3/4年双轨", d22, "uniform_fixed", 5),
        ("service_plus_village", "1/3/4年双轨", d_vill, "uniform_fixed", 5),
        ("order_alone_dual", "1/3/4年双轨", d22, "uniform_fixed", 28),
        ("order_plus_village", "1/3/4年双轨", d_vill, "uniform_fixed", 28),
        ("portfolio29_1yr", "统一1年短租", d_1yr, "regional_fixed", 29),
        ("portfolio29_dual", "1/3/4年双轨", d22, "regional_fixed", 29),
        ("portfolio29_3yr", "统一3年长租", d_3yr, "regional_fixed", 29),
    ]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        ep_map = {(ep["policy_label"], ep["seed"]): ep for ep in pool.map(
            _episode_worker, [(sd, regions, d_use, mode, lbl, a_idx, None, True) for lbl, _, d_use, mode, a_idx in arms for sd in eval_seeds]
        )}

    rows = []
    for lbl, reg_lbl, _, _, _ in arms:
        p0_key = "p0_1yr" if reg_lbl == "统一1年短租" else ("p0_3yr" if reg_lbl == "统一3年长租" else "p0_dual")
        gd_l, sp_l, wd_l, succ_l, unm_l, unpl_l = [], [], [], [], [], []
        for sd in eval_seeds:
            ep, ep0 = ep_map[(lbl, sd)], ep_map[(p0_key, sd)]
            ny = len(ep["main_joint"])
            gd_l.append((fsum(r["grain_kg"] for r in ep["main_joint"]) - fsum(r["grain_kg"] for r in ep0["main_joint"])) / 1000.0)
            sp_l.append(fsum(r["spending"] for r in ep["main_joint"]) / 10_000.0)
            wd_l.append((fsum(r["grain_kg"] for r in ep["withdrawal_joint"]) - fsum(r["grain_kg"] for r in ep0["withdrawal_joint"])) / 1000.0)
            succ_l.append(fsum(r["fate_2_strict_succession_grain_mu"] for r in ep["main_joint"]) / ny)
            unm_l.append(fsum(r["unmatched_area_mu"] for r in ep["main_joint"]) / ny)
            unpl_l.append(fsum(r["fate_5_leased_unplanted_mu"] for r in ep["main_joint"]) / ny)

        gm, glo, ghi = paired_ci(gd_l, T_CRIT_DF47)
        wm, wlo, whi = paired_ci(wd_l, T_CRIT_DF47)
        sp_m = float(np.mean(sp_l))
        rows.append({
            "arm": lbl,
            "lease_regime": reg_lbl,
            "grain_gain_vs_p0_tons": round(gm, 2),
            "grain_gain_ci95": f"[{glo:.2f}, {ghi:.2f}]",
            "spending_8yr_wan": round(sp_m, 3),
            "kg_per_100yuan_spent": round((gm * 1000.0) / (sp_m * 100.0), 2) if sp_m > 1e-4 else 0.0,
            "strict_succession_mu_yr": round(float(np.mean(succ_l)), 2),
            "unmatched_idle_mu_yr": round(float(np.mean(unm_l)), 2),
            "leased_unplanted_mu_yr": round(float(np.mean(unpl_l)), 2),
            "withdrawal_yr7_8_gain_tons": round(wm, 2),
            "withdrawal_yr7_8_ci95": f"[{wlo:.2f}, {whi:.2f}]",
        })
    write_csv(out_dir / "exp3_village_leases_v2_2.csv", rows)
    return rows


def _arbitrage_worker(args: tuple) -> dict:
    sd, regs, d_arm, arm_lbl, joint_act = args
    m = AgriSuccessionMesaModel(sd, regs, d_arm, policy_mode="fixed", fixed_action_idx=0)
    east_g, east_cash_mu, cent_g, arb_kg, leak_y, loc_sub_y, pool_init, pool_ded, gap_max, spend_tot = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    pair_acc: dict[str, float] = defaultdict(float)
    for _ in range(d_arm.main_years):
        j, r_list, _ = m.step(explicit_joint_action=joint_act)
        east_g += r_list[0]["grain_kg"] / 1000.0
        east_cash_mu += r_list[0]["cash_physical_mu"] / d_arm.main_years
        cent_g += r_list[1]["grain_kg"] / 1000.0
        arb_kg += j["interprov_arbitrage_grain_kg"]
        leak_y += j["interprov_arbitrage_subsidy_leakage_yuan"]
        loc_sub_y += j["local_verified_order_subsidy_yuan"]
        pool_init += j["central_surplus_pool_initial_kg"]
        pool_ded += j["central_surplus_pool_deducted_kg"]
        gap_max = max(gap_max, abs(j["bilateral_conservation_gap_kg"]))
        spend_tot += j["spending"]
        for k, v in m.last_interprov_pair_flows.items():
            pair_acc[k] += v / 1000.0
    return {
        "arm": arm_lbl, "seed": sd,
        "east_grain_tons": east_g, "east_cash_mu_yr": east_cash_mu, "central_grain_tons": cent_g,
        "arb_tons": arb_kg / 1000.0, "leak_wan": leak_y / 10_000.0, "loc_sub_wan": loc_sub_y / 10_000.0,
        "pool_init_tons": pool_init / 1000.0, "pool_ded_tons": pool_ded / 1000.0,
        "gap_max_kg": gap_max, "spend_wan": spend_tot / 10_000.0, "pair_tons": dict(pair_acc),
    }


def run_exp4_interprovincial_arbitrage_v2_2(regions, out_dir: Path, d22: V2Design, eval_seeds: list[int]) -> tuple[list[dict], list[dict]]:
    d_arb = d22.with_overrides(enable_interprovincial_arbitrage=True)
    east_only = (("order_farming", 75_000), ("none", 0), ("none", 0), ("none", 0))
    nat_unif = AgriSuccessionMesaModel(eval_seeds[0], regions, d_arb).menu[28]
    p0_act = (("none", 0), ("none", 0), ("none", 0), ("none", 0))

    arms = [
        ("p0_no_order_bonus", d_arb, p0_act),
        ("east_order_zero_bonus_control", d_arb.with_overrides(order_max_price_bonus_ratio=0.0, order_quota_mode="loose_area_quota"), east_only),
        ("east_order_loose_quota", d_arb.with_overrides(order_quota_mode="loose_area_quota"), east_only),
        ("east_order_verified_cap_400", d_arb.with_overrides(order_quota_mode="verified_grain_area_cap", order_verified_cap_kg_per_sown_mu=400.0), east_only),
        ("east_order_verified_cap_350", d_arb.with_overrides(order_quota_mode="verified_grain_area_cap", order_verified_cap_kg_per_sown_mu=350.0), east_only),
        ("east_order_verified_cap_400_village", d_arb.with_overrides(order_quota_mode="verified_grain_area_cap", order_verified_cap_kg_per_sown_mu=400.0, enable_village_coordination=True), east_only),
        ("national_order_loose_quota", d_arb.with_overrides(order_quota_mode="loose_area_quota"), nat_unif),
        ("national_order_verified_cap_400", d_arb.with_overrides(order_quota_mode="verified_grain_area_cap", order_verified_cap_kg_per_sown_mu=400.0), nat_unif),
        ("sens_freight_009_loose", d_arb.with_overrides(order_quota_mode="loose_area_quota", interprov_transport_cost_yuan_kg=0.09), east_only),
        ("sens_freight_018_loose", d_arb.with_overrides(order_quota_mode="loose_area_quota", interprov_transport_cost_yuan_kg=0.18), east_only),
    ]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        res_all = list(pool.map(_arbitrage_worker, [(sd, regions, d_u, lbl, act) for lbl, d_u, act in arms for sd in eval_seeds]))

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in res_all:
        by_arm[r["arm"]].append(r)

    p0_east = [r["east_grain_tons"] for r in sorted(by_arm["p0_no_order_bonus"], key=lambda x: x["seed"])]
    p0_cash = [r["east_cash_mu_yr"] for r in sorted(by_arm["p0_no_order_bonus"], key=lambda x: x["seed"])]

    summary_rows = []
    pair_rows = []
    for lbl, _, _ in arms:
        lst = sorted(by_arm[lbl], key=lambda x: x["seed"])
        eg_diff = [lst[i]["east_grain_tons"] - p0_east[i] for i in range(len(lst))]
        cm_diff = [lst[i]["east_cash_mu_yr"] - p0_cash[i] for i in range(len(lst))]
        egm, eglo, eghi = paired_ci(eg_diff, T_CRIT_DF47)
        akm, aklo, akhi = paired_ci([x["arb_tons"] for x in lst], T_CRIT_DF47)
        leak_m = float(np.mean([x["leak_wan"] for x in lst]))
        loc_m = float(np.mean([x["loc_sub_wan"] for x in lst]))
        tot_ord_m = max(1e-9, leak_m + loc_m)
        gap_max = max(x["gap_max_kg"] for x in lst)
        assert isclose(gap_max, 0.0, abs_tol=1e-6)
        summary_rows.append({
            "arm": lbl,
            "east_grain_8yr_tons": round(float(np.mean([x["east_grain_tons"] for x in lst])), 2),
            "east_grain_gain_vs_p0_tons": round(egm, 2),
            "east_grain_gain_ci95": f"[{eglo:.2f}, {eghi:.2f}]",
            "east_cash_mu_diff_yr": round(float(np.mean(cm_diff)), 2),
            "central_grain_8yr_tons": round(float(np.mean([x["central_grain_tons"] for x in lst])), 2),
            "central_surplus_pool_tons": round(float(np.mean([x["pool_init_tons"] for x in lst])), 2),
            "central_pool_deducted_tons": round(float(np.mean([x["pool_ded_tons"] for x in lst])), 2),
            "interprov_arbitrage_tons": round(akm, 2),
            "interprov_arbitrage_ci95": f"[{aklo:.2f}, {akhi:.2f}]",
            "spending_8yr_wan": round(float(np.mean([x["spend_wan"] for x in lst])), 3),
            "local_order_sub_wan": round(loc_m, 3),
            "arb_subsidy_leakage_wan": round(leak_m, 3),
            "arb_leakage_share_pct": round((leak_m / tot_ord_m) * 100.0 if (leak_m + loc_m) > 1e-6 else 0.0, 2),
            "bilateral_conservation_gap_kg": round(gap_max, 8),
        })
        if lbl in ("east_order_loose_quota", "east_order_verified_cap_400", "east_order_verified_cap_350"):
            all_pairs = sorted({pk for x in lst for pk in x["pair_tons"]})
            for pk in all_pairs:
                pair_rows.append({
                    "arm": lbl,
                    "border_pair": pk,
                    "arbitrage_8yr_tons": round(float(np.mean([x["pair_tons"].get(pk, 0.0) for x in lst])), 2),
                })

    write_csv(out_dir / "exp4_interprovincial_order_arbitrage_v2_2.csv", summary_rows)
    write_csv(out_dir / "exp4_border_pair_flows_v2_2.csv", pair_rows)
    return summary_rows, pair_rows


def run_exp5_government_6way_v2_2(regions, out_dir: Path, d22: V2Design, eval_seeds: list[int]) -> dict:
    dev_seeds = list(range(d22.dev_seeds[0], d22.dev_seeds[1] + 1))
    train_seeds = list(range(d22.train_seeds[0], d22.train_seeds[1] + 1))

    dev_sel = select_dev_fixed_policies(dev_seeds, regions, d22, max_workers=N_WORKERS)
    write_csv(out_dir / "exp5_dev_30actions_leaderboard_v2_2.csv", dev_sel["leaderboard"])

    fqi_fit = fit_v2_government_fqi(train_seeds, regions, d22, max_workers=N_WORKERS)
    u_idx = dev_sel["best_uniform_fixed_idx"]
    r_idx = dev_sel["best_regional_fixed_idx"]
    write_json(out_dir / "exp5_fqi_training_summary_v2_2.json", {
        "n_train_seeds": fqi_fit["n_train_seeds"],
        "n_actions": fqi_fit["n_actions"],
        "best_uniform_fixed_idx": u_idx,
        "best_regional_fixed_idx": r_idx,
        "fit_r2_q": fqi_fit["fit_r2_q"],
        "fit_r2_myopic": fqi_fit["fit_r2_myopic"],
    })

    six_arms = [
        ("p0", "p0", 0, None),
        ("uniform_fixed", "uniform_fixed", u_idx, None),
        ("regional_fixed", "regional_fixed", r_idx, None),
        ("transparent_rule", "transparent_rule", 0, None),
        ("myopic", "myopic", 0, fqi_fit["myopic_weights"]),
        ("gov_rl", "gov_rl", 0, fqi_fit["q_weights"]),
    ]

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        ep_list = list(pool.map(_episode_worker, [
            (sd, regions, d22, mode, lbl, f_idx, w_mat, True)
            for lbl, mode, f_idx, w_mat in six_arms
            for sd in eval_seeds
        ]))
    ep_store = {(ep["policy_label"], ep["seed"]): ep for ep in ep_list}

    # Action trajectories (full seed x year transparency!)
    traj_rows = []
    yearly_rows = []
    for lbl, _, _, _ in six_arms:
        for yr_i in range(d22.main_years):
            g_yr = [ep_store[(lbl, sd)]["main_joint"][yr_i]["grain_kg"] / 1000.0 for sd in eval_seeds]
            g0_yr = [ep_store[("p0", sd)]["main_joint"][yr_i]["grain_kg"] / 1000.0 for sd in eval_seeds]
            sp_yr = [ep_store[(lbl, sd)]["main_joint"][yr_i]["spending"] / 10_000.0 for sd in eval_seeds]
            acts_yr = [int(ep_store[(lbl, sd)]["main_joint"][yr_i]["action_idx"]) for sd in eval_seeds]
            gm, glo, ghi = paired_ci([g_yr[k] - g0_yr[k] for k in range(len(eval_seeds))], T_CRIT_DF47)
            yearly_rows.append({
                "policy_label": lbl,
                "window": "main",
                "year": yr_i + 1,
                "grain_tons_mean": round(float(np.mean(g_yr)), 2),
                "grain_gain_vs_p0_tons": round(gm, 2),
                "grain_gain_ci95_low": round(glo, 2),
                "grain_gain_ci95_high": round(ghi, 2),
                "spending_wan_mean": round(float(np.mean(sp_yr)), 3),
                "modal_action_idx": int(max(set(acts_yr), key=acts_yr.count)),
            })
        for wd_k in range(d22.main_years - d22.withdrawal_after):
            g_yr = [ep_store[(lbl, sd)]["withdrawal_joint"][wd_k]["grain_kg"] / 1000.0 for sd in eval_seeds]
            g0_yr = [ep_store[("p0", sd)]["withdrawal_joint"][wd_k]["grain_kg"] / 1000.0 for sd in eval_seeds]
            gm, glo, ghi = paired_ci([g_yr[k] - g0_yr[k] for k in range(len(eval_seeds))], T_CRIT_DF47)
            yearly_rows.append({
                "policy_label": lbl,
                "window": "withdrawal",
                "year": d22.withdrawal_after + wd_k + 1,
                "grain_tons_mean": round(float(np.mean(g_yr)), 2),
                "grain_gain_vs_p0_tons": round(gm, 2),
                "grain_gain_ci95_low": round(glo, 2),
                "grain_gain_ci95_high": round(ghi, 2),
                "spending_wan_mean": 0.0,
                "modal_action_idx": 0,
            })
        for sd in eval_seeds:
            ep = ep_store[(lbl, sd)]
            traj_rows.append({
                "policy_label": lbl,
                "seed": sd,
                **{f"yr{r['year']}_action": int(r["action_idx"]) for r in ep["main_joint"]},
                "grain_8yr_tons": round(fsum(r["grain_kg"] for r in ep["main_joint"]) / 1000.0, 2),
                "spending_8yr_wan": round(fsum(r["spending"] for r in ep["main_joint"]) / 10_000.0, 3),
                "wd_yr7_8_tons": round(fsum(r["grain_kg"] for r in ep["withdrawal_joint"]) / 1000.0, 2),
            })
    write_csv(out_dir / "exp5_government_6way_v2_2_yearly.csv", yearly_rows)
    write_csv(out_dir / "exp5_government_6way_v2_2_action_trajectories.csv", traj_rows)

    summary_rows = []
    for lbl, _, _, _ in six_arms:
        g_l, gd_l, sp_l, eff_l, wd_d_l, wd_tot_l, pe_l, succ_l = [], [], [], [], [], [], [], []
        for sd in eval_seeds:
            ep, ep0 = ep_store[(lbl, sd)], ep_store[("p0", sd)]
            gt = fsum(r["grain_kg"] for r in ep["main_joint"]) / 1000.0
            g0 = fsum(r["grain_kg"] for r in ep0["main_joint"]) / 1000.0
            sp = fsum(r["spending"] for r in ep["main_joint"]) / 10_000.0
            wt = fsum(r["grain_kg"] for r in ep["withdrawal_joint"]) / 1000.0
            w0 = fsum(r["grain_kg"] for r in ep0["withdrawal_joint"]) / 1000.0
            g_l.append(gt); gd_l.append(gt - g0); sp_l.append(sp)
            if sp > 1e-4:
                eff_l.append(((gt - g0) * 1000.0) / (sp * 100.0))
            wd_tot_l.append(wt); wd_d_l.append(wt - w0)
            succ_l.append(fsum(r["fate_2_strict_succession_grain_mu"] for r in ep["main_joint"]) / d22.main_years)
            pe_l.append(fsum(
                symmetric_rent_decomposition(
                    ep["main_joint"][k]["leased_area_mu"], ep["main_joint"][k]["total_rent_yuan"],
                    ep0["main_joint"][k]["leased_area_mu"], ep0["main_joint"][k]["total_rent_yuan"],
                    ep["parcel_rents_by_year"][k + 1], ep0["parcel_rents_by_year"][k + 1],
                )["price_effect_yuan"]
                for k in range(d22.main_years)
            ) / 10_000.0)

        gm, glo, ghi = paired_ci(gd_l, T_CRIT_DF47)
        spm, splo, sphi = paired_ci(sp_l, T_CRIT_DF47)
        efm, eflo, efhi = paired_ci(eff_l, T_CRIT_DF47) if eff_l else (0.0, 0.0, 0.0)
        wdm, wdlo, wdhi = paired_ci(wd_d_l, T_CRIT_DF47)
        acts_seq = [int(max(
            set(int(ep_store[(lbl, sd)]["main_joint"][yr_i]["action_idx"]) for sd in eval_seeds),
            key=lambda a: sum(1 for sd in eval_seeds if int(ep_store[(lbl, sd)]["main_joint"][yr_i]["action_idx"]) == a),
        )) for yr_i in range(d22.main_years)]

        summary_rows.append({
            "policy_label": lbl,
            "grain_8yr_tons": round(float(np.mean(g_l)), 2),
            "grain_gain_vs_p0_tons": round(gm, 2),
            "grain_gain_ci95": f"[{glo:.2f}, {ghi:.2f}]",
            "spending_8yr_wan": round(spm, 3),
            "spending_ci95": f"[{splo:.3f}, {sphi:.3f}]",
            "kg_per_100yuan_spent": round((gm * 1000.0) / (spm * 100.0), 2) if spm > 1e-4 else 0.0,
            "kg_per_100yuan_ci95": f"[{eflo:.2f}, {efhi:.2f}]" if eff_l else "—",
            "withdrawal_yr7_8_tons": round(float(np.mean(wd_tot_l)), 2),
            "withdrawal_yr7_8_gain_tons": round(wdm, 2),
            "withdrawal_gain_ci95": f"[{wdlo:.2f}, {wdhi:.2f}]",
            "pure_price_cap_pct": round(float(np.mean(pe_l)) / max(1e-4, spm) * 100.0, 2) if spm > 1e-4 else 0.0,
            "strict_succession_mu_yr": round(float(np.mean(succ_l)), 2),
            "modal_action_seq": str(acts_seq),
        })
    write_csv(out_dir / "exp5_government_6way_v2_2_summary.csv", summary_rows)

    contrasts = []
    for ref_lbl in ("p0", "uniform_fixed", "regional_fixed", "transparent_rule", "myopic"):
        gd, spd, wdd = [], [], []
        for sd in eval_seeds:
            e_rl, e_rf = ep_store[("gov_rl", sd)], ep_store[(ref_lbl, sd)]
            gd.append((fsum(r["grain_kg"] for r in e_rl["main_joint"]) - fsum(r["grain_kg"] for r in e_rf["main_joint"])) / 1000.0)
            spd.append((fsum(r["spending"] for r in e_rl["main_joint"]) - fsum(r["spending"] for r in e_rf["main_joint"])) / 10_000.0)
            wdd.append((fsum(r["grain_kg"] for r in e_rl["withdrawal_joint"]) - fsum(r["grain_kg"] for r in e_rf["withdrawal_joint"])) / 1000.0)
        gm, glo, ghi = paired_ci(gd, T_CRIT_DF47)
        sm, slo, shi = paired_ci(spd, T_CRIT_DF47)
        wm, wlo, whi = paired_ci(wdd, T_CRIT_DF47)
        contrasts.append({
            "contrast": f"gov_rl_minus_{ref_lbl}",
            "grain_8yr_diff_tons": round(gm, 2),
            "grain_ci95": f"[{glo:.2f}, {ghi:.2f}]",
            "spending_8yr_diff_wan": round(sm, 3),
            "spending_ci95": f"[{slo:.3f}, {shi:.3f}]",
            "withdrawal_diff_tons": round(wm, 2),
            "withdrawal_ci95": f"[{wlo:.2f}, {whi:.2f}]",
        })
    write_csv(out_dir / "exp5_government_6way_v2_2_contrasts.csv", contrasts)
    return {"summary": summary_rows, "contrasts": contrasts, "u_idx": u_idx, "r_idx": r_idx}


def main() -> None:
    t0 = time.perf_counter()
    project_root = resolve_project_data_root(Path(__file__))
    out_dir = Path(__file__).resolve().parents[1] / "docs/Mesa农业经营接续V2重构/结果_V2_2净版"
    out_dir.mkdir(parents=True, exist_ok=True)

    d20 = default_v2_design()
    d22 = clean_v2_2_design()
    regions22, _ = load_v2_calibrated_regions(project_root, d22)
    eval_seeds = list(range(d22.eval_seeds[0], d22.eval_seeds[1] + 1))

    print("[1/5] Running Stage 1 (2011-2015 Conditional & 5-Yr Recursive Validation + Ablations)...")
    run_exp0_stage1_v2_2(project_root, out_dir, d20, d22)

    print("[2/5] Running Exp 1 & 2 (Liquidity Split, Rent Decomposition & Equal-Spending Efficiency)...")
    run_exp1_and_exp2_v2_2(regions22, out_dir, d22, eval_seeds)

    print("[3/5] Running Exp 3 (Village Coordination & Dual-Track vs Uniform Lease Comparison)...")
    run_exp3_village_and_leases_v2_2(regions22, out_dir, d22, eval_seeds)

    print("[4/5] Running Exp 4 (Interprovincial Order-Grain Arbitrage vs Verified Grain-Area Cap)...")
    run_exp4_interprovincial_arbitrage_v2_2(regions22, out_dir, d22, eval_seeds)

    print("[5/5] Running Exp 5 (Zero-Water 6-Way Government Dynamic Policy Benchmark, 48 Holdout Seeds)...")
    res5 = run_exp5_government_6way_v2_2(regions22, out_dir, d22, eval_seeds)

    elapsed = time.perf_counter() - t0
    write_json(out_dir / "v2_2_run_manifest.json", {
        "design_id": d22.id,
        "elapsed_seconds": round(elapsed, 2),
        "n_eval_seeds": len(eval_seeds),
        "best_uniform_fixed_idx": res5["u_idx"],
        "best_regional_fixed_idx": res5["r_idx"],
    })
    print(f"ALL V2.2 ZERO-WATER EXPERIMENTS COMPLETED IN {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
