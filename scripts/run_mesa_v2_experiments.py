"""End-to-End Experiment Suite Runner for Mesa Agri-ABM V2 (`mesa_agri_abm`).

Executes all 6 scientific & experimental deliverables on the unified `AgriSuccessionMesaModel`:
- Stage 1: 2011-2015 One-Step Conditional & 5-Year Recursive Survey Cohort Validation + Ablations
- Exp 1: Fixed-Contract vs. Endogenous Market Two-Step Decomposition + Cash Fungibility vs. Legacy Rent-Ban
- Exp 2: Symmetric Rent Price-Area Decomposition + Common-Parcel Rent Diff + 6-Fate Ledger + Iso-Expenditure
- Exp 3: Boundary Falsification Grid (Friction/Liquidity/Labor/Scale-Orgs/2nd-Price) + Budget & Rate Gradients
- Exp 4: 1/2/3-Year Lease Duration & Post-Withdrawal Beneficiary Parcel Cohort Tracking (Years 7-9)
- Exp 5: 6-Way Government Dynamic Policy Benchmark (`P0`, `Uniform-Fixed`, `Regional-Fixed`,
         `Transparent-Rule`, `Myopic`, `Gov-RL`) on 48 independent holdout seeds + Interactive HTML Export
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from math import fsum, sqrt
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from mesa_agri_abm import (
    AgriSuccessionMesaModel,
    V2Design,
    default_v2_design,
    fit_v2_government_fqi,
    load_v2_calibrated_regions,
    run_policy_episode,
    select_dev_fixed_policies,
    symmetric_rent_decomposition,
)
from mesa_agri_abm.calibration import resolve_project_data_root
from mesa_agri_abm.interactive_inspector import export_inspector_html
from mesa_agri_abm.pmp_production import SubsidySchedule, solve_actor_pmp_v2

T_CRIT_DF47 = 2.0117
T_CRIT_DF31 = 2.0395
T_CRIT_DF23 = 2.0687
N_WORKERS = min(8, max(2, (os.cpu_count() or 4) - 2))


def _episode_worker(args: tuple) -> dict:
    sd, regions, design, mode, label, f_idx, w_mat, branch_wd = args
    return run_policy_episode(
        sd, regions, design, policy_mode=mode, policy_label=label,
        fixed_action_idx=f_idx, learned_weights=w_mat, branch_withdrawal=branch_wd,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def paired_ci(diffs: list[float], t_crit: float) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    m = float(np.mean(arr))
    if len(arr) < 2:
        return m, m, m
    se = float(np.std(arr, ddof=1) / sqrt(len(arr)))
    return m, m - t_crit * se, m + t_crit * se


def load_survey_panel_2010_2015(project_root: Path) -> pd.DataFrame:
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


def eval_cohort_metrics(sim_by_actor: dict[str, dict], survey_yr: pd.DataFrame, seed: int, year: int, arm: str) -> dict:
    comp_mask = survey_yr[["sow_food", "sow_cash"]].notna().all(axis=1) & survey_yr[["sow_food", "sow_cash"]].ge(0).all(axis=1)
    comp = survey_yr.loc[comp_mask]
    n_comp = len(comp)

    obs_tot = comp["sow_food"] + comp["sow_cash"]
    obs_sown = int(obs_tot.gt(0).sum())
    obs_mixed = int((comp["sow_food"].gt(0) & comp["sow_cash"].gt(0)).sum())
    obs_g_mu = float(comp["sow_food"].sum())
    obs_c_mu = float(comp["sow_cash"].sum())
    obs_pos_sh = (comp.loc[obs_tot > 0, "sow_food"] / obs_tot.loc[obs_tot > 0]).to_numpy()
    obs_q = np.percentile(obs_pos_sh, [25, 50, 75]) if len(obs_pos_sh) else np.zeros(3)

    sim_sown = 0
    sim_mixed = 0
    sim_g_mu = 0.0
    sim_c_mu = 0.0
    sim_sh = []
    for aid in comp.index:
        rec = sim_by_actor.get(aid, {"sown_grain_mu": 0.0, "sown_cash_mu": 0.0, "grain_kg": 0.0, "operated_physical_mu": 0.0})
        sg = float(rec["sown_grain_mu"])
        sc = float(rec["sown_cash_mu"])
        st = sg + sc
        sim_g_mu += sg
        sim_c_mu += sc
        if st > 1e-6:
            sim_sown += 1
            sim_sh.append(sg / st)
            if sg > 1e-6 and sc > 1e-6:
                sim_mixed += 1

    sim_q = np.percentile(sim_sh, [25, 50, 75]) if sim_sh else np.zeros(3)
    q_mae = float(np.mean(np.abs(sim_q - obs_q)))
    kg_mask = comp["prod_food"].notna() & comp["prod_food"].ge(0)
    obs_kg = float(comp.loc[kg_mask, "prod_food"].sum())
    sim_kg = fsum(float(sim_by_actor.get(aid, {"grain_kg": 0.0})["grain_kg"]) for aid in comp.loc[kg_mask].index)

    return {
        "arm": arm,
        "seed": seed,
        "year": year,
        "comparable_n": n_comp,
        "survey_sown": obs_sown,
        "sim_sown": sim_sown,
        "survey_mixed": obs_mixed,
        "sim_mixed": sim_mixed,
        "survey_grain_mu_seasons": round(obs_g_mu, 2),
        "sim_grain_mu_seasons": round(sim_g_mu, 2),
        "survey_grain_tons": round(obs_kg / 1000.0, 2),
        "sim_grain_tons": round(sim_kg / 1000.0, 2),
        "participation_err_pp": round(abs(sim_sown - obs_sown) / n_comp, 4),
        "mixed_err_pp": round(abs(sim_mixed - obs_mixed) / n_comp, 4),
        "grain_share_q_mae": round(q_mae, 4),
        "grain_mu_rel_err": round(abs(sim_g_mu - obs_g_mu) / max(1.0, obs_g_mu), 4),
    }


def run_stage1_validation(project_root: Path, out_dir: Path, base_design: V2Design) -> dict:
    s1_dir = out_dir / "stage1_validation"
    regions, meta = load_v2_calibrated_regions(project_root, base_design)
    write_json(s1_dir / "calibration_summary.json", meta)
    panel = load_survey_panel_2010_2015(project_root)

    # 1. One-Step Conditional Household Prediction (2011-2015)
    hh_map = {}
    tech_map = {}
    for r in regions:
        for h in r.households:
            hh_map[h.id] = h
            tech_map[h.id] = r.tech

    no_sub = SubsidySchedule("none", 0.0, 50.0, 250.0, True, 0.0)
    one_step_rows = []
    for yr in range(2011, 2016):
        prev_df = panel.loc[yr - 1]
        cur_df = panel.loc[yr]
        drift = (1.0 + base_design.real_offfarm_wage_drift) ** (yr - 2010)
        sim_map = {}
        for aid, h in hh_map.items():
            if aid in cur_df.index and pd.notna(cur_df.loc[aid, "area_e"]) and float(cur_df.loc[aid, "area_e"]) >= 0:
                op_mu = float(cur_df.loc[aid, "area_e"])
            elif aid in prev_df.index and pd.notna(prev_df.loc[aid, "area_e"]) and float(prev_df.loc[aid, "area_e"]) >= 0:
                op_mu = float(prev_df.loc[aid, "area_e"])
            else:
                op_mu = h.own_physical_mu

            ref_sh = h.ref_grain_share
            if aid in prev_df.index:
                ps_g = float(prev_df.loc[aid, "sow_food"]) if pd.notna(prev_df.loc[aid, "sow_food"]) else 0.0
                ps_c = float(prev_df.loc[aid, "sow_cash"]) if pd.notna(prev_df.loc[aid, "sow_cash"]) else 0.0
                if ps_g + ps_c > 1e-6:
                    ref_sh = ps_g / (ps_g + ps_c)

            pl = solve_actor_pmp_v2(
                op_mu, 0.0, h, tech_map[aid], base_design, no_sub,
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
        one_step_rows.append(eval_cohort_metrics(sim_map, cur_df, 0, yr, "v2_one_step_conditional"))
    write_csv(s1_dir / "one_step_conditional_2011_2015.csv", one_step_rows)

    # 2. Five-Year Recursive Trajectories & Mechanism Ablations (2011-2015)
    ablation_specs = [
        ("v2_full_mesa", base_design, True),
        ("v2_closed_sample", base_design, False),
        ("ablation_no_subsistence_foodc", base_design.with_overrides(enable_subsistence_grain_split=False), True),
        ("ablation_no_machinery_service", base_design.with_overrides(enable_machinery_service=False, allow_hired_labor=False), True),
        ("ablation_near_lp_gamma5", base_design.with_overrides(pmp_gamma_yuan_per_sown_mu=5.0), True),
        ("ablation_no_scale_orgs", base_design.with_overrides(enable_scale_organizations=False), False),
    ]
    rec_summary_rows = []
    for arm_name, d_arm, v_exch in ablation_specs:
        regs_arm = regions if d_arm.enable_scale_organizations == base_design.enable_scale_organizations else load_v2_calibrated_regions(project_root, d_arm)[0]
        seeds = list(range(301, 321)) if arm_name == "v2_full_mesa" else [571001, 571002, 571003, 571004]
        by_yr: dict[int, list[dict]] = defaultdict(list)
        ne_idle_2015 = []
        for sd in seeds:
            m = AgriSuccessionMesaModel(sd, regs_arm, d_arm, policy_mode="p0")
            for step_i in range(1, 6):
                cal_yr = 2010 + step_i
                m.step(explicit_action_idx=0, collect_micro_ledger=True, village_exchange=v_exch)
                sim_actors = {}
                for r_obj in m.regions:
                    for ar in r_obj.last_actor_rows:
                        if ar["is_survey_household"]:
                            sim_actors[ar["actor"]] = ar
                    if cal_yr == 2015 and r_obj.name == "东北":
                        ne_idle_2015.append(r_obj.observed["idle_rate"])
                by_yr[cal_yr].append(eval_cohort_metrics(sim_actors, panel.loc[cal_yr], sd, cal_yr, arm_name))

        for cal_yr, m_list in sorted(by_yr.items()):
            rec_summary_rows.append({
                "arm": arm_name,
                "n_seeds": len(seeds),
                "year": cal_yr,
                "survey_sown": m_list[0]["survey_sown"],
                "sim_sown_mean": round(float(np.mean([x["sim_sown"] for x in m_list])), 2),
                "survey_mixed": m_list[0]["survey_mixed"],
                "sim_mixed_mean": round(float(np.mean([x["sim_mixed"] for x in m_list])), 2),
                "survey_grain_mu_seasons": m_list[0]["survey_grain_mu_seasons"],
                "sim_grain_mu_seasons_mean": round(float(np.mean([x["sim_grain_mu_seasons"] for x in m_list])), 2),
                "survey_grain_tons": m_list[0]["survey_grain_tons"],
                "sim_grain_tons_mean": round(float(np.mean([x["sim_grain_tons"] for x in m_list])), 2),
                "participation_err_pp": round(float(np.mean([x["participation_err_pp"] for x in m_list])), 4),
                "mixed_err_pp": round(float(np.mean([x["mixed_err_pp"] for x in m_list])), 4),
                "grain_share_q_mae": round(float(np.mean([x["grain_share_q_mae"] for x in m_list])), 4),
                "ne_idle_rate_2015": round(float(np.mean(ne_idle_2015)), 4) if cal_yr == 2015 and ne_idle_2015 else None,
            })
    write_csv(s1_dir / "recursive_and_ablations_2011_2015.csv", rec_summary_rows)
    return {"one_step": one_step_rows, "recursive": rec_summary_rows}


def _exp1_all_inst_seed_worker(args: tuple) -> dict[str, tuple]:
    sd, regions, base_design, d_legacy_ban, inst_actions = args
    m_p0 = AgriSuccessionMesaModel(sd, regions, base_design, policy_mode="p0")
    en_models = {name: AgriSuccessionMesaModel(sd, regions, base_design, policy_mode="fixed", fixed_action_idx=a_idx) for name, a_idx in inst_actions}
    ban_models = {name: AgriSuccessionMesaModel(sd, regions, d_legacy_ban, policy_mode="fixed", fixed_action_idx=a_idx) for name, a_idx in inst_actions}

    acc = {name: [0.0] * 11 for name, _ in inst_actions}
    ny = base_design.main_years
    for _ in range(ny):
        m_p0_open = m_p0.fork()
        j0, _, snaps0 = m_p0.step(explicit_action_idx=0, record_contract_snapshots=True)
        g0 = j0["grain_kg"] / 1000.0
        r0 = j0["total_rent_yuan"] / 10_000.0
        l0 = j0["leased_area_mu"] / ny
        s0 = j0["fate_2_strict_succession_grain_mu"] / ny

        for name, a_idx in inst_actions:
            m_fc = m_p0_open.fork()
            j_fc, _, _ = m_fc.step(explicit_action_idx=a_idx, fixed_contract_refs=snaps0)
            j_en, _, _ = en_models[name].step(explicit_action_idx=a_idx)
            j_ban, _, _ = ban_models[name].step(explicit_action_idx=a_idx)

            g_fc = j_fc["grain_kg"] / 1000.0
            g_en = j_en["grain_kg"] / 1000.0
            v = acc[name]
            v[0] += (g_fc - g0)
            v[1] += (g_en - g_fc)
            v[2] += (g_en - g0)
            v[3] += j_fc["spending"] / 10_000.0
            v[4] += j_en["spending"] / 10_000.0
            v[5] += j_ban["spending"] / 10_000.0
            v[6] += (j_en["total_rent_yuan"] / 10_000.0 - r0)
            v[7] += (j_ban["total_rent_yuan"] / 10_000.0 - r0)
            v[8] += (j_en["leased_area_mu"] / ny - l0)
            v[9] += (j_ban["leased_area_mu"] / ny - l0)
            v[10] += (j_en["fate_2_strict_succession_grain_mu"] / ny - s0)

    return {k: tuple(val) for k, val in acc.items()}


def run_exp1_fixed_vs_endogenous_and_fungibility(
    regions, out_dir: Path, base_design: V2Design, eval_seeds: list[int]
) -> list[dict]:
    """Experiment 1: Fixed-Contract vs. Endogenous Market Two-Step Decomposition + Fungibility vs. Legacy Ban."""
    e1_dir = out_dir / "exp1_fixed_vs_endogenous"
    inst_actions = [
        ("producer_pre", 1),
        ("producer_post", 2),
        ("scale_takeover_pre", 3),
        ("scale_takeover_post", 4),
        ("service_trusteeship", 5),
    ]
    d_legacy_ban = base_design.with_overrides(cash_subsidy_fungible_in_bidding=False)

    tasks = [(sd, regions, base_design, d_legacy_ban, inst_actions) for sd in eval_seeds]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        seed_maps = list(pool.map(_exp1_all_inst_seed_worker, tasks))

    rows = []
    for inst_name, _ in inst_actions:
        res_list = [sm[inst_name] for sm in seed_maps]
        fc_grain_diffs = [x[0] for x in res_list]
        mkt_realloc_diffs = [x[1] for x in res_list]
        tot_grain_diffs = [x[2] for x in res_list]
        fc_spend = [x[3] for x in res_list]
        en_spend = [x[4] for x in res_list]
        en_rent_diffs = [x[6] for x in res_list]
        ban_rent_diffs = [x[7] for x in res_list]
        en_leased_diffs = [x[8] for x in res_list]
        ban_leased_diffs = [x[9] for x in res_list]
        en_strict_succ_diffs = [x[10] for x in res_list]

        fc_m, fc_lo, fc_hi = paired_ci(fc_grain_diffs, T_CRIT_DF47)
        re_m, re_lo, re_hi = paired_ci(mkt_realloc_diffs, T_CRIT_DF47)
        tot_m, tot_lo, tot_hi = paired_ci(tot_grain_diffs, T_CRIT_DF47)

        rows.append({
            "instrument": inst_name,
            "fixed_contract_grain_gain_tons": round(fc_m, 3),
            "fixed_contract_ci95": f"[{fc_lo:.2f}, {fc_hi:.2f}]",
            "market_reallocation_grain_gain_tons": round(re_m, 3),
            "market_reallocation_ci95": f"[{re_lo:.2f}, {re_hi:.2f}]",
            "total_endogenous_grain_gain_tons": round(tot_m, 3),
            "total_endogenous_ci95": f"[{tot_lo:.2f}, {tot_hi:.2f}]",
            "fixed_contract_spending_wan": round(float(np.mean(fc_spend)), 3),
            "endogenous_fungible_spending_wan": round(float(np.mean(en_spend)), 3),
            "endogenous_fungible_rent_diff_wan": round(float(np.mean(en_rent_diffs)), 4),
            "legacy_rent_ban_rent_diff_wan": round(float(np.mean(ban_rent_diffs)), 4),
            "endogenous_leased_diff_mu_yr": round(float(np.mean(en_leased_diffs)), 2),
            "legacy_rent_ban_leased_diff_mu_yr": round(float(np.mean(ban_leased_diffs)), 2),
            "strict_succession_diff_mu_yr": round(float(np.mean(en_strict_succ_diffs)), 2),
        })

    write_csv(e1_dir / "exp1_two_step_decomposition.csv", rows)
    return rows


def run_exp2_rent_decomposition_6fates_and_iso_spend(
    regions, out_dir: Path, base_design: V2Design, eval_seeds: list[int]
) -> tuple[list[dict], list[dict]]:
    """Experiment 2: Symmetric Rent Price-Area Decomposition + Common-Plot Rent Diff + 6 Fates + Iso-Expenditure."""
    e2_dir = out_dir / "exp2_rent_and_6fates"
    arms = [
        ("p0", "p0", 0),
        ("producer_pre", "uniform_fixed", 1),
        ("producer_post", "uniform_fixed", 2),
        ("scale_takeover_pre", "uniform_fixed", 3),
        ("scale_takeover_post", "uniform_fixed", 4),
        ("service_trusteeship", "uniform_fixed", 5),
    ]

    ep_by_arm_seed: dict[tuple[str, int], dict] = {}
    ep2_tasks = [
        (sd, regions, base_design, mode, label, f_idx, None, False)
        for label, mode, f_idx in arms
        for sd in eval_seeds
    ]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for ep_res in pool.map(_episode_worker, ep2_tasks):
            ep_by_arm_seed[(ep_res["policy_label"], ep_res["seed"])] = ep_res

    summary_rows = []
    for label, _, _ in arms:
        g_tot, g_surv, g_ff, g_coop, g_corp = [], [], [], [], []
        sp_tot, wheat_t, rice_t, maize_t, soy_t = [], [], [], [], []
        sub_t, mkt_t = [], []
        f1, f2, f3, f3a, f4, f5, f6 = [], [], [], [], [], [], []
        leased_mu, unit_r, idi_vals = [], [], []
        tot_rent_diff, price_eff, area_eff, comm_mu, comm_unit_diff = [], [], [], [], []

        for sd in eval_seeds:
            ep = ep_by_arm_seed[(label, sd)]
            ep0 = ep_by_arm_seed[("p0", sd)]
            mj = ep["main_joint"]
            mj0 = ep0["main_joint"]
            ny = len(mj)

            g_tot.append(fsum(r["grain_kg"] for r in mj) / 1000.0)
            g_surv.append(fsum(r["survey_600_grain_kg"] for r in mj) / 1000.0)
            g_ff.append(fsum(r["family_farm_grain_kg"] for r in mj) / 1000.0)
            g_coop.append(fsum(r["cooperative_grain_kg"] for r in mj) / 1000.0)
            g_corp.append(fsum(r["agri_company_grain_kg"] for r in mj) / 1000.0)
            sp_tot.append(fsum(r["spending"] for r in mj) / 10_000.0)

            wheat_t.append(fsum(r["wheat_kg"] for r in mj) / 1000.0)
            rice_t.append(fsum(r["rice_kg"] for r in mj) / 1000.0)
            maize_t.append(fsum(r["maize_kg"] for r in mj) / 1000.0)
            soy_t.append(fsum(r["soybean_kg"] for r in mj) / 1000.0)
            sub_t.append(fsum(r["subsistence_grain_kg"] for r in mj) / 1000.0)
            mkt_t.append(fsum(r["marketed_grain_kg"] for r in mj) / 1000.0)

            f1.append(fsum(r["fate_1_incumbent_grain_mu"] for r in mj) / ny)
            f2.append(fsum(r["fate_2_strict_succession_grain_mu"] for r in mj) / ny)
            f3.append(fsum(r["fate_3_expanded_grain_mu"] for r in mj) / ny)
            f3a.append(fsum(r["fate_3a_idle_recovered_grain_mu"] for r in mj) / ny)
            f4.append(fsum(r["fate_4_leased_cash_mu"] for r in mj) / ny)
            f5.append(fsum(r["fate_5_leased_unplanted_mu"] for r in mj) / ny)
            f6.append(fsum(r["fate_6_own_idle_mu"] for r in mj) / ny)
            leased_mu.append(fsum(r["leased_area_mu"] for r in mj) / ny)
            unit_r.append(float(np.mean([r["unit_rent_yuan_mu"] for r in mj])))
            idi_vals.append(float(np.mean([r["idi_score"] for r in mj])))

            # Year-by-year symmetric rent decomposition summed over 8 years
            p_eff_sd, a_eff_sd, t_diff_sd = 0.0, 0.0, 0.0
            c_mu_sd, c_udiff_sd = [], []
            for yr_idx in range(ny):
                yr_num = yr_idx + 1
                dec = symmetric_rent_decomposition(
                    mj[yr_idx]["leased_area_mu"], mj[yr_idx]["total_rent_yuan"],
                    mj0[yr_idx]["leased_area_mu"], mj0[yr_idx]["total_rent_yuan"],
                    ep["parcel_rents_by_year"][yr_num], ep0["parcel_rents_by_year"][yr_num],
                )
                p_eff_sd += dec["price_effect_yuan"]
                a_eff_sd += dec["area_effect_yuan"]
                t_diff_sd += dec["total_rent_diff_yuan"]
                c_mu_sd.append(dec["common_leased_mu"])
                c_udiff_sd.append(dec["common_unit_rent_diff_yuan_mu"])

            tot_rent_diff.append(t_diff_sd / 10_000.0)
            price_eff.append(p_eff_sd / 10_000.0)
            area_eff.append(a_eff_sd / 10_000.0)
            comm_mu.append(float(np.mean(c_mu_sd)))
            comm_unit_diff.append(float(np.mean(c_udiff_sd)))

        sp_mean = float(np.mean(sp_tot))
        g_diff = float(np.mean(g_tot) - np.mean([fsum(r["grain_kg"] for r in ep_by_arm_seed[("p0", sd)]["main_joint"]) / 1000.0 for sd in eval_seeds]))
        pure_price_cap_pct = (float(np.mean(price_eff)) / sp_mean * 100.0) if sp_mean > 1e-4 else 0.0
        gross_rent_ratio_pct = (float(np.mean(tot_rent_diff)) / sp_mean * 100.0) if sp_mean > 1e-4 else 0.0

        summary_rows.append({
            "policy": label,
            "grain_8yr_tons": round(float(np.mean(g_tot)), 2),
            "grain_gain_vs_p0_tons": round(g_diff, 2),
            "spending_8yr_wan": round(sp_mean, 3),
            "kg_per_100yuan_spent": round((g_diff * 1000.0) / (sp_mean * 100.0), 2) if sp_mean > 1e-4 else 0.0,
            "survey_600_grain_tons": round(float(np.mean(g_surv)), 2),
            "family_farm_grain_tons": round(float(np.mean(g_ff)), 2),
            "cooperative_grain_tons": round(float(np.mean(g_coop)), 2),
            "agri_company_grain_tons": round(float(np.mean(g_corp)), 2),
            "wheat_tons": round(float(np.mean(wheat_t)), 1),
            "rice_tons": round(float(np.mean(rice_t)), 1),
            "maize_tons": round(float(np.mean(maize_t)), 1),
            "soybean_tons": round(float(np.mean(soy_t)), 1),
            "subsistence_grain_tons": round(float(np.mean(sub_t)), 1),
            "marketed_grain_tons": round(float(np.mean(mkt_t)), 1),
            "fate1_incumbent_grain_mu_yr": round(float(np.mean(f1)), 2),
            "fate2_strict_succession_mu_yr": round(float(np.mean(f2)), 2),
            "fate3_expanded_grain_mu_yr": round(float(np.mean(f3)), 2),
            "fate3a_idle_recovered_mu_yr": round(float(np.mean(f3a)), 2),
            "fate4_leased_cash_mu_yr": round(float(np.mean(f4)), 2),
            "fate5_leased_unplanted_mu_yr": round(float(np.mean(f5)), 2),
            "fate6_own_idle_mu_yr": round(float(np.mean(f6)), 2),
            "leased_area_mu_yr": round(float(np.mean(leased_mu)), 2),
            "unit_rent_yuan_mu": round(float(np.mean(unit_r)), 2),
            "total_rent_diff_wan": round(float(np.mean(tot_rent_diff)), 4),
            "rent_price_effect_wan": round(float(np.mean(price_eff)), 4),
            "rent_area_effect_wan": round(float(np.mean(area_eff)), 4),
            "pure_price_capitalization_pct": round(pure_price_cap_pct, 2),
            "gross_rent_to_spend_pct": round(gross_rent_ratio_pct, 2),
            "common_leased_mu_yr": round(float(np.mean(comm_mu)), 2),
            "common_unit_rent_diff_yuan_mu": round(float(np.mean(comm_unit_diff)), 2),
            "idi_score": round(float(np.mean(idi_vals)), 4),
        })
    write_csv(e2_dir / "exp2_rent_and_6fates_summary.csv", summary_rows)

    # Iso-Expenditure (Equal Actual Spending) Comparison calibrated to match `service_trusteeship` / `scale` realized outlay
    target_wan = next(r["spending_8yr_wan"] for r in summary_rows if r["policy"] == "scale_takeover_pre")
    iso_budget_annual = int(round((target_wan * 10_000.0) / base_design.main_years))
    d_iso = base_design.with_overrides(annual_common_budget_yuan=iso_budget_annual)
    p0_base_tons = next(r["grain_8yr_tons"] for r in summary_rows if r["policy"] == "p0")

    iso_rows = []
    for label, a_idx in (("producer_pre", 1), ("scale_takeover_pre", 3), ("service_trusteeship", 5)):
        if label == "scale_takeover_pre":
            # Already at target_wan under baseline budget
            ref_r = next(r for r in summary_rows if r["policy"] == label)
            iso_rows.append({
                "policy": label,
                "annual_budget_cap_yuan": base_design.annual_common_budget_yuan,
                "realized_8yr_spending_wan": ref_r["spending_8yr_wan"],
                "grain_gain_vs_p0_tons": ref_r["grain_gain_vs_p0_tons"],
                "kg_per_100yuan_spent": ref_r["kg_per_100yuan_spent"],
                "pure_price_capitalization_pct": ref_r["pure_price_capitalization_pct"],
            })
        else:
            g_list, sp_list, pe_list = [], [], []
            for sd in eval_seeds:
                ep_i = run_policy_episode(sd, regions, d_iso, policy_mode="uniform_fixed", policy_label=label, fixed_action_idx=a_idx, branch_withdrawal=False)
                ep0 = ep_by_arm_seed[("p0", sd)]
                g_list.append(fsum(r["grain_kg"] for r in ep_i["main_joint"]) / 1000.0)
                sp_list.append(fsum(r["spending"] for r in ep_i["main_joint"]) / 10_000.0)
                pe_sd = 0.0
                for yr_i in range(base_design.main_years):
                    dec = symmetric_rent_decomposition(
                        ep_i["main_joint"][yr_i]["leased_area_mu"], ep_i["main_joint"][yr_i]["total_rent_yuan"],
                        ep0["main_joint"][yr_i]["leased_area_mu"], ep0["main_joint"][yr_i]["total_rent_yuan"],
                        ep_i["parcel_rents_by_year"][yr_i + 1], ep0["parcel_rents_by_year"][yr_i + 1],
                    )
                    pe_sd += dec["price_effect_yuan"]
                pe_list.append(pe_sd / 10_000.0)
            sp_m = float(np.mean(sp_list))
            gd_m = float(np.mean(g_list)) - p0_base_tons
            iso_rows.append({
                "policy": label,
                "annual_budget_cap_yuan": iso_budget_annual,
                "realized_8yr_spending_wan": round(sp_m, 3),
                "grain_gain_vs_p0_tons": round(gd_m, 2),
                "kg_per_100yuan_spent": round((gd_m * 1000.0) / max(1e-4, sp_m * 100.0), 2),
                "pure_price_capitalization_pct": round(float(np.mean(pe_list)) / max(1e-4, sp_m) * 100.0, 2),
            })
    write_csv(e2_dir / "exp2_iso_expenditure_comparison.csv", iso_rows)
    return summary_rows, iso_rows


def run_exp3_boundary_and_gradients(
    project_root: Path, regions, out_dir: Path, base_design: V2Design, seeds_24: list[int]
) -> tuple[list[dict], list[dict]]:
    """Experiment 3: Falsification Boundary Grid + Multi-Budget & Scale-Subsidy Rate Gradients."""
    e3_dir = out_dir / "exp3_boundary_and_gradients"

    env_scenarios = [
        ("baseline_v2", base_design, False),
        ("high_search_friction_042", base_design.with_overrides(search_friction_prob=0.42), False),
        ("abundant_working_cap_18x", base_design.with_overrides(working_cap_scale=1.80), True),
        ("tight_family_labor_065x", base_design.with_overrides(family_labor_scale=0.65), True),
        ("pure_600_no_scale_orgs", base_design.with_overrides(enable_scale_organizations=False), True),
        ("regmas_second_price_auction", base_design.with_overrides(pricing_rule="second_price"), False),
        ("no_single_operator_cap", base_design.with_overrides(enable_single_operator_cap=False), False),
        ("no_anti_abandonment_rule", base_design.with_overrides(fertility_anti_abandonment_rule=False), False),
        ("no_background_fertility_sub", base_design.with_overrides(enable_background_fertility_subsidy=False), False),
    ]

    grid_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for sc_name, d_sc, need_recal in env_scenarios:
            regs_sc = load_v2_calibrated_regions(project_root, d_sc)[0] if need_recal else regions
            sc_arms = [("p0", "p0", 0), ("producer_pre", "uniform_fixed", 1), ("producer_post", "uniform_fixed", 2), ("scale_takeover_pre", "uniform_fixed", 3), ("service_trusteeship", "uniform_fixed", 5)]
            sc_tasks = [(sd, regs_sc, d_sc, m_mode, p_lbl, a_idx, None, False) for p_lbl, m_mode, a_idx in sc_arms for sd in seeds_24]
            sc_map = {(ep["policy_label"], ep["seed"]): ep for ep in pool.map(_episode_worker, sc_tasks)}

            res_by_pol: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
            for sd in seeds_24:
                ep0 = sc_map[("p0", sd)]
                g0 = fsum(r["grain_kg"] for r in ep0["main_joint"]) / 1000.0
                for p_lbl in ("producer_pre", "producer_post", "scale_takeover_pre", "service_trusteeship"):
                    ep = sc_map[(p_lbl, sd)]
                    g_p = fsum(r["grain_kg"] for r in ep["main_joint"]) / 1000.0
                    sp_p = fsum(r["spending"] for r in ep["main_joint"]) / 10_000.0
                    pe_sd = fsum(
                        symmetric_rent_decomposition(
                            ep["main_joint"][k]["leased_area_mu"], ep["main_joint"][k]["total_rent_yuan"],
                            ep0["main_joint"][k]["leased_area_mu"], ep0["main_joint"][k]["total_rent_yuan"],
                            ep["parcel_rents_by_year"][k + 1], ep0["parcel_rents_by_year"][k + 1],
                        )["price_effect_yuan"]
                        for k in range(d_sc.main_years)
                    ) / 10_000.0
                    succ_mu = fsum(r["fate_2_strict_succession_grain_mu"] for r in ep["main_joint"]) / d_sc.main_years
                    res_by_pol[p_lbl].append((g_p - g0, sp_p, pe_sd, succ_mu))

            pre_post_gap = float(np.mean([x[0] for x in res_by_pol["producer_pre"]]) - np.mean([x[0] for x in res_by_pol["producer_post"]]))
            for p_lbl in ("producer_pre", "producer_post", "scale_takeover_pre", "service_trusteeship"):
                vals = res_by_pol[p_lbl]
                sp_m = float(np.mean([x[1] for x in vals]))
                pe_m = float(np.mean([x[2] for x in vals]))
                grid_rows.append({
                    "scenario": sc_name,
                    "policy": p_lbl,
                    "grain_gain_vs_p0_tons": round(float(np.mean([x[0] for x in vals])), 2),
                    "spending_8yr_wan": round(sp_m, 3),
                    "pure_price_cap_pct": round(pe_m / max(1e-4, sp_m) * 100.0, 2),
                    "strict_succession_mu_yr": round(float(np.mean([x[3] for x in vals])), 2),
                    "pre_minus_post_producer_tons": round(pre_post_gap, 2),
                })
    write_csv(e3_dir / "exp3_boundary_grid.csv", grid_rows)

    # Budget Envelope & Scale Subsidy Rate Ceiling Gradients (including legacy 98,953 budget & 85 CNY/mu cap!)
    gradient_specs = [
        ("budget_75k", base_design.with_overrides(annual_common_budget_yuan=75_000)),
        ("budget_98953_v1_iso", base_design.with_overrides(annual_common_budget_yuan=98_953)),
        ("budget_150k_baseline", base_design),
        ("budget_250k", base_design.with_overrides(annual_common_budget_yuan=250_000)),
        ("scale_cap_85_legacy", base_design.with_overrides(scale_takeover_cap_yuan_per_mu=85.0)),
        ("scale_cap_150", base_design.with_overrides(scale_takeover_cap_yuan_per_mu=150.0)),
        ("scale_cap_200_baseline", base_design.with_overrides(scale_takeover_cap_yuan_per_mu=200.0)),
        ("scale_cap_280", base_design.with_overrides(scale_takeover_cap_yuan_per_mu=280.0)),
    ]
    grad_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        # P0 does not depend on budget or scale_cap! Run P0 once on seeds_24!
        ep0_base = {
            ep["seed"]: ep
            for ep in pool.map(_episode_worker, [(sd, regions, base_design, "p0", "p0", 0, None, False) for sd in seeds_24])
        }
        for g_name, d_g in gradient_specs:
            g_tasks = [
                (sd, regions, d_g, "uniform_fixed", p_lbl, a_idx, None, False)
                for p_lbl, a_idx in (("producer_pre", 1), ("scale_takeover_pre", 3), ("service_trusteeship", 5))
                for sd in seeds_24
            ]
            g_map = {(ep["policy_label"], ep["seed"]): ep for ep in pool.map(_episode_worker, g_tasks)}
            for p_lbl in ("producer_pre", "scale_takeover_pre", "service_trusteeship"):
                g_diffs, sp_vals, pe_vals, leased_vals = [], [], [], []
                for sd in seeds_24:
                    ep0 = ep0_base[sd]
                    ep = g_map[(p_lbl, sd)]
                    g_diffs.append((fsum(r["grain_kg"] for r in ep["main_joint"]) - fsum(r["grain_kg"] for r in ep0["main_joint"])) / 1000.0)
                    sp_vals.append(fsum(r["spending"] for r in ep["main_joint"]) / 10_000.0)
                    pe_vals.append(fsum(
                        symmetric_rent_decomposition(
                            ep["main_joint"][k]["leased_area_mu"], ep["main_joint"][k]["total_rent_yuan"],
                            ep0["main_joint"][k]["leased_area_mu"], ep0["main_joint"][k]["total_rent_yuan"],
                            ep["parcel_rents_by_year"][k + 1], ep0["parcel_rents_by_year"][k + 1],
                        )["price_effect_yuan"]
                        for k in range(d_g.main_years)
                    ) / 10_000.0)
                    leased_vals.append(fsum(r["leased_area_mu"] for r in ep["main_joint"]) / d_g.main_years)

                sp_m = float(np.mean(sp_vals))
                gd_m = float(np.mean(g_diffs))
                grad_rows.append({
                    "gradient_arm": g_name,
                    "annual_budget_yuan": d_g.annual_common_budget_yuan,
                    "scale_cap_yuan_mu": d_g.scale_takeover_cap_yuan_per_mu,
                    "policy": p_lbl,
                    "grain_gain_vs_p0_tons": round(gd_m, 2),
                    "spending_8yr_wan": round(sp_m, 3),
                    "budget_exec_pct": round(sp_m * 10_000.0 / (d_g.main_years * d_g.annual_common_budget_yuan) * 100.0, 1),
                    "kg_per_100yuan_spent": round((gd_m * 1000.0) / max(1e-4, sp_m * 100.0), 2),
                    "pure_price_cap_pct": round(float(np.mean(pe_vals)) / max(1e-4, sp_m) * 100.0, 2),
                    "leased_area_mu_yr": round(float(np.mean(leased_vals)), 2),
                })
    write_csv(e3_dir / "exp3_budget_and_rate_gradient.csv", grad_rows)
    return grid_rows, grad_rows


def run_exp4_lease_duration_and_cohort_withdrawal(
    regions, out_dir: Path, base_design: V2Design, seeds_32: list[int]
) -> list[dict]:
    """Experiment 4: 1/2/3-Year Lease Duration & Post-Withdrawal Years +1/+2/+3 Beneficiary Cohort Tracking."""
    e4_dir = out_dir / "exp4_lease_and_withdrawal"
    specs = [
        ("lease_1yr_carry_on", base_design.with_overrides(lease_duration_years=1, main_years=9, withdrawal_after=6)),
        ("lease_2yr_carry_on", base_design.with_overrides(lease_duration_years=2, main_years=9, withdrawal_after=6)),
        ("lease_3yr_carry_on", base_design.with_overrides(lease_duration_years=3, main_years=9, withdrawal_after=6)),
        ("lease_2yr_no_cash_carry", base_design.with_overrides(lease_duration_years=2, main_years=9, withdrawal_after=6, enable_cash_carryover=False)),
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for sp_name, d_sp in specs:
            e4_arms = [("p0", "p0", 0), ("producer_pre", "uniform_fixed", 1), ("scale_takeover_pre", "uniform_fixed", 3), ("service_trusteeship", "uniform_fixed", 5)]
            e4_tasks = [(sd, regions, d_sp, m_mode, p_lbl, a_idx, None, True) for p_lbl, m_mode, a_idx in e4_arms for sd in seeds_32]
            e4_map = {(ep["policy_label"], ep["seed"]): ep for ep in pool.map(_episode_worker, e4_tasks)}

            for p_lbl in ("producer_pre", "scale_takeover_pre", "service_trusteeship"):
                wd_y1, wd_y2, wd_y3, wd_3yr_tot = [], [], [], []
                coh_tracked, coh_same_op_y1, coh_same_op_y3, coh_grain_y1, coh_grain_y3 = [], [], [], [], []
                unplanted_leased_active = []

                for sd in seeds_32:
                    ep0 = e4_map[("p0", sd)]
                    ep = e4_map[(p_lbl, sd)]
                    w_pol = ep["withdrawal_joint"]
                    w_p0 = ep0["withdrawal_joint"]
                    d1 = (w_pol[0]["grain_kg"] - w_p0[0]["grain_kg"]) / 1000.0
                    d2 = (w_pol[1]["grain_kg"] - w_p0[1]["grain_kg"]) / 1000.0
                    d3 = (w_pol[2]["grain_kg"] - w_p0[2]["grain_kg"]) / 1000.0
                    wd_y1.append(d1)
                    wd_y2.append(d2)
                    wd_y3.append(d3)
                    wd_3yr_tot.append(d1 + d2 + d3)

                    coh_tracked.append(w_pol[0]["cohort_tracked_mu"])
                    coh_same_op_y1.append(w_pol[0]["cohort_retained_same_op_mu"])
                    coh_same_op_y3.append(w_pol[2]["cohort_retained_same_op_mu"])
                    coh_grain_y1.append(w_pol[0]["cohort_still_grain_mu"])
                    coh_grain_y3.append(w_pol[2]["cohort_still_grain_mu"])
                    unplanted_leased_active.append(fsum(r["fate_5_leased_unplanted_mu"] for r in ep["main_joint"][:6]) / 6.0)

                tot_m, tot_lo, tot_hi = paired_ci(wd_3yr_tot, T_CRIT_DF31)
                tr_mu = max(1e-6, float(np.mean(coh_tracked)))
                rows.append({
                    "contract_spec": sp_name,
                    "lease_years": d_sp.lease_duration_years,
                    "cash_carryover": int(d_sp.enable_cash_carryover),
                    "policy": p_lbl,
                    "active_leased_unplanted_mu_yr": round(float(np.mean(unplanted_leased_active)), 2),
                    "post_stop_yr1_grain_diff_tons": round(float(np.mean(wd_y1)), 2),
                    "post_stop_yr2_grain_diff_tons": round(float(np.mean(wd_y2)), 2),
                    "post_stop_yr3_grain_diff_tons": round(float(np.mean(wd_y3)), 2),
                    "post_stop_3yr_total_diff_tons": round(tot_m, 2),
                    "post_stop_3yr_ci95": f"[{tot_lo:.2f}, {tot_hi:.2f}]",
                    "beneficiary_cohort_mu": round(float(np.mean(coh_tracked)), 2),
                    "cohort_same_op_share_yr1_pct": round(float(np.mean(coh_same_op_y1)) / tr_mu * 100.0, 1),
                    "cohort_same_op_share_yr3_pct": round(float(np.mean(coh_same_op_y3)) / tr_mu * 100.0, 1),
                    "cohort_still_grain_share_yr1_pct": round(float(np.mean(coh_grain_y1)) / tr_mu * 100.0, 1),
                    "cohort_still_grain_share_yr3_pct": round(float(np.mean(coh_grain_y3)) / tr_mu * 100.0, 1),
                })
    write_csv(e4_dir / "exp4_lease_duration_withdrawal.csv", rows)
    return rows


def run_exp5_government_6way_benchmark(
    project_root: Path, regions, out_dir: Path, base_design: V2Design, eval_seeds: list[int]
) -> dict:
    """Experiment 5: 6-Way Government Policy Benchmark on 48 Independent Holdout Seeds + HTML Inspector."""
    e5_dir = out_dir / "exp5_government_6way_rl"
    dev_seeds = list(range(base_design.dev_seeds[0], base_design.dev_seeds[1] + 1))
    train_seeds = list(range(base_design.train_seeds[0], base_design.train_seeds[1] + 1))

    dev_sel = select_dev_fixed_policies(dev_seeds, regions, base_design)
    write_csv(e5_dir / "dev_25actions_leaderboard.csv", dev_sel["leaderboard"])

    fqi_fit = fit_v2_government_fqi(train_seeds, regions, base_design)
    write_json(
        e5_dir / "fqi_training_summary.json",
        {
            "n_train_seeds": fqi_fit["n_train_seeds"],
            "best_uniform_fixed_idx": dev_sel["best_uniform_fixed_idx"],
            "best_regional_fixed_idx": dev_sel["best_regional_fixed_idx"],
            "fit_r2_q": fqi_fit["fit_r2_q"],
            "fit_r2_myopic": fqi_fit["fit_r2_myopic"],
        },
    )

    u_idx = dev_sel["best_uniform_fixed_idx"]
    r_idx = dev_sel["best_regional_fixed_idx"]

    six_arms = [
        ("p0", "p0", 0, None),
        ("uniform_fixed", "uniform_fixed", u_idx, None),
        ("regional_fixed", "regional_fixed", r_idx, None),
        ("transparent_rule", "transparent_rule", 0, None),
        ("myopic", "myopic", 0, fqi_fit["myopic_weights"]),
        ("gov_rl", "gov_rl", 0, fqi_fit["q_weights"]),
    ]

    all_joint_rows = []
    all_reg_rows = []
    ep_store: dict[tuple[str, int], dict] = {}

    ep5_tasks = [
        (sd, regions, base_design, mode, label, f_idx, w_mat, True)
        for label, mode, f_idx, w_mat in six_arms
        for sd in eval_seeds
    ]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for ep in pool.map(_episode_worker, ep5_tasks):
            ep_store[(ep["policy_label"], ep["seed"])] = ep
            all_joint_rows.extend(ep["main_joint"])
            all_joint_rows.extend(ep["withdrawal_joint"])
            all_reg_rows.extend(ep["main_regions"])
            all_reg_rows.extend(ep["withdrawal_regions"])

    write_csv(e5_dir / "holdout_joint_trajectories.csv", all_joint_rows)
    write_csv(e5_dir / "holdout_regional_trajectories.csv", all_reg_rows)

    # Summarize 6-way leaderboard & paired CIs
    six_summary = []
    for label, _, _, _ in six_arms:
        g_8yr, sp_8yr, wd_2yr, idi_m, pe_8yr, succ_yr = [], [], [], [], [], []
        reg_grains: dict[str, list[float]] = defaultdict(list)
        reg_spends: dict[str, list[float]] = defaultdict(list)

        for sd in eval_seeds:
            ep = ep_store[(label, sd)]
            ep0 = ep_store[("p0", sd)]
            g_8yr.append(fsum(r["grain_kg"] for r in ep["main_joint"]) / 1000.0)
            sp_8yr.append(fsum(r["spending"] for r in ep["main_joint"]) / 10_000.0)
            wd_2yr.append(fsum(r["grain_kg"] for r in ep["withdrawal_joint"]) / 1000.0)
            idi_m.append(float(np.mean([r["idi_score"] for r in ep["main_joint"]])))
            succ_yr.append(fsum(r["fate_2_strict_succession_grain_mu"] for r in ep["main_joint"]) / base_design.main_years)
            pe_8yr.append(fsum(
                symmetric_rent_decomposition(
                    ep["main_joint"][k]["leased_area_mu"], ep["main_joint"][k]["total_rent_yuan"],
                    ep0["main_joint"][k]["leased_area_mu"], ep0["main_joint"][k]["total_rent_yuan"],
                    ep["parcel_rents_by_year"][k + 1], ep0["parcel_rents_by_year"][k + 1],
                )["price_effect_yuan"]
                for k in range(base_design.main_years)
            ) / 10_000.0)
            for rr in ep["main_regions"]:
                reg_grains[rr["region"]].append(rr["grain_kg"] / 1000.0)
                reg_spends[rr["region"]].append(rr["spending"] / 10_000.0)

        p0_g = float(np.mean([fsum(r["grain_kg"] for r in ep_store[("p0", sd)]["main_joint"]) / 1000.0 for sd in eval_seeds]))
        p0_wd = float(np.mean([fsum(r["grain_kg"] for r in ep_store[("p0", sd)]["withdrawal_joint"]) / 1000.0 for sd in eval_seeds]))
        g_m = float(np.mean(g_8yr))
        sp_m = float(np.mean(sp_8yr))
        wd_m = float(np.mean(wd_2yr))

        six_summary.append({
            "policy_label": label,
            "grain_8yr_tons": round(g_m, 2),
            "grain_gain_vs_p0_tons": round(g_m - p0_g, 2),
            "spending_8yr_wan": round(sp_m, 3),
            "kg_per_100yuan_spent": round(((g_m - p0_g) * 1000.0) / (sp_m * 100.0), 2) if sp_m > 1e-4 else 0.0,
            "pure_price_cap_pct": round(float(np.mean(pe_8yr)) / max(1e-4, sp_m) * 100.0, 2) if sp_m > 1e-4 else 0.0,
            "strict_succession_mu_yr": round(float(np.mean(succ_yr)), 2),
            "idi_score": round(float(np.mean(idi_m)), 4),
            "withdrawal_yr7_8_tons": round(wd_m, 2),
            "withdrawal_gain_vs_p0_tons": round(wd_m - p0_wd, 2),
            "east_grain_8yr_tons": round(float(np.sum(reg_grains["东部"]) / len(eval_seeds)), 2),
            "central_grain_8yr_tons": round(float(np.sum(reg_grains["中部"]) / len(eval_seeds)), 2),
            "west_grain_8yr_tons": round(float(np.sum(reg_grains["西部"]) / len(eval_seeds)), 2),
            "northeast_grain_8yr_tons": round(float(np.sum(reg_grains["东北"]) / len(eval_seeds)), 2),
            "east_spend_8yr_wan": round(float(np.sum(reg_spends["东部"]) / len(eval_seeds)), 3),
            "central_spend_8yr_wan": round(float(np.sum(reg_spends["中部"]) / len(eval_seeds)), 3),
            "west_spend_8yr_wan": round(float(np.sum(reg_spends["西部"]) / len(eval_seeds)), 3),
            "northeast_spend_8yr_wan": round(float(np.sum(reg_spends["东北"]) / len(eval_seeds)), 3),
        })
    write_csv(e5_dir / "six_way_policy_summary.csv", six_summary)

    # Paired contrasts for Gov-RL vs every benchmark
    contrasts = []
    for ref_lbl in ("p0", "uniform_fixed", "regional_fixed", "transparent_rule", "myopic"):
        g_d = [
            fsum(r["grain_kg"] for r in ep_store[("gov_rl", sd)]["main_joint"]) / 1000.0
            - fsum(r["grain_kg"] for r in ep_store[(ref_lbl, sd)]["main_joint"]) / 1000.0
            for sd in eval_seeds
        ]
        sp_d = [
            fsum(r["spending"] for r in ep_store[("gov_rl", sd)]["main_joint"]) / 10_000.0
            - fsum(r["spending"] for r in ep_store[(ref_lbl, sd)]["main_joint"]) / 10_000.0
            for sd in eval_seeds
        ]
        wd_d = [
            fsum(r["grain_kg"] for r in ep_store[("gov_rl", sd)]["withdrawal_joint"]) / 1000.0
            - fsum(r["grain_kg"] for r in ep_store[(ref_lbl, sd)]["withdrawal_joint"]) / 1000.0
            for sd in eval_seeds
        ]
        gm, glo, ghi = paired_ci(g_d, T_CRIT_DF47)
        sm, slo, shi = paired_ci(sp_d, T_CRIT_DF47)
        wm, wlo, whi = paired_ci(wd_d, T_CRIT_DF47)
        contrasts.append({
            "contrast": f"gov_rl_minus_{ref_lbl}",
            "grain_8yr_diff_tons": round(gm, 3),
            "grain_ci95": f"[{glo:.2f}, {ghi:.2f}]",
            "spending_8yr_diff_wan": round(sm, 3),
            "spending_ci95": f"[{slo:.3f}, {shi:.3f}]",
            "withdrawal_diff_tons": round(wm, 3),
            "withdrawal_ci95": f"[{wlo:.2f}, {whi:.2f}]",
        })
    write_csv(e5_dir / "gov_rl_paired_contrasts.csv", contrasts)

    # Export real-time Mesa state interactive HTML inspector
    html_path = out_dir.parent / "交互式检查台/Mesa农业经营接续交互检查台.html"
    export_inspector_html(
        html_path,
        project_root,
        seed=eval_seeds[0],
        design=base_design,
        learned_q_weights=fqi_fit["q_weights"],
        uniform_fixed_idx=u_idx,
        regional_fixed_idx=r_idx,
    )

    return {
        "six_summary": six_summary,
        "contrasts": contrasts,
        "best_uniform_fixed_idx": u_idx,
        "best_regional_fixed_idx": r_idx,
        "inspector_html": str(html_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=resolve_project_data_root(Path(__file__)))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/Mesa农业经营接续V2重构/结果",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    design = default_v2_design()
    regions, _ = load_v2_calibrated_regions(args.project_root, design)
    eval_seeds = list(range(design.eval_seeds[0], design.eval_seeds[1] + 1))

    print("[1/5] Running Stage 1 (2011-2015 Conditional & Recursive Validation + Ablations)...")
    run_stage1_validation(args.project_root, args.out_dir, design)

    print("[2/5] Running Exp 1 (Fixed-Contract vs. Endogenous Market + Fungibility vs. Legacy Rent-Ban)...")
    run_exp1_fixed_vs_endogenous_and_fungibility(regions, args.out_dir, design, eval_seeds)

    print("[3/5] Running Exp 2 (Symmetric Rent Price-Area Decomposition + 6 Fates + Iso-Expenditure)...")
    run_exp2_rent_decomposition_6fates_and_iso_spend(regions, args.out_dir, design, eval_seeds)

    print("[4/5] Running Exp 3 & Exp 4 (Boundary Falsification Grid, Gradients & 1/2/3-Yr Cohort Withdrawal)...")
    run_exp3_boundary_and_gradients(args.project_root, regions, args.out_dir, design, eval_seeds[:24])
    run_exp4_lease_duration_and_cohort_withdrawal(regions, args.out_dir, design, eval_seeds[:32])

    print("[5/5] Running Exp 5 (6-Way Government Policy Benchmark & Live Mesa HTML Inspector Export)...")
    res5 = run_exp5_government_6way_benchmark(args.project_root, regions, args.out_dir, design, eval_seeds)

    elapsed = time.perf_counter() - t0
    write_json(
        args.out_dir / "v2_run_manifest.json",
        {
            "design_id": design.id,
            "elapsed_seconds": round(elapsed, 2),
            "n_eval_seeds": len(eval_seeds),
            "best_uniform_fixed_idx": res5["best_uniform_fixed_idx"],
            "best_regional_fixed_idx": res5["best_regional_fixed_idx"],
            "inspector_html": res5["inspector_html"],
        },
    )
    print(f"ALL V2 EXPERIMENTS COMPLETED IN {elapsed:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
