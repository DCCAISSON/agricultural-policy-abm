"""Formal V2.1 Institutional, Intertemporal & Asymmetric-Information Experiment Suite (`mesa_agri_abm`).

Preserves all frozen V2 outputs (`docs/Mesa农业经营接续V2重构/结果/`) untouched and writes V2.1 outputs to
`docs/Mesa农业经营接续V2重构/结果_V2_1深化/`:
1. `stage1_v2_vs_v2_1_cohort_2011_2015.csv`: Pre-2011 NFPS individual aging/migrant/exit-inertia fit vs. V2
2. `exp_f5_institutional_moderators_and_leases.csv`: Village coordination, Order grain, Socialized services & Dual-track leases
3. `exp_asymmetric_info_adverse_selection_moral_hazard.csv`: Coupled climate-shock signal noise, Adverse Selection, Moral Hazard & Beidou Cross-Verification
4. `exp_rl_necessity_standard_6way.csv` & `exp_rl_necessity_asymmetric_6way.csv` + paired 95% CIs on 48 independent holdout seeds (`574001-574048`)
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import fsum
import os
from pathlib import Path
import time

import numpy as np

from mesa_agri_abm import (
    AgriSuccessionMesaModel,
    V2Design,
    default_v2_design,
    fit_v2_government_fqi,
    load_v2_calibrated_regions,
    run_policy_episode,
    select_dev_fixed_policies,
    symmetric_rent_decomposition,
    upgraded_v2_1_design,
)
from mesa_agri_abm.calibration import resolve_project_data_root
from scripts.run_mesa_v2_experiments import (
    T_CRIT_DF31,
    T_CRIT_DF47,
    _episode_worker,
    eval_cohort_metrics,
    load_survey_panel_2010_2015,
    paired_ci,
    write_csv,
    write_json,
)

N_WORKERS = min(6, max(2, (os.cpu_count() or 4) - 2))


def _stage1_seed_worker(args: tuple) -> list[dict]:
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


def run_v2_1_stage1_comparison(project_root: Path, out_dir: Path, d20: V2Design, d21: V2Design) -> list[dict]:
    panel = load_survey_panel_2010_2015(project_root)
    regs20, _ = load_v2_calibrated_regions(project_root, d20)
    regs21, meta21 = load_v2_calibrated_regions(project_root, d21)
    write_json(out_dir / "v2_1_calibration_metadata.json", meta21)

    seeds = list(range(301, 321))
    specs = [
        ("v2_0_baseline", regs20, d20, True),
        ("v2_1_aging_migrant_exit_inertia", regs21, d21, True),
        ("v2_1_ablation_no_exit_inertia", regs21, d21.with_overrides(restart_cultivation_fixed_days=0.0, restart_cultivation_fixed_yuan=0.0), True),
        ("v2_1_ablation_no_rent_foodc_offset", regs21, d21.with_overrides(rent_offsets_subsistence_wedge=False, dynamic_expected_rent_update=False), True),
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for arm_name, regs, d_spec, v_exch in specs:
            s_seeds = seeds if arm_name.startswith("v2_1_aging") or arm_name == "v2_0_baseline" else seeds[:8]
            tasks = [(sd, regs, d_spec, arm_name, v_exch) for sd in s_seeds]
            by_yr: dict[int, list[dict]] = defaultdict(list)
            ne_2015: list[float] = []
            for seed_list in pool.map(_stage1_seed_worker, tasks):
                for item in seed_list:
                    cy = item["year"]
                    by_yr[cy].append(eval_cohort_metrics(item["sim_actors"], panel.loc[cy], item["seed"], cy, arm_name))
                    if cy == 2015:
                        ne_2015.append(item["ne_idle"])

            for cy, m_list in sorted(by_yr.items()):
                rows.append({
                    "arm": arm_name,
                    "n_seeds": len(s_seeds),
                    "year": cy,
                    "survey_sown": m_list[0]["survey_sown"],
                    "sim_sown_mean": round(float(np.mean([x["sim_sown"] for x in m_list])), 2),
                    "participation_err_pp": round(float(np.mean([x["participation_err_pp"] for x in m_list])) * 100.0, 2),
                    "survey_mixed": m_list[0]["survey_mixed"],
                    "sim_mixed_mean": round(float(np.mean([x["sim_mixed"] for x in m_list])), 2),
                    "mixed_err_pp": round(float(np.mean([x["mixed_err_pp"] for x in m_list])) * 100.0, 2),
                    "survey_grain_tons": m_list[0]["survey_grain_tons"],
                    "sim_grain_tons_mean": round(float(np.mean([x["sim_grain_tons"] for x in m_list])), 2),
                    "grain_tons_rel_err_pct": round(
                        abs(float(np.mean([x["sim_grain_tons"] for x in m_list])) - m_list[0]["survey_grain_tons"])
                        / max(1.0, m_list[0]["survey_grain_tons"]) * 100.0,
                        2,
                    ),
                    "grain_share_q_mae_pct": round(float(np.mean([x["grain_share_q_mae"] for x in m_list])) * 100.0, 2),
                    "ne_idle_rate_2015_pct": round(float(np.mean(ne_2015)) * 100.0, 2) if cy == 2015 else None,
                })
    write_csv(out_dir / "stage1_v2_vs_v2_1_cohort_2011_2015.csv", rows)
    return rows


def run_f5_institutional_moderators_and_lease_regimes(
    regs21, out_dir: Path, d21: V2Design, seeds_32: list[int]
) -> list[dict]:
    """Test Village Collective Coordination, Order Grain, Socialized Services, and Short/Long/Dual-Track Leases."""
    arms = [
        ("p0_dual_track", d21, "p0", 0),
        ("scale_takeover_alone", d21, "fixed", 3),
        ("scale_plus_village_coord", d21.with_overrides(enable_village_coordination=True), "fixed", 3),
        ("order_farming_alone", d21, "fixed", 28),
        ("service_trusteeship_accum", d21, "fixed", 5),
        ("village_coord_plus_service", d21.with_overrides(enable_village_coordination=True), "fixed", 5),
        ("portfolio25_dual_track_lease", d21, "fixed", 25),
        ("portfolio25_short_1yr_lease", d21.with_overrides(enable_heterogeneous_lease_terms=False, lease_duration_years=1), "fixed", 25),
        ("portfolio25_long_3yr_lease", d21.with_overrides(enable_heterogeneous_lease_terms=False, lease_duration_years=3), "fixed", 25),
    ]

    tasks = [
        (sd, regs21, d_arm, mode, label, f_idx, None, True)
        for label, d_arm, mode, f_idx in arms
        for sd in seeds_32
    ]
    ep_map: dict[tuple[str, int], dict] = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for ep in pool.map(_episode_worker, tasks):
            ep_map[(ep["policy_label"], ep["seed"])] = ep

    p0_g = float(np.mean([fsum(r["grain_kg"] for r in ep_map[("p0_dual_track", sd)]["main_joint"]) / 1000.0 for sd in seeds_32]))
    p0_wd = float(np.mean([fsum(r["grain_kg"] for r in ep_map[("p0_dual_track", sd)]["withdrawal_joint"]) / 1000.0 for sd in seeds_32]))

    rows = []
    for label, d_arm, _, _ in arms:
        g_list, sp_list, wd_list, pe_list, succ_list, unm_list, l_unp_list, k_svc_list, idi_list = (
            [], [], [], [], [], [], [], [], []
        )
        for sd in seeds_32:
            ep = ep_map[(label, sd)]
            ep0 = ep_map[("p0_dual_track", sd)]
            mj = ep["main_joint"]
            mj0 = ep0["main_joint"]
            ny = len(mj)
            g_list.append(fsum(r["grain_kg"] for r in mj) / 1000.0)
            sp_list.append(fsum(r["spending"] for r in mj) / 10_000.0)
            wd_list.append(fsum(r["grain_kg"] for r in ep["withdrawal_joint"]) / 1000.0)
            succ_list.append(fsum(r["fate_2_strict_succession_grain_mu"] for r in mj) / ny)
            unm_list.append(fsum(r["unmatched_area_mu"] for r in mj) / ny)
            l_unp_list.append(fsum(r["fate_5_leased_unplanted_mu"] for r in mj) / ny)
            k_svc_list.append(mj[-1]["service_capacity_idx"])
            idi_list.append(float(np.mean([r["idi_score"] for r in mj])))
            pe_sd = fsum(
                symmetric_rent_decomposition(
                    mj[k]["leased_area_mu"], mj[k]["total_rent_yuan"],
                    mj0[k]["leased_area_mu"], mj0[k]["total_rent_yuan"],
                    ep["parcel_rents_by_year"][k + 1], ep0["parcel_rents_by_year"][k + 1],
                )["price_effect_yuan"]
                for k in range(ny)
            ) / 10_000.0
            pe_list.append(pe_sd)

        gm = float(np.mean(g_list))
        spm = float(np.mean(sp_list))
        wdm = float(np.mean(wd_list))
        rows.append({
            "arm": label,
            "village_coord": int(d_arm.enable_village_coordination),
            "lease_regime": "dual_track_1_3_4yr" if d_arm.enable_heterogeneous_lease_terms else f"uniform_{d_arm.lease_duration_years}yr",
            "grain_8yr_tons": round(gm, 2),
            "grain_gain_vs_p0_tons": round(gm - p0_g, 2),
            "spending_8yr_wan": round(spm, 3),
            "kg_per_100yuan_spent": round(((gm - p0_g) * 1000.0) / (spm * 100.0), 2) if spm > 1e-4 else 0.0,
            "pure_price_cap_pct": round(float(np.mean(pe_list)) / max(1e-4, spm) * 100.0, 2) if spm > 1e-4 else 0.0,
            "strict_succession_mu_yr": round(float(np.mean(succ_list)), 2),
            "unmatched_released_mu_yr": round(float(np.mean(unm_list)), 2),
            "leased_unplanted_mu_yr": round(float(np.mean(l_unp_list)), 2),
            "end_service_capacity_idx": round(float(np.mean(k_svc_list)), 3),
            "idi_score": round(float(np.mean(idi_list)), 4),
            "withdrawal_yr7_8_gain_tons": round(wdm - p0_wd, 2),
        })
    write_csv(out_dir / "exp_f5_institutional_moderators_and_leases.csv", rows)
    return rows


def run_asymmetric_info_experiment(
    project_root: Path, out_dir: Path, d21: V2Design, seeds_32: list[int]
) -> list[dict]:
    """Identify Adverse Selection, Disaster-Amplified Moral Hazard, and Cross-Verification / Budget Feedback Repair."""
    d_asym = d21.with_overrides(enable_asymmetric_info_and_verification=True)
    regs_asym, _ = load_v2_calibrated_regions(project_root, d_asym)

    arms = [
        ("p0_no_inc_sub", d_asym, "p0", 0),
        ("scale_pre_local_audit", d_asym.with_overrides(cross_verify_enabled=False, enable_performance_budget_linkage=False), "fixed", 3),
        ("scale_pre_beidou_xver", d_asym.with_overrides(cross_verify_enabled=True, enable_performance_budget_linkage=False), "fixed", 3),
        ("scale_pre_xver_perf_budget", d_asym.with_overrides(cross_verify_enabled=True, enable_performance_budget_linkage=True), "fixed", 3),
        ("producer_pre_local_audit", d_asym.with_overrides(cross_verify_enabled=False), "fixed", 1),
        ("producer_pre_beidou_xver", d_asym.with_overrides(cross_verify_enabled=True), "fixed", 1),
        ("order_farming_self_screen", d_asym.with_overrides(cross_verify_enabled=False), "fixed", 28),
        ("service_trust_local_audit", d_asym.with_overrides(cross_verify_enabled=False), "fixed", 5),
        ("service_trust_beidou_xver", d_asym.with_overrides(cross_verify_enabled=True), "fixed", 5),
    ]

    tasks = [
        (sd, regs_asym, d_arm, mode, label, f_idx, None, False)
        for label, d_arm, mode, f_idx in arms
        for sd in seeds_32
    ]
    ep_map: dict[tuple[str, int], dict] = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for ep in pool.map(_episode_worker, tasks):
            ep_map[(ep["policy_label"], ep["seed"])] = ep

    p0_g = float(np.mean([fsum(r["grain_kg"] for r in ep_map[("p0_no_inc_sub", sd)]["main_joint"]) / 1000.0 for sd in seeds_32]))

    rows = []
    for label, d_arm, _, _ in arms:
        g_tot, sp_tot, opp_lease, opp_sub_sh, overclaim_w, mh_loss_tot = [], [], [], [], [], []
        mh_disaster_yr, mh_normal_yr = [], []
        fines_w = []
        for sd in seeds_32:
            ep = ep_map[(label, sd)]
            mj = ep["main_joint"]
            mr = ep["main_regions"]
            ny = len(mj)
            g_tot.append(fsum(r["grain_kg"] for r in mj) / 1000.0)
            sp = fsum(r["spending"] for r in mj)
            sp_tot.append(sp / 10_000.0)
            opp_l = fsum(r["opportunistic_leased_mu"] for r in mj) / ny
            opp_lease.append(opp_l)
            opp_s = fsum(r["opportunistic_subsidy_yuan"] for r in mj)
            opp_sub_sh.append((opp_s / sp * 100.0) if sp > 1.0 else 0.0)
            overclaim_w.append(fsum(r["ineligible_overclaim_net_yuan"] for r in mj) / 10_000.0)
            mh_loss_tot.append(fsum(r["moral_hazard_grain_loss_kg"] for r in mj) / 1000.0)
            fines_w.append(fsum(r["fines_clawed_back_yuan"] for r in mj) / 10_000.0)

            for rr in mr:
                if rr["weather_mult"] < 1.0:
                    mh_disaster_yr.append(rr["moral_hazard_grain_loss_kg"])
                else:
                    mh_normal_yr.append(rr["moral_hazard_grain_loss_kg"])

        gm = float(np.mean(g_tot))
        spm = float(np.mean(sp_tot))
        rows.append({
            "arm": label,
            "cross_verify": int(d_arm.cross_verify_enabled),
            "perf_budget_feedback": int(d_arm.enable_performance_budget_linkage),
            "grain_8yr_tons": round(gm, 2),
            "grain_gain_vs_p0_tons": round(gm - p0_g, 2),
            "spending_8yr_wan": round(spm, 3),
            "kg_per_100yuan_spent": round(((gm - p0_g) * 1000.0) / (spm * 100.0), 2) if spm > 1e-4 else 0.0,
            "adverse_sel_opportunistic_leased_mu_yr": round(float(np.mean(opp_lease)), 2),
            "adverse_sel_opportunistic_subsidy_share_pct": round(float(np.mean(opp_sub_sh)), 2),
            "ineligible_overclaim_leakage_wan": round(float(np.mean(overclaim_w)), 3),
            "fines_clawed_back_wan": round(float(np.mean(fines_w)), 3),
            "moral_hazard_8yr_grain_loss_tons": round(float(np.mean(mh_loss_tot)), 2),
            "moral_hazard_loss_disaster_reg_yr_kg": round(float(np.mean(mh_disaster_yr)) if mh_disaster_yr else 0.0, 1),
            "moral_hazard_loss_normal_reg_yr_kg": round(float(np.mean(mh_normal_yr)) if mh_normal_yr else 0.0, 1),
        })
    write_csv(out_dir / "exp_asymmetric_info_adverse_selection_moral_hazard.csv", rows)
    return rows


def run_six_way_rl_benchmark(
    tag: str,
    regions,
    out_dir: Path,
    design: V2Design,
    train_seeds: list[int],
    eval_seeds: list[int],
) -> dict:
    dev_seeds = list(range(design.dev_seeds[0], design.dev_seeds[0] + 6))
    dev_sel = select_dev_fixed_policies(dev_seeds, regions, design, max_workers=N_WORKERS)
    write_csv(out_dir / f"dev_menu_leaderboard_{tag}.csv", dev_sel["leaderboard"])

    fqi_fit = fit_v2_government_fqi(train_seeds, regions, design, max_workers=N_WORKERS)
    write_json(
        out_dir / f"fqi_training_summary_{tag}.json",
        {
            "tag": tag,
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

    tasks = [
        (sd, regions, design, mode, label, f_idx, w_mat, True)
        for label, mode, f_idx, w_mat in six_arms
        for sd in eval_seeds
    ]
    ep_store: dict[tuple[str, int], dict] = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for ep in pool.map(_episode_worker, tasks):
            ep_store[(ep["policy_label"], ep["seed"])] = ep

    p0_g = float(np.mean([fsum(r["grain_kg"] for r in ep_store[("p0", sd)]["main_joint"]) / 1000.0 for sd in eval_seeds]))
    p0_wd = float(np.mean([fsum(r["grain_kg"] for r in ep_store[("p0", sd)]["withdrawal_joint"]) / 1000.0 for sd in eval_seeds]))

    summary_rows = []
    for label, _, _, _ in six_arms:
        g_8yr, sp_8yr, wd_2yr, idi_m, pe_8yr, k_svc, ov_w, mh_t = [], [], [], [], [], [], [], []
        sample_acts = []
        for sd in eval_seeds:
            ep = ep_store[(label, sd)]
            ep0 = ep_store[("p0", sd)]
            mj = ep["main_joint"]
            mj0 = ep0["main_joint"]
            g_8yr.append(fsum(r["grain_kg"] for r in mj) / 1000.0)
            sp_8yr.append(fsum(r["spending"] for r in mj) / 10_000.0)
            wd_2yr.append(fsum(r["grain_kg"] for r in ep["withdrawal_joint"]) / 1000.0)
            idi_m.append(float(np.mean([r["idi_score"] for r in mj])))
            k_svc.append(mj[-1]["service_capacity_idx"])
            ov_w.append(fsum(r["ineligible_overclaim_net_yuan"] for r in mj) / 10_000.0)
            mh_t.append(fsum(r["moral_hazard_grain_loss_kg"] for r in mj) / 1000.0)
            if not sample_acts:
                sample_acts = [r["action_idx"] for r in mj]
            pe_8yr.append(
                fsum(
                    symmetric_rent_decomposition(
                        mj[k]["leased_area_mu"], mj[k]["total_rent_yuan"],
                        mj0[k]["leased_area_mu"], mj0[k]["total_rent_yuan"],
                        ep["parcel_rents_by_year"][k + 1], ep0["parcel_rents_by_year"][k + 1],
                    )["price_effect_yuan"]
                    for k in range(design.main_years)
                ) / 10_000.0
            )

        gm = float(np.mean(g_8yr))
        spm = float(np.mean(sp_8yr))
        wdm = float(np.mean(wd_2yr))
        summary_rows.append({
            "env_tag": tag,
            "policy_label": label,
            "grain_8yr_tons": round(gm, 2),
            "grain_gain_vs_p0_tons": round(gm - p0_g, 2),
            "spending_8yr_wan": round(spm, 3),
            "kg_per_100yuan_spent": round(((gm - p0_g) * 1000.0) / (spm * 100.0), 2) if spm > 1e-4 else 0.0,
            "pure_price_cap_pct": round(float(np.mean(pe_8yr)) / max(1e-4, spm) * 100.0, 2) if spm > 1e-4 else 0.0,
            "end_service_capacity_idx": round(float(np.mean(k_svc)), 3),
            "ineligible_overclaim_wan": round(float(np.mean(ov_w)), 3),
            "moral_hazard_loss_tons": round(float(np.mean(mh_t)), 2),
            "idi_score": round(float(np.mean(idi_m)), 4),
            "withdrawal_yr7_8_tons": round(wdm, 2),
            "withdrawal_gain_vs_p0_tons": round(wdm - p0_wd, 2),
            "sample_seed0_actions": str(sample_acts),
        })
    write_csv(out_dir / f"exp_rl_necessity_{tag}_6way.csv", summary_rows)

    t_crit = T_CRIT_DF47 if len(eval_seeds) >= 48 else T_CRIT_DF31
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
        gm, glo, ghi = paired_ci(g_d, t_crit)
        sm, slo, shi = paired_ci(sp_d, t_crit)
        wm, wlo, whi = paired_ci(wd_d, t_crit)
        contrasts.append({
            "env_tag": tag,
            "contrast": f"gov_rl_minus_{ref_lbl}",
            "grain_8yr_diff_tons": round(gm, 3),
            "grain_ci95": f"[{glo:.2f}, {ghi:.2f}]",
            "spending_8yr_diff_wan": round(sm, 3),
            "spending_ci95": f"[{slo:.3f}, {shi:.3f}]",
            "withdrawal_diff_tons": round(wm, 3),
            "withdrawal_ci95": f"[{wlo:.2f}, {whi:.2f}]",
        })
    write_csv(out_dir / f"exp_rl_contrasts_{tag}.csv", contrasts)
    return {"summary": summary_rows, "contrasts": contrasts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=resolve_project_data_root(Path(__file__)))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/Mesa农业经营接续V2重构/结果_V2_1深化",
    )
    parser.add_argument("--skip-stage1", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    d20 = default_v2_design()
    d21 = upgraded_v2_1_design()
    regs21, _ = load_v2_calibrated_regions(args.project_root, d21)
    eval_seeds_48 = list(range(d21.eval_seeds[0], d21.eval_seeds[1] + 1))
    train_seeds_24 = list(range(d21.train_seeds[0], d21.train_seeds[0] + 24))

    if not args.skip_stage1:
        print("[1/4] Stage 1 Cohort Validation: V2.0 vs V2.1 (Pre-2011 Individual Aging, Migrant & Exit Inertia)...")
        run_v2_1_stage1_comparison(args.project_root, args.out_dir, d20, d21)
        print(f"      Done at {time.perf_counter() - t0:.1f}s")

    print("[2/4] Institutional Moderators (Village Coordination, Order Grain, Services) & Dual-Track Leases...")
    run_f5_institutional_moderators_and_lease_regimes(regs21, args.out_dir, d21, eval_seeds_48[:32])
    print(f"      Done at {time.perf_counter() - t0:.1f}s")

    print("[3/4] Coupled Natural-Social Signal Noise: Adverse Selection, Moral Hazard & Beidou Cross-Verification...")
    run_asymmetric_info_experiment(args.project_root, args.out_dir, d21, eval_seeds_48[:32])
    print(f"      Done at {time.perf_counter() - t0:.1f}s")

    print("[4/4] 6-Way Government RL Necessity Benchmark (48 Holdout Seeds: Standard Intertemporal & Asymmetric Info)...")
    res_std = run_six_way_rl_benchmark("intertemporal_v2_1", regs21, args.out_dir, d21, train_seeds_24, eval_seeds_48)

    d_asym = d21.with_overrides(enable_asymmetric_info_and_verification=True)
    regs_asym, _ = load_v2_calibrated_regions(args.project_root, d_asym)
    res_asym = run_six_way_rl_benchmark(
        "coupled_asymmetric_v2_1", regs_asym, args.out_dir, d_asym, train_seeds_24[:18], eval_seeds_48[:32]
    )

    elapsed = round(time.perf_counter() - t0, 2)
    write_json(
        args.out_dir / "v2_1_run_manifest.json",
        {
            "design_id": d21.id,
            "elapsed_seconds": elapsed,
            "n_eval_seeds_standard": len(eval_seeds_48),
            "n_eval_seeds_asymmetric": 32,
            "std_contrasts": res_std["contrasts"],
            "asym_contrasts": res_asym["contrasts"],
        },
    )
    print(f"ALL V2.1 UPGRADE EXPERIMENTS COMPLETED IN {elapsed:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
