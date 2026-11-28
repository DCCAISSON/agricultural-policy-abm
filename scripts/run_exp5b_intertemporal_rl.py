"""Zero-Water Exp 5B: Intertemporal Government RL (Paid 3-Yr Village Campaigns + 28-Dim State + Double-Q MLP-FQI).

Runs strictly on `AgriSuccessionMesaModel` with 100% real fiscal deduction (105,000 yuan/campaign out of
150,000 yuan/yr shared clearing cap), zero free yield externalities, exogenous P0-ref Bellman rewards,
disjoint dev/train/eval seeds (12 dev / 32 train x 2 passes / 48 holdout eval `574001-574048`), and matched Ridge-FQI ablation.
"""

from concurrent.futures import ProcessPoolExecutor
import csv
from math import fsum
import os
from pathlib import Path
import time

import numpy as np

from mesa_agri_abm import (
    fit_v2_government_mlp_fqi,
    intertemporal_v2_2_design,
    load_v2_calibrated_regions,
    select_dev_fixed_policies,
    symmetric_rent_decomposition,
)
from mesa_agri_abm.calibration import resolve_project_data_root
from scripts.run_mesa_v2_experiments import (
    T_CRIT_DF47,
    _episode_worker,
    paired_ci,
    write_csv,
    write_json,
)

N_WORKERS = min(16, max(2, (os.cpu_count() or 4) - 4))
ROLLOUT_CACHE_PATH = Path(
    r"C:\Users\lenovo\.gemini\antigravity\brain\7cc286b5-e692-4f69-9b24-fd2e5cbe9b6c\scratch\exp5b_rollouts_p0ref.pkl"
)


def main() -> None:
    t0 = time.perf_counter()
    project_root = resolve_project_data_root(Path(__file__))
    out_dir = Path(__file__).resolve().parents[1] / "docs/Mesa农业经营接续V2重构/结果_V2_2净版"
    out_dir.mkdir(parents=True, exist_ok=True)

    d5b = intertemporal_v2_2_design()
    regions, _ = load_v2_calibrated_regions(project_root, d5b)
    dev_seeds = list(range(d5b.dev_seeds[0], d5b.dev_seeds[1] + 1))
    train_seeds = list(range(d5b.train_seeds[0], d5b.train_seeds[1] + 1))
    eval_seeds = list(range(d5b.eval_seeds[0], d5b.eval_seeds[1] + 1))

    lb_path = out_dir / "exp5b_dev_36actions_leaderboard.csv"
    if lb_path.exists():
        with lb_path.open("r", encoding="utf-8", newline="") as fh:
            lb_rows = [
                {
                    "action_idx": int(r["action_idx"]),
                    "menu_spec": r["menu_spec"],
                    "dev_mean_8yr_grain_tons": float(r["dev_mean_8yr_grain_tons"]),
                    "dev_mean_8yr_spending_wan": float(r["dev_mean_8yr_spending_wan"]),
                }
                for r in csv.DictReader(fh)
            ]
        if len(lb_rows) == 36:
            uniform_idxs = list(range(1, 6)) + [28]
            regional_idxs = [i for i in range(6, 36) if i not in uniform_idxs]
            score_fn = lambda r: (
                r["dev_mean_8yr_grain_tons"] - d5b.gov_fiscal_cost_weight * (r["dev_mean_8yr_spending_wan"] / 10.0),
                -r["dev_mean_8yr_spending_wan"],
            )
            u_idx = int(max((lb_rows[i] for i in uniform_idxs), key=score_fn)["action_idx"])
            r_idx = int(max((lb_rows[i] for i in regional_idxs), key=score_fn)["action_idx"])
            print(f"[Exp5B 1/3] Reused cached 36-action dev leaderboard (u_idx={u_idx}, r_idx={r_idx}).")
        else:
            lb_rows = []
    else:
        lb_rows = []

    if not lb_rows:
        print("[Exp5B 1/3] Dev leaderboard across 36 actions (12 disjoint dev seeds)...")
        dev_sel = select_dev_fixed_policies(dev_seeds, regions, d5b, max_workers=N_WORKERS)
        write_csv(lb_path, dev_sel["leaderboard"])
        u_idx = dev_sel["best_uniform_fixed_idx"]
        r_idx = dev_sel["best_regional_fixed_idx"]

    print("[Exp5B 2/3] Fitting Cross-Fitted Double-Q Tree-Based Batch FQI (gamma=0 vs gamma=0.96) + matched Ridge-FQI...")
    fqi_fit = fit_v2_government_mlp_fqi(
        train_seeds,
        regions,
        d5b,
        n_rollout_passes=2,
        max_workers=N_WORKERS,
        rollout_cache_path=ROLLOUT_CACHE_PATH,
    )
    write_json(out_dir / "exp5b_mlp_fqi_training_summary.json", {
        "n_train_seeds": fqi_fit["n_train_seeds"],
        "n_rollout_episodes": fqi_fit["n_rollout_episodes"],
        "n_actions": fqi_fit["n_actions"],
        "best_uniform_fixed_idx": u_idx,
        "best_regional_fixed_idx": r_idx,
        "fit_r2_tree_q": fqi_fit["fit_r2_tree_q"],
        "fit_r2_tree_myopic": fqi_fit["fit_r2_tree_myopic"],
        "fit_r2_ridge_q": fqi_fit["fit_r2_ridge_q"],
        "oos_r2_tree_q": fqi_fit["oos_r2_tree_q"],
        "oos_r2_ridge_q": fqi_fit["oos_r2_ridge_q"],
        "tree_q_weights": fqi_fit["tree_q_weights"],
        "tree_myopic_weights": fqi_fit["tree_myopic_weights"],
        "ridge_q_weights": fqi_fit["ridge_q_weights"],
    })

    arms = [
        ("p0", "p0", 0, None),
        ("uniform_fixed", "uniform_fixed", u_idx, None),
        ("regional_fixed", "regional_fixed", r_idx, None),
        ("transparent_rule", "transparent_rule", 0, None),
        ("myopic_tree", "myopic", 0, fqi_fit["tree_myopic_weights"]),
        ("gov_rl_tree", "gov_rl", 0, fqi_fit["tree_q_weights"]),
        ("gov_rl_ridge", "gov_rl", 0, fqi_fit["ridge_q_weights"]),
    ]

    print("[Exp5B 3/3] Evaluating 7 strategies across 48 independent holdout seeds (`574001-574048`)...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        ep_list = list(pool.map(_episode_worker, [
            (sd, regions, d5b, mode, lbl, f_idx, w_mat, True)
            for lbl, mode, f_idx, w_mat in arms
            for sd in eval_seeds
        ]))
    ep_store = {(ep["policy_label"], ep["seed"]): ep for ep in ep_list}

    for ep in ep_list:
        for row in ep["main_joint"]:
            assert row["spending"] <= 150_000.0 + 1e-2, f"Budget breached: {row}"
        for row in ep["withdrawal_joint"]:
            assert row["spending"] == 0.0, f"Withdrawal spending nonzero: {row}"

    traj_rows = []
    yearly_rows = []
    for lbl, _, _, _ in arms:
        for yr_i in range(d5b.main_years):
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
        for wd_k in range(d5b.main_years - d5b.withdrawal_after):
            g_yr = [ep_store[(lbl, sd)]["withdrawal_joint"][wd_k]["grain_kg"] / 1000.0 for sd in eval_seeds]
            g0_yr = [ep_store[("p0", sd)]["withdrawal_joint"][wd_k]["grain_kg"] / 1000.0 for sd in eval_seeds]
            gm, glo, ghi = paired_ci([g_yr[k] - g0_yr[k] for k in range(len(eval_seeds))], T_CRIT_DF47)
            yearly_rows.append({
                "policy_label": lbl,
                "window": "withdrawal",
                "year": d5b.withdrawal_after + wd_k + 1,
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
    write_csv(out_dir / "exp5b_intertemporal_yearly.csv", yearly_rows)
    write_csv(out_dir / "exp5b_intertemporal_action_trajectories.csv", traj_rows)

    summary_rows = []
    for lbl, _, _, _ in arms:
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
            succ_l.append(fsum(r["fate_2_strict_succession_grain_mu"] for r in ep["main_joint"]) / d5b.main_years)
            pe_l.append(fsum(
                symmetric_rent_decomposition(
                    ep["main_joint"][k]["leased_area_mu"], ep["main_joint"][k]["total_rent_yuan"],
                    ep0["main_joint"][k]["leased_area_mu"], ep0["main_joint"][k]["total_rent_yuan"],
                    ep["parcel_rents_by_year"][k + 1], ep0["parcel_rents_by_year"][k + 1],
                )["price_effect_yuan"]
                for k in range(d5b.main_years)
            ) / 10_000.0)

        gm, glo, ghi = paired_ci(gd_l, T_CRIT_DF47)
        spm, splo, sphi = paired_ci(sp_l, T_CRIT_DF47)
        efm, eflo, efhi = paired_ci(eff_l, T_CRIT_DF47) if eff_l else (0.0, 0.0, 0.0)
        wdm, wdlo, wdhi = paired_ci(wd_d_l, T_CRIT_DF47)
        acts_seq = [int(max(
            set(int(ep_store[(lbl, sd)]["main_joint"][yr_i]["action_idx"]) for sd in eval_seeds),
            key=lambda a: sum(1 for sd in eval_seeds if int(ep_store[(lbl, sd)]["main_joint"][yr_i]["action_idx"]) == a),
        )) for yr_i in range(d5b.main_years)]

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
    write_csv(out_dir / "exp5b_intertemporal_6way_summary.csv", summary_rows)

    contrasts = []
    for ref_lbl in ("p0", "uniform_fixed", "regional_fixed", "transparent_rule", "myopic_tree", "gov_rl_ridge"):
        gd, spd, wdd = [], [], []
        for sd in eval_seeds:
            e_rl, e_rf = ep_store[("gov_rl_tree", sd)], ep_store[(ref_lbl, sd)]
            gd.append((fsum(r["grain_kg"] for r in e_rl["main_joint"]) - fsum(r["grain_kg"] for r in e_rf["main_joint"])) / 1000.0)
            spd.append((fsum(r["spending"] for r in e_rl["main_joint"]) - fsum(r["spending"] for r in e_rf["main_joint"])) / 10_000.0)
            wdd.append((fsum(r["grain_kg"] for r in e_rl["withdrawal_joint"]) - fsum(r["grain_kg"] for r in e_rf["withdrawal_joint"])) / 1000.0)
        gm, glo, ghi = paired_ci(gd, T_CRIT_DF47)
        sm, slo, shi = paired_ci(spd, T_CRIT_DF47)
        wm, wlo, whi = paired_ci(wdd, T_CRIT_DF47)
        contrasts.append({
            "contrast": f"gov_rl_tree_minus_{ref_lbl}",
            "grain_8yr_diff_tons": round(gm, 2),
            "grain_ci95": f"[{glo:.2f}, {ghi:.2f}]",
            "spending_8yr_diff_wan": round(sm, 3),
            "spending_ci95": f"[{slo:.3f}, {shi:.3f}]",
            "withdrawal_diff_tons": round(wm, 2),
            "withdrawal_ci95": f"[{wlo:.2f}, {whi:.2f}]",
        })
    write_csv(out_dir / "exp5b_intertemporal_6way_contrasts.csv", contrasts)

    elapsed = time.perf_counter() - t0
    print(f"EXP 5B COMPLETED IN {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
