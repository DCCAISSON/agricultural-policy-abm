"""Stage 2 Shared-Budget Government RL & Multi-Channel Transmission Evaluation for Competing Model.

Trains Backward Ridge-FQI (`q`) and Same-Information Myopic (`myopic`) on disjoint training seeds,
selects `strong_fixed` strictly on `fixed_dev_seeds`, and evaluates all 21 fixed actions + dynamic
planners + Year 7-8 withdrawal + Year-1 allocation pulse on 48 disjoint holdout seeds (`574001..574048`).
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
from hashlib import sha256
import json
from math import fsum, sqrt
from pathlib import Path
import platform
from statistics import mean, stdev

import numpy as np
import pandas as pd
import scipy.stats as stats

from competing_pmp.budget_and_rl import (
    create_worlds,
    discrete_state_code,
    extract_planner_features,
    fit_backward_fqi,
    fork_all_actions,
    make_shared_budget,
    select_planner_action,
)
from competing_pmp.calibration_2010 import load_calibrated_regions
from competing_pmp.config import CompetingDesign, default_design

EVAL_METRICS = (
    "grain_kg", "grain_area", "cash_crop_area", "planted_physical_mu", "idle_area",
    "sown_grain_mu_seasons", "sown_cash_mu_seasons",
    "retained_grain", "successor_grain", "entrant_grain", "continuity_area",
    "offered_area", "posted_area", "matched_area", "unfilled_area", "leased_area",
    "held_area", "default_area", "leased_idle", "rent_total", "avg_rent_per_leased_mu",
    "hired_days_total", "hired_cost_total", "smallholder_cultivating_share",
    "owner_income", "owner_ag_cash", "owner_rent_income", "owner_rent_paid",
    "owner_support", "committed", "spending",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    opener = (lambda p: gzip.open(p, "wt", encoding="utf-8", newline="")) if path.suffix == ".gz" else (
        lambda p: p.open("w", encoding="utf-8", newline="")
    )
    with opener(path) as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _init_worker(regions: tuple, design: CompetingDesign) -> None:
    global G_REGIONS, G_DESIGN, G_BUDGET, G_MENU
    G_REGIONS = regions
    G_DESIGN = design
    G_BUDGET = make_shared_budget(regions, design)
    G_MENU = G_BUDGET.menu()


def _dev_fixed_seed(seed: int) -> list[tuple[float, float]]:
    """Simulate all 21 fixed actions over 8 years by caching distinct local regional paths."""
    roots = create_worlds(seed, G_REGIONS, G_DESIGN)
    local_paths = {}
    for r_i, root in enumerate(roots):
        for loc_act in dict.fromkeys(act[r_i] for act in G_MENU):
            w = root.fork()
            local_paths[(r_i, loc_act)] = [
                w.step(loc_act[0], float(loc_act[1]), ledger=False)[0]
                for _ in range(G_DESIGN.main_years)
            ]
    totals = []
    for act in G_MENU:
        g_sum = fsum(r["grain_kg"] for r_i, loc in enumerate(act) for r in local_paths[(r_i, loc)])
        s_sum = fsum(r["spending"] for r_i, loc in enumerate(act) for r in local_paths[(r_i, loc)])
        totals.append((g_sum, s_sum))
    return totals


def _train_rollout_seed(seed: int) -> list[tuple[int, int, list[float], list[float], list[list[float]]]]:
    """Generate 8-year exploratory trajectory with exact 21-action counterfactual branches at each year."""
    rng = np.random.default_rng([seed, G_DESIGN.exploration_stream])
    exp_actions = rng.integers(len(G_MENU), size=G_DESIGN.main_years)
    ws = create_worlds(seed, G_REGIONS, G_DESIGN)
    out = []
    for yr_idx, a_exp in enumerate(exp_actions, start=1):
        phi = extract_planner_features(ws, G_DESIGN)
        g_vec, _, next_ws = fork_all_actions(ws, G_BUDGET, G_MENU)
        next_phis = [extract_planner_features(nw, G_DESIGN).tolist() for nw in next_ws]
        out.append((seed, yr_idx, phi.tolist(), g_vec.tolist(), next_phis))
        ws = next_ws[int(a_exp)]
    return out


def _eval_seed(job: tuple[int, dict]) -> list[dict]:
    """Evaluate all 21 fixed actions + Q + Myopic + Withdrawal + Pulse on one holdout seed."""
    seed, frozen = job
    rows: list[dict] = []

    # 1. All 21 fixed actions via exact local path cache
    roots = create_worlds(seed, G_REGIONS, G_DESIGN)
    local_paths = {}
    for r_i, root in enumerate(roots):
        for loc_act in dict.fromkeys(act[r_i] for act in G_MENU):
            w = root.fork()
            local_paths[(r_i, loc_act)] = [
                w.step(loc_act[0], float(loc_act[1]), ledger=(seed == G_DESIGN.evaluation_seeds[0]))[0]
                for _ in range(G_DESIGN.main_years)
            ]
    for a_idx, act in enumerate(G_MENU):
        strat = f"fixed_{a_idx:02d}"
        for yr in range(1, G_DESIGN.main_years + 1):
            for r_i, reg in enumerate(G_REGIONS):
                out_r = local_paths[(r_i, act[r_i])][yr - 1]
                rows.append({
                    "seed": seed, "mode": "main", "strategy": strat, "year": yr, "region": reg.name,
                    "action": a_idx, "state": f"fixed_y{yr}", "reason": "fixed",
                    "pred_q_kg": "", "pred_myopic_kg": "", "physical_mu": reg.physical_area,
                    "policy": out_r["policy"], "reservation": out_r["reservation"],
                    "rate_announced_mu": out_r["rate_announced_mu"],
                    **{k: out_r[k] for k in EVAL_METRICS},
                })

    # 2. Dynamic strategies (main 8-year, withdrawal after year 6, opening pulse)
    strong_name = f"fixed_{frozen['strong_fixed']:02d}"
    prop_name = "fixed_17"  # Area-proportional grain_pre
    wd_strats = tuple(dict.fromkeys((strong_name, prop_name, "myopic", "q")))
    plan_runs = [
        *(("main", s) for s in ("myopic", "q", "myopic_repeat", "q_repeat")),
        *(("withdrawal", s) for s in wd_strats),
        *(("pulse", s) for s in ("central_then_prop", "west_then_prop")),
    ]
    audit_ledger = (seed == G_DESIGN.evaluation_seeds[0])

    for mode, strat in plan_runs:
        ws = create_worlds(seed, G_REGIONS, G_DESIGN)
        for yr in range(1, G_DESIGN.main_years + 1):
            scode = discrete_state_code(ws, G_DESIGN)
            if mode == "pulse":
                if yr == 1:
                    a_idx, reason, pq, pm = (5 if strat == "central_then_prop" else 9), "pulse", "", ""
                else:
                    a_idx, reason, pq, pm = 17, "common_followup", "", ""
            else:
                stopped = (mode == "withdrawal" and yr > G_DESIGN.withdrawal_after)
                a_idx, reason, pq_v, pm_v = select_planner_action(strat, yr, ws, G_DESIGN, frozen, stopped=stopped)
                pq = "" if reason in ("fixed", "forced_stop") else round(pq_v, 2)
                pm = "" if reason in ("fixed", "forced_stop") else round(pm_v, 2)

            step_outs = G_BUDGET.step(ws, G_MENU[a_idx], ledger=audit_ledger)
            for reg, out_r in zip(G_REGIONS, step_outs, strict=True):
                rows.append({
                    "seed": seed, "mode": mode, "strategy": strat, "year": yr, "region": reg.name,
                    "action": a_idx, "state": scode, "reason": reason,
                    "pred_q_kg": pq, "pred_myopic_kg": pm, "physical_mu": reg.physical_area,
                    "policy": out_r["policy"], "reservation": out_r["reservation"],
                    "rate_announced_mu": out_r["rate_announced_mu"],
                    **{k: out_r[k] for k in EVAL_METRICS},
                })
    return rows


def _paired_ci(diffs: list[float], bonf_k: int = 1) -> tuple[float, float, float]:
    n = len(diffs)
    m = mean(diffs)
    sd = stdev(diffs) if n > 1 else 0.0
    t_crit = float(stats.t.ppf(1.0 - 0.025 / bonf_k, df=max(1, n - 1)))
    half = t_crit * sd / sqrt(max(1, n))
    return m, m - half, m + half


def run_pipeline(
    project_root: Path, out_dir: Path, workers: int, *, eval_only: bool = False, reaggregate_only: bool = False
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    design = default_design()
    regions, cal_meta = load_calibrated_regions(project_root, design)
    budget = make_shared_budget(regions, design)
    menu = budget.menu()

    dev_seeds = list(range(design.fixed_dev_seeds[0], design.fixed_dev_seeds[1] + 1))
    train_seeds = list(range(design.training_seeds[0], design.training_seeds[1] + 1))
    eval_seeds = list(range(design.evaluation_seeds[0], design.evaluation_seeds[1] + 1))
    assert len(set(dev_seeds) & set(train_seeds)) == 0
    assert len(set(dev_seeds) & set(eval_seeds)) == 0
    assert len(set(train_seeds) & set(eval_seeds)) == 0

    print(f"[Stage 2] Budget = {budget.common_yuan} yuan/yr, 21 actions, workers = {workers}", flush=True)

    if (eval_only or reaggregate_only) and (out_dir / "frozen_planner.json").exists():
        frozen = json.loads((out_dir / "frozen_planner.json").read_text(encoding="utf-8"))
        assert frozen["design_sha256"] == design.fingerprint(), "Frozen planner design hash mismatch"
        strong_fixed = int(frozen["strong_fixed"])
        main_fit = frozen["main_fit"]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(regions, design)) as pool:
            print("1. Developing fixed benchmark on 16 disjoint dev seeds...", flush=True)
            dev_res = list(pool.map(_dev_fixed_seed, dev_seeds))
            dev_scores = [mean(s[i][0] for s in dev_res) for i in range(len(menu))]
            strong_fixed = max(range(len(dev_scores)), key=dev_scores.__getitem__)

            print(f"   Selected strong_fixed = fixed_{strong_fixed:02d} ({dev_scores[strong_fixed]/1000:.2f} tons)", flush=True)
            print(f"2. Generating counterfactual FQI rollouts across {len(train_seeds)} training seeds...", flush=True)
            raw_rollouts = [item for block in pool.map(_train_rollout_seed, train_seeds) for item in block]

        rollouts_main = {y: [] for y in range(1, design.main_years + 1)}
        rollouts_rep = {y: [] for y in range(1, design.main_years + 1)}
        for s, yr, phi_l, g_l, nphi_l in raw_rollouts:
            tup = (np.asarray(phi_l), np.asarray(g_l), [np.asarray(x) for x in nphi_l])
            rollouts_main[yr].append(tup)
            if s % 4 != 0:
                rollouts_rep[yr].append(tup)

        main_fit = fit_backward_fqi(rollouts_main, design, len(menu))
        rep_fit = fit_backward_fqi(rollouts_rep, design, len(menu))

        frozen = {
            "design_sha256": design.fingerprint(),
            "common_budget_yuan": budget.common_yuan,
            "menu": menu,
            "strong_fixed": strong_fixed,
            "dev_fixed_grain_kg": [round(x, 2) for x in dev_scores],
            "main_fit": main_fit,
            "repeat_fit": rep_fit,
        }
        _write_json(out_dir / "frozen_planner.json", frozen)

    if reaggregate_only and (out_dir / "region_year.csv.gz").exists():
        with gzip.open(out_dir / "region_year.csv.gz", "rt", encoding="utf-8", newline="") as fh:
            raw_rows = list(csv.DictReader(fh))
        seen_keys: dict[tuple, dict] = {}
        eval_rows = []
        num_cols = {"physical_mu", "reservation", "rate_announced_mu", *EVAL_METRICS}
        int_cols = {"seed", "year", "action"}
        for r in raw_rows:
            parsed = {
                k: (int(v) if k in int_cols else (float(v) if k in num_cols else v))
                for k, v in r.items()
            }
            pk = (parsed["seed"], parsed["mode"], parsed["strategy"], parsed["year"], parsed["region"])
            if pk in seen_keys:
                assert seen_keys[pk] == parsed, f"Non-identical duplicate row at {pk}"
                continue
            seen_keys[pk] = parsed
            eval_rows.append(parsed)
        _write_csv(out_dir / "region_year.csv.gz", eval_rows)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(regions, design)) as pool:
            print(f"3. Evaluating all policies on {len(eval_seeds)} disjoint holdout seeds...", flush=True)
            eval_rows = [r for block in pool.map(_eval_seed, [(s, frozen) for s in eval_seeds]) for r in block]
        _write_csv(out_dir / "region_year.csv.gz", eval_rows)

    # Aggregate 8-year totals by (mode, strategy, seed) and by region
    df = pd.DataFrame(eval_rows)
    main_df = df[df["mode"] == "main"]
    wd_df = df[(df["mode"] == "withdrawal") & (df["year"] > design.withdrawal_after)]

    seed_tot = main_df.groupby(["strategy", "seed"], as_index=False)[
        ["grain_kg", "spending", "committed", "successor_grain", "entrant_grain",
         "retained_grain", "leased_area", "rent_total", "idle_area", "hired_days_total"]
    ].sum()
    wd_tot = wd_df.groupby(["strategy", "seed"], as_index=False)[["grain_kg", "spending"]].sum()

    # Strategy summary table + 4-channel fiscal transmission diagnostics vs P0 (`fixed_00`)
    p0_by_seed = seed_tot[seed_tot["strategy"] == "fixed_00"].set_index("seed")
    p0_wd_seed = main_df[(main_df["strategy"] == "fixed_00") & (main_df["year"] > design.withdrawal_after)].groupby("seed")["grain_kg"].sum()

    strat_summary = []
    key_strats = [
        "fixed_00", "fixed_17", "fixed_18", "fixed_19", "fixed_20",
        f"fixed_{strong_fixed:02d}", "myopic", "q", "myopic_repeat", "q_repeat",
    ]
    seen_s = set()
    ordered_strats = [s for s in key_strats if not (s in seen_s or seen_s.add(s))] + [
        f"fixed_{i:02d}" for i in range(21) if f"fixed_{i:02d}" not in seen_s
    ]

    for st_name in ordered_strats:
        sub = seed_tot[seed_tot["strategy"] == st_name].set_index("seed")
        if sub.empty:
            continue
        g_tons = float(sub["grain_kg"].mean() / 1000.0)
        sp_wan = float(sub["spending"].mean() / 10000.0)
        dg_tons = float((sub["grain_kg"] - p0_by_seed["grain_kg"]).mean() / 1000.0)
        drent_wan = float((sub["rent_total"] - p0_by_seed["rent_total"]).mean() / 10000.0)
        sp_yuan = float(sub["spending"].mean())

        # Transmission diagnostics
        me_kg_per_100yuan = (dg_tons * 1000.0) / (sp_yuan / 100.0) if sp_yuan > 1.0 else 0.0
        rent_cap_ratio = (drent_wan * 10000.0) / sp_yuan if sp_yuan > 1.0 else 0.0
        succ_share = float(sub["successor_grain"].mean() / max(1.0, sub["grain_kg"].mean()))
        ent_share = float(sub["entrant_grain"].mean() / max(1.0, sub["grain_kg"].mean()))
        unspent_share = 1.0 - (sp_yuan / (8.0 * budget.common_yuan)) if st_name != "fixed_00" else 0.0

        wd_sub = wd_tot[wd_tot["strategy"] == st_name].set_index("seed")
        wd_tons = float(wd_sub["grain_kg"].mean() / 1000.0) if not wd_sub.empty else ""
        wd_gain_vs_p0 = float((wd_sub["grain_kg"] - p0_wd_seed).mean() / 1000.0) if not wd_sub.empty else ""

        strat_summary.append({
            "strategy": st_name,
            "grain_8yr_tons": round(g_tons, 2),
            "spending_8yr_wanyuan": round(sp_wan, 3),
            "delta_grain_vs_P0_tons": round(dg_tons, 2),
            "grain_kg_per_100yuan_spent": round(me_kg_per_100yuan, 2),
            "rent_capitalization_ratio": round(rent_cap_ratio, 4),
            "successor_grain_share": round(succ_share, 4),
            "entrant_grain_share": round(ent_share, 4),
            "budget_unspent_attenuation": round(unspent_share, 4),
            "idle_area_mean_mu_yr": round(float(sub["idle_area"].mean() / 8.0), 2),
            "leased_area_mean_mu_yr": round(float(sub["leased_area"].mean() / 8.0), 2),
            "withdrawal_yr7_8_tons": round(wd_tons, 2) if wd_tons != "" else "",
            "withdrawal_vs_P0_tons": round(wd_gain_vs_p0, 2) if wd_gain_vs_p0 != "" else "",
        })
    _write_csv(out_dir / "strategy_transmission_summary.csv", strat_summary)

    # Regional breakdown for Q, Myopic, Strong Fixed, Area-Proportional Fixed, and P0
    reg_breakdown = []
    for st_name in dict.fromkeys(("fixed_00", "fixed_17", f"fixed_{strong_fixed:02d}", "myopic", "q")):
        sub_r = main_df[main_df["strategy"] == st_name].groupby(["region", "seed"], as_index=False)[
            ["grain_kg", "spending", "leased_area", "idle_area", "successor_grain"]
        ].sum()
        wd_r = wd_df[wd_df["strategy"] == st_name].groupby(["region", "seed"], as_index=False)["grain_kg"].sum()
        for reg_name in design.region_order:
            sr = sub_r[sub_r["region"] == reg_name]
            wr = wd_r[wd_r["region"] == reg_name]
            reg_breakdown.append({
                "strategy": st_name,
                "region": reg_name,
                "grain_8yr_tons": round(float(sr["grain_kg"].mean() / 1000.0), 2),
                "spending_8yr_wanyuan": round(float(sr["spending"].mean() / 10000.0), 3),
                "successor_grain_8yr_tons": round(float(sr["successor_grain"].mean() / 1000.0), 2),
                "leased_mu_per_yr": round(float(sr["leased_area"].mean() / 8.0), 2),
                "idle_mu_per_yr": round(float(sr["idle_area"].mean() / 8.0), 2),
                "withdrawal_yr7_8_tons": round(float(wr["grain_kg"].mean() / 1000.0), 2) if not wr.empty else "",
            })
    _write_csv(out_dir / "regional_breakdown.csv", reg_breakdown)

    # Primary Bonferroni-adjusted paired comparisons (k=6 familywise tests, matching sprint_protocol.json)
    q_main = seed_tot[seed_tot["strategy"] == "q"].set_index("seed")
    q_wd = wd_tot[wd_tot["strategy"] == "q"].set_index("seed")
    paired_rows = []
    for comp_label, comp_st in (
        ("strong_fixed", f"fixed_{strong_fixed:02d}"),
        ("same_info_myopic", "myopic"),
        ("area_prop_grain_pre", "fixed_17"),
    ):
        c_main = seed_tot[seed_tot["strategy"] == comp_st].set_index("seed")
        c_wd = wd_tot[wd_tot["strategy"] == comp_st].set_index("seed")
        bonf_k = 6 if comp_label in ("strong_fixed", "same_info_myopic") else 1

        dg = ((q_main["grain_kg"] - c_main["grain_kg"]) / 1000.0).tolist()
        ds = ((q_main["spending"] - c_main["spending"]) / 10000.0).tolist()
        dw = ((q_wd["grain_kg"] - c_wd["grain_kg"]) / 1000.0).tolist()

        mg, lg, hg = _paired_ci(dg, bonf_k)
        ms, ls, hs = _paired_ci(ds, bonf_k)
        mw, lw, hw = _paired_ci(dw, bonf_k)
        paired_rows.append({
            "comparison": f"q_minus_{comp_label}",
            "comparator": comp_st,
            "bonferroni_k": bonf_k,
            "delta_grain_tons": round(mg, 2),
            "grain_ci_low": round(lg, 2),
            "grain_ci_high": round(hg, 2),
            "delta_spending_wanyuan": round(ms, 3),
            "spending_ci_low": round(ls, 3),
            "spending_ci_high": round(hs, 3),
            "delta_withdrawal_tons": round(mw, 2),
            "withdrawal_ci_low": round(lw, 2),
            "withdrawal_ci_high": round(hw, 2),
        })
    _write_csv(out_dir / "paired_comparisons.csv", paired_rows)

    # Action frequency & fallback audit
    act_counts = (
        main_df[main_df["strategy"].isin(["q", "myopic", "q_repeat", "myopic_repeat"])]
        .groupby(["strategy", "action", "reason"], as_index=False)
        .size()
        .rename(columns={"size": "region_year_rows"})
    )
    act_counts["decision_count"] = act_counts["region_year_rows"] // 4
    _write_csv(out_dir / "planner_action_counts.csv", act_counts.to_dict("records"))

    receipt = {
        "status": "completed_and_verified",
        "design_sha256": design.fingerprint(),
        "common_annual_budget_yuan": budget.common_yuan,
        "strong_fixed_selected_on_dev": f"fixed_{strong_fixed:02d}",
        "eval_seeds_count": len(eval_seeds),
        "region_year_rows": len(eval_rows),
        "q_fallback_count": int((main_df.loc[main_df["strategy"] == "q", "reason"] != "fqi_supported").sum() // 4),
        "fqi_year_r2": main_fit["fit_r2"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    _write_json(out_dir / "evaluation_receipt.json", receipt)
    print("Stage 2 complete! Paired results:", flush=True)
    for r in paired_rows:
        print(" ", r, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--reaggregate-only", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.project, args.out, args.workers, eval_only=args.eval_only, reaggregate_only=args.reaggregate_only)
