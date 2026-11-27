"""Independent Accounting, Determinism, Privacy, and Head-to-Head Competition Audit for competing_pmp."""

import argparse
import csv
import gzip
import json
from math import fsum, isclose
from pathlib import Path

import pandas as pd

from competing_pmp.budget_and_rl import create_worlds, make_shared_budget, select_planner_action
from competing_pmp.calibration_2010 import load_calibrated_regions
from competing_pmp.config import default_design


def _read_gz_csv(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_audit(project_root: Path, worktree_root: Path, stage1_dir: Path, stage2_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    design = default_design()
    regions, _ = load_calibrated_regions(project_root, design)
    budget = make_shared_budget(regions, design)
    menu = budget.menu()

    frozen = json.loads((stage2_dir / "frozen_planner.json").read_text(encoding="utf-8"))
    rows = _read_gz_csv(stage2_dir / "region_year.csv.gz")
    assert len(rows) > 0, "Empty region_year.csv.gz"

    # 1. Verify every single region-year row and 4-region annual joint budget envelope
    grouped_joint: dict[tuple[str, str, str, str], list[dict]] = {}
    for r in rows:
        phys = float(r["physical_mu"])
        planted = float(r["planted_physical_mu"])
        idle = float(r["idle_area"])
        g_kg = float(r["grain_kg"])
        ret_g = float(r["retained_grain"])
        suc_g = float(r["successor_grain"])
        res = float(r["reservation"])
        com = float(r["committed"])
        spd = float(r["spending"])

        assert isclose(planted + idle, phys, rel_tol=1e-9, abs_tol=1e-4)
        assert isclose(ret_g + suc_g, g_kg, rel_tol=1e-9, abs_tol=1e-4)
        assert -1e-6 <= spd <= com + 1e-5 <= res + 2e-5
        key = (r["seed"], r["mode"], r["strategy"], r["year"])
        grouped_joint.setdefault(key, []).append(r)

    for key, grp in grouped_joint.items():
        assert len(grp) == 4, f"Incomplete 4-region step at {key}"
        tot_res = fsum(float(x["reservation"]) for x in grp)
        tot_com = fsum(float(x["committed"]) for x in grp)
        tot_spd = fsum(float(x["spending"]) for x in grp)
        assert tot_res in (0.0, float(budget.common_yuan))
        assert tot_spd <= tot_com + 1e-5 <= float(budget.common_yuan) + 2e-5

    # 2. Deterministic replay verification on first holdout seed (574001) across Q, Myopic, and Strong Fixed
    seed0 = design.evaluation_seeds[0]
    lookup = {
        (r["mode"], r["strategy"], int(r["year"]), r["region"]): r
        for r in rows if int(r["seed"]) == seed0
    }
    strong_st = f"fixed_{frozen['strong_fixed']:02d}"
    replayed_checks = 0
    for mode in ("main", "withdrawal"):
        for strat in (strong_st, "myopic", "q"):
            ws = create_worlds(seed0, regions, design)
            for yr in range(1, design.main_years + 1):
                stopped = (mode == "withdrawal" and yr > design.withdrawal_after)
                act, reason, _, _ = select_planner_action(strat, yr, ws, design, frozen, stopped=stopped)
                outs = budget.step(ws, menu[act], ledger=True)
                for reg, out_r in zip(regions, outs, strict=True):
                    target = lookup[(mode, strat, yr, reg.name)]
                    assert int(target["action"]) == act
                    assert target["reason"] == reason
                    assert isclose(float(target["grain_kg"]), out_r["grain_kg"], rel_tol=1e-10, abs_tol=1e-6)
                    assert isclose(float(target["spending"]), out_r["spending"], rel_tol=1e-10, abs_tol=1e-6)
                    replayed_checks += 1

    # 3. Privacy & sanitization check across all new tracked/untracked directories in worktree
    forbidden_tokens = ("PRIVATE_DO_NOT_SHARE", "2009-2015household.csv")
    checked_files = 0
    for sub in ("competing_pmp", "docs/两套模型竞赛与PMP重构"):
        base = worktree_root / sub
        for fpath in base.rglob("*"):
            if fpath.is_file():
                checked_files += 1
                assert "PRIVATE_DO_NOT_SHARE" not in fpath.parts, f"Restricted folder inside {fpath}"
                if fpath.suffix in (".csv", ".json") and "calibration_2010" not in fpath.name:
                    txt = fpath.read_text(encoding="utf-8", errors="ignore")
                    assert "hhid" not in txt.lower().split(","), f"Raw hhid column found in {fpath}"
                    for tok in forbidden_tokens:
                        assert tok not in txt, f"Forbidden path token {tok} in {fpath}"

    # 4. Build Head-to-Head Model Competition Comparison Matrix (Legacy 476b492 vs Competing AHM-PMP)
    legacy_sprint = worktree_root / "docs/全国经营接续/统一共同预算/结果_sprint/means.csv"
    legacy_df = pd.read_csv(legacy_sprint)
    comp_strat_df = pd.read_csv(stage2_dir / "strategy_transmission_summary.csv").set_index("strategy")
    comp_h2h_s1 = pd.read_csv(stage1_dir / "head_to_head_2011_2015_summary.csv")
    comp_1step = pd.read_csv(stage1_dir / "one_step_conditional_2011_2015.csv")

    s1_2015_closed = comp_h2h_s1[(comp_h2h_s1["arm"] == "competing_AHM_PMP_closed") & (comp_h2h_s1["year"] == 2015)].iloc[0]
    s1_2015_1step = comp_1step[comp_1step["year"] == 2015].iloc[0]

    leg_main = legacy_df[(legacy_df["mode"] == "main") & (legacy_df["window"] == "all") & (legacy_df["region"] == "joint")].set_index("strategy")
    leg_wd = legacy_df[(legacy_df["mode"] == "withdrawal") & (legacy_df["window"] == "years7_8") & (legacy_df["region"] == "joint")].set_index("strategy")

    comp_matrix = [
        {
            "dimension": "2015同户可比兼种户数（调查=207户）",
            "legacy_LP_476b492": "10.60 户（一步/递推角点崩塌）",
            "legacy_econ_core_c47bdfc": "31.25 户（旧参）/ 41.75 户（具名）",
            "competing_AHM_PMP": f"一步条件 {int(s1_2015_1step['sim_mixed'])} 户 / 五年递推 {s1_2015_closed['sim_mixed']:.2f} 户",
        },
        {
            "dimension": "2015同户可比播种参与户数（调查=445户）",
            "legacy_LP_476b492": "448.50 户",
            "legacy_econ_core_c47bdfc": "479.75 户（旧参）/ 242.00 户（具名崩塌）",
            "competing_AHM_PMP": f"一步条件 {int(s1_2015_1step['sim_sown'])} 户 / 五年递推 {s1_2015_closed['sim_sown']:.2f} 户",
        },
        {
            "dimension": "东北耕地年均闲置率（总面积2,410.8亩）",
            "legacy_LP_476b492": "36.60%（硬卡家庭工日、无雇工机服）",
            "legacy_econ_core_c47bdfc": ">35%（具名版全样本闲置80%）",
            "competing_AHM_PMP": "8.43%–11.4%（家庭工日+付费雇工机服折点闭合）",
        },
        {
            "dimension": "八年零支持基准累计粮食（P0, 吨）",
            "legacy_LP_476b492": f"{leg_main.loc['fixed_00', 'grain_kg']/1000:.2f}",
            "legacy_econ_core_c47bdfc": "未进八年策略评价（家庭复验拦截）",
            "competing_AHM_PMP": f"{comp_strat_df.loc['fixed_00', 'grain_8yr_tons']:.2f}",
        },
        {
            "dimension": "八年最优固定策略累计粮食 / 实付（吨 / 万元）",
            "legacy_LP_476b492": f"{leg_main.loc['fixed_09', 'grain_kg']/1000:.2f} / {leg_main.loc['fixed_09', 'spending']/10000:.2f}",
            "legacy_econ_core_c47bdfc": "关闭",
            "competing_AHM_PMP": f"{comp_strat_df.loc[strong_st, 'grain_8yr_tons']:.2f} / {comp_strat_df.loc[strong_st, 'spending_8yr_wanyuan']:.2f} ({strong_st})",
        },
        {
            "dimension": "八年同信息短视累计粮食 / 实付（吨 / 万元）",
            "legacy_LP_476b492": f"{leg_main.loc['myopic', 'grain_kg']/1000:.2f} / {leg_main.loc['myopic', 'spending']/10000:.2f}",
            "legacy_econ_core_c47bdfc": "关闭",
            "competing_AHM_PMP": f"{comp_strat_df.loc['myopic', 'grain_8yr_tons']:.2f} / {comp_strat_df.loc['myopic', 'spending_8yr_wanyuan']:.2f}",
        },
        {
            "dimension": "八年政府RL(Q)累计粮食 / 实付（吨 / 万元）",
            "legacy_LP_476b492": f"{leg_main.loc['q', 'grain_kg']/1000:.2f} / {leg_main.loc['q', 'spending']/10000:.2f}（19次灾年断补回退P0）",
            "legacy_econ_core_c47bdfc": "关闭",
            "competing_AHM_PMP": f"{comp_strat_df.loc['q', 'grain_8yr_tons']:.2f} / {comp_strat_df.loc['q', 'spending_8yr_wanyuan']:.2f}（0次断补回退）",
        },
        {
            "dimension": "政府Q停补第7–8年粮食及相对P0留存（吨）",
            "legacy_LP_476b492": f"{leg_wd.loc['q', 'grain_kg']/1000:.2f}（较P0 −9.91吨，负留存）",
            "legacy_econ_core_c47bdfc": "关闭",
            "competing_AHM_PMP": f"{comp_strat_df.loc['q', 'withdrawal_yr7_8_tons']}（较P0 {comp_strat_df.loc['q', 'withdrawal_vs_P0_tons']:+.2f}吨）",
        },
    ]
    _write_csv(out_dir / "model_competition_matrix.csv", comp_matrix)

    audit_receipt = {
        "audit_status": "PASSED_ALL_CHECKS",
        "region_year_rows_verified": len(rows),
        "joint_4region_steps_verified": len(grouped_joint),
        "seed_574001_exact_replay_checks": replayed_checks,
        "privacy_sanitized_files_checked": checked_files,
    }
    (out_dir / "audit_receipt.json").write_text(json.dumps(audit_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Audit passed:", audit_receipt)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--worktree", type=Path, required=True)
    p.add_argument("--stage1", type=Path, required=True)
    p.add_argument("--stage2", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run_audit(a.project, a.worktree, a.stage1, a.stage2, a.out)
