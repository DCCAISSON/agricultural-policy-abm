# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pyyaml", "mesa"]
# ///
# python scripts/build_n1_income_delivery.py --run-id n1_runtime_20260908_v1
"""Build source-selected cases and arithmetic checks from an existing N1 run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import fmean, median
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v5_r6_fixed_empirical_cohort import DEFAULT_NFPS, _read_nfps
from scripts.run_n1_action_income_cases import OUTPUT_ROOT, digest, write_csv

Row = dict[str, str]
COMPONENTS: Final = (
    ("sales_yuan", "sales_yuan", 1), ("support_yuan", "support_paid_yuan", 1),
    ("rent_out_yuan", "rent_received_yuan", 1), ("off_farm_yuan", "off_farm_income_yuan", 1),
    ("common_income_yuan", "common_income_yuan", 1),
    ("production_cost_yuan", "production_cost_yuan", -1),
    ("service_cost_yuan", "service_fee_yuan", -1), ("rent_in_yuan", "rent_paid_yuan", -1),
    ("management_cost_yuan", "management_cost_yuan", -1),
    ("conversion_cost_yuan", "conversion_cost_yuan", -1),
)
RESOURCES: Final = (
    ("area_mu", "operated_area_mu"), ("family_labor_days", "family_labor_used_days"),
    ("hired_labor_days", "hired_labor_days"), ("hired_labor_cost_yuan", "hired_labor_cost_yuan"),
    ("service_area_mu", "service_area_mu"), ("grain_output_kg", "grain_output_kg"),
    ("cash_output_kg", "cash_output_kg"), ("net_income", "net_income_yuan"),
)


def read_rows(path: Path) -> list[Row]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def select_initial_cases(initial: list[Row]) -> list[Row]:
    households = sorted((r for r in initial if r["operator_type"] == "household"), key=lambda r: r["operator_id"])
    selected = []
    for role in ("no_transfer", "rent_in", "both"):
        row = next(r for r in households if r["source_land_role"] == role)
        selected.append({"case_role": f"source_{role}", "selection_rule": f"lowest_id_initial_source_land_role={role}", **row})
    for field, role in (("family_labor_days", "low_labor_per_mu"), ("working_capital_yuan", "low_cash_per_mu"), ("off_farm_day_cap", "restricted_nonfarm_opportunity")):
        candidates = [r for r in households if float(r["operated_area_mu"]) > 0 and r["operator_id"] not in {s["operator_id"] for s in selected}]
        score = lambda r: float(r[field]) * float(r["off_farm_wage_level"]) if field == "off_farm_day_cap" else float(r[field]) / float(r["operated_area_mu"])
        row = min(candidates, key=lambda r: (score(r), r["operator_id"]))
        selected.append({"case_role": role, "selection_rule": f"minimum_initial_{field}_opportunity_or_per_mu_then_id", **row})
    return selected


def arithmetic_check(row: Row, realized: bool) -> Decimal:
    value = sum((Decimal(row[real if realized else expected]) * sign for expected, real, sign in COMPONENTS), Decimal(0))
    if realized:
        value += Decimal(row["labor_sale_income_yuan"])
    return value - Decimal(row["net_income_yuan" if realized else "net_income"])


def build(run_id: str, baseline_id: str) -> None:
    run, baseline = OUTPUT_ROOT / run_id, OUTPUT_ROOT / baseline_id
    out = run / "income_delivery_v3"
    out.mkdir(exist_ok=False)
    initial, realized, expected, traces = (read_rows(run / f"{name}.csv") for name in ("initial_actors", "realized_actors", "action_accounts", "decision_trace"))
    cases = select_initial_cases(initial)
    selected_ids = {r["operator_id"]: r for r in cases}
    raw = _read_nfps(DEFAULT_NFPS, {int(r["source_row"]) for r in cases})
    write_csv(out / "initial_case_selection.csv", cases)
    write_csv(out / "nfps_source_rows.csv", [{"operator_id": r["operator_id"], "source_row": r["source_row"], "source_path": str(DEFAULT_NFPS.relative_to(ROOT)), **raw[int(r["source_row"]) ]} for r in cases])
    write_csv(out / "case_action_accounts.csv", [{"case_role": selected_ids[r["operator_id"]]["case_role"], **r} for r in expected if r["operator_id"] in selected_ids])
    write_csv(out / "case_realized_accounts.csv", [{"case_role": selected_ids[r["operator_id"]]["case_role"], **r} for r in realized if r["operator_id"] in selected_ids])
    realized_by_key = {(r["year"], r["operator_id"]): r for r in realized}
    assert len(realized_by_key) == len(realized)
    trace_by_key = {(r["year"], r["operator_id"]): r for r in traces}
    assert len(trace_by_key) == len(traces)
    assert set(trace_by_key) == set(realized_by_key)
    aligned, differences, arithmetic = [], [], []
    for row in expected:
        arithmetic.append({"account": "expected", "year": row["year"], "operator_id": row["operator_id"], "action": row["action"], "residual_yuan": str(arithmetic_check(row, False))})
        if row["selected"] != "True":
            continue
        key = row["year"], row["operator_id"]
        actual, trace = realized_by_key[key], trace_by_key[key]
        chosen = row["action"] == actual["action"] == trace["action"]
        aligned.append({"year": row["year"], "operator_id": row["operator_id"], "selected_action": row["action"], "realized_action": actual["action"], "same_year_same_action": chosen, "chosen_feasible": row["feasible"], "feasible_rank": row["feasible_rank"], "inertia_yuan": trace["inertia_yuan"], "previous_action": trace["previous_action"]})
        if row["operator_id"] in selected_ids:
            for exp, real in (*[(a, b) for a, b, _ in COMPONENTS], *RESOURCES):
                differences.append({"year": row["year"], "operator_id": row["operator_id"], "action": row["action"], "expected_field": exp, "realized_field": real, "expected_value": row[exp], "realized_value": actual[real], "realized_minus_expected": float(actual[real]) - float(row[exp]), "decision_working_capital_yuan": row["working_capital_available"], "decision_required_cash_yuan": row["working_capital_required"], "realized_cash_cost_yuan": sum(float(actual[b]) for _, b, sign in COMPONENTS if sign < 0), "labor_fulfillment_ratio": actual["labor_fulfillment_ratio"]})
            differences.append({"year": row["year"], "operator_id": row["operator_id"], "action": row["action"], "expected_field": "farm_labor_sale_not_in_forecast", "realized_field": "labor_sale_income_yuan", "expected_value": 0, "realized_value": actual["labor_sale_income_yuan"], "realized_minus_expected": actual["labor_sale_income_yuan"]})
    arithmetic.extend({"account": "realized", "year": r["year"], "operator_id": r["operator_id"], "action": r["action"], "residual_yuan": str(arithmetic_check(r, True))} for r in realized)
    write_csv(out / "annual_action_alignment.csv", aligned)
    write_csv(out / "income_arithmetic_check.csv", arithmetic)
    write_csv(out / "case_resource_differences.csv", differences)
    with (run / "land_bid_accounts.csv").open(encoding="utf-8-sig", newline="") as handle:
        lease_checks = [r for r in csv.DictReader(handle) if r.get("record_type") == "lease_execution_gate"]
    write_csv(out / "lease_execution_gate.csv", lease_checks)
    accepted_leases = [r for r in lease_checks if r["lease_gate_passed"] == "True"]
    assert all(float(r["after_working_capital_required"]) <= float(r["after_working_capital_available"]) + 1e-8 for r in accepted_leases)
    assert all(float(r["after_hired_labor_days"]) <= float(r["after_hired_labor_available_days"]) + 1e-8 for r in accepted_leases)
    previous = {(r["year"], r["operator_id"]): r for r in read_rows(baseline / "realized_actors.csv")}
    comparison = []
    for row in realized:
        key = row["year"], row["operator_id"]
        old = previous.get(key)
        if old is None:
            continue
        comparison.append({"year": row["year"], "operator_id": row["operator_id"], "operator_type": row["operator_type"], "old_action": old["action"], "new_action": row["action"], "action_changed": old["action"] != row["action"], "old_income_yuan": old["net_income_yuan"], "new_income_yuan": row["net_income_yuan"], "old_area_mu": old["operated_area_mu"], "new_area_mu": row["operated_area_mu"]})
    write_csv(out / "matched_actor_comparison.csv", comparison)
    model_data_comparison(out, (baseline, run))
    hand = read_rows(OUTPUT_ROOT / "n1_hand_20260908_v2/hand_action_accounts.csv")
    write_csv(out / "hand_feasibility_and_single_charge_tests.csv", hand)
    primary = cases[0]["operator_id"]
    first_year = min(r["year"] for r in expected)
    figure_rows = [r for r in expected if r["operator_id"] == primary and r["year"] == first_year]
    write_csv(out / "figure_expected_action_source.csv", figure_rows)
    write_csv(out / "figure_component_source.csv", [r for r in differences if r["operator_id"] == primary and r["year"] == first_year])
    draw_figure(out, figure_rows, realized_by_key[(first_year, primary)])
    maximum_residual = max(abs(Decimal(r["residual_yuan"])) for r in arithmetic)
    summary = {
        "run_id": run_id, "baseline_run_id": baseline_id, "cases": len(cases), "selected_ids": list(selected_ids),
        "case_selection_uses": "initial state only; low-resource roles are initialized proxies, not observed shortages",
        "tenure_boundary": "NFPS source rent-in/out metadata is not materialized as household lease-in tenure at initialization",
        "source_weight_boundary": "600 equal runtime weights of one after source-weighted cohort selection",
        "expected_rows": len(expected), "realized_rows": len(realized), "annual_alignment_rows": len(aligned),
        "max_absolute_arithmetic_residual_yuan": str(maximum_residual),
        "all_selected_feasible": all(r["chosen_feasible"] == "True" for r in aligned),
        "same_year_same_action": all(r["same_year_same_action"] for r in aligned),
        "matched_actor_years": len(comparison), "changed_action_actor_years": sum(r["action_changed"] for r in comparison),
        "first_year_actions": dict(Counter(r["action"] for r in realized if r["year"] == first_year)),
        "lease_execution_gate_rows": len(lease_checks), "lease_gate_accepted_rows": len(accepted_leases),
        "lease_gate_rejected_rows": len(lease_checks) - len(accepted_leases),
        "lease_gate_status": "accepted_rows_pass_full_cash_and_hired_labor_constraints" if lease_checks else "no_execution_gate_evidence_in_this_version",
        "hand_rows": len(hand), "source_sha256": digest(DEFAULT_NFPS),
        "realized_income_formula": "expected component formula plus realized farm-labor-sale income; no second deduction of hired labor costs",
        "run_summary_path": str((run / "run_summary.json").relative_to(ROOT)),
        "artifact_hashes": {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    }
    (out / "delivery_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    assert maximum_residual < Decimal("1e-7")
    assert summary["same_year_same_action"] and summary["all_selected_feasible"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def model_data_comparison(out: Path, runs: tuple[Path, Path]) -> None:
    source = ROOT / "results/empirical/three_paper_execution_20260908/nfps/run_20260908_d1_nfps_r2"
    moments = {r["period"]: r for r in read_rows(source / "early_empirical_moments_for_b19_p3.csv")}
    facts = {r["year"]: r for r in read_rows(source / "early_land_income_labor_facts.csv")}
    rows = []
    for run in runs:
        realized = read_rows(run / "realized_actors.csv")
        for year in sorted({r["year"] for r in realized}):
            households = [r for r in realized if r["year"] == year and r["operator_type"] == "household"]
            areas = [float(r["operated_area_mu"]) for r in households]
            statistics = {
                "mean_operated_area_mu": (fmean(areas), moments[year]["mean_operated_area_mu"], facts[year]["operated_area_valid_n"]),
                "median_operated_area_mu": (median(areas), moments[year]["median_operated_area_mu"], facts[year]["operated_area_valid_n"]),
                "rent_in_household_share": (fmean(float(r["rented_in_area_mu"]) > 1e-8 for r in households), moments[year]["rent_in_household_share"], facts[year]["raw_rows"]),
                "household_net_income_mean_yuan": (fmean(float(r["net_income_yuan"]) for r in households), facts[year]["household_net_income_mean_yuan"], facts[year]["household_net_income_valid_n"]),
            }
            if "rented_out_area_mu" in households[0]:
                statistics["rent_out_household_share"] = (fmean(float(r["rented_out_area_mu"]) > 1e-8 for r in households), moments[year]["rent_out_household_share"], facts[year]["raw_rows"])
            for metric, (model_value, empirical, n) in statistics.items():
                rows.append({"run_id": run.name, "year": year, "metric": metric, "model_value": model_value, "empirical_value": empirical, "signed_error": model_value - float(empirical), "absolute_error": abs(model_value - float(empirical)), "runtime_household_n": len(households), "empirical_denominator_n": n, "model_source_path": str((run / "realized_actors.csv").relative_to(ROOT)), "empirical_source_path": str((source / ("early_land_income_labor_facts.csv" if "income" in metric else "early_empirical_moments_for_b19_p3.csv")).relative_to(ROOT)), "comparison_status": "descriptive_same_year_different_samples_and_account_boundaries_not_validation_fit"})
    write_csv(out / "model_data_moment_comparison.csv", rows)


def draw_figure(out: Path, options: list[Row], actual: Row) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = {"maintain_grain": "Grain", "expand_cash": "Cash crop", "exit": "Exit farming"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    bars = axes[0].bar([labels[r["action"]] for r in options], [float(r["net_income"]) / 1000 for r in options], color=["#4f9d69" if r["selected"] == "True" else "#2f6f9f" for r in options])
    for bar, row in zip(bars, options):
        if row["feasible"] != "True":
            bar.set_hatch("///")
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{float(row['net_income']) / 1000:.2f}\n{'selected' if row['selected'] == 'True' else 'infeasible' if row['feasible'] != 'True' else 'feasible'}", ha="center", va="bottom", fontsize=9)
    axes[0].set(ylabel="Expected household net income (1,000 yuan)", title="A  Complete action accounts")
    axes[0].set_ylim(top=max(float(r["net_income"]) for r in options) / 1000 * 1.25)
    selected = next(r for r in options if r["selected"] == "True")
    names = ("Sales", "Support", "Rent received", "Nonfarm", "Common", "Production", "Service", "Rent paid", "Management", "Conversion")
    used = [(name, a, b, sign) for name, (a, b, sign) in zip(names, COMPONENTS) if float(selected[a]) != 0 or float(actual[b]) != 0]
    if float(actual["labor_sale_income_yuan"]) != 0:
        selected = {**selected, "farm_labor_sale_not_forecast": "0"}
        used.append(("Farm wage", "farm_labor_sale_not_forecast", "labor_sale_income_yuan", 1))
    x = np.arange(len(used))
    axes[1].bar(x - 0.18, [float(selected[a]) * sign / 1000 for _, a, _, sign in used], 0.36, label="Expected", color="#2f6f9f")
    axes[1].bar(x + 0.18, [float(actual[b]) * sign / 1000 for _, _, b, sign in used], 0.36, label="Realized", color="#4f9d69")
    axes[1].set(xticks=x, xticklabels=[name for name, _, _, _ in used], ylabel="Signed income component (1,000 yuan)", title=f"B  Selected action: {labels[selected['action']]}")
    axes[1].legend(frameon=False)
    fig.text(0.52, -0.03, "Farm wages are settled after matching, not forecast in the action account.", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", color="#d7e0ea", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(f"NFPS-anchored actor {actual['operator_id']} | {actual['year']} | two-year engineering run", fontsize=12)
    fig.savefig(out / "nfps_case_action_and_realization.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline", default="n1_legacy_20260908")
    args = parser.parse_args()
    build(args.run_id, args.baseline)


if __name__ == "__main__":
    main()
