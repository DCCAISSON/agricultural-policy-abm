"""Reconcile plot-use bounds and summarize paired annual policy branches."""
import argparse
import math
from pathlib import Path

from scripts.run_national_succession_transmission import OLD, TARGET, old, rows_to_csv


def change(a, b):
    assert a.keys() == b.keys()
    return {k: b[k] - a[k] for k in a}


def snapshot(directory):
    data = old.read(directory / "actual_plot_use.json")
    assert not data["reconciliation"]["unreconciled_operators"]
    p = {p["patch_id"]: p for p in data["patches"]}
    assert len(p) == len(data["patches"])
    return p, data["reconciliation"]["harvest"]


def common_land(a, b):
    assert a.keys() == b.keys()
    assert all(a[k]["area_mu"] == b[k]["area_mu"] for k in a)
    return {k for k in a if a[k]["operator_id"] == b[k]["operator_id"]
            and a[k]["cultivated_area_mu"] > 1e-8
            and math.isclose(a[k]["cultivated_area_mu"], b[k]["cultivated_area_mu"],
                             rel_tol=0, abs_tol=1e-8)}


def bounds(plots, inside):
    # Reconstructed operator totals are exact after harvest reconciliation.
    # Never allocate the unidentified crop mix to individual plots.
    by_operator = {}
    for k, p in plots.items():
        row = by_operator.setdefault(p["operator_id"], [0.0] * 5)
        i = 0 if k in inside else 2
        row[i] += p["grain_area_mu"] or 0
        row[i + 1] += p["unknown_crop_area_mu"]
        row[4] += p["allocated_grain_area_mu"]
    low, high = [], []
    for g, u, og, ou, total in by_operator.values():
        assert g + og - 1e-6 <= total <= g + u + og + ou + 1e-6
        low.append(max(g, total - og - ou))
        high.append(min(g + u, total - og))
    return math.fsum(low), math.fsum(high)


def decomposition(base, treat):
    a, av = snapshot(base)
    b, bv = snapshot(treat)
    common = common_land(a, b)
    al, ah = bounds(a, common)
    bl, bh = bounds(b, common)
    low, high = bl - ah, bh - al
    A, B = av["grain_area_mu"], bv["grain_area_mu"]
    Q, R = av["grain_output_kg"], bv["grain_output_kg"]
    delta = B - A
    uy, vy = Q / A, R / B
    rate = (uy + vy) / 2
    yield_component = (A + B) / 2 * (vy - uy)
    assert math.isclose(delta * rate + yield_component, R - Q, abs_tol=1e-6)
    return {"grain_area_change_mu": delta, "output_change_kg": R - Q,
            "yield_change_kg_per_mu": vy - uy, "total_area_component_kg": delta * rate,
            "common_operator_use_lower_mu": low, "common_operator_use_upper_mu": high,
            "changed_operator_or_cultivation_lower_mu": delta - high,
            "changed_operator_or_cultivation_upper_mu": delta - low,
            "common_operator_use_lower_kg_at_mean_yield": low * rate,
            "common_operator_use_upper_kg_at_mean_yield": high * rate,
            "changed_operator_or_cultivation_lower_kg_at_mean_yield": (delta - high) * rate,
            "changed_operator_or_cultivation_upper_kg_at_mean_yield": (delta - low) * rate,
            "aggregate_yield_component_kg": yield_component}


def read_year(path, year, file):
    return old.read(path / f"year_{year:02d}" / (file + ".json"))


def performance(path, year):
    row = read_year(path, year, "government_sar")
    outcome = row["outcome_observations"]
    snap = read_year(path, year, "operator_snapshot")
    payments = read_year(path, year, "fiscal_account")["payments_made"]
    area = snap["grain_area_mu"]
    return {"grain_kg": outcome["grain_output_kg"], "grain_area_mu": area,
            "grain_yield_kg_per_mu": outcome["grain_output_kg"] / area,
            "h0_cash_yuan": outcome["h0_mean_net_income_yuan"],
            "organization_grain_kg": snap["professional_grain_output_kg"],
            "trial_paid_yuan": math.fsum(p["paid_yuan"] for p in payments if p["layer"] == "trial"),
            "fiscal_cost_yuan": outcome["fiscal_cost_yuan"],
            "model_reward": row["reward"]}


def income_totals(path, year):
    rows = [r for r in old.read(path / "household_income.json") if r["step"] == year]
    assert len(rows) == 600 and all("net_income_at_harvest_yuan" in r for r in rows)
    groups = {"crop_sales": (1, ("sales_yuan",)),
              "off_farm_and_hired_out_wages": (1, ("off_farm_income_yuan", "labor_sale_income_yuan")),
              "rent_net": (1, ("rent_received_yuan",)),
              "support": (1, ("support_at_harvest_yuan", "verified_support_received_yuan")),
              "production_service_conversion": (-1, ("production_cost_yuan", "service_cost_yuan", "conversion_cost_yuan")),
              "common_income": (1, ("common_income_yuan",))}
    out = {group: sign * math.fsum(r.get(k, 0) for r in rows for k in keys) / len(rows)
           for group, (sign, keys) in groups.items()}
    out["rent_net"] -= math.fsum(r.get("rent_paid_yuan", 0) for r in rows) / len(rows)
    net = math.fsum(r["cash_net"] for r in rows) / len(rows)
    out["unexplained_account_residual"] = net - math.fsum(out.values())
    assert abs(out["unexplained_account_residual"]) < 1e-6
    out["cash_net"] = net
    assert math.isclose(net, performance(path, year)["h0_cash_yuan"], abs_tol=1e-6)
    return out


def audit(output, experiment=None):
    output.mkdir(parents=True)
    results = []
    for seed in (31, 47):
        for policy in ("area", "scale80"):
            for year in range(1, 9):
                d = f"year_{year:02d}"
                results.append(dict(seed=seed, policy=policy, year=year, **decomposition(
                    OLD / f"eval_{seed}_base" / d, OLD / f"eval_{seed}_{policy}" / d)))
    rows_to_csv(output / "system_decomposition.csv", results)
    if experiment is None:
        return
    runs = old.read(experiment / "COMPLETED.json")
    assert len(runs) == 15
    years_checked = payment_rows = 0
    for run in runs:
        path = experiment / run["name"]
        for year in range(1, run["years"] + 1):
            fiscal = read_year(path, year, "fiscal_account")
            state = read_year(path, year, "government_sar")
            p = fiscal["payments_made"]
            assert all(r["paid_yuan"] == r["verified_payable_yuan"] for r in p)
            total = math.fsum(r["paid_yuan"] for r in p) + fiscal["implementation_cost_yuan"]
            assert math.isclose(total, state["outcome_observations"]["fiscal_cost_yuan"], abs_tol=1e-6)
            assert total <= state["outcome_observations"]["fiscal_envelope_yuan"] + 1e-6
            assert not state["q_updated"]
            years_checked += 1
            payment_rows += len(p)
    actions, horizons = [], []
    for seed in (31, 47):
        base = experiment / f"branch_{seed}_0"
        prior = performance(base, 1)
        for action in range(3):
            path = experiment / f"branch_{seed}_{action}"
            future = []
            for year in range(2, 5):
                p = performance(path, year)
                future.append(p)
                actions.append(dict(seed=seed, year=year, action_in_year2=action,
                                    prior_grain_kg=prior["grain_kg"], prior_grain_area_mu=prior["grain_area_mu"],
                                    prior_yield_kg_per_mu=prior["grain_yield_kg_per_mu"], **p,
                                    **{f"delta_{k}": v for k, v in change(performance(base, year), p).items()}))
            horizons.append(dict(seed=seed, action_in_year2=action,
                grain_kg=math.fsum(p["grain_kg"] for p in future),
                h0_mean_cash_yuan=math.fsum(p["h0_cash_yuan"] for p in future) / 3,
                trial_paid_yuan=math.fsum(p["trial_paid_yuan"] for p in future),
                fiscal_cost_yuan=math.fsum(p["fiscal_cost_yuan"] for p in future),
                discounted_reward=math.fsum(.95**i * p["model_reward"] for i, p in enumerate(future))))
    rows_to_csv(output / "decision_branches.csv", actions)
    rows_to_csv(output / "decision_horizons.csv", horizons)
    mechanisms = []
    for run in runs:
        name = run["name"]
        if run["seed"] != 47:
            continue
        path = experiment / name
        target = read_year(path, 2, "target")
        decision, = target["decision"]
        p = performance(path, 2)
        account = target["cash_account"] or {}
        mechanisms.append(dict(path=name, target=TARGET, previous_action=decision["previous_action"],
            action=decision["action"], inertia_yuan=decision["inertia_yuan"],
            year1_payment_yuan=target["year1_trial_paid"],
            year2_held_aside_yuan=target["cash_offsets"].get(TARGET, {}).get("held_aside", 0.0),
            grain_expected_support_yuan=decision["maintain_grain_support_yuan"],
            other_expected_support_yuan=decision["expand_cash_support_yuan"],
            grain_return_yuan=decision["grain_return"], other_return_yuan=decision["cash_return"],
            planning_working_cash_yuan=decision["maintain_grain_working_capital_available"],
            grain_plan_area_mu=decision["maintain_grain_grain_area_mu"],
            other_plan_grain_area_mu=decision["expand_cash_grain_area_mu"],
            cash_available_before_production_yuan=account.get("opening_available_yuan"),
            production_cash_committed_yuan=account.get("committed_yuan"),
            held_area_mu=math.fsum(p["area_mu"] for p in snapshot(path / "year_02")[0].values()
                                    if p["operator_id"] == TARGET),
            target_grain_area_mu=target["grain_area"], target_grain_kg=target["grain_kg"], **p))
    rows_to_csv(output / "first_divergence_and_rate_anchors.csv", mechanisms)
    paired = [
        ("branch_47_0", "branch_47_2"),
        ("mechanism_no_new_leases_base", "mechanism_no_new_leases_scale"),
        ("branch_47_2", "mechanism_no_cash"),
        ("branch_47_2", "mechanism_no_expectation"),
        ("branch_47_2", "mechanism_neither"),
    ]
    rows_to_csv(output / "mechanism_plot_differences.csv", [
        dict(base=a, compared=b, **decomposition(
            experiment / a / "year_02", experiment / b / "year_02")) for a, b in paired])
    base, scale = (experiment / f"original_47_{a}" for a in (0, 2))
    incomes = []
    for year in range(1, 9):
        for a, p in ((0, base), (2, scale)):
            archived = performance(OLD / f"eval_47_{'base' if a == 0 else 'scale80'}", year)
            actual = performance(p, year)
            assert all(math.isclose(actual[k], archived[k], abs_tol=1e-5) for k in actual)
        incomes.append(dict(year=year, **change(income_totals(base, year), income_totals(scale, year))))
    rows_to_csv(output / "household_income_differences.csv", incomes)
    old.write(output / "audit.json", {"old_system_pairs": len(results), "new_branch_years": len(actions),
              "mechanism_rows": len(mechanisms), "replicated_original_years": 16,
              "h0_income_years": len(incomes), "fiscal_years_checked": years_checked,
              "payment_rows_checked": payment_rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--experiment", type=Path)
    args = parser.parse_args()
    audit(args.output, args.experiment)
