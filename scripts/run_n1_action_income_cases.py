# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml", "numpy", "mesa"]
# ///
# python scripts/run_n1_action_income_cases.py --run-id n1_legacy_20260908
"""Observe two real flagship years without replacing any economic calculation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Final
from unittest.mock import patch

import yaml

ROOT: Final = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_builder import ModelBuilder
from scripts.paper_b_flagship_runtime import DEFAULT_COHORT, DEFAULT_OVERLAY, run_worldline
from scripts.paper_b_flagship_scenarios import flagship_scenarios
from scripts.u2_unified_paper_model_runtime import DEFAULT_R6_MANIFEST, selected_parameters

OUTPUT_ROOT: Final = ROOT / "results/empirical/three_paper_execution_20260908/national"
SOURCES: Final = (
    "core/fixed_rule_runtime.py", "agents/farmer_agent.py", "agents/enterprise_agent.py",
    "core/model.py", "core/model_builder.py", "scripts/paper_b_flagship_runtime.py",
    "scripts/run_n1_action_income_cases.py", "agents/village_collective.py", "modules/land_patch.py",
    "scripts/n1_income_fixtures.py", "scripts/build_n1_income_delivery.py", "scripts/run_n1_hand_accounts.py",
    "tests/test_n1_action_income.py", "tests/test_n1_lease_execution.py",
)
INPUTS: Final = (
    "config/master_config.yaml", "config/experiments/v4_8_national_fixed_rule.yaml",
    "data/derived_runtime/v5_r6/nfps_fixed_empirical_cohort_2010_k600.json",
    "data/derived_runtime/v4_8/cfps_entrant_feature_library_2010_2016.json",
    "config/ssp_scenarios.yaml",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def actor_row(actor, year: int) -> dict:
    source = getattr(actor, "_microdata_source", {})
    kind = "household" if hasattr(actor, "cash_net_income") else "scale"
    return {
        "year": year, "operator_id": str(actor.unique_id), "operator_type": kind,
        **{f"source_{key}": value for key, value in source.items()},
        "cohort_agent_id": source.get("cohort_agent_id", ""),
        "source_observation_id": source.get("source_observation_id", ""),
        "source_row": source.get("source_row", ""),
        "source_land_role": source.get("land_role", ""),
        "source_agri_income": source.get("agri_income"),
        "source_net_income": source.get("net_income"),
        "operated_area_mu": actor.get_operated_land_area(),
        "owned_area_mu": sum(p.area for p in actor.model.land_patches if p.owner is actor),
        "rented_in_area_mu": sum(p.area for p in actor.model.land_patches if p.leased_to is actor and p.owner is not actor),
        "rented_out_area_mu": sum(p.area for p in actor.model.land_patches if p.owner is actor and p.leased_to not in (None, actor)),
        "self_leased_area_mu": sum(p.area for p in actor.model.land_patches if p.leased_to is actor and p.owner is actor),
        "capital_yuan": actor.capital,
        "working_capital_yuan": actor.balance_sheet_snapshot()["working_capital"],
        "family_labor_days": getattr(actor, "total_labor_units", 0.0),
        "off_farm_wage_level": getattr(actor, "off_farm_wage_level", None),
        "off_farm_day_cap": actor._get_max_off_farm_days(actor.total_labor_units) if kind == "household" else 0.0,
        "livelihood": str(getattr(getattr(actor, "livelihood_strategy", None), "value", "")),
        "action": getattr(actor, "fixed_rule_current_action", ""),
        "crop": str(getattr(getattr(actor, "planned_crop", None), "value", "")),
        "grain_area_mu": getattr(actor, "grain_area", 0.0),
        "grain_output_kg": getattr(actor, "grain_production", 0.0),
        "cash_output_kg": getattr(actor, "cash_crop_production", 0.0),
        "sales_yuan": getattr(actor, "agricultural_income", getattr(actor, "agricultural_revenue", 0.0)),
        "planting_net_yuan": getattr(actor, "planting_net_income", None),
        "net_income_yuan": getattr(actor, "cash_net_income", getattr(actor, "profit", 0.0)),
        "support_paid_yuan": getattr(actor, "subsidy_income", getattr(actor, "subsidy_revenue", 0.0)),
        "rent_received_yuan": getattr(actor, "land_rental_income", 0.0),
        "rent_paid_yuan": getattr(actor, "land_rental_expense", getattr(actor, "land_rental_cost", 0.0)),
        "production_cost_yuan": getattr(actor, "cash_production_cost", getattr(actor, "production_cost", 0.0)),
        "management_cost_yuan": getattr(actor, "management_cost", 0.0),
        "service_fee_yuan": getattr(actor, "service_fee_expense" if kind == "household" else "service_expense", 0.0),
        "common_income_yuan": getattr(actor, "fixed_rule_common_income_yuan", 0.0),
        "conversion_cost_yuan": getattr(actor, "fixed_rule_conversion_expense", 0.0),
        "service_area_mu": sum(c.area for c in getattr(actor, "service_contracts", [])),
        "family_labor_used_days": getattr(actor, "family_labor_used", 0.0),
        "hired_labor_days": getattr(actor, "hired_labor_days", 0.0),
        "hired_labor_cost_yuan": getattr(actor, "hired_labor_cost", 0.0),
        "labor_sold_days": getattr(actor, "labor_sold_days", 0.0),
        "labor_sale_income_yuan": getattr(actor, "labor_sale_income", 0.0),
        "off_farm_income_yuan": getattr(actor, "off_farm_income", 0.0),
        "realized_off_farm_days": getattr(actor, "realized_off_farm_labor_days", 0.0),
        "labor_fulfillment_ratio": getattr(actor, "fixed_rule_labor_fulfillment", None),
        "labor_need_days": getattr(actor, "fixed_rule_labor_need", None),
        "desired_area_mu": getattr(actor, "fixed_rule_desired_area", None),
        "exit_pending": getattr(actor, "fixed_rule_exit_pending", False),
        "representation_weight_q": getattr(actor, "representation_weight_q", 1.0),
    }


def run_audit(run_id: str, overlay: Path) -> dict:
    out = OUTPUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=False)
    paths = [ROOT / name for name in (*SOURCES, *INPUTS)] + [DEFAULT_R6_MANIFEST, overlay]
    before = {str(p.relative_to(ROOT)): digest(p) for p in dict.fromkeys(paths)}
    for name in SOURCES:
        target = out / "code_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    initial: list[dict] = []
    realized: list[dict] = []
    accounts: list[dict] = []
    traces: list[dict] = []
    bids: list[dict] = []
    stages: list[dict] = []
    effective_rule: dict = {}
    models = []
    seen_actors = {}
    build = ModelBuilder.build

    def observed_build(builder):
        model = build(builder)
        models.append(model)
        effective_rule.update(model.config.get("runtime_fixed_rule", {}))
        actors = [*model.farmers, *model.agricultural_enterprises]
        initial.extend(actor_row(a, model.simulation_base_year) for a in actors)
        prior = getattr(model, "_completed_production_year_snapshot_hook", None)

        def capture(current):
            if callable(prior):
                prior(current)
            year = current.simulation_base_year + current.step_count
            for actor in [*current.farmers, *current.agricultural_enterprises]:
                seen_actors[str(actor.unique_id)] = actor
                realized.append(actor_row(actor, year))
                options = getattr(actor, "fixed_rule_action_accounts", ())
                ranked = sorted((a for a in options if a.feasible), key=lambda a: a.score, reverse=True)
                for account in options:
                    accounts.append({
                        "year": year, "operator_id": str(actor.unique_id),
                        "operator_type": "household" if hasattr(actor, "cash_net_income") else "scale",
                        **asdict(account), "net_income": account.net_income, "score": account.score,
                        "working_capital_required": account.working_capital_required,
                        "feasible": account.feasible, "constraint_reason": account.constraint_reason,
                        "feasible_rank": ranked.index(account) + 1 if account.feasible else None,
                        "selected": account.action == actor.fixed_rule_current_action,
                    })
                bids.extend(getattr(actor, "fixed_rule_bid_accounts", ()))
            traces[:] = current.v4_8_fixed_rule_trace
            stages[:] = current.v4_8_market_stage_trace

        model._completed_production_year_snapshot_hook = capture
        return model

    scenario = next(s for s in flagship_scenarios() if s.scenario_id == "ssp2_s350_c1")
    parameters = selected_parameters()
    if overlay != DEFAULT_OVERLAY:
        with overlay.open(encoding="utf-8") as handle:
            rule = yaml.safe_load(handle)["decision_rule"]
        parameters["fixed_rule_inertia_margin"] = float(rule["minimum_inertia_margin_yuan_per_mu"])
    start = time.perf_counter()
    with patch.object(ModelBuilder, "build", observed_build):
        result = run_worldline(scenario, 17, 2, parameters, overlay_path=overlay)
    write_csv(out / "initial_actors.csv", initial)
    write_csv(out / "realized_actors.csv", realized)
    write_csv(out / "action_accounts.csv", accounts)
    write_csv(out / "decision_trace.csv", traces)
    write_csv(out / "land_bid_accounts.csv", bids)
    write_csv(out / "market_stages.csv", stages)
    write_csv(out / "annual_observations.csv", [dict(row) for row in result.annual_rows])
    model = models[0]
    remaining_ids = {str(a.unique_id) for a in [*model.farmers, *model.agricultural_enterprises]}
    write_csv(out / "population_events.csv", [
        {"operator_id": identity, "last_production_year": max(r["year"] for r in realized if r["operator_id"] == identity),
         "runtime_status": str(getattr(actor, "status", "")), "last_direct_farming_action": getattr(actor, "fixed_rule_current_action", ""),
         "removed_after_annual_collection": True}
        for identity, actor in seen_actors.items() if identity not in remaining_ids
    ])
    (out / "population_dynamics.json").write_text(json.dumps(model.population_dynamics_history, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    after = {str(p.relative_to(ROOT)): digest(p) for p in dict.fromkeys(paths)}
    summary = {
        "run_id": run_id, "scenario_id": scenario.scenario_id,
        "scenario": "SSP2; support reference 350 yuan/mu; coordination scale 1.0",
        "runtime": "paper_b_flagship_runtime.run_worldline -> u3_runtime_config -> ModelBuilder.build -> MainModel.step",
        "seed": 17, "years": 2, "elapsed_seconds": time.perf_counter() - start,
        "initial_households": sum(r["operator_type"] == "household" for r in initial),
        "initial_enterprises": sum(r["operator_type"] == "scale" for r in initial),
        "realized_actor_years": len(realized), "action_account_rows": len(accounts),
        "decision_rows": len(traces), "land_bid_rows": len(bids),
        "first_year_actions": dict(Counter(r["action"] for r in traces[:len(initial)])),
        "effective_parameters": parameters, "effective_decision_rule": effective_rule,
        "inertia_status": "new_engineering_setting_not_reestimated" if overlay != DEFAULT_OVERLAY else "legacy_R6_fit_for_old_rule_only",
        "source_hashes_before": before, "source_hashes_after": after,
        "read_only_inputs_unchanged": before == after,
        "claim_boundary": "Two-year mechanism debugging only; not policy effect or external validation.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if before != after:
        raise RuntimeError("An audited source changed while the worldline was running")
    return {key: value for key, value in summary.items() if not key.startswith("source_hashes")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.run_id, args.overlay.resolve()), indent=2))


if __name__ == "__main__":
    main()
