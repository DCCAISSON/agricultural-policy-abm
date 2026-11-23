#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_builder import ModelBuilder


def runtime_config(overlay: Path) -> dict[str, Any]:  # noqa: DICT_OK
    return {
        "experiment": {"seed": 4803, "steps": 3, "base_year": 2009, "overlay_path": str(overlay)},
        "scale": {
            "n_farmers": 10,
            "n_enterprises": 1,
            "grid_width": 20,
            "grid_height": 20,
            "width": 20,
            "height": 20,
        },
        "initial_conditions": {"data_source": "config"},
        "calibration": {"apply_calibrated_params": False},
        "environment": {
            "market": {"base_prices": {"grain": 2.5, "cash_crop": 3.0}},
            "land": {
                "patches_per_farmer_dist": {"type": "randint", "min": 1, "max": 2},
                "area_dist": {"type": "uniform", "min": 0.8, "max": 1.2},
                "soil_quality_dist": {"type": "uniform", "min": 0.9, "max": 1.1},
            },
        },
        "policy": {"active_scenario": "baseline_cash", "policy_mode": "rule", "rl_enabled": False},
        "mechanism_switches": {
            "rent_market_enabled": True,
            "crop_choice_enabled": True,
            "acquaintance_network_enabled": True,
            "use_learning_farmer": True,
            "use_learning_enterprise": True,
            "use_reputation_village": True,
            "use_extended_government_rl": True,
        },
        "agents": {
            "farmer": {
                "initial_land_mean": 1.0,
                "initial_capital_dist": {"type": "lognormal", "mean": 9.2, "std": 0.1},
                "farming_ability_dist": {"type": "uniform", "min": 0.9, "max": 1.1},
            },
            "enterprise": {
                "type_distribution": {"COOPERATIVE": 1.0},
                "types": {
                    "COOPERATIVE": {
                        "initial_capital_dist": {"type": "lognormal", "mean": 13.8, "std": 0.1}
                    }
                },
                "population_dynamics": {"initial_operated_land_share": 0.0},
            },
            "village_collective": {"enabled": False},
        },
        "land": {"min_patch_area": 1.0, "max_patch_area": 1.0, "patches_per_farmer_mean": 1.0},
        "production": {
            "grain": {"yield_per_mu": 500},
            "cash_crop": {"yield_per_mu": 800},
            "base_cost_per_mu": 300.0,
            "agricultural_service": {"enabled": False},
        },
        "labor_market": {"enabled": False},
        "population_dynamics": {"enabled": False},
        "enterprise_demography": {"enabled": False},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = ModelBuilder(config=runtime_config(args.overlay)).build()
    annual_observables = []
    for _year in range(3):
        before_farmer_ids = [str(farmer.unique_id) for farmer in model.farmers]
        before_enterprise_ids = [str(enterprise.unique_id) for enterprise in model.agricultural_enterprises]
        model.step()
        current_year = 2009 + int(model.step_count)
        year_trace = [
            row
            for row in model.v4_8_fixed_rule_trace
            if int(row["year"]) == current_year
        ]
        annual_observables.append(
            {
                "year": current_year,
                "active_farmers_before": before_farmer_ids,
                "active_enterprises_before": before_enterprise_ids,
                "trace_count": len(year_trace),
                "targeted_operator_ids": [row["operator_id"] for row in year_trace],
                "active_farmers_after": [str(farmer.unique_id) for farmer in model.farmers],
                "enterprise_fixed_states": [
                    {
                        "operator_id": str(enterprise.unique_id),
                        "fixed_rule_current_action": getattr(enterprise, "fixed_rule_current_action", None),
                        "fixed_rule_scale_decision": getattr(enterprise, "fixed_rule_scale_decision", None),
                        "target_expansion_area": float(getattr(enterprise, "target_expansion_area", 0.0) or 0.0),
                    }
                    for enterprise in model.agricultural_enterprises
                ],
            }
        )
    payload = {
        "runtime_surface": "ModelBuilder.build + MainModel.step",
        "fixed_rule_trace": model.v4_8_fixed_rule_trace,
        "market_stage_trace": model.v4_8_market_stage_trace,
        "annual_observables": annual_observables,
        "runtime_fixed_rule": model.config["runtime_fixed_rule"],
        "mechanism_switches": model.config["mechanism_switches"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
