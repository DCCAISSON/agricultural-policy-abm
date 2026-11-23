from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final

import yaml

from core.fixed_rule_runtime import apply_experiment_overlay, load_experiment_overlay
from core.model import MainModel
from core.model_builder import ModelBuilder
from core.reproducibility import RandomController
from scripts.run_v4_8_national_calibration_validation import (
    BASELINE_SCENARIO,
    _apply_runtime_parameters,
    _initial_runtime_observation,
    _runtime_observation,
    _stable_hash,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
MASTER_CONFIG: Final = PROJECT_ROOT / "config" / "master_config.yaml"
ENTRANT_LIBRARY: Final = (
    PROJECT_ROOT
    / "data"
    / "derived_runtime"
    / "v4_8"
    / "cfps_entrant_feature_library_2010_2016.json"
)


class R6RuntimeError(RuntimeError):
    pass


def _project_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def cohort_size(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    farmers = payload.get("farmers") if isinstance(payload, dict) else None
    if not isinstance(farmers, list) or not farmers:
        raise R6RuntimeError("fixed cohort has no farmer rows")
    return len(farmers)


def runtime_config(
    overlay: Path,
    cohort_path: Path,
    seed: int,
    years: int,
) -> dict[str, Any]:
    master = yaml.safe_load(MASTER_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(master, dict):
        raise R6RuntimeError("master_config.yaml must be a mapping")
    config = copy.deepcopy(master)
    apply_experiment_overlay(config, load_experiment_overlay(overlay))
    size = cohort_size(cohort_path)
    experiment = config.setdefault("experiment", {})
    experiment.update({"seed": seed, "steps": years})
    scale = config.setdefault("scale", {})
    scale["n_farmers"] = size
    scale["n_enterprises"] = max(
        1,
        int(round(size * float(scale.get("enterprise_ratio", 0.10) or 0.10))),
    )
    scale["agent_representation_weight"] = 1.0
    scale["grid_width"] = min(int(scale.get("grid_width", 100) or 100), 50)
    scale["grid_height"] = min(int(scale.get("grid_height", 100) or 100), 50)
    scale["width"] = scale["grid_width"]
    scale["height"] = scale["grid_height"]
    initial = config.setdefault("initial_conditions", {})
    initial.update(
        {
            "data_source": "nfps",
            "microdata_driven": True,
            "fixed_cohort_path": _project_relative(cohort_path),
        }
    )
    policy = config.setdefault("policy", {})
    policy["active_scenario"] = BASELINE_SCENARIO
    policy["policy_mode"] = "rule"
    policy["rl_enabled"] = False
    government = policy.setdefault("government", {})
    government["policy_mode"] = "rule"
    government["adaptive_adjustment_enabled"] = False
    population = config.setdefault("population_dynamics", {})
    population.setdefault("entry_mechanism", {})["entrant_feature_library"] = (
        _project_relative(ENTRANT_LIBRARY)
    )
    config.setdefault("calibration", {})["apply_calibrated_params"] = False
    config["r6_runtime_config_source"] = (
        "master_config_copy_plus_v4_8_fixed_rule_overlay_plus_fixed_empirical_cohort"
    )
    config.pop("runtime_smoke", None)
    return config


def _validate_r6_runtime_config(config: dict[str, Any]) -> None:
    initial = config.get("initial_conditions", {})
    policy = config.get("policy", {})
    government = policy.get("government", {}) if isinstance(policy, dict) else {}
    entry = config.get("population_dynamics", {}).get("entry_mechanism", {})
    scale = config.get("scale", {})
    raw_cohort_path = initial.get("fixed_cohort_path") if isinstance(initial, dict) else None
    match raw_cohort_path:
        case str() if raw_cohort_path:
            fixed_cohort_path = Path(raw_cohort_path)
            if not fixed_cohort_path.is_absolute():
                fixed_cohort_path = PROJECT_ROOT / fixed_cohort_path
            fixed_cohort_size = cohort_size(fixed_cohort_path)
        case _:
            fixed_cohort_size = 0
    valid = (
        config.get("r6_runtime_config_source")
        == "master_config_copy_plus_v4_8_fixed_rule_overlay_plus_fixed_empirical_cohort"
        and not config.get("runtime_smoke")
        and isinstance(initial, dict)
        and initial.get("data_source") == "nfps"
        and initial.get("microdata_driven") is True
        and bool(initial.get("fixed_cohort_path"))
        and isinstance(entry, dict)
        and bool(entry.get("entrant_feature_library"))
        and isinstance(policy, dict)
        and policy.get("policy_mode") == "rule"
        and policy.get("rl_enabled") is False
        and isinstance(government, dict)
        and government.get("adaptive_adjustment_enabled") is False
        and int(scale.get("n_farmers", 0)) == fixed_cohort_size
        and float(scale.get("agent_representation_weight", 0.0)) == 1.0
    )
    if not valid:
        raise R6RuntimeError("R6 fixed-cohort runtime contract is not satisfied")


def run_case(
    overlay: Path,
    cohort_path: Path,
    seed: int,
    years: int,
    parameters: dict[str, float],
) -> tuple[dict[str, str], ...]:
    RandomController.set_seed(seed)
    config = runtime_config(overlay, cohort_path, seed, years)
    _apply_runtime_parameters(config, parameters)
    _validate_r6_runtime_config(config)
    config_hash = _stable_hash(config)
    with Path(os.devnull).open("w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            model = ModelBuilder(config=config).build()
            initial_hash = hashlib.sha256(
                json.dumps(
                    [
                        {
                            "id": str(getattr(farmer, "unique_id", "")),
                            "source": str(
                                getattr(farmer, "source_representation_weight_q", "")
                            ),
                            "patches": [
                                float(getattr(patch, "area", 0.0) or 0.0)
                                for patch in getattr(farmer, "land_patches", []) or []
                            ],
                        }
                        for farmer in model.farmers
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            rows = [
                _initial_runtime_observation(
                    model,
                    seed,
                    BASELINE_SCENARIO,
                    config_hash,
                    initial_hash,
                )
            ]
            captured: dict[str, str] = {}
            current_year_index = 0
            prior_hook = getattr(model, "_completed_production_year_snapshot_hook", None)

            def capture(current_model: MainModel) -> None:
                if callable(prior_hook):
                    prior_hook(current_model)
                captured.clear()
                captured.update(
                    _runtime_observation(
                        current_model,
                        seed,
                        BASELINE_SCENARIO,
                        current_year_index,
                        config_hash,
                        initial_hash,
                    )
                )

            setattr(model, "_completed_production_year_snapshot_hook", capture)
            for year_index in range(1, years + 1):
                current_year_index = year_index
                model.step()
                if not captured:
                    raise R6RuntimeError("annual observation hook did not run")
                rows.append(dict(captured))
    return tuple(rows)
