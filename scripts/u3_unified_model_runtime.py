#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
# How to run:
# python -m pytest -q tests/test_u3_unified_model_qualification.py

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final

from core.model import MainModel
from core.model_builder import ModelBuilder
from core.reproducibility import RandomController
from scripts.run_v4_8_national_calibration_validation import (
    _initial_runtime_observation,
    _runtime_observation,
    _stable_hash,
)
from scripts.u2_unified_paper_model_runtime import (
    DEFAULT_COHORT,
    DEFAULT_OVERLAY,
    U2_SCENARIOS,
    u2_runtime_config,
)


U3_SCENARIO_ID: Final = "u3_baseline_service_environment"


def u3_runtime_config(
    seed: int,
    years: int,
    parameters: dict[str, float],
    *,
    cohort_path: Path = DEFAULT_COHORT,
    overlay_path: Path = DEFAULT_OVERLAY,
) -> dict[str, Any]:
    config = u2_runtime_config(
        seed,
        years,
        service_capacity_scale=U2_SCENARIOS[0].service_capacity_scale,
        cohort_path=cohort_path,
        overlay_path=overlay_path,
        parameters=parameters,
    )
    config["unified_paper_model"]["runtime_stage"] = (
        "U3_unified_model_dynamic_qualification"
    )
    config["u3_runtime_config_source"] = (
        "u2_unified_model_plus_frozen_candidate_parameters"
    )
    return config


def _initial_state_hash(model: MainModel) -> str:
    payload = [
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
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mark_u3_boundary(row: dict[str, str]) -> dict[str, str]:
    return {
        **row,
        "u2_unified_model_enabled": "true",
        "service_environment_only": "true",
        "paper_model_exit_mode": "livelihood_exit_only",
        "game_enabled": "false",
        "learning_enabled": "false",
    }


def run_moment_case(
    seed: int,
    years: int,
    parameters: dict[str, float],
    *,
    cohort_path: Path = DEFAULT_COHORT,
    overlay_path: Path = DEFAULT_OVERLAY,
) -> tuple[dict[str, str], ...]:
    RandomController.set_seed(seed)
    config = u3_runtime_config(
        seed,
        years,
        parameters,
        cohort_path=cohort_path,
        overlay_path=overlay_path,
    )
    config_hash = _stable_hash(config)
    with Path(os.devnull).open("w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            model = ModelBuilder(config=config).build()
            initial_hash = _initial_state_hash(model)
            rows = [
                _mark_u3_boundary(
                    _initial_runtime_observation(
                        model,
                        seed,
                        U3_SCENARIO_ID,
                        config_hash,
                        initial_hash,
                    )
                )
            ]
            captured: dict[str, str] = {}
            current_year_index = 0
            prior_hook = getattr(
                model, "_completed_production_year_snapshot_hook", None
            )

            def capture(current_model: MainModel) -> None:
                if callable(prior_hook):
                    _ = prior_hook(current_model)
                captured.clear()
                captured.update(
                    _mark_u3_boundary(
                        _runtime_observation(
                            current_model,
                            seed,
                            U3_SCENARIO_ID,
                            current_year_index,
                            config_hash,
                            initial_hash,
                        )
                    )
                )

            setattr(model, "_completed_production_year_snapshot_hook", capture)
            for year_index in range(1, years + 1):
                current_year_index = year_index
                model.step()
                if not captured:
                    msg = "annual observation hook did not run"
                    raise RuntimeError(msg)
                rows.append(dict(captured))
    return tuple(rows)
