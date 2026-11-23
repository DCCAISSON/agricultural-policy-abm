# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
# --- How to run ---
# python -m pytest -q tests/test_paper_b_flagship_counterfactual.py

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias, cast

from core.model import MainModel
from core.model_builder import ModelBuilder
from core.reproducibility import RandomController
from modules.production_succession_accounting import ProductionSuccessionAccount
from scripts.paper_b_flagship_observation import (
    FlagshipModel,
    initial_state_hash,
    operator_snapshot,
    operator_weights,
    observe_final_use,
    stable_hash,
)
from scripts.paper_b_flagship_scenarios import (
    Config,
    FlagshipScenario,
    apply_annual_exogenous_path,
    apply_scenario_config,
)
from scripts.u3_unified_model_runtime import u3_runtime_config


Scalar: TypeAlias = str | int | float | bool | None
Row: TypeAlias = dict[str, Scalar]
ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_COHORT: Final = (
    ROOT / "data/derived_runtime/v5_r6/nfps_fixed_empirical_cohort_2010_k600.json"
)
DEFAULT_OVERLAY: Final = ROOT / "config/experiments/v4_8_national_fixed_rule.yaml"


@dataclass(frozen=True, slots=True)
class WorldlineResult:
    scenario_id: str
    seed: int
    annual_rows: tuple[Row, ...]


class FlagshipRuntimeError(RuntimeError):
    pass


def run_worldline(
    scenario: FlagshipScenario,
    seed: int,
    years: int,
    parameters: dict[str, float],
    *,
    cohort_path: Path = DEFAULT_COHORT,
    overlay_path: Path = DEFAULT_OVERLAY,
) -> WorldlineResult:
    if years < 2:
        raise FlagshipRuntimeError("flagship runtime requires at least two years")
    RandomController.set_seed(seed)
    config = cast(
        Config,
        u3_runtime_config(
            seed,
            years,
            parameters,
            cohort_path=cohort_path,
            overlay_path=overlay_path,
        ),
    )
    apply_scenario_config(config, scenario)
    config_hash = stable_hash(config)
    observations: dict[int, Row] = {}
    pre_household_mass: dict[int, float] = {}
    exit_mass: dict[int, float] = {}
    with Path(os.devnull).open("w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            model = cast(
                FlagshipModel,
                cast(object, ModelBuilder(config=config).build()),
            )
            initial_hash = initial_state_hash(model)
            account = ProductionSuccessionAccount()
            setattr(model, "production_succession_account", account)
            setattr(
                model.population_manager.land_bank,
                "production_succession_account",
                account,
            )
            prior_hook = getattr(
                model, "_completed_production_year_snapshot_hook", None
            )

            def capture(current_model: MainModel) -> None:
                if callable(prior_hook):
                    _ = prior_hook(current_model)
                runtime_model = cast(FlagshipModel, cast(object, current_model))
                simulation_year = int(
                    runtime_model.simulation_base_year + runtime_model.step_count
                )
                observe_final_use(runtime_model, account, simulation_year)
                observations[simulation_year] = operator_snapshot(runtime_model, account)

            setattr(model, "_completed_production_year_snapshot_hook", capture)
            for year_index in range(1, years + 1):
                apply_annual_exogenous_path(model, year_index - 1)
                year = int(model.simulation_base_year + year_index)
                weights = operator_weights(model.farmers)
                pre_household_mass[year] = sum(weights.values())
                model.step()
                surviving_operator_ids = {
                    str(operator.unique_id) for operator in model.farmers
                }
                exit_mass[year] = removed_operator_mass(
                    weights, surviving_operator_ids
                )
    years_tuple = tuple(sorted(observations))
    destinations = {
        row.release_year: row for row in account.annual_land_destination(years_tuple)
    }
    grain_accounts = {
        row.release_year: row for row in account.grain_retention_account(years_tuple)
    }
    annual_rows: list[Row] = []
    for year_index, year in enumerate(years_tuple, start=1):
        destination = destinations[year]
        grain = grain_accounts[year]
        observable_grain = max(
            0.0,
            grain.released_grain_area_mu - grain.unresolved_final_use_area_mu,
        )
        annual_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "ssp_code": scenario.ssp_code,
                "ssp_name": scenario.ssp_name,
                "coordination_code": scenario.coordination_code,
                "coordination_scale": scenario.coordination_scale,
                "support_yuan_per_mu": scenario.support_yuan_per_mu,
                "runtime_seed": seed,
                "year_index": year_index,
                "simulation_year": year,
                "runtime_input_hash": config_hash,
                "initial_state_sha256": initial_hash,
                "released_area_mu": destination.released_area_mu,
                "allocated_area_mu": destination.total_absorbed_area_mu,
                "unabsorbed_area_mu": destination.unabsorbed_area_mu,
                "land_absorption_rate": _ratio(
                    destination.total_absorbed_area_mu,
                    destination.released_area_mu,
                ),
                "released_grain_area_mu": grain.released_grain_area_mu,
                "observable_released_grain_area_mu": observable_grain,
                "retained_grain_area_mu": grain.retained_grain_area_mu,
                "grain_succession_rate": _ratio(
                    grain.retained_grain_area_mu, observable_grain
                ),
                "unresolved_grain_final_use_area_mu": (
                    grain.unresolved_final_use_area_mu
                ),
                "exit_operator_mass": exit_mass[year],
                "active_household_mass_pre_step": pre_household_mass[year],
                "exit_rate": _ratio(exit_mass[year], pre_household_mass[year]),
                **observations[year],
            }
        )
    return WorldlineResult(scenario.scenario_id, seed, tuple(annual_rows))


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def removed_operator_mass(
    pre_step_weights: Mapping[str, float], surviving_operator_ids: Set[str]
) -> float:
    return sum(
        weight
        for operator_id, weight in pre_step_weights.items()
        if operator_id not in surviving_operator_ids
    )
