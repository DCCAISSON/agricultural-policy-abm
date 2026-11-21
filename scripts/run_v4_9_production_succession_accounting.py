# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
# --- How to run ---
# python scripts/run_v4_9_production_succession_accounting.py --overlay config/experiments/v4_8_national_fixed_rule.yaml --seeds 0:2 --years 6 --n-farmers 100 --output results/intermediate/v4_9_production_succession_20260814
"""# noqa: SIZE_OK - bounded runtime and manifest form one audit surface."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, TypeAlias

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model_builder import ModelBuilder  # noqa: E402
from core.reproducibility import RandomController  # noqa: E402
from modules.production_succession_accounting import (  # noqa: E402
    FinalUseObservation,
    FinalUsePatch,
    ProductionSuccessionAccount,
)
from scripts.run_v4_8_national_calibration_validation import (  # noqa: E402
    BASELINE_SCENARIO,
    _validate_formal_runtime_config,
    runtime_config,
)
from scripts.v5_r5_runtime_contract import sha256_file  # noqa: E402


DEFAULT_OUTPUT: Final = (
    PROJECT_ROOT / "results" / "intermediate" / "v4_9_production_succession_20260814"
)
DEFAULT_OVERLAY: Final = (
    PROJECT_ROOT / "config" / "experiments" / "v4_8_national_fixed_rule.yaml"
)
PROTECTED_CONFIGS: Final = (
    Path("config/master_config.yaml"),
    Path("config/policy_params.json"),
)
SOURCE_FILES: Final = (
    Path("modules/production_succession_accounting.py"),
    Path("modules/population_dynamics_v2.py"),
    Path("core/model.py"),
    Path("core/model_builder.py"),
    Path("scripts/run_v4_9_production_succession_accounting.py"),
    Path("scripts/v5_r5_runtime_contract.py"),
)
Scalar: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class RunnerError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class SeedRun:
    account: ProductionSuccessionAccount
    weighted_candidate_trace: tuple[dict[str, Scalar], ...]
    weighted_annual_stats: tuple[dict[str, Scalar], ...]


class CliArgs(argparse.Namespace):
    overlay: Path = DEFAULT_OVERLAY
    seeds: str = "0:2"
    years: int = 6
    n_farmers: int = 100
    output: Path = DEFAULT_OUTPUT


def _parse_seeds(specification: str) -> tuple[int, ...]:
    parts = specification.split(":")
    if len(parts) == 1:
        return (int(parts[0]),)
    if len(parts) != 2:
        raise RunnerError("seeds must be one integer or an inclusive start:end range")
    start, end = (int(part) for part in parts)
    if start < 0 or end < start or end - start > 9:
        raise RunnerError("M2 allows at most ten nonnegative bounded seeds")
    return tuple(range(start, end + 1))


def _observe_final_use(
    model, account: ProductionSuccessionAccount, observation_year: int
) -> None:
    observed = {event.release_event_id for event in account.final_uses}
    patches = list(getattr(model, "land_patches", []) or [])
    for allocation in account.allocations:
        if (
            allocation.release_event_id in observed
            or observation_year <= allocation.allocation_year
        ):
            continue
        matched = [
            patch
            for patch in patches
            if str(getattr(patch, "production_succession_release_event_id", "") or "")
            == allocation.release_event_id
        ]
        if not matched:
            continue
        final_use_patches = []
        for patch in matched:
            operator = getattr(patch, "current_operator", None)
            crop = None
            if operator is not None:
                crop = getattr(patch, "current_crop", None)
                if crop is None:
                    crop = getattr(operator, "planned_crop", None)
            final_use_patches.append(
                FinalUsePatch(
                    patch_id=str(patch.get_patch_id()),
                    area_mu=float(getattr(patch, "area", 0.0) or 0.0),
                    crop_label=(
                        "unoperated"
                        if operator is None
                        else None
                        if crop is None
                        else str(getattr(crop, "value", crop))
                    ),
                    final_operator_id=str(getattr(operator, "unique_id", "unknown")),
                )
            )
        account.observe_final_use(
            FinalUseObservation(
                release_event_id=allocation.release_event_id,
                observation_year=observation_year,
                patches=tuple(final_use_patches),
            )
        )


def _run_seed(overlay: Path, seed: int, years: int, n_farmers: int) -> SeedRun:
    RandomController.set_seed(seed)
    config = runtime_config(overlay, seed=seed, years=years, n_farmers=n_farmers)
    config["policy"]["active_scenario"] = BASELINE_SCENARIO
    _validate_formal_runtime_config(config)
    config["experiment"].pop("overlay_path", None)
    model = ModelBuilder(config=config).build()
    population_manager = getattr(model, "population_manager", None)
    if population_manager is None:
        raise RunnerError("population manager is required for M2")
    account = ProductionSuccessionAccount()
    setattr(model, "production_succession_account", account)
    population_manager.land_bank.production_succession_account = account
    prior_hook = getattr(model, "_completed_production_year_snapshot_hook", None)

    def capture(current_model) -> None:
        if callable(prior_hook):
            prior_hook(current_model)
        current_year = int(
            current_model.simulation_base_year + current_model.step_count
        )
        _observe_final_use(current_model, account, current_year)

    setattr(model, "_completed_production_year_snapshot_hook", capture)
    for _ in range(years):
        model.step()
    return SeedRun(
        account=account,
        weighted_candidate_trace=tuple(
            getattr(population_manager, "weighted_allocation_trace", []) or []
        ),
        weighted_annual_stats=tuple(
            getattr(population_manager, "weighted_allocation_annual_stats", []) or []
        ),
    )


def run(
    overlay: Path, seeds: tuple[int, ...], years: int, n_farmers: int, output: Path
) -> None:
    if years < 2 or years > 10 or n_farmers < 20 or n_farmers > 300:
        raise RunnerError("M2 requires 2-10 years and 20-300 farmers")
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, list[dict[str, Scalar]]] = {
        "annual_land_destination.csv": [],
        "channel_assignment_trace.csv": [],
        "production_succession_metrics.csv": [],
        "organizational_succession_composition.csv": [],
        "grain_retention_account.csv": [],
        "weighted_runtime_candidate_trace.csv": [],
        "weighted_runtime_annual_account.csv": [],
    }
    for seed in seeds:
        seed_run = _run_seed(overlay, seed, years, n_farmers)
        account = seed_run.account
        artifacts["weighted_runtime_candidate_trace.csv"].extend(
            {"runtime_seed": seed, **row} for row in seed_run.weighted_candidate_trace
        )
        artifacts["weighted_runtime_annual_account.csv"].extend(
            {"runtime_seed": seed, **row} for row in seed_run.weighted_annual_stats
        )
        release_years = tuple(range(2011, 2011 + years))
        annual = account.annual_land_destination(release_years)
        grains = account.grain_retention_account(release_years)
        artifacts["annual_land_destination.csv"].extend(
            {"runtime_seed": seed, **asdict(row)} for row in annual
        )
        artifacts["grain_retention_account.csv"].extend(
            {"runtime_seed": seed, **asdict(row)} for row in grains
        )
        artifacts["production_succession_metrics.csv"].extend(
            {
                "runtime_seed": seed,
                "release_year": row.release_year,
                "psr": row.psr,
                "psg_area_mu": row.psg_area_mu,
                "market_intermediated_share": row.market_intermediated_share,
            }
            for row in annual
        )
        artifacts["organizational_succession_composition.csv"].extend(
            {"runtime_seed": seed, **asdict(row)}
            for row in account.organizational_composition()
        )
        allocations = {event.release_event_id: event for event in account.allocations}
        final_uses = {event.release_event_id: event for event in account.final_uses}
        for release in account.releases:
            allocation = allocations.get(release.release_event_id)
            final_use = final_uses.get(release.release_event_id)
            artifacts["channel_assignment_trace.csv"].append(
                {
                    "runtime_seed": seed,
                    **asdict(release),
                    "patch_ids": "|".join(release.patch_ids),
                    "allocation_event_id": allocation.allocation_event_id
                    if allocation
                    else "",
                    "allocation_year": allocation.allocation_year
                    if allocation
                    else None,
                    "channel": allocation.channel.value if allocation else "unabsorbed",
                    "destination_operator_id": allocation.destination_operator_id
                    if allocation
                    else "",
                    "destination_operator_type": allocation.destination_operator_type
                    if allocation
                    else "",
                    "succession_role": allocation.succession_role
                    if allocation
                    else "unabsorbed",
                    "absorbed_area_mu": allocation.absorbed_area_mu
                    if allocation
                    else 0.0,
                    "final_use_observation_year": final_use.observation_year
                    if final_use
                    else None,
                    "final_crop_labels": "|".join(
                        "" if patch.crop_label is None else patch.crop_label
                        for patch in final_use.patches
                    )
                    if final_use
                    else "",
                    "final_operator_ids": "|".join(
                        patch.final_operator_id for patch in final_use.patches
                    )
                    if final_use
                    else "",
                }
            )
    for name, rows in artifacts.items():
        fieldnames = tuple(rows[0]) if rows else ("runtime_seed", "release_year")
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "stage": "V4.9_M2",
        "run_status": "completed_pending_validation",
        "runtime_surface": "ModelBuilder.build + MainModel.step",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "overlay": str(overlay),
        "overlay_sha256": sha256_file(overlay),
        "seed_set": list(seeds),
        "years": years,
        "n_farmers": n_farmers,
        "source_sha256": {
            str(path).replace("\\", "/"): sha256_file(PROJECT_ROOT / path)
            for path in SOURCE_FILES
        },
        "protected_config_sha256": {
            str(path).replace("\\", "/"): sha256_file(PROJECT_ROOT / path)
            for path in PROTECTED_CONFIGS
        },
        "artifact_sha256": {name: sha256_file(output / name) for name in artifacts},
        "claim_boundary": "Internal event accounting only; not empirical calibration or causal policy identification.",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run bounded V4.9 M2 production-succession accounting."
    )
    _ = parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    _ = parser.add_argument("--seeds", default="0:2")
    _ = parser.add_argument("--years", type=int, default=6)
    _ = parser.add_argument("--n-farmers", type=int, default=100)
    _ = parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(namespace=CliArgs())
    try:
        run(
            args.overlay.resolve(),
            _parse_seeds(args.seeds),
            args.years,
            args.n_farmers,
            args.output.resolve(),
        )
    except (OSError, UnicodeError, ValueError, RunnerError) as error:
        print(f"BLOCK_PRODUCTION_SUCCESSION_ACCOUNTING: {error}", file=sys.stderr)
        return 2
    print("M2_ACCOUNTING_RUN_COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
