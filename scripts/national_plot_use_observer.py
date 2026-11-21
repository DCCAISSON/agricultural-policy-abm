"""Read-only plot-use observer for the national fixed-rule cash runway."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import fsum, isclose, isfinite
from typing import TYPE_CHECKING, Literal, TypedDict

from scripts.run_national_formal_revision_20261118 import annual_actors

if TYPE_CHECKING:
    from core.model import MainModel


class PlotEvidenceError(ValueError):
    """The observed harvest and plot accounts do not support the requested evidence."""


class HarvestTotals(TypedDict):
    cultivated_area_mu: float
    grain_area_mu: float
    economic_crop_area_mu: float
    grain_output_kg: float
    economic_output_model_units: float


class PlotUse(TypedDict):
    patch_id: str
    area_mu: float
    operator_id: str | None
    cash_account_area_mu: float
    cultivated_area_mu: float
    uncultivated_area_mu: float
    cultivation_basis: Literal[
        "proportional_cash_account", "no_cultivation_in_harvest", "unreconciled_cash_account",
    ]
    crop_evidence: Literal[
        "not_cultivated", "unidentified", "area_conservation_single_crop",
        "area_conservation_single_cultivated_patch", "unreconciled",
    ]
    grain_area_mu: float | None
    economic_crop_area_mu: float | None
    unknown_crop_area_mu: float
    allocated_grain_area_mu: float
    allocated_economic_crop_area_mu: float
    allocated_grain_output_kg: float
    allocated_economic_output_model_units: float


class Reconciliation(TypedDict):
    harvest: HarvestTotals
    allocated: HarvestTotals
    residual_harvest_minus_allocated: dict[str, float]
    unreconciled_operators: dict[str, dict[str, float]]
    plot_area_mu: float
    uncultivated_area_mu: float
    unknown_crop_area_mu: float
    settlement_count: int


class PlotUseSnapshot(TypedDict):
    year: int
    patches: list[PlotUse]
    reconciliation: Reconciliation


def _check_close(expected: float, observed: float, label: str) -> None:
    if not _close(expected, observed):
        raise PlotEvidenceError(f"{label}: expected {expected}, observed {observed}")


def _close(expected: float, observed: float) -> bool:
    return isclose(expected, observed, rel_tol=1e-10, abs_tol=1e-7)


def snapshot_plot_use(current: MainModel) -> PlotUseSnapshot:
    """Call in the pre-demography annual hook under the actual-harvest observer.

    Cultivation follows the cash account's proportional area rule, gated by
    actual production. Crop areas are identified only by area conservation;
    output remains a proportional proxy even when crop area is identified.
    """
    step = int(current.step_count)
    records = getattr(current, "revision_harvest_records", None)
    if not records or any(row["step"] != step for row in records.values()):
        raise PlotEvidenceError(f"Missing or stale harvest observations for step {step}")

    totals: dict[str, HarvestTotals] = {}
    for operator_id, actor in annual_actors(current).items():
        if operator_id not in records:
            continue
        row = records[operator_id]
        grain, economic = float(actor.grain_area), float(actor.cash_crop_area)
        cultivated = grain + economic if hasattr(actor, "crop_portfolio") else float(actor.current_production_area)
        values = (cultivated, grain, economic, row["grain_physical_kg"], row["cash_output_model_units"])
        if any(not isfinite(value) for value in values) or min(values[:3]) < 0:
            raise PlotEvidenceError(f"Invalid harvest accounts for {operator_id}: {values}")
        totals[operator_id] = HarvestTotals(
            cultivated_area_mu=cultivated, grain_area_mu=grain, economic_crop_area_mu=economic,
            grain_output_kg=values[3], economic_output_model_units=values[4],
        )
    if totals.keys() != records.keys():
        raise PlotEvidenceError(f"Unmatched harvest operators: {sorted(records.keys() - totals.keys())}")

    patches: list[PlotUse] = []
    by_operator: dict[str, list[PlotUse]] = defaultdict(list)
    seen: set[str] = set()
    for patch in current.land_patches:
        patch_id = str(patch.get_patch_id())
        if patch_id in seen or getattr(patch, "preproduction_cash_step", None) != step:
            raise PlotEvidenceError(f"Duplicate patch or missing current cash-area account: {patch_id}")
        seen.add(patch_id)
        area, funded = float(patch.area), float(patch.preproduction_cultivated_area_mu)
        if not (isfinite(area) and isfinite(funded) and 0 <= funded <= area + 1e-7):
            raise PlotEvidenceError(f"Invalid cash-account area for {patch_id}: {funded} / {area}")
        operator = patch.current_operator or patch.leased_to or patch.owner
        operator_id = None if operator is None else str(operator.unique_id)
        harvest = None if operator_id is None else totals.get(operator_id)
        cultivated = funded if harvest and harvest["cultivated_area_mu"] > 0 else 0.0
        share = cultivated / harvest["cultivated_area_mu"] if harvest and cultivated > 0 else 0.0
        # shortcut: output is only operator-level; replace proxies with dated plot harvest records.
        plot = PlotUse(
            patch_id=patch_id, area_mu=area, operator_id=operator_id, cash_account_area_mu=funded,
            cultivated_area_mu=cultivated, uncultivated_area_mu=area - cultivated,
            cultivation_basis="proportional_cash_account" if cultivated else "no_cultivation_in_harvest",
            crop_evidence="unidentified" if cultivated else "not_cultivated",
            grain_area_mu=None if cultivated else 0.0, economic_crop_area_mu=None if cultivated else 0.0,
            unknown_crop_area_mu=cultivated,
            allocated_grain_area_mu=0.0 if harvest is None else share * harvest["grain_area_mu"],
            allocated_economic_crop_area_mu=0.0 if harvest is None else share * harvest["economic_crop_area_mu"],
            allocated_grain_output_kg=0.0 if harvest is None else share * harvest["grain_output_kg"],
            allocated_economic_output_model_units=(
                0.0 if harvest is None else share * harvest["economic_output_model_units"]
            ),
        )
        patches.append(plot)
        if operator_id is not None:
            by_operator[operator_id].append(plot)

    unreconciled: dict[str, dict[str, float]] = {}
    for operator_id, harvest in totals.items():
        operated = by_operator[operator_id]
        allocated = _allocated(operated)
        residual = {key: amount - allocated[key] for key, amount in harvest.items()
                    if not _close(amount, allocated[key])}
        if residual:
            unreconciled[operator_id] = residual
            for plot in operated:
                plot["cultivation_basis"] = "unreconciled_cash_account"
                plot["crop_evidence"] = "unreconciled"
                plot["grain_area_mu"] = plot["economic_crop_area_mu"] = None
            continue
        positive = [p for p in operated if p["cultivated_area_mu"] > 0]
        grain, economic = harvest["grain_area_mu"], harvest["economic_crop_area_mu"]
        cultivated = harvest["cultivated_area_mu"]
        if positive and cultivated == grain + economic:
            if grain == 0 or economic == 0:
                for plot in positive:
                    plot["crop_evidence"] = "area_conservation_single_crop"
                    plot["grain_area_mu"] = plot["cultivated_area_mu"] if grain else 0.0
                    plot["economic_crop_area_mu"] = plot["cultivated_area_mu"] if economic else 0.0
                    plot["unknown_crop_area_mu"] = 0.0
            elif len(positive) == 1:
                plot = positive[0]
                plot["crop_evidence"] = "area_conservation_single_cultivated_patch"
                plot["grain_area_mu"], plot["economic_crop_area_mu"] = grain, economic
                plot["unknown_crop_area_mu"] = 0.0
    allocated = _allocated(patches)
    harvest = HarvestTotals(
        cultivated_area_mu=fsum(row["cultivated_area_mu"] for row in totals.values()),
        grain_area_mu=fsum(row["grain_area_mu"] for row in totals.values()),
        economic_crop_area_mu=fsum(row["economic_crop_area_mu"] for row in totals.values()),
        grain_output_kg=fsum(row["grain_output_kg"] for row in totals.values()),
        economic_output_model_units=fsum(row["economic_output_model_units"] for row in totals.values()),
    )
    unknown = fsum(p["unknown_crop_area_mu"] for p in patches)
    _check_close(allocated["cultivated_area_mu"], unknown + fsum(
        (p["grain_area_mu"] or 0.0) + (p["economic_crop_area_mu"] or 0.0) for p in patches
    ), "annual identified and unknown crop area")
    land_area, uncultivated = fsum(p["area_mu"] for p in patches), fsum(p["uncultivated_area_mu"] for p in patches)
    _check_close(land_area, uncultivated + allocated["cultivated_area_mu"], "annual land area")
    return PlotUseSnapshot(
        year=int(current.simulation_base_year) + step, patches=patches,
        reconciliation=Reconciliation(
            harvest=harvest, allocated=allocated,
            residual_harvest_minus_allocated={key: amount - allocated[key] for key, amount in harvest.items()},
            unreconciled_operators=unreconciled, plot_area_mu=land_area,
            uncultivated_area_mu=uncultivated, unknown_crop_area_mu=unknown,
            settlement_count=len(records),
        ),
    )


def _allocated(plots: Sequence[PlotUse]) -> HarvestTotals:
    return HarvestTotals(
        cultivated_area_mu=fsum(p["cultivated_area_mu"] for p in plots),
        grain_area_mu=fsum(p["allocated_grain_area_mu"] for p in plots),
        economic_crop_area_mu=fsum(p["allocated_economic_crop_area_mu"] for p in plots),
        grain_output_kg=fsum(p["allocated_grain_output_kg"] for p in plots),
        economic_output_model_units=fsum(p["allocated_economic_output_model_units"] for p in plots),
    )


class ReleaseRow(TypedDict):
    """Required fields from land_succession.json's releases."""

    release_event_id: str
    release_year: int
    source_operator_id: str
    release_reason: str
    patch_ids: Sequence[str]


class ReleaseDestination(TypedDict):
    release_event_id: str | None
    release_reason: str | None
    source_operator_id: str | None
    source_matches_prior_operator: bool | None
    origin: PlotUse | None
    destination: PlotUse | None


class PairedReleaseDestination(TypedDict):
    seed: int
    patch_id: str
    release_year: int
    destination_year: int
    reference: ReleaseDestination
    comparison: ReleaseDestination


def paired_release_destinations(
    reference: Sequence[PlotUseSnapshot],
    comparison: Sequence[PlotUseSnapshot],
    *,
    reference_releases: Sequence[ReleaseRow],
    comparison_releases: Sequence[ReleaseRow],
    seed: int,
) -> list[PairedReleaseDestination]:
    """Compare the union of logged releases on a caller-verified same-seed pair.

    Matching is by patch and release year, not by release event ID. The origin
    and next-year rows carry their own crop-area evidence basis. To select
    released old grain, require a matched source and positive known origin grain.
    """
    index = []
    for snapshots, releases in ((reference, reference_releases), (comparison, comparison_releases)):
        years = {s["year"]: {p["patch_id"]: p for p in s["patches"]} for s in snapshots}
        events = {(r["release_year"], patch_id): r for r in releases for patch_id in r["patch_ids"]}
        if (len(years) != len(snapshots) or any(len(years[s["year"]]) != len(s["patches"]) for s in snapshots)
                or len(events) != sum(len(r["patch_ids"]) for r in releases)):
            raise PlotEvidenceError("Duplicate annual plot or release observations")
        index.append((years, events))

    (ref_years, ref_events), (cmp_years, cmp_events) = index
    for year in ref_years.keys() & cmp_years.keys():
        if {k: p["area_mu"] for k, p in ref_years[year].items()} != {
            k: p["area_mu"] for k, p in cmp_years[year].items()
        }:
            raise PlotEvidenceError(f"Paired plot identities or areas differ in {year}")

    rows: list[PairedReleaseDestination] = []
    for year, patch_id in sorted(ref_events.keys() | cmp_events.keys()):
        sides: list[ReleaseDestination] = []
        for years, events in index:
            event = events.get((year, patch_id))
            origin = years.get(year, {}).get(patch_id)
            destination = years.get(year + 1, {}).get(patch_id)
            source = None if event is None else event["source_operator_id"]
            sides.append(ReleaseDestination(
                release_event_id=None if event is None else event["release_event_id"],
                release_reason=None if event is None else event["release_reason"],
                source_operator_id=source, origin=None if origin is None else origin.copy(),
                source_matches_prior_operator=None if origin is None or source is None else (
                    source == origin["operator_id"]
                ),
                destination=None if destination is None else destination.copy(),
            ))
        rows.append(PairedReleaseDestination(
            seed=seed, patch_id=patch_id, release_year=year, destination_year=year + 1,
            reference=sides[0], comparison=sides[1],
        ))
    return rows
