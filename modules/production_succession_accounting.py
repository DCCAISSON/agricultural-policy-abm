"""# noqa: SIZE_OK - one event ledger owns release, allocation, and final-use invariants."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Final


AREA_TOLERANCE: Final = 1e-9


@unique
class Channel(StrEnum):
    MARKET = "market"
    DIRECT = "direct"
    VILLAGE = "village"
    ADMINISTRATIVE = "administrative"


@dataclass(frozen=True, slots=True)
class ProductionSuccessionAccountingError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class ReleaseRegistration:
    release_year: int
    source_operator_id: str
    source_operator_type: str
    release_reason: str
    area_mu: float
    patch_ids: tuple[str, ...]
    was_grain_land: bool | None


@dataclass(frozen=True, slots=True)
class ReleaseEvent:
    release_event_id: str
    release_year: int
    source_operator_id: str
    source_operator_type: str
    release_reason: str
    area_mu: float
    patch_ids: tuple[str, ...]
    was_grain_land: bool | None


@dataclass(frozen=True, slots=True)
class AllocationEvent:
    allocation_event_id: str
    release_event_id: str
    allocation_year: int
    channel: Channel
    destination_operator_id: str
    destination_operator_type: str
    succession_role: str
    absorbed_area_mu: float


@dataclass(frozen=True, slots=True)
class FinalUsePatch:
    patch_id: str
    area_mu: float
    crop_label: str | None
    final_operator_id: str


@dataclass(frozen=True, slots=True)
class FinalUseObservation:
    release_event_id: str
    observation_year: int
    patches: tuple[FinalUsePatch, ...]


@dataclass(frozen=True, slots=True)
class AnnualLandDestination:
    release_year: int
    released_area_mu: float
    market_absorbed_area_mu: float
    direct_absorbed_area_mu: float
    village_absorbed_area_mu: float
    administrative_absorbed_area_mu: float
    total_absorbed_area_mu: float
    unabsorbed_area_mu: float
    late_absorbed_area_mu: float
    residual_area_mu: float
    psr: float | None
    psg_area_mu: float
    market_intermediated_share: float | None
    zero_release_year: bool


@dataclass(frozen=True, slots=True)
class GrainRetentionAccount:
    release_year: int
    released_grain_area_mu: float
    retained_grain_area_mu: float
    unresolved_origin_area_mu: float
    unresolved_final_use_area_mu: float
    grain_retention_rate: float | None
    final_use_observation_year: int | None


@dataclass(frozen=True, slots=True)
class SuccessionComposition:
    release_year: int
    succession_role: str
    absorbed_area_mu: float
    share_of_absorbed: float


def grain_status(crop_label: str | None) -> bool | None:
    if crop_label is None or not crop_label.strip():
        return None
    label = crop_label.strip().lower()
    if any(token in label for token in ("grain", "rice", "wheat", "maize", "corn", "粮", "稻", "麦", "玉米")):
        return True
    if any(
        token in label
        for token in (
            "cash",
            "vegetable",
            "fruit",
            "unoperated",
            "abandoned",
            "fallow",
            "经济",
            "蔬菜",
            "水果",
            "撂荒",
            "休耕",
        )
    ):
        return False
    return None


class ProductionSuccessionAccount:
    """Mutable event ledger that observes land allocation without changing it."""

    def __init__(self) -> None:
        self._releases: dict[str, ReleaseEvent] = {}
        self._allocations: dict[str, AllocationEvent] = {}
        self._final_uses: dict[str, FinalUseObservation] = {}
        self._release_sequence = 0

    @property
    def releases(self) -> tuple[ReleaseEvent, ...]:
        return tuple(self._releases.values())

    @property
    def allocations(self) -> tuple[AllocationEvent, ...]:
        return tuple(self._allocations.values())

    @property
    def final_uses(self) -> tuple[FinalUseObservation, ...]:
        return tuple(self._final_uses.values())

    def record_release(self, registration: ReleaseRegistration) -> ReleaseEvent:
        if registration.area_mu <= 0.0:
            raise ProductionSuccessionAccountingError("released area must be positive")
        self._release_sequence += 1
        identity = "|".join(
            (
                str(registration.release_year),
                registration.source_operator_id,
                str(self._release_sequence),
                f"{registration.area_mu:.12f}",
                *registration.patch_ids,
            )
        )
        event_id = f"release_{registration.release_year}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
        event = ReleaseEvent(
            release_event_id=event_id,
            release_year=registration.release_year,
            source_operator_id=registration.source_operator_id,
            source_operator_type=registration.source_operator_type,
            release_reason=registration.release_reason,
            area_mu=registration.area_mu,
            patch_ids=registration.patch_ids,
            was_grain_land=registration.was_grain_land,
        )
        self._releases[event_id] = event
        return event

    def record_allocation(self, event: AllocationEvent) -> None:
        release = self._releases.get(event.release_event_id)
        if release is None:
            raise ProductionSuccessionAccountingError("allocation references unknown release")
        if event.allocation_event_id in self._allocations:
            raise ProductionSuccessionAccountingError("allocation event already exists")
        if event.absorbed_area_mu <= 0.0:
            raise ProductionSuccessionAccountingError("absorbed area must be positive")
        allocated_area = sum(
            allocation.absorbed_area_mu
            for allocation in self._allocations.values()
            if allocation.release_event_id == event.release_event_id
        )
        if allocated_area + event.absorbed_area_mu - release.area_mu > AREA_TOLERANCE:
            raise ProductionSuccessionAccountingError("allocation exceeds released area")
        if event.allocation_year < release.release_year:
            raise ProductionSuccessionAccountingError("allocation precedes release year")
        self._allocations[event.allocation_event_id] = event

    def observe_final_use(self, observation: FinalUseObservation) -> None:
        allocations = [
            event
            for event in self._allocations.values()
            if event.release_event_id == observation.release_event_id
        ]
        if not allocations:
            raise ProductionSuccessionAccountingError("final use references unallocated release")
        if observation.observation_year <= max(
            event.allocation_year for event in allocations
        ):
            raise ProductionSuccessionAccountingError("final use must be observed after allocation year")
        if observation.release_event_id in self._final_uses:
            raise ProductionSuccessionAccountingError("final use is already observed")
        observed_area = sum(patch.area_mu for patch in observation.patches)
        allocated_area = sum(event.absorbed_area_mu for event in allocations)
        if observed_area - allocated_area > AREA_TOLERANCE:
            raise ProductionSuccessionAccountingError("final-use area exceeds absorbed area")
        self._final_uses[observation.release_event_id] = observation

    def annual_land_destination(self, years: tuple[int, ...]) -> list[AnnualLandDestination]:
        rows: list[AnnualLandDestination] = []
        for year in sorted(set(years)):
            releases = [event for event in self._releases.values() if event.release_year == year]
            release_ids = {event.release_event_id for event in releases}
            allocations = [
                event
                for event in self._allocations.values()
                if event.release_event_id in release_ids
            ]
            same_year = [event for event in allocations if event.allocation_year == year]
            by_channel = {channel: sum(event.absorbed_area_mu for event in same_year if event.channel is channel) for channel in Channel}
            released = sum(event.area_mu for event in releases)
            absorbed = sum(by_channel.values())
            unabsorbed = max(0.0, released - absorbed)
            late_absorbed = sum(event.absorbed_area_mu for event in allocations if event.allocation_year > year)
            rows.append(
                AnnualLandDestination(
                    year, released, by_channel[Channel.MARKET], by_channel[Channel.DIRECT],
                    by_channel[Channel.VILLAGE], by_channel[Channel.ADMINISTRATIVE], absorbed,
                    unabsorbed, late_absorbed, released - absorbed - unabsorbed,
                    absorbed / released if released > 0.0 else None, unabsorbed,
                    by_channel[Channel.MARKET] / absorbed if absorbed > 0.0 else None,
                    released == 0.0,
                )
            )
        return rows

    def grain_retention_account(self, years: tuple[int, ...]) -> list[GrainRetentionAccount]:
        rows: list[GrainRetentionAccount] = []
        for year in sorted(set(years)):
            releases = [event for event in self._releases.values() if event.release_year == year]
            grain_releases = [event for event in releases if event.was_grain_land is True]
            released_grain = sum(event.area_mu for event in grain_releases)
            unresolved_origin = sum(event.area_mu for event in releases if event.was_grain_land is None)
            retained = 0.0
            unresolved_final = 0.0
            observation_years: list[int] = []
            for release in grain_releases:
                allocations = [
                    event
                    for event in self._allocations.values()
                    if event.release_event_id == release.release_event_id
                    and event.allocation_year == year
                ]
                if not allocations:
                    continue
                allocated_area = sum(event.absorbed_area_mu for event in allocations)
                final_use = self._final_uses.get(release.release_event_id)
                if final_use is None:
                    unresolved_final += allocated_area
                    continue
                observation_years.append(final_use.observation_year)
                observed_patches = final_use.patches
                observed_area = sum(patch.area_mu for patch in observed_patches)
                unresolved_final += max(0.0, allocated_area - observed_area)
                for patch in observed_patches:
                    status = grain_status(patch.crop_label)
                    if status is True:
                        retained += patch.area_mu
                    elif status is None:
                        unresolved_final += patch.area_mu
            rate = retained / released_grain if released_grain > 0.0 and unresolved_final <= AREA_TOLERANCE else None
            rows.append(
                GrainRetentionAccount(
                    year, released_grain, retained, unresolved_origin, unresolved_final, rate,
                    max(observation_years) if observation_years else None,
                )
            )
        return rows

    def organizational_composition(self) -> list[SuccessionComposition]:
        rows: list[SuccessionComposition] = []
        years = sorted({event.release_year for event in self._releases.values()})
        for year in years:
            same_year = [event for event in self._allocations.values() if event.allocation_year == year and self._releases[event.release_event_id].release_year == year]
            total = sum(event.absorbed_area_mu for event in same_year)
            roles = sorted({event.succession_role for event in same_year})
            for role in roles:
                area = sum(event.absorbed_area_mu for event in same_year if event.succession_role == role)
                rows.append(SuccessionComposition(year, role, area, area / total if total > 0.0 else 0.0))
        return rows
