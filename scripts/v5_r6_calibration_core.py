from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Final


PROBABILITY_TOLERANCE: Final = 1e-10


@dataclass(frozen=True, slots=True)
class CohortSource:
    observation_id: str
    source_row: int
    representation_weight_q: float
    common_region: str
    farm_size: str
    land_role: str
    operated_area_mu: float


@dataclass(frozen=True, slots=True)
class CohortMember:
    cohort_agent_id: str
    source_observation_id: str
    source_row: int
    source_representation_weight_q: float
    cohort_probability_q: float
    runtime_representation_weight: float
    common_region: str
    farm_size: str
    land_role: str
    operated_area_mu: float


@dataclass(frozen=True, slots=True)
class MomentResidual:
    moment_id: str
    simulated_value: float
    target_value: float
    standardization: float


class R6CalibrationError(RuntimeError):
    pass


def _cell(source: CohortSource) -> tuple[str, str, str]:
    return source.common_region, source.farm_size, source.land_role


def _cell_counts(
    sources: tuple[CohortSource, ...],
    cohort_size: int,
) -> dict[tuple[str, str, str], int]:
    masses: dict[tuple[str, str, str], float] = defaultdict(float)
    for source in sources:
        masses[_cell(source)] += source.representation_weight_q
    exact = {cell: mass * cohort_size for cell, mass in masses.items()}
    counts = {cell: math.floor(value) for cell, value in exact.items()}
    remaining = cohort_size - sum(counts.values())
    ranked = sorted(
        exact,
        key=lambda cell: (-(exact[cell] - counts[cell]), cell),
    )
    for cell in ranked[:remaining]:
        counts[cell] += 1
    return counts


def _systematic_cell_draws(
    sources: tuple[CohortSource, ...],
    draws: int,
) -> tuple[CohortSource, ...]:
    if draws == 0:
        return ()
    ordered = tuple(sorted(sources, key=lambda row: (row.source_row, row.observation_id)))
    total = math.fsum(row.representation_weight_q for row in ordered)
    points = tuple((index + 0.5) * total / draws for index in range(draws))
    selected: list[CohortSource] = []
    cumulative = 0.0
    point_index = 0
    for source in ordered:
        cumulative += source.representation_weight_q
        while point_index < draws and points[point_index] <= cumulative:
            selected.append(source)
            point_index += 1
    if len(selected) != draws:
        selected.extend([ordered[-1]] * (draws - len(selected)))
    return tuple(selected)


def build_fixed_cohort(
    sources: tuple[CohortSource, ...],
    cohort_size: int,
) -> tuple[CohortMember, ...]:
    if cohort_size < 1:
        raise R6CalibrationError("cohort_size must be positive")
    if not sources:
        raise R6CalibrationError("eligible source population is empty")
    if any(row.representation_weight_q <= 0.0 for row in sources):
        raise R6CalibrationError("source representation weights must be positive")
    total_weight = math.fsum(row.representation_weight_q for row in sources)
    if not math.isclose(total_weight, 1.0, abs_tol=PROBABILITY_TOLERANCE):
        raise R6CalibrationError("source representation weights must sum to one")

    grouped: dict[tuple[str, str, str], list[CohortSource]] = defaultdict(list)
    for source in sources:
        grouped[_cell(source)].append(source)
    counts = _cell_counts(sources, cohort_size)
    selected = tuple(
        source
        for cell in sorted(grouped)
        for source in _systematic_cell_draws(tuple(grouped[cell]), counts[cell])
    )
    if len(selected) != cohort_size:
        raise R6CalibrationError("fixed cohort size does not match its contract")
    probability = 1.0 / cohort_size
    return tuple(
        CohortMember(
            cohort_agent_id=f"r6_{index:04d}",
            source_observation_id=source.observation_id,
            source_row=source.source_row,
            source_representation_weight_q=source.representation_weight_q,
            cohort_probability_q=probability,
            runtime_representation_weight=1.0,
            common_region=source.common_region,
            farm_size=source.farm_size,
            land_role=source.land_role,
            operated_area_mu=source.operated_area_mu,
        )
        for index, source in enumerate(selected, 1)
    )


def smm_objective(rows: tuple[MomentResidual, ...]) -> float:
    if not rows:
        raise R6CalibrationError("SMM objective requires at least one moment")
    if any(row.standardization <= 0.0 for row in rows):
        raise R6CalibrationError("moment standardization must be positive")
    return math.fsum(
        ((row.simulated_value - row.target_value) / row.standardization) ** 2
        for row in rows
    )
