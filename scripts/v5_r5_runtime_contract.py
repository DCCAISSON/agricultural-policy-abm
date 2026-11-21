from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias


SeedYear: TypeAlias = tuple[int, int]


@dataclass(frozen=True, slots=True)
class AccountingValidationError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class WeightedRuntimeSummary:
    candidate_rows: int
    annual_rows: int
    max_formula_error_mu: float
    max_account_residual_mu: float


@dataclass(frozen=True, slots=True)
class WeightedRuntimeEvidence:
    candidates: Sequence[Mapping[str, str]]
    annual_rows: Sequence[Mapping[str, str]]
    m2_annual_by_key: Mapping[SeedYear, Mapping[str, str]]
    m2_trace: Sequence[Mapping[str, str]]
    expected_keys: frozenset[SeedYear]


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise AccountingValidationError(detail)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise AccountingValidationError(
            f"invalid numeric field {field!r}: {row}"
        ) from error
    _require(math.isfinite(value), f"non-finite numeric field {field!r}: {row}")
    return value


def _key(row: Mapping[str, str]) -> SeedYear:
    return int(_float(row, "runtime_seed")), int(_float(row, "year"))


def _tolerance(area_mu: float) -> float:
    return max(1e-9, 1e-10 * abs(area_mu))


def _same(actual: float, expected: float, area_mu: float) -> bool:
    return abs(actual - expected) <= _tolerance(area_mu)


def validate_final_use(row: Mapping[str, str]) -> None:
    crops = row.get("final_crop_labels", "").split("|")
    operators = row.get("final_operator_ids", "").split("|")
    _require(all(crop.strip() for crop in crops), "observed final use lacks crop")
    _require(len(crops) == len(operators), "final operator and crop counts differ")
    for crop, operator in zip(crops, operators, strict=True):
        if crop == "unoperated":
            _require(operator == "unknown", "unoperated final use has an operator")
        else:
            _require(
                bool(operator.strip()) and operator != "unknown",
                "observed final use lacks operator",
            )


def validate_weighted_runtime(
    evidence: WeightedRuntimeEvidence,
) -> WeightedRuntimeSummary:
    candidates = evidence.candidates
    expected_keys = evidence.expected_keys
    _require(bool(candidates), "weighted runtime candidate trace is empty")
    weighted_demand_by_key: dict[SeedYear, float] = {}
    candidate_keys: set[SeedYear] = set()
    max_formula_error = 0.0
    for row in candidates:
        key = _key(row)
        candidate_keys.add(key)
        _require(
            key in expected_keys,
            f"weighted candidate is outside declared seed-years: {key}",
        )
        _require(
            row.get("area_unit") == "mu",
            f"weighted candidate area unit is not mu: {key}",
        )
        _require(
            bool(row.get("operator_id", "").strip()),
            f"weighted candidate lacks operator: {key}",
        )
        weight = _float(row, "representation_weight_q")
        feasible = _float(row, "feasible_expansion_mu")
        weighted_capacity = _float(row, "weighted_capacity_mu")
        expected_return = _float(row, "expected_return_yuan_per_mu")
        _require(
            weight >= 0.0 and feasible >= 0.0, f"negative weighted demand input: {key}"
        )
        formula_error = abs(weighted_capacity - weight * feasible)
        max_formula_error = max(max_formula_error, formula_error)
        _require(
            _same(weighted_capacity, weight * feasible, weighted_capacity),
            f"weighted capacity formula fails: {key}",
        )
        eligible = row.get("eligible_bid") == "True"
        _require(
            row.get("eligible_bid") in {"True", "False"}
            and eligible == (expected_return > 0.0 and weighted_capacity > 0.0),
            f"weighted eligibility mismatch: {key}",
        )
        if eligible:
            weighted_demand_by_key[key] = (
                weighted_demand_by_key.get(key, 0.0) + weighted_capacity
            )
    allocated_by_calendar: dict[SeedYear, float] = {}
    for row in evidence.m2_trace:
        if not row.get("allocation_event_id", "").strip():
            continue
        key = int(_float(row, "runtime_seed")), int(_float(row, "allocation_year"))
        _require(
            key in expected_keys, f"allocation is outside declared seed-years: {key}"
        )
        allocated_by_calendar[key] = allocated_by_calendar.get(key, 0.0) + _float(
            row,
            "absorbed_area_mu",
        )

    available_by_calendar: dict[SeedYear, float] = {}
    ending_stock_by_calendar: dict[SeedYear, float] = {}
    for seed in sorted({key[0] for key in expected_keys}):
        stock = 0.0
        for key in sorted(key for key in expected_keys if key[0] == seed):
            stock += _float(evidence.m2_annual_by_key[key], "released_area_mu")
            available_by_calendar[key] = stock
            allocated = allocated_by_calendar.get(key, 0.0)
            _require(
                allocated <= stock + _tolerance(stock),
                f"calendar allocation exceeds available stock: {key}",
            )
            stock -= allocated
            ending_stock_by_calendar[key] = stock
    required_allocation_keys = {
        key
        for key, available in available_by_calendar.items()
        if available > _tolerance(0.0)
    }
    _require(
        required_allocation_keys <= candidate_keys,
        "weighted candidate trace does not cover positive-release seed-years",
    )

    annual_by_key = {_key(row): row for row in evidence.annual_rows}
    _require(
        len(annual_by_key) == len(evidence.annual_rows),
        "weighted annual account contains duplicate seed-year rows",
    )
    _require(
        set(annual_by_key) == candidate_keys,
        "weighted annual account does not match candidate-trace seed-years",
    )
    max_account_residual = 0.0
    for key, row in annual_by_key.items():
        _require(
            row.get("area_unit") == "mu", f"weighted annual area unit is not mu: {key}"
        )
        released = _float(row, "released_area_mu")
        allocated = _float(row, "allocated_area_mu")
        unabsorbed = _float(row, "unabsorbed_area_mu")
        eligible_demand = _float(row, "eligible_weighted_demand_mu")
        farmers = _float(row, "allocated_to_farmers_mu")
        enterprises = _float(row, "allocated_to_enterprises_mu")
        new_entrants = _float(row, "allocated_to_new_entrants_mu")
        _require(
            _same(released, available_by_calendar[key], released),
            f"weighted released area mismatch: {key}",
        )
        _require(
            _same(allocated, allocated_by_calendar.get(key, 0.0), released),
            f"weighted allocated area mismatch: {key}",
        )
        _require(
            _same(unabsorbed, ending_stock_by_calendar[key], released),
            f"weighted unabsorbed area mismatch: {key}",
        )
        _require(
            _same(
                eligible_demand, weighted_demand_by_key.get(key, 0.0), eligible_demand
            ),
            f"eligible weighted demand mismatch: {key}",
        )
        _require(
            allocated <= eligible_demand + _tolerance(eligible_demand),
            f"allocation exceeds eligible weighted demand: {key}",
        )
        _require(
            _same(allocated, farmers + enterprises, released),
            f"recipient-type allocation mismatch: {key}",
        )
        _require(
            new_entrants <= farmers + _tolerance(farmers),
            f"new-entrant allocation exceeds farmer allocation: {key}",
        )
        residual = released - allocated - unabsorbed
        max_account_residual = max(max_account_residual, abs(residual))
        _require(
            abs(residual) <= _tolerance(released),
            f"weighted annual identity fails: {key}",
        )

    return WeightedRuntimeSummary(
        candidate_rows=len(candidates),
        annual_rows=len(evidence.annual_rows),
        max_formula_error_mu=max_formula_error,
        max_account_residual_mu=max_account_residual,
    )
