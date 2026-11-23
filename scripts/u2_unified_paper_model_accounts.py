from __future__ import annotations

from typing import Any

from core.model import MainModel


def classify_household_action(action: str) -> str:
    return {
        "maintain_grain": "SelfFarm",
        "rent_out": "RentOut",
        "expand_cash": "Expand",
        "contract": "ShrinkOrExit",
        "exit": "ShrinkOrExit",
    }.get(action, "Unclassified")


def classify_professional_action(action: str) -> str:
    return {
        "maintain_grain": "Maintain",
        "expand_cash": "Expand",
        "contract": "Shrink",
        "exit": "ExitLocalOperation",
    }.get(action, "Unclassified")


def mass_by_action(
    trace_rows: list[dict[str, Any]],
    weights: dict[str, float],
    operator_type: str,
) -> dict[str, float]:
    result = {
        "SelfFarm": 0.0,
        "RentOut": 0.0,
        "Expand": 0.0,
        "ShrinkOrExit": 0.0,
        "Maintain": 0.0,
        "Shrink": 0.0,
        "ExitLocalOperation": 0.0,
    }
    classifier = (
        classify_household_action
        if operator_type == "household"
        else classify_professional_action
    )
    for row in trace_rows:
        if row.get("operator_type") != operator_type:
            continue
        paper_action = classifier(str(row.get("action", "")))
        if paper_action in result:
            result[paper_action] += weights.get(str(row.get("operator_id", "")), 1.0)
    return result


def land_stats(model: MainModel, year: int) -> dict[str, Any]:
    population_manager = getattr(model, "population_manager", None)
    history = getattr(population_manager, "annual_stats", []) or []
    for row in reversed(history):
        if int(row.get("year", -1)) == year:
            weighted = row.get("weighted_land_allocation") or {}
            if isinstance(weighted, dict):
                return weighted
    return {
        "year": year,
        "area_unit": "mu",
        "released_area_mu": 0.0,
        "eligible_weighted_demand_mu": 0.0,
        "allocated_area_mu": 0.0,
        "unabsorbed_area_mu": 0.0,
        "allocation_trace": [],
    }


def allocation_hhi(allocation_rows: list[dict[str, Any]]) -> float:
    total = sum(
        float(row.get("absorbed_area_mu", 0.0) or 0.0) for row in allocation_rows
    )
    if total <= 0.0:
        return 0.0
    by_recipient: dict[str, float] = {}
    for row in allocation_rows:
        recipient = str(row.get("recipient_id", ""))
        by_recipient[recipient] = by_recipient.get(recipient, 0.0) + float(
            row.get("absorbed_area_mu", 0.0) or 0.0
        )
    return sum((area / total) ** 2 for area in by_recipient.values())
