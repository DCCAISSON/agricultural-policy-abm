# -*- coding: utf-8 -*-
"""Canonical farmer income helpers shared across model, diagnostics, and reporting."""

from __future__ import annotations

from typing import Any


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def get_farmer_gross_income(farmer: Any, default: float = 0.0) -> float:
    gross_income = _coerce_float(getattr(farmer, "gross_income", None))
    if gross_income is not None:
        return gross_income

    total_income = _coerce_float(getattr(farmer, "total_income", None))
    if total_income is not None:
        return total_income

    return float(default)


def get_farmer_net_income(farmer: Any, default: float = 0.0) -> float:
    cash_net_income = _coerce_float(getattr(farmer, "cash_net_income", None))
    if cash_net_income is not None:
        return cash_net_income

    net_income = _coerce_float(getattr(farmer, "net_income", None))
    if net_income is not None:
        return net_income
    return get_farmer_gross_income(farmer, default=default)


def get_farmer_economic_profit(farmer: Any, default: float = 0.0) -> float:
    economic_profit = _coerce_float(getattr(farmer, "economic_profit", None))
    if economic_profit is not None:
        return economic_profit
    return get_farmer_net_income(farmer, default=default)


def get_farmer_income_signal(farmer: Any, default: float = 0.0) -> float:
    """Decision-facing welfare signal: prefer net income, fall back to gross."""
    return get_farmer_net_income(farmer, default=default)


def get_farmer_off_farm_income(farmer: Any, default: float = 0.0) -> float:
    off_farm_income = _coerce_float(getattr(farmer, "off_farm_income", None))
    if off_farm_income is not None:
        return off_farm_income
    return float(default)


def get_farmer_agriculture_linked_income(farmer: Any) -> float:
    if hasattr(farmer, "_get_agriculture_linked_income"):
        try:
            return max(0.0, float(farmer._get_agriculture_linked_income()))
        except (TypeError, ValueError):
            pass

    agricultural_income = _coerce_float(getattr(farmer, "agricultural_income", None)) or 0.0
    subsidy_income = _coerce_float(getattr(farmer, "subsidy_income", None)) or 0.0
    agricultural_wage_income = _coerce_float(getattr(farmer, "labor_sale_income", None)) or 0.0
    return max(0.0, agricultural_income + subsidy_income + agricultural_wage_income)


def get_farmer_planting_income_for_validation(farmer: Any, default: float = 0.0) -> float:
    """
    NFPS planting income is closer to a net crop-income concept than to raw
    crop sales revenue. Use a dedicated comparable field when present.
    """
    planting_net_income = _coerce_float(getattr(farmer, "planting_net_income", None))
    if planting_net_income is not None:
        return max(0.0, planting_net_income)

    agricultural_income = _coerce_float(getattr(farmer, "agricultural_income", None)) or 0.0
    cash_production_cost = _coerce_float(getattr(farmer, "cash_production_cost", None))
    if cash_production_cost is None:
        cash_production_cost = _coerce_float(getattr(farmer, "production_cost", None)) or 0.0
    service_fee_expense = _coerce_float(getattr(farmer, "service_fee_expense", None)) or 0.0

    return max(0.0, agricultural_income - cash_production_cost - service_fee_expense)


def get_farmer_livelihood_agriculture_ratio(farmer: Any, default: float = 0.5) -> float:
    if hasattr(farmer, "_get_livelihood_agricultural_income_ratio"):
        try:
            return max(0.0, min(1.0, float(farmer._get_livelihood_agricultural_income_ratio())))
        except (TypeError, ValueError):
            pass

    agriculture_linked_income = get_farmer_agriculture_linked_income(farmer)
    total_income = _coerce_float(getattr(farmer, "gross_income", None))
    if total_income is None:
        total_income = _coerce_float(getattr(farmer, "total_income", None))
    if total_income is None:
        total_income = _coerce_float(getattr(farmer, "income_total", None))

    if total_income is None or total_income <= 0:
        off_farm_income = max(0.0, get_farmer_off_farm_income(farmer, default=0.0))
        property_income = max(0.0, _coerce_float(getattr(farmer, "land_rental_income", None)) or 0.0)
        transfer_income = max(0.0, _coerce_float(getattr(farmer, "transfer_income", None)) or 0.0)
        non_farm_business_income = max(
            0.0,
            _coerce_float(getattr(farmer, "non_farm_business_income", None)) or 0.0,
        )
        total_income = (
            agriculture_linked_income
            + off_farm_income
            + property_income
            + transfer_income
            + non_farm_business_income
        )

    if total_income > 0:
        return max(0.0, min(1.0, agriculture_linked_income / total_income))
    if agriculture_linked_income > 0:
        return 1.0
    non_ag_income = (
        max(0.0, get_farmer_off_farm_income(farmer, default=0.0))
        + max(0.0, _coerce_float(getattr(farmer, "land_rental_income", None)) or 0.0)
        + max(0.0, _coerce_float(getattr(farmer, "transfer_income", None)) or 0.0)
        + max(0.0, _coerce_float(getattr(farmer, "non_farm_business_income", None)) or 0.0)
    )
    if non_ag_income > 0:
        return 0.0
    return float(default)


def get_farmer_operated_land_area(farmer: Any) -> float:
    if hasattr(farmer, "get_operated_land_area"):
        try:
            return float(farmer.get_operated_land_area())
        except (TypeError, ValueError):
            pass

    if hasattr(farmer, "land_patches") and getattr(farmer, "land_patches", None):
        try:
            return float(sum(getattr(patch, "area", 0.0) for patch in farmer.land_patches))
        except TypeError:
            pass

    for attr_name in ("operated_land_area", "land_area", "land_area_mu"):
        area_value = _coerce_float(getattr(farmer, attr_name, None))
        if area_value is not None:
            return area_value

    return 0.0


def get_farmer_livelihood_strategy_name(farmer: Any, default: str = "Unknown") -> str:
    strategy = getattr(getattr(farmer, "livelihood_strategy", None), "value", None)
    if strategy is None:
        strategy = getattr(farmer, "livelihood_strategy", default)
    return str(strategy or default)


def is_active_farming_household(
    farmer: Any,
    min_operated_area: float = 1.0,
    min_agriculture_income: float = 1.0,
) -> bool:
    operated_area = get_farmer_operated_land_area(farmer)
    agriculture_linked_income = get_farmer_agriculture_linked_income(farmer)
    return (
        operated_area >= float(min_operated_area or 0.0)
        or agriculture_linked_income >= float(min_agriculture_income or 0.0)
    )


def is_pure_farmer(
    farmer: Any,
    max_off_farm_income_ratio: float = 0.2,
    min_operated_area: float = 1.0,
    min_agriculture_income: float = 1.0,
) -> bool:
    if not is_active_farming_household(
        farmer,
        min_operated_area=min_operated_area,
        min_agriculture_income=min_agriculture_income,
    ):
        return False

    strategy_name = get_farmer_livelihood_strategy_name(farmer, default="")
    if strategy_name:
        return strategy_name == "Pure_Farmer"

    agriculture_ratio = get_farmer_livelihood_agriculture_ratio(farmer, default=0.5)
    return (1.0 - agriculture_ratio) < float(max_off_farm_income_ratio or 0.0)


def is_large_operator(farmer: Any, min_operated_area: float = 30.0) -> bool:
    return get_farmer_operated_land_area(farmer) >= float(min_operated_area or 0.0)


def is_professional_large_operator(
    farmer: Any,
    min_operated_area: float = 30.0,
    min_agricultural_income_ratio: float = 0.78,
    max_off_farm_income: float = 12000.0,
) -> bool:
    if not is_large_operator(farmer, min_operated_area=min_operated_area):
        return False

    strategy_name = get_farmer_livelihood_strategy_name(farmer)
    if strategy_name == "Non_Farmer":
        return False

    off_farm_income = max(0.0, get_farmer_off_farm_income(farmer, default=0.0))
    agriculture_linked_income = get_farmer_agriculture_linked_income(farmer)
    if agriculture_linked_income > 0.0 or off_farm_income > 0.0:
        agri_dependence = get_farmer_livelihood_agriculture_ratio(farmer, default=0.5)
    else:
        agri_dependence = getattr(getattr(farmer, "characteristics", None), "agricultural_dependence", None)
        if agri_dependence is None:
            agri_dependence = get_farmer_livelihood_agriculture_ratio(farmer, default=0.5)

    return (
        float(agri_dependence) >= float(min_agricultural_income_ratio or 0.0)
        and off_farm_income <= float(max_off_farm_income or 0.0)
    )
