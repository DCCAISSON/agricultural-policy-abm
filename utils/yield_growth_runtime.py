from __future__ import annotations

from typing import Any, Mapping


def is_grain_crop(crop_type: Any) -> bool:
    crop_name = str(getattr(crop_type, "value", crop_type)).lower()
    return crop_name == "grain"


def resolve_crop_annual_yield_growth(
    tech_progress: Mapping[str, Any] | None,
    crop_type: Any,
    *,
    default: float = 0.015,
) -> float:
    """Return the explicit exogenous yield-trend multiplier rate.

    The model has a separate endogenous technology channel:
    ``annual_tfp_growth`` raises the technology frontier and
    ``yield_factors.technology.max_improvement`` translates adopted
    technology into yield. The crop-specific annual yield trend is therefore
    opt-in, so it cannot silently double-count the same technology progress.
    """
    if not isinstance(tech_progress, Mapping):
        return 0.0
    if not bool(tech_progress.get("enabled", True)):
        return 0.0
    if not bool(tech_progress.get("apply_exogenous_yield_trend_multiplier", False)):
        return 0.0

    growth_key = (
        "grain_annual_yield_growth_rate"
        if is_grain_crop(crop_type)
        else "cash_crop_annual_yield_growth_rate"
    )
    raw_value = tech_progress.get(
        growth_key,
        tech_progress.get("annual_yield_growth_rate", default),
    )
    try:
        return float(raw_value or 0.0)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "is_grain_crop",
    "resolve_crop_annual_yield_growth",
]
