from __future__ import annotations

from typing import Any, Dict, Optional

DEFAULT_BASE_NON_FARM_WAGE = 190.0


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_base_non_farm_wage(
    config: Dict[str, Any],
    default: float = DEFAULT_BASE_NON_FARM_WAGE,
) -> float:
    """
    Resolve the canonical non-farm base wage.

    `labor_market.base_non_farm_wage` is treated as the single source of truth,
    while `environment.market.base_non_farm_wage` is kept as a synchronized alias
    for legacy modules that still read it.
    """
    labor_market_cfg = (config or {}).get("labor_market", {}) or {}
    env_market_cfg = ((config or {}).get("environment", {}) or {}).get("market", {}) or {}

    labor_value = _safe_float(labor_market_cfg.get("base_non_farm_wage"))
    if labor_value is not None:
        return labor_value

    env_value = _safe_float(env_market_cfg.get("base_non_farm_wage"))
    if env_value is not None:
        return env_value

    return float(default)


def sync_base_non_farm_wage(
    config: Dict[str, Any],
    default: float = DEFAULT_BASE_NON_FARM_WAGE,
) -> float:
    """Synchronize legacy wage config aliases to one canonical value."""
    if config is None:
        return float(default)

    if "labor_market" not in config or config["labor_market"] is None:
        config["labor_market"] = {}
    if "environment" not in config or config["environment"] is None:
        config["environment"] = {}
    if "market" not in config["environment"] or config["environment"]["market"] is None:
        config["environment"]["market"] = {}

    canonical = resolve_base_non_farm_wage(config, default=default)
    config["labor_market"]["base_non_farm_wage"] = canonical
    config["environment"]["market"]["base_non_farm_wage"] = canonical
    return canonical
