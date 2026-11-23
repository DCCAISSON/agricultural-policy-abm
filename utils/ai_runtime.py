from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, MutableMapping


AI_MECHANISMS = (
    "information_governance",
    "service_coordination",
    "productivity_shock",
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _ai_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _as_mapping((config or {}).get("ai", {}))


def mechanism_enabled(config: Mapping[str, Any] | None, mechanism: str) -> bool:
    """Return whether a named AI mechanism is explicitly active.

    The old ``ai.enabled`` flag is treated only as a legacy fallback for
    ``productivity_shock`` when the new namespace is absent. This preserves old
    experiments without letting the main policy baseline silently include every
    possible AI interpretation.
    """

    ai_cfg = _ai_config(config)
    mechanism_cfg = _as_mapping(ai_cfg.get(mechanism, {}))
    if "enabled" in mechanism_cfg:
        return _as_bool(mechanism_cfg.get("enabled"))
    if mechanism == "productivity_shock" and mechanism not in ai_cfg:
        return _as_bool(ai_cfg.get("enabled", False))
    return False


def productivity_params(config: Mapping[str, Any] | None) -> dict[str, float]:
    ai_cfg = _ai_config(config)
    legacy_enterprise = _as_mapping(ai_cfg.get("enterprise", {}))
    legacy_farmer = _as_mapping(ai_cfg.get("farmer", {}))
    legacy_labor = _as_mapping(ai_cfg.get("labor_market", {}))
    modern = _as_mapping(ai_cfg.get("productivity_shock", {}))
    return {
        "tfp_boost": _as_float(modern.get("tfp_boost", legacy_enterprise.get("tfp_boost", 0.0))),
        "labor_substitution": _as_float(
            modern.get("labor_substitution", legacy_enterprise.get("labor_substitution", 0.0))
        ),
        "input_saving": _as_float(modern.get("input_saving", legacy_enterprise.get("input_saving", 0.0))),
        "non_farm_job_shock": _as_float(
            modern.get("non_farm_job_shock", legacy_farmer.get("non_farm_job_shock", 0.0))
        ),
        "wage_penalty": _as_float(modern.get("wage_penalty", legacy_farmer.get("wage_penalty", 0.0))),
        "adoption_lag": _as_float(modern.get("adoption_lag", legacy_farmer.get("adoption_lag", 0.0))),
        "non_farm_demand_penalty": _as_float(
            modern.get("non_farm_demand_penalty", legacy_labor.get("non_farm_demand_penalty", 0.0))
        ),
        "wage_floor": _as_float(modern.get("wage_floor", legacy_labor.get("wage_floor", 0.50)), 0.50),
    }


def information_governance_params(config: Mapping[str, Any] | None) -> dict[str, float]:
    cfg = _as_mapping(_ai_config(config).get("information_governance", {}))
    return {
        "damage_detection_accuracy": _as_float(cfg.get("damage_detection_accuracy", 0.0)),
        "misallocation_detection_rate": _as_float(cfg.get("misallocation_detection_rate", 0.0)),
        "duplicate_subsidy_detection_rate": _as_float(cfg.get("duplicate_subsidy_detection_rate", 0.0)),
        "leakage_reduction_per_detection": _as_float(cfg.get("leakage_reduction_per_detection", 0.35)),
    }


def service_coordination_params(config: Mapping[str, Any] | None) -> dict[str, float]:
    cfg = _as_mapping(_ai_config(config).get("service_coordination", {}))
    dispatch_efficiency = _as_float(
        cfg.get("service_dispatch_efficiency", cfg.get("service_matching_boost", 0.0))
    )
    return {
        "service_dispatch_efficiency": dispatch_efficiency,
        "service_matching_boost": dispatch_efficiency,
        "service_radius_boost": _as_float(cfg.get("service_radius_boost", 0.0)),
        "service_cost_reduction": _as_float(cfg.get("service_cost_reduction", 0.0)),
    }


def set_ai_mechanism(
    config: MutableMapping[str, Any],
    mechanism: str,
    enabled: bool = True,
    *,
    exclusive: bool = True,
) -> None:
    """Set one AI mechanism switch in a mutable runtime config."""

    if mechanism not in (*AI_MECHANISMS, "none"):
        raise ValueError(f"Unknown AI mechanism: {mechanism}")
    ai_cfg = config.setdefault("ai", {})
    ai_cfg["enabled"] = False
    if exclusive:
        for key in AI_MECHANISMS:
            ai_cfg.setdefault(key, {})["enabled"] = False
    if mechanism == "none":
        return
    ai_cfg.setdefault(mechanism, {})["enabled"] = bool(enabled)


def set_legacy_ai_enabled(config: MutableMapping[str, Any], enabled: bool) -> None:
    """Compatibility shim for historical ``ai_enabled`` treatment keys.

    ``True`` maps to the old runtime effect, now named ``productivity_shock``.
    The information-governance and service-coordination mechanisms must be
    requested explicitly.
    """

    set_ai_mechanism(config, "productivity_shock" if enabled else "none", enabled=bool(enabled))


def describe_ai_runtime(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "active_mechanisms": [key for key in AI_MECHANISMS if mechanism_enabled(config, key)],
        "information_governance": {
            "enabled": mechanism_enabled(config, "information_governance"),
            **information_governance_params(config),
        },
        "service_coordination": {
            "enabled": mechanism_enabled(config, "service_coordination"),
            **service_coordination_params(config),
        },
        "productivity_shock": {
            "enabled": mechanism_enabled(config, "productivity_shock"),
            **productivity_params(config),
        },
    }


def config_with_ai_mechanism(
    config: Mapping[str, Any],
    mechanism: str,
    enabled: bool = True,
    *,
    exclusive: bool = True,
) -> dict[str, Any]:
    cfg = deepcopy(dict(config))
    set_ai_mechanism(cfg, mechanism, enabled=enabled, exclusive=exclusive)
    return cfg
