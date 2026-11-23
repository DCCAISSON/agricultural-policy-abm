from __future__ import annotations

import json
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from canonical.constants.policy_constants import (
    DEFAULT_SCALE_SUBSIDY_BASE_RATE,
    NO_POLICY_ANNUAL_SUBSIDY_BUDGET,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_LIBRARY_PATH = PROJECT_ROOT / "config" / "policy_params.json"


class ConfigWarning(UserWarning):
    """Warning for missing runtime config values filled by canonical defaults."""


def load_policy_library(policy_library_path: Optional[Path] = None) -> Dict[str, Any]:
    policy_path = Path(policy_library_path) if policy_library_path else DEFAULT_POLICY_LIBRARY_PATH
    with open(policy_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_policy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    policy_cfg = config.setdefault("policy", {})
    policy_cfg.setdefault("active_scenario", "baseline_cash")
    policy_cfg.setdefault("overrides", {})
    if policy_cfg["overrides"] is None:
        policy_cfg["overrides"] = {}
    return policy_cfg


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Return base updated by overlay without mutating either input."""
    merged = deepcopy(base or {})
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_grain_subsidy_aliases(
    policy_state: Dict[str, Any],
    *,
    context: str = "policy_state",
) -> Dict[str, Any]:
    """Normalize grain subsidy naming.

    ``grain_subsidies_rate`` is the canonical runtime key. ``subsidy_rate`` is
    retained only as a legacy alias for old configs and downstream code. If both
    are present and differ, canonical wins and a ConfigWarning is emitted.
    """
    resolved = deepcopy(policy_state or {})
    canonical_present = "grain_subsidies_rate" in resolved
    legacy_present = "subsidy_rate" in resolved

    if canonical_present and legacy_present:
        canonical_value = _as_optional_float(resolved.get("grain_subsidies_rate"))
        legacy_value = _as_optional_float(resolved.get("subsidy_rate"))
        if (
            canonical_value is not None
            and legacy_value is not None
            and abs(canonical_value - legacy_value) > 1e-12
        ):
            warnings.warn(
                (
                    f"{context} defines both grain_subsidies_rate={canonical_value} "
                    f"and legacy subsidy_rate={legacy_value}; using grain_subsidies_rate."
                ),
                ConfigWarning,
                stacklevel=2,
            )
        resolved["subsidy_rate"] = resolved.get("grain_subsidies_rate")
    elif canonical_present:
        resolved["subsidy_rate"] = resolved.get("grain_subsidies_rate")
    elif legacy_present:
        warnings.warn(
            (
                f"{context} uses legacy subsidy_rate without grain_subsidies_rate; "
                "copying it to canonical grain_subsidies_rate."
            ),
            ConfigWarning,
            stacklevel=2,
        )
        resolved["grain_subsidies_rate"] = resolved.get("subsidy_rate")

    return resolved


def resolve_fiscal_system_config(
    config: Dict[str, Any],
    *,
    create: bool = False,
    sync: bool = False,
) -> Dict[str, Any]:
    """Resolve fiscal config across root and legacy agents.fiscal_system paths.

    The root ``fiscal_system`` path is the canonical path. Existing project
    configs often store the same block under ``agents.fiscal_system``. This
    resolver merges both paths with root values taking precedence. When
    ``create`` or ``sync`` is true, the merged config is mirrored to both paths.
    """
    agents_cfg = config.setdefault("agents", {}) if create else (config.get("agents", {}) or {})
    legacy_cfg = agents_cfg.setdefault("fiscal_system", {}) if create else (agents_cfg.get("fiscal_system", {}) or {})
    root_cfg = config.setdefault("fiscal_system", {}) if create else (config.get("fiscal_system", {}) or {})

    merged = _deep_merge(legacy_cfg, root_cfg)

    if create or sync:
        root_target = config.setdefault("fiscal_system", {})
        legacy_target = config.setdefault("agents", {}).setdefault("fiscal_system", {})
        root_target.clear()
        root_target.update(deepcopy(merged))
        legacy_target.clear()
        legacy_target.update(deepcopy(merged))
        return root_target

    return merged


def _sync_fiscal_system_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return resolve_fiscal_system_config(config, create=True, sync=True)


def _get_scale_subsidy_config_readonly(config: Dict[str, Any]) -> Dict[str, Any]:
    fiscal_cfg = resolve_fiscal_system_config(config, create=False)
    return (
        fiscal_cfg.get("budget_allocation", {})
        .get("scale_subsidy", {})
        or {}
    )


def _set_scale_subsidy(
    config: Dict[str, Any],
    *,
    enabled: bool,
    base_rate: float,
) -> Dict[str, Any]:
    fiscal_cfg = resolve_fiscal_system_config(config, create=True)
    budget_cfg = fiscal_cfg.setdefault("budget_allocation", {})
    scale_cfg = budget_cfg.setdefault("scale_subsidy", {})
    scale_cfg["enabled"] = bool(enabled)
    scale_cfg["base_rate"] = float(base_rate)
    synced_fiscal_cfg = _sync_fiscal_system_config(config)
    return synced_fiscal_cfg["budget_allocation"]["scale_subsidy"]


def _set_fiscal_budget(config: Dict[str, Any], annual_budget: float) -> None:
    fiscal_cfg = resolve_fiscal_system_config(config, create=True)
    fiscal_cfg["annual_subsidy_budget"] = float(annual_budget)
    _sync_fiscal_system_config(config)


def _set_grain_reserve_budget(config: Dict[str, Any], annual_budget: float) -> None:
    agents_cfg = config.setdefault("agents", {})
    agents_cfg.setdefault("grain_reserve_bureau", {})["annual_budget"] = float(annual_budget)
    agents_cfg.setdefault("grain_reserve", {})["annual_budget"] = float(annual_budget)
    config.setdefault("policy", {}).setdefault("grain_reserve", {})["annual_budget"] = float(annual_budget)


def _normalize_policy_state(scenario_id: Optional[str], policy_state: Dict[str, Any]) -> Dict[str, Any]:
    resolved = normalize_grain_subsidy_aliases(
        policy_state or {},
        context=f"policy scenario '{scenario_id or 'unknown'}'",
    )
    scenario_label = str(scenario_id or "").strip().lower()
    explicit_disable = resolved.get("disable_grain_intervention")
    if explicit_disable is None:
        try:
            floor_value = float(resolved.get("grain_price_floor", 0.0) or 0.0)
        except (TypeError, ValueError):
            floor_value = 0.0
        try:
            subsidy_rate = float(
                resolved.get(
                    "grain_subsidies_rate",
                    resolved.get("subsidy_rate", 0.0),
                ) or 0.0
            )
        except (TypeError, ValueError):
            subsidy_rate = 0.0
        disable_grain_intervention = (
            scenario_label in {"no_policy", "none", "market_only"}
            or (floor_value <= 0.0 and subsidy_rate <= 0.0)
        )
    else:
        disable_grain_intervention = bool(explicit_disable) or scenario_label in {
            "no_policy",
            "none",
            "market_only",
        }

    resolved["disable_grain_intervention"] = disable_grain_intervention
    if disable_grain_intervention:
        resolved["grain_price_floor"] = 0.0
        resolved["grain_subsidies_rate"] = 0.0
        resolved["subsidy_rate"] = 0.0
        resolved["base_subsidy_per_mu"] = 0.0
        if "annual_subsidy_budget" in resolved:
            resolved["annual_subsidy_budget"] = 0.0
    return resolved


def resolve_policy_state(
    config: Dict[str, Any],
    policy_library: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    policy_cfg = ensure_policy_config(config)
    scenario_id = policy_cfg.get("active_scenario") or "baseline_cash"
    library = policy_library or load_policy_library()
    scenarios = library.get("scenarios", {})
    if scenario_id not in scenarios:
        raise KeyError(f"Policy scenario '{scenario_id}' not found in policy_params.json")

    resolved = deepcopy(scenarios[scenario_id])
    resolved.update(deepcopy(policy_cfg.get("overrides") or {}))
    resolved = _normalize_policy_state(scenario_id, resolved)
    return scenario_id, resolved


def apply_policy_runtime(
    config: Dict[str, Any],
    scenario_id: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    merge_overrides: bool = False,
) -> Dict[str, Any]:
    policy_cfg = ensure_policy_config(config)
    if scenario_id is not None:
        policy_cfg["active_scenario"] = scenario_id
        gov_cfg = policy_cfg.setdefault("government", {})
        gov_cfg["scenario_id"] = scenario_id
        if str(scenario_id).strip().lower() == "no_policy":
            gov_cfg["enable_regional_heterogeneity"] = False

    if overrides is not None:
        if merge_overrides:
            merged = deepcopy(policy_cfg.get("overrides") or {})
            merged.update(deepcopy(overrides))
            policy_cfg["overrides"] = merged
        else:
            policy_cfg["overrides"] = deepcopy(overrides)

    _, resolved = resolve_policy_state(config)
    scenario_label = str(policy_cfg.get("active_scenario") or "").strip().lower()
    full_fiscal_shutdown = (
        scenario_label in {"no_policy", "none", "market_only"}
        or bool(resolved.get("disable_all_public_support", False))
        or bool(resolved.get("disable_all_fiscal_support", False))
        or bool(resolved.get("disable_all_subsidies", False))
    )

    # Keep root fiscal_system and legacy agents.fiscal_system synchronized.
    resolve_fiscal_system_config(config, create=True, sync=True)

    explicit_budget = _as_optional_float(resolved.get("annual_subsidy_budget"))
    if full_fiscal_shutdown:
        _set_fiscal_budget(config, NO_POLICY_ANNUAL_SUBSIDY_BUDGET)
        _set_scale_subsidy(config, enabled=False, base_rate=0.0)
        _set_grain_reserve_budget(config, NO_POLICY_ANNUAL_SUBSIDY_BUDGET)
        policy_cfg.setdefault("overrides", {})["annual_subsidy_budget"] = NO_POLICY_ANNUAL_SUBSIDY_BUDGET
        config.setdefault("government", {})["annual_subsidy_budget"] = NO_POLICY_ANNUAL_SUBSIDY_BUDGET
    elif explicit_budget is not None:
        _set_fiscal_budget(config, explicit_budget)

    return config


def clear_policy_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    ensure_policy_config(config)["overrides"] = {}
    return config


def ensure_scale_subsidy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    fiscal_cfg = resolve_fiscal_system_config(config, create=True)
    budget_cfg = fiscal_cfg.setdefault("budget_allocation", {})
    scale_cfg = budget_cfg.setdefault("scale_subsidy", {})
    scale_cfg.setdefault("enabled", True)
    if "base_rate" not in scale_cfg or scale_cfg.get("base_rate") is None:
        warnings.warn(
            (
                "Missing fiscal_system.budget_allocation.scale_subsidy.base_rate; "
                f"using canonical default {DEFAULT_SCALE_SUBSIDY_BASE_RATE}."
            ),
            ConfigWarning,
            stacklevel=2,
        )
        scale_cfg["base_rate"] = DEFAULT_SCALE_SUBSIDY_BASE_RATE
    synced_fiscal_cfg = _sync_fiscal_system_config(config)
    return synced_fiscal_cfg["budget_allocation"]["scale_subsidy"]


def apply_scale_subsidy_runtime(
    config: Dict[str, Any],
    *,
    enabled: Optional[bool] = None,
    base_rate: Optional[float] = None,
    base_rate_multiplier: Optional[float] = None,
) -> Dict[str, Any]:
    if base_rate is not None and base_rate_multiplier is not None:
        raise ValueError("Specify either base_rate or base_rate_multiplier, not both")

    scale_cfg = ensure_scale_subsidy_config(config)
    if enabled is not None:
        scale_cfg["enabled"] = bool(enabled)

    if base_rate is not None:
        scale_cfg["base_rate"] = float(base_rate)
    elif base_rate_multiplier is not None:
        scale_cfg["base_rate"] = float(scale_cfg.get("base_rate", 0.0)) * float(base_rate_multiplier)

    _sync_fiscal_system_config(config)
    return scale_cfg


def get_effective_grain_subsidy_per_mu(config: Dict[str, Any]) -> float:
    _, resolved = resolve_policy_state(config)
    return float(resolved.get("base_subsidy_per_mu", 0.0)) * float(resolved.get("grain_subsidies_rate", 0.0))


def apply_grain_subsidy_multiplier(
    config: Dict[str, Any],
    multiplier: float,
) -> Dict[str, Any]:
    scenario_id, resolved = resolve_policy_state(config)
    rate = float(resolved.get("grain_subsidies_rate", 0.0))
    base_subsidy = float(resolved.get("base_subsidy_per_mu", 0.0))

    overrides = deepcopy(ensure_policy_config(config).get("overrides") or {})
    overrides["grain_subsidies_rate"] = rate
    overrides["base_subsidy_per_mu"] = base_subsidy * float(multiplier)

    apply_policy_runtime(config, scenario_id=scenario_id, overrides=overrides)
    return resolve_policy_state(config)[1]


def set_effective_grain_subsidy_per_mu(
    config: Dict[str, Any],
    effective_subsidy_per_mu: float,
) -> Dict[str, Any]:
    scenario_id, resolved = resolve_policy_state(config)
    rate = float(resolved.get("grain_subsidies_rate", 0.0))
    target_value = float(effective_subsidy_per_mu)
    if target_value < 0:
        raise ValueError("effective_subsidy_per_mu must be non-negative")

    overrides = deepcopy(ensure_policy_config(config).get("overrides") or {})
    if target_value == 0:
        overrides["grain_subsidies_rate"] = rate
        overrides["base_subsidy_per_mu"] = 0.0
    else:
        if rate <= 0:
            raise ValueError(
                f"Cannot set an absolute grain subsidy under scenario '{scenario_id}' because grain_subsidies_rate={rate}"
            )
        overrides["grain_subsidies_rate"] = rate
        overrides["base_subsidy_per_mu"] = target_value / rate

    apply_policy_runtime(config, scenario_id=scenario_id, overrides=overrides)
    return resolve_policy_state(config)[1]


def snapshot_policy_runtime_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(config)
    scenario_id, resolved = resolve_policy_state(cfg)
    scale_cfg = _get_scale_subsidy_config_readonly(cfg)
    fiscal_cfg = resolve_fiscal_system_config(cfg, create=False)
    reserve_cfg = (
        cfg.get("agents", {})
        .get("grain_reserve_bureau", {})
        or cfg.get("agents", {}).get("grain_reserve", {})
        or {}
    )
    return {
        "active_scenario": scenario_id,
        "disable_grain_intervention": bool(resolved.get("disable_grain_intervention", False)),
        "grain_price_floor": float(resolved.get("grain_price_floor", 0.0) or 0.0),
        "grain_subsidies_rate": float(resolved.get("grain_subsidies_rate", 0.0)),
        "base_subsidy_per_mu": float(resolved.get("base_subsidy_per_mu", 0.0)),
        "effective_grain_subsidy_per_mu": (
            float(resolved.get("grain_subsidies_rate", 0.0)) * float(resolved.get("base_subsidy_per_mu", 0.0))
        ),
        "scale_subsidy_enabled": bool(scale_cfg.get("enabled", False)),
        "scale_subsidy_base_rate": float(scale_cfg.get("base_rate", 0.0)),
        "annual_subsidy_budget": float(fiscal_cfg.get("annual_subsidy_budget", 0.0) or 0.0),
        "grain_reserve_annual_budget": float(reserve_cfg.get("annual_budget", 0.0) or 0.0),
    }


def snapshot_policy_runtime_from_model(model: Any) -> Dict[str, Any]:
    env = getattr(model, "environment", None)
    if env is None:
        raise AttributeError("Model has no environment; cannot audit active policy state")

    active_policy = deepcopy(getattr(env, "active_policy", {}) or {})
    model_config = getattr(model, "config", {}) or {}
    fiscal_cfg = resolve_fiscal_system_config(model_config, create=False)
    scale_cfg = (
        fiscal_cfg
        .get("budget_allocation", {})
        .get("scale_subsidy", {})
        or {}
    )
    reserve_cfg = (
        model_config.get("agents", {})
        .get("grain_reserve_bureau", {})
        or model_config.get("agents", {}).get("grain_reserve", {})
        or {}
    )
    return {
        "active_scenario": getattr(env, "active_policy_id", None) or getattr(model, "active_policy_id", None),
        "disable_grain_intervention": bool(active_policy.get("disable_grain_intervention", False)),
        "grain_price_floor": float(active_policy.get("grain_price_floor", 0.0) or 0.0),
        "grain_subsidies_rate": float(active_policy.get("grain_subsidies_rate", 0.0)),
        "base_subsidy_per_mu": float(active_policy.get("base_subsidy_per_mu", 0.0)),
        "effective_grain_subsidy_per_mu": (
            float(active_policy.get("grain_subsidies_rate", 0.0))
            * float(active_policy.get("base_subsidy_per_mu", 0.0))
        ),
        "scale_subsidy_enabled": bool(scale_cfg.get("enabled", False)),
        "scale_subsidy_base_rate": float(scale_cfg.get("base_rate", 0.0)),
        "annual_subsidy_budget": float(fiscal_cfg.get("annual_subsidy_budget", 0.0) or 0.0),
        "grain_reserve_annual_budget": float(reserve_cfg.get("annual_budget", 0.0) or 0.0),
    }


def assert_policy_runtime_matches(
    model: Any,
    expected_snapshot: Dict[str, Any],
    *,
    tolerance: float = 1e-9,
) -> Dict[str, Any]:
    actual = snapshot_policy_runtime_from_model(model)
    mismatches = []

    default_keys = (
        "active_scenario",
        "disable_grain_intervention",
        "grain_price_floor",
        "grain_subsidies_rate",
        "base_subsidy_per_mu",
        "effective_grain_subsidy_per_mu",
        "scale_subsidy_enabled",
        "scale_subsidy_base_rate",
        "annual_subsidy_budget",
        "grain_reserve_annual_budget",
    )
    for key in tuple(k for k in default_keys if k in expected_snapshot):
        expected_value = expected_snapshot.get(key)
        actual_value = actual.get(key)
        if isinstance(expected_value, float):
            if abs(actual_value - expected_value) > tolerance:
                mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
        else:
            if actual_value != expected_value:
                mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")

    if mismatches:
        raise RuntimeError("Policy runtime injection mismatch detected: " + "; ".join(mismatches))

    return actual
