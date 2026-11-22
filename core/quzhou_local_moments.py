from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from core.quzhou_overlay_v0 import PROJECT_ROOT, QUZHOU_LOCAL_CASE_ID, QuzhouOverlayError

DEFAULT_QUZHOU_V1_LOCAL_MOMENTS_PATH = PROJECT_ROOT / "data" / "quzhou" / "quzhou_v1_local_moments.yaml"
DEFAULT_QUZHOU_V1_LOCAL_MOMENTS_RELATIVE_PATH = "data/quzhou/quzhou_v1_local_moments.yaml"
LOCAL_QUZHOU_ONLY = "local_quzhou_only"
LOCAL_MOMENTS_ONLY_NOT_ACTIVE = "local_moments_only_not_active"
CONFIG_LEVEL_SMALL_SMOKE = "config_level_small_smoke"
QUZHOU_V1_ACTIVATE_MECHANISMS = (
    "order_grain_purchase_price",
    "order_grain_extra_subsidy",
    "rice_scale_subsidy",
    "land_rent_average",
    "early_rice_yield",
    "early_rice_cost",
    "socialized_service_fee",
    "insurance_claim_loss",
)
QUZHOU_V1_FORBIDDEN_NON_ACTIVATE_MECHANISMS = (
    "outside_bidder_rent",
    "short_contract_duration",
    "service_capacity",
    "disaster_area",
    "village_collective_redistribution",
    "digital_monitoring",
    "farmer_household_heterogeneity",
    "enterprise_service_provider_finance",
    "agricultural_wage",
    "non_farm_wage",
    "yield_linked_subsidy",
    "AESI_climate_sustainability",
    "PPO_MARL_learning",
    "government_q_extended",
    "full_all_agent_MARL",
    "national_forecasting",
    "full_general_equilibrium",
)


def _validate_quzhou_v1_local_moments(local_moments: Mapping[str, Any]) -> None:
    metadata = local_moments.get("metadata", {}) if isinstance(local_moments, Mapping) else {}
    if not isinstance(metadata, Mapping):
        raise QuzhouOverlayError("Quzhou v1 local moments metadata must be a mapping.")

    status = metadata.get("status")
    if status != LOCAL_MOMENTS_ONLY_NOT_ACTIVE:
        raise QuzhouOverlayError(
            f"Quzhou v1 local moments status must be {LOCAL_MOMENTS_ONLY_NOT_ACTIVE!r}; got {status!r}."
        )

    if metadata.get("evidence_boundary") != LOCAL_QUZHOU_ONLY:
        raise QuzhouOverlayError(
            f"Quzhou v1 local moments must declare evidence_boundary: {LOCAL_QUZHOU_ONLY!r}."
        )

    if metadata.get("source_scope") != "activate_in_v1_only":
        raise QuzhouOverlayError("Quzhou v1 local moments must declare source_scope: activate_in_v1_only.")

    if metadata.get("national_config_mutation_allowed") is not False:
        raise QuzhouOverlayError("Quzhou v1 local moments must declare national_config_mutation_allowed: false.")

    if metadata.get("policy_params_mutation_allowed") is not False:
        raise QuzhouOverlayError("Quzhou v1 local moments must declare policy_params_mutation_allowed: false.")

    if metadata.get("raw_data_committed") is not False:
        raise QuzhouOverlayError("Quzhou v1 local moments must declare raw_data_committed: false.")

    activate_mechanisms = local_moments.get("activate_in_v1_mechanisms")
    if tuple(activate_mechanisms or ()) != QUZHOU_V1_ACTIVATE_MECHANISMS:
        raise QuzhouOverlayError("Quzhou v1 local moments must list exactly the activate_in_v1 mechanisms.")

    moments = local_moments.get("moments", {})
    if not isinstance(moments, Mapping):
        raise QuzhouOverlayError("Quzhou v1 local moments moments must be a mapping.")

    moment_keys = set(moments.keys())
    expected_keys = set(QUZHOU_V1_ACTIVATE_MECHANISMS)
    if moment_keys != expected_keys:
        missing = sorted(expected_keys - moment_keys)
        extra = sorted(moment_keys - expected_keys)
        raise QuzhouOverlayError(
            f"Quzhou v1 local moments key mismatch; missing={missing}, extra={extra}."
        )

    forbidden = sorted(moment_keys.intersection(QUZHOU_V1_FORBIDDEN_NON_ACTIVATE_MECHANISMS))
    if forbidden:
        raise QuzhouOverlayError(f"Quzhou v1 local moments contain non-activate mechanisms: {forbidden}.")


def load_quzhou_v1_local_moments(path: str | Path | None = None) -> dict[str, Any]:
    """Load validated Quzhou v1 local moments without activating national config values."""
    moments_path = Path(path) if path is not None else DEFAULT_QUZHOU_V1_LOCAL_MOMENTS_PATH
    with moments_path.open("r", encoding="utf-8") as handle:
        local_moments = yaml.safe_load(handle) or {}

    if not isinstance(local_moments, dict):
        raise QuzhouOverlayError(f"Quzhou v1 local moments must be a mapping: {moments_path}")

    _validate_quzhou_v1_local_moments(local_moments)
    return local_moments


def extract_quzhou_v1_local_moments_payload(local_moments: Mapping[str, Any]) -> dict[str, Any]:
    """Return a local-only Quzhou v1 moments payload for copied-config smoke."""
    _validate_quzhou_v1_local_moments(local_moments)
    return {
        "metadata": deepcopy(dict(local_moments.get("metadata", {}))),
        "cleaned_tables": deepcopy(dict(local_moments.get("cleaned_tables", {}))),
        "activate_in_v1_mechanisms": list(QUZHOU_V1_ACTIVATE_MECHANISMS),
        "moments": deepcopy(dict(local_moments.get("moments", {}))),
    }


def activate_quzhou_v1_local_moments_for_local_case(
    national_config: Mapping[str, Any],
    local_moments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach Quzhou v1 local moments to a copied config for config-level small smoke."""
    candidate_moments = deepcopy(local_moments if local_moments is not None else load_quzhou_v1_local_moments())
    _validate_quzhou_v1_local_moments(candidate_moments)

    copied_config = deepcopy(dict(national_config or {}))
    payload = extract_quzhou_v1_local_moments_payload(candidate_moments)

    copied_config["local_case"] = {
        "active": True,
        "case_id": QUZHOU_LOCAL_CASE_ID,
        "case_type": "quzhou_local_case_v1",
        "source_local_moments": DEFAULT_QUZHOU_V1_LOCAL_MOMENTS_RELATIVE_PATH,
        "evidence_boundary": LOCAL_QUZHOU_ONLY,
        "activation_mode": CONFIG_LEVEL_SMALL_SMOKE,
        "config_level_activation_only": True,
        "applies_to": "copied_config_only",
        "national_baseline_mutation_allowed": False,
        "policy_params_mutation_allowed": False,
        "active_runtime_keys_modified": [],
        "activated_mechanisms": list(QUZHOU_V1_ACTIVATE_MECHANISMS),
        "non_activate_mechanisms_activated": [],
        "quzhou_v1_local_moments": payload,
        "claim_boundary": {
            "can_claim": [
                "Quzhou v1 local moments can be attached to a copied config for config-level small smoke.",
                "National master config remains unchanged by this helper.",
                "National policy params remain unchanged by this helper.",
                "Only activate_in_v1 local moments are attached under local_case.",
            ],
            "cannot_claim": [
                "Quzhou v1 is calibrated.",
                "Quzhou v1 is validated.",
                "Quzhou v1 smoke is a policy recommendation.",
                "Quzhou v1 local moments are national evidence.",
                "The helper runs PPO, government_q_extended, full MARL, or long experiments.",
            ],
        },
    }
    return copied_config


__all__ = [
    "CONFIG_LEVEL_SMALL_SMOKE",
    "DEFAULT_QUZHOU_V1_LOCAL_MOMENTS_PATH",
    "DEFAULT_QUZHOU_V1_LOCAL_MOMENTS_RELATIVE_PATH",
    "LOCAL_MOMENTS_ONLY_NOT_ACTIVE",
    "LOCAL_QUZHOU_ONLY",
    "QUZHOU_V1_ACTIVATE_MECHANISMS",
    "QUZHOU_V1_FORBIDDEN_NON_ACTIVATE_MECHANISMS",
    "activate_quzhou_v1_local_moments_for_local_case",
    "extract_quzhou_v1_local_moments_payload",
    "load_quzhou_v1_local_moments",
]
