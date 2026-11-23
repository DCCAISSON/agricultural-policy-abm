from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUZHOU_OVERLAY_PATH = PROJECT_ROOT / "config" / "quzhou" / "quzhou_overlay.yaml"
DEFAULT_QUZHOU_OVERLAY_RELATIVE_PATH = "config/quzhou/quzhou_overlay.yaml"
CANDIDATE_NOT_ACTIVE = "candidate_not_active"
LOCAL_OVERLAY_ONLY = "local_overlay_only"
QUZHOU_LOCAL_CASE_ID = "quzhou_2026_case"
CONFIG_ACTIVATION_SMOKE_ONLY = "config_activation_smoke_only"


class QuzhouOverlayError(ValueError):
    """Raised when the Quzhou local overlay is malformed or accidentally active."""


def _validate_candidate_overlay(overlay: Mapping[str, Any]) -> None:
    metadata = overlay.get("metadata", {}) if isinstance(overlay, Mapping) else {}
    status = metadata.get("status") if isinstance(metadata, Mapping) else None
    if status != CANDIDATE_NOT_ACTIVE:
        raise QuzhouOverlayError(
            f"Quzhou overlay status must be {CANDIDATE_NOT_ACTIVE!r}; got {status!r}."
        )

    if metadata.get("active_in_national_baseline") is not False:
        raise QuzhouOverlayError("Quzhou overlay must declare active_in_national_baseline: false.")

    if metadata.get("national_config_mutation_allowed") is not False:
        raise QuzhouOverlayError("Quzhou overlay must declare national_config_mutation_allowed: false.")

    if metadata.get("evidence_boundary") != LOCAL_OVERLAY_ONLY:
        raise QuzhouOverlayError(
            f"Quzhou overlay must declare evidence_boundary: {LOCAL_OVERLAY_ONLY!r}."
        )


def _candidate_value(candidates: Mapping[str, Any], section: str, key: str) -> Any:
    section_data = candidates.get(section, {}) if isinstance(candidates, Mapping) else {}
    item = section_data.get(key, {}) if isinstance(section_data, Mapping) else {}
    if not isinstance(item, Mapping) or "value" not in item:
        raise QuzhouOverlayError(f"Missing Quzhou candidate value: {section}.{key}")
    return item["value"]


def _candidate_value_range(candidates: Mapping[str, Any], section: str, key: str) -> list[Any]:
    section_data = candidates.get(section, {}) if isinstance(candidates, Mapping) else {}
    item = section_data.get(key, {}) if isinstance(section_data, Mapping) else {}
    value_range = item.get("value_range") if isinstance(item, Mapping) else None
    if not isinstance(value_range, list) or len(value_range) != 2:
        raise QuzhouOverlayError(f"Missing Quzhou candidate value_range: {section}.{key}")
    return list(value_range)


def _candidate_summary_value(
    candidates: Mapping[str, Any],
    section: str,
    key: str,
    summary_key: str,
) -> Any:
    section_data = candidates.get(section, {}) if isinstance(candidates, Mapping) else {}
    item = section_data.get(key, {}) if isinstance(section_data, Mapping) else {}
    if not isinstance(item, Mapping) or summary_key not in item:
        raise QuzhouOverlayError(f"Missing Quzhou candidate summary value: {section}.{key}.{summary_key}")
    return item[summary_key]


def extract_quzhou_v0_local_case_parameters(overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the minimal v0 Quzhou local-case diagnostic parameters."""
    _validate_candidate_overlay(overlay)
    candidates = overlay.get("parameter_candidates", {})
    if not isinstance(candidates, Mapping):
        raise QuzhouOverlayError("Quzhou overlay parameter_candidates must be a mapping.")

    return {
        "grain_reserve": {
            "order_purchase_price_yuan_per_jin": _candidate_value(
                candidates,
                "grain_reserve",
                "order_purchase_price_yuan_per_jin",
            ),
            "order_extra_subsidy_yuan_per_jin": _candidate_value(
                candidates,
                "grain_reserve",
                "order_extra_subsidy_yuan_per_jin",
            ),
        },
        "land_market": {
            "meeting_average_rent_yuan_per_mu_year_range": _candidate_value_range(
                candidates,
                "land_market",
                "meeting_average_rent_yuan_per_mu_year",
            ),
        },
        "service_market": {
            "full_socialized_trusteeship_fee_yuan_per_mu": _candidate_value(
                candidates,
                "service_market",
                "full_socialized_trusteeship_fee_yuan_per_mu",
            ),
        },
        "production": {
            "early_rice_sample_area_weighted_yield_kg_per_mu": _candidate_summary_value(
                candidates,
                "early_rice_2025",
                "yield_kg_per_mu_summary",
                "area_weighted_mean",
            ),
        },
    }


def load_quzhou_overlay(path: str | Path | None = None) -> dict[str, Any]:
    """Load the Quzhou candidate overlay without applying it to the national config."""
    overlay_path = Path(path) if path is not None else DEFAULT_QUZHOU_OVERLAY_PATH
    with overlay_path.open("r", encoding="utf-8") as handle:
        overlay = yaml.safe_load(handle) or {}

    if not isinstance(overlay, dict):
        raise QuzhouOverlayError(f"Quzhou overlay must be a mapping: {overlay_path}")

    _validate_candidate_overlay(overlay)
    return overlay


def attach_quzhou_overlay_snapshot(
    national_config: Mapping[str, Any],
    overlay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copied config with the candidate overlay attached under local_overlays."""
    candidate_overlay = deepcopy(overlay if overlay is not None else load_quzhou_overlay())
    _validate_candidate_overlay(candidate_overlay)

    copied_config = deepcopy(dict(national_config or {}))
    local_overlays = copied_config.setdefault("local_overlays", {})
    local_overlays["quzhou"] = candidate_overlay
    return copied_config


def activate_quzhou_overlay_for_local_case(
    national_config: Mapping[str, Any],
    overlay: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Activate Quzhou v0 diagnostics on a copied config for local-case smoke only."""
    candidate_overlay = deepcopy(overlay if overlay is not None else load_quzhou_overlay())
    _validate_candidate_overlay(candidate_overlay)

    copied_config = attach_quzhou_overlay_snapshot(national_config, candidate_overlay)
    metadata = candidate_overlay.get("metadata", {})
    v0_parameters = extract_quzhou_v0_local_case_parameters(candidate_overlay)

    copied_config["local_case"] = {
        "active": True,
        "case_id": QUZHOU_LOCAL_CASE_ID,
        "case_type": "quzhou_local_case_v0",
        "source_overlay": DEFAULT_QUZHOU_OVERLAY_RELATIVE_PATH,
        "evidence_boundary": metadata.get("evidence_boundary", LOCAL_OVERLAY_ONLY),
        "activation_mode": CONFIG_ACTIVATION_SMOKE_ONLY,
        "applies_to": "copied_config_only",
        "national_baseline_mutation_allowed": False,
        "active_runtime_keys_modified": [],
        "quzhou_v0_parameters": v0_parameters,
        "claim_boundary": {
            "can_claim": [
                "Quzhou local overlay can be activated on a copied config for diagnostic local-case smoke.",
                "National master config remains unchanged by this helper.",
                "Quzhou parameters are local candidate/smoke parameters only.",
            ],
            "cannot_claim": [
                "Quzhou model is calibrated.",
                "Quzhou policy recommendation is ready.",
                "Quzhou result is national evidence.",
                "Quzhou smoke validates causal policy effects.",
                "Local smoke replaces S1-S4 formal evidence.",
            ],
        },
    }
    return copied_config


__all__ = [
    "CANDIDATE_NOT_ACTIVE",
    "CONFIG_ACTIVATION_SMOKE_ONLY",
    "DEFAULT_QUZHOU_OVERLAY_PATH",
    "DEFAULT_QUZHOU_OVERLAY_RELATIVE_PATH",
    "LOCAL_OVERLAY_ONLY",
    "PROJECT_ROOT",
    "QUZHOU_LOCAL_CASE_ID",
    "QuzhouOverlayError",
    "activate_quzhou_overlay_for_local_case",
    "attach_quzhou_overlay_snapshot",
    "extract_quzhou_v0_local_case_parameters",
    "load_quzhou_overlay",
]
