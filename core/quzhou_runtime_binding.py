from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from core.quzhou_local_moments import (
    DEFAULT_QUZHOU_V1_LOCAL_MOMENTS_RELATIVE_PATH,
    LOCAL_QUZHOU_ONLY,
    QUZHOU_V1_ACTIVATE_MECHANISMS,
    QUZHOU_V1_FORBIDDEN_NON_ACTIVATE_MECHANISMS,
    load_quzhou_v1_local_moments,
)
from core.quzhou_overlay_v0 import PROJECT_ROOT, QUZHOU_LOCAL_CASE_ID, QuzhouOverlayError

DEFAULT_QUZHOU_V1_RUNTIME_BINDING_MAP_PATH = PROJECT_ROOT / "project_management" / "QUZHOU_V1_RUNTIME_BINDING_MAP_20260706.csv"
DEFAULT_QUZHOU_V1_RUNTIME_BINDING_MAP_RELATIVE_PATH = "project_management/QUZHOU_V1_RUNTIME_BINDING_MAP_20260706.csv"
DEFAULT_QUZHOU_V1_ACTIVE_PROMOTION_DECISION_MATRIX_PATH = PROJECT_ROOT / "project_management" / "QUZHOU_V1_ACTIVE_PROMOTION_DECISION_MATRIX_20260707.csv"
DEFAULT_QUZHOU_V1_ACTIVE_PROMOTION_DECISION_MATRIX_RELATIVE_PATH = "project_management/QUZHOU_V1_ACTIVE_PROMOTION_DECISION_MATRIX_20260707.csv"
QUZHOU_V1_RUNTIME_BINDING_SMOKE = "quzhou_v1_runtime_binding_smoke"
BIND_NOW = "bind_now"
DIAGNOSTIC_ONLY = "diagnostic_only"
NOT_BINDABLE_YET = "not_bindable_yet"
PROMOTE_TO_ACTIVE = "promote_to_active"
REMAIN_DIAGNOSTIC = "remain_diagnostic"
PRIOR_ONLY = "prior_only"
OUT_OF_SCOPE = "out_of_scope"
RUNTIME_BINDING_STATUSES = (BIND_NOW, DIAGNOSTIC_ONLY, NOT_BINDABLE_YET)
PROMOTION_DECISIONS = (PROMOTE_TO_ACTIVE, REMAIN_DIAGNOSTIC, PRIOR_ONLY, OUT_OF_SCOPE)
PROMOTION_DECISION_MECHANISMS = (
    "order_grain_extra_subsidy",
    "early_rice_cost",
    "insurance_claim_loss",
    "land_rent_average",
    "socialized_service_fee",
)
PROMOTION_REQUIRED_FIELDS = (
    "mechanism",
    "current_status",
    "current_code_path",
    "current_config_path",
    "local_data_available",
    "row_level_data_available",
    "unit_consistency",
    "active_path_candidate",
    "promotion_decision",
    "reason",
    "claim_boundary",
    "required_test",
)


def _split_target_paths(target_text: str) -> list[str]:
    return [item.strip() for item in str(target_text or "").split(";") if item.strip()]


def _get_nested(mapping: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = mapping
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested(mapping: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = mapping
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _required_text(row: Mapping[str, str], field: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise QuzhouOverlayError(f"Quzhou runtime binding row missing {field!r}: {row}")
    return value


def load_quzhou_v1_runtime_binding_map(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load and validate the Quzhou v1 runtime binding map."""
    map_path = Path(path) if path is not None else DEFAULT_QUZHOU_V1_RUNTIME_BINDING_MAP_PATH
    with map_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    if len(rows) != len(QUZHOU_V1_ACTIVATE_MECHANISMS):
        raise QuzhouOverlayError(f"Quzhou runtime binding map must have 8 rows; got {len(rows)}.")

    mechanisms = [row.get("mechanism", "").strip() for row in rows]
    if tuple(mechanisms) != QUZHOU_V1_ACTIVATE_MECHANISMS:
        raise QuzhouOverlayError("Quzhou runtime binding map must follow activate_in_v1 mechanism order.")

    for row in rows:
        status = _required_text(row, "runtime_binding_status")
        if status not in RUNTIME_BINDING_STATUSES:
            raise QuzhouOverlayError(f"Unsupported runtime binding status {status!r}.")
        for field in (
            "local_moment_path",
            "local_unit",
            "copied_config_target_path",
            "active_code_path",
            "unit_conversion",
            "fallback_behavior",
            "validation_probe",
            "claim_boundary",
        ):
            _required_text(row, field)
    return rows


def load_quzhou_v1_active_promotion_decision_matrix(path: str | Path | None = None) -> list[dict[str, str]]:
    """Load the bounded Quzhou active-promotion decision matrix."""
    matrix_path = Path(path) if path is not None else DEFAULT_QUZHOU_V1_ACTIVE_PROMOTION_DECISION_MATRIX_PATH
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    mechanisms = [row.get("mechanism", "").strip() for row in rows]
    if tuple(mechanisms) != PROMOTION_DECISION_MECHANISMS:
        raise QuzhouOverlayError("Quzhou promotion decision matrix must follow the five core mechanism order.")
    for row in rows:
        for field in PROMOTION_REQUIRED_FIELDS:
            _required_text(row, field)
        if row["promotion_decision"] not in PROMOTION_DECISIONS:
            raise QuzhouOverlayError(f"Unsupported promotion decision {row['promotion_decision']!r}.")
        if row["promotion_decision"] == PROMOTE_TO_ACTIVE and row["current_status"] != BIND_NOW:
            raise QuzhouOverlayError(f"Promoted mechanism must already be bind_now copied-runtime capable: {row['mechanism']}")
    return rows


def _build_binding_record(*, row: Mapping[str, str], target_path: str, before_value: Any, after_value: Any) -> dict[str, Any]:
    return {
        "mechanism": row["mechanism"],
        "runtime_binding_status": row["runtime_binding_status"],
        "target_path": target_path,
        "before_value": before_value,
        "after_value": after_value,
        "source_moment": row["local_moment_path"],
        "unit": row["local_unit"],
        "active_code_path": row["active_code_path"],
        "unit_conversion": row["unit_conversion"],
        "fallback_behavior": row["fallback_behavior"],
        "validation_probe": row["validation_probe"],
        "claim_boundary": row["claim_boundary"],
    }


def _build_diagnostic_record(row: Mapping[str, str], value: Any) -> dict[str, Any]:
    return {
        "mechanism": row["mechanism"],
        "runtime_binding_status": row["runtime_binding_status"],
        "target_path": row["copied_config_target_path"],
        "source_moment": row["local_moment_path"],
        "value": value,
        "unit": row["local_unit"],
        "active_code_path": row["active_code_path"],
        "unit_conversion": row["unit_conversion"],
        "fallback_behavior": row["fallback_behavior"],
        "validation_probe": row["validation_probe"],
        "claim_boundary": row["claim_boundary"],
    }


def _promotion_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "source_active_promotion_decision_matrix": DEFAULT_QUZHOU_V1_ACTIVE_PROMOTION_DECISION_MATRIX_RELATIVE_PATH,
        "promoted_active_mechanisms": [row["mechanism"] for row in rows if row["promotion_decision"] == PROMOTE_TO_ACTIVE],
        "retained_diagnostic_mechanisms": [row["mechanism"] for row in rows if row["promotion_decision"] == REMAIN_DIAGNOSTIC],
        "prior_only_mechanisms": [row["mechanism"] for row in rows if row["promotion_decision"] == PRIOR_ONLY],
        "out_of_scope_mechanisms": [row["mechanism"] for row in rows if row["promotion_decision"] == OUT_OF_SCOPE],
        "promotion_decisions": {
            row["mechanism"]: {
                "current_status": row["current_status"],
                "active_path_candidate": row["active_path_candidate"],
                "promotion_decision": row["promotion_decision"],
                "reason": row["reason"],
                "claim_boundary": row["claim_boundary"],
                "required_test": row["required_test"],
            }
            for row in rows
        },
    }


def bind_quzhou_v1_local_moments_to_copied_runtime_config(
    national_config: Mapping[str, Any],
    local_moments: Mapping[str, Any] | None = None,
    binding_map: list[dict[str, str]] | None = None,
    promotion_matrix: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Bind Quzhou v1 local moments to copied runtime-config targets.

    Unsupported mechanisms are recorded as diagnostic-only and do not write active
    runtime config keys. The input national_config is never mutated.
    """
    source_moments = deepcopy(local_moments if local_moments is not None else load_quzhou_v1_local_moments())
    rows = deepcopy(binding_map if binding_map is not None else load_quzhou_v1_runtime_binding_map())
    promotion_rows = deepcopy(
        promotion_matrix if promotion_matrix is not None else load_quzhou_v1_active_promotion_decision_matrix()
    )
    loaded_mechanisms = set(source_moments.get("moments", {}).keys())
    if loaded_mechanisms != set(QUZHOU_V1_ACTIVATE_MECHANISMS):
        raise QuzhouOverlayError("Quzhou local moments do not match activate_in_v1 mechanisms.")

    copied_config = deepcopy(dict(national_config or {}))
    binding_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    diagnostic_moments: dict[str, dict[str, Any]] = {}
    active_runtime_keys_modified: list[str] = []

    for row in rows:
        mechanism = row["mechanism"]
        if mechanism in QUZHOU_V1_FORBIDDEN_NON_ACTIVATE_MECHANISMS:
            raise QuzhouOverlayError(f"Forbidden mechanism cannot be runtime-bound: {mechanism}")
        value = _get_nested(source_moments, row["local_moment_path"])
        if value is None:
            raise QuzhouOverlayError(f"Missing local moment value for {mechanism}: {row['local_moment_path']}")

        if row["runtime_binding_status"] == BIND_NOW:
            for target_path in _split_target_paths(row["copied_config_target_path"]):
                before_value = _get_nested(copied_config, target_path)
                _set_nested(copied_config, target_path, value)
                active_runtime_keys_modified.append(target_path)
                binding_records.append(_build_binding_record(row=row, target_path=target_path, before_value=before_value, after_value=value))
        else:
            diagnostic_records.append(_build_diagnostic_record(row, value))
            diagnostic_moments[mechanism] = {
                "value": value,
                "unit": row["local_unit"],
                "source_moment": row["local_moment_path"],
                "target_path": row["copied_config_target_path"],
                "claim_boundary": row["claim_boundary"],
            }

    promotion_summary = _promotion_summary(promotion_rows)
    local_case = copied_config.setdefault("local_case", {})
    local_case.update({
        "active": True,
        "case_id": QUZHOU_LOCAL_CASE_ID,
        "case_type": "quzhou_local_case_v1_runtime_binding",
        "smoke_mode": QUZHOU_V1_RUNTIME_BINDING_SMOKE,
        "evidence_boundary": LOCAL_QUZHOU_ONLY,
        "source_local_moments": DEFAULT_QUZHOU_V1_LOCAL_MOMENTS_RELATIVE_PATH,
        "source_runtime_binding_map": DEFAULT_QUZHOU_V1_RUNTIME_BINDING_MAP_RELATIVE_PATH,
        "source_active_promotion_decision_matrix": DEFAULT_QUZHOU_V1_ACTIVE_PROMOTION_DECISION_MATRIX_RELATIVE_PATH,
        "applies_to": "copied_config_only",
        "national_baseline_mutation_allowed": False,
        "policy_params_mutation_allowed": False,
        "active_runtime_keys_modified": active_runtime_keys_modified,
        "quzhou_v1_active_promotion": promotion_summary,
        "quzhou_v1_runtime_binding": {
            "smoke_mode": QUZHOU_V1_RUNTIME_BINDING_SMOKE,
            "binding_map_rows": len(rows),
            "bind_now_mechanisms": sorted({record["mechanism"] for record in binding_records}),
            "diagnostic_only_mechanisms": sorted({record["mechanism"] for record in diagnostic_records}),
            "active_runtime_keys_modified": active_runtime_keys_modified,
            "binding_records": binding_records,
            "diagnostic_records": diagnostic_records,
            "diagnostic_moments": diagnostic_moments,
            "active_promotion": promotion_summary,
            "claim_boundary": {
                "can_claim": [
                    "Quzhou v1 local moments have a controlled copied-config runtime binding map.",
                    "Evidence-supported active paths are promoted only on a copied Quzhou runtime config.",
                    "Unsupported mechanisms remain diagnostic_only.",
                    "National baseline config is unchanged by the helper.",
                ],
                "cannot_claim": [
                    "Quzhou v1 is calibrated.",
                    "Quzhou v1 is externally validated.",
                    "Quzhou v1 produces policy recommendations.",
                    "Quzhou evidence is national evidence.",
                    "PPO, government_q_extended, or full MARL was run.",
                    "The smoke result proves policy effect.",
                ],
            },
        },
    })
    return copied_config


__all__ = [
    "BIND_NOW",
    "DEFAULT_QUZHOU_V1_ACTIVE_PROMOTION_DECISION_MATRIX_PATH",
    "DEFAULT_QUZHOU_V1_ACTIVE_PROMOTION_DECISION_MATRIX_RELATIVE_PATH",
    "DEFAULT_QUZHOU_V1_RUNTIME_BINDING_MAP_PATH",
    "DEFAULT_QUZHOU_V1_RUNTIME_BINDING_MAP_RELATIVE_PATH",
    "DIAGNOSTIC_ONLY",
    "NOT_BINDABLE_YET",
    "PROMOTE_TO_ACTIVE",
    "PROMOTION_DECISION_MECHANISMS",
    "REMAIN_DIAGNOSTIC",
    "QUZHOU_V1_RUNTIME_BINDING_SMOKE",
    "RUNTIME_BINDING_STATUSES",
    "bind_quzhou_v1_local_moments_to_copied_runtime_config",
    "load_quzhou_v1_active_promotion_decision_matrix",
    "load_quzhou_v1_runtime_binding_map",
]
