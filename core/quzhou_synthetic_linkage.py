from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.quzhou_game_payoff import _float, farmer_type_profile, land_area, profile_float, profile_for_type

SYNTHETIC_LINKAGE_STATUS = "synthetic_linkage_for_model_world_not_real_household_plot_match"
SYNTHETIC_LINKAGE_FIELDS = [
    "synthetic_farmer_id", "farmer_type", "village_code", "village_name", "assigned_plot_count",
    "assigned_paddy_area_mu", "assigned_non_paddy_area_mu", "operated_area_mu", "order_grain_eligible",
    "service_access_level", "insurance_exposure_level", "land_rent_pressure_level", "matching_basis", "evidence_level", "claim_boundary",
]


def linkage_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "eligible", "assumed_eligible"}


def build_synthetic_linkage_rows(plot_rows: Sequence[Mapping[str, object]], farmer_count: int, params: Mapping[str, object], *, seed: int = 0) -> list[dict[str, object]]:
    sorted_plots = sorted(plot_rows, key=lambda row: (0 if linkage_truthy(row.get("is_paddy_field")) else 1, str(row.get("village_code") or ""), str(row.get("plot_id") or "")))
    rows: list[dict[str, object]] = []
    for index in range(farmer_count):
        farmer_type, _ = farmer_type_profile(index, params)
        plot = sorted_plots[index] if index < len(sorted_plots) else {}
        is_paddy = linkage_truthy(plot.get("is_paddy_field"))
        area = round(float(plot.get("plot_area_mu") or 0.0), 6) if plot else 0.0
        paddy_area = area if is_paddy else 0.0
        non_paddy_area = 0.0 if is_paddy else area
        service_level = "high_mechanism_from_service_contract_sample" if index % 4 in {1, 2} else "medium_mechanism_from_service_contract_sample"
        exposure_level = ("high" if index % 5 == 0 else "medium" if index % 2 == 0 else "low") + "_risk_diagnostic_from_insurance_claim_summary"
        rows.append({
            "synthetic_farmer_id": f"farmer_{index}", "farmer_type": farmer_type,
            "village_code": plot.get("village_code") or "missing_or_unusable", "village_name": plot.get("village_name") or "missing_or_unusable",
            "assigned_plot_count": 1 if plot else 0, "assigned_paddy_area_mu": round(paddy_area, 6),
            "assigned_non_paddy_area_mu": round(non_paddy_area, 6), "operated_area_mu": round(paddy_area + non_paddy_area, 6),
            "order_grain_eligible": index % 3 != 0, "service_access_level": service_level, "insurance_exposure_level": exposure_level,
            "land_rent_pressure_level": "high" if area >= 10.0 else "medium" if area >= 3.0 else "low",
            "matching_basis": f"seed={seed};deterministic_sorted_village_plot_area;paddy_first;same_village_linkage_world;service_from_contract_fee_mechanism;insurance_from_claim_summary_risk_diagnostic",
            "evidence_level": SYNTHETIC_LINKAGE_STATUS,
            "claim_boundary": "synthetic_linkage_for_model_world_not_real_household_plot_match;not_calibration;not_policy_recommendation",
        })
    return rows


def initialize_farmer_states(model: object, params: Mapping[str, object], synthetic_linkage_rows: Sequence[Mapping[str, object]] | None = None) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    linkage_by_id = {str(row.get("synthetic_farmer_id")): row for row in (synthetic_linkage_rows or [])}
    for index, farmer in enumerate(getattr(model, "farmers", []) or []):
        farmer_id = str(getattr(farmer, "unique_id", f"farmer_{index}"))
        linkage = linkage_by_id.get(farmer_id)
        if linkage:
            farmer_type = str(linkage.get("farmer_type") or farmer_type_profile(index, params)[0])
            profile = profile_for_type(farmer_type, params)
            area = max(0.0001, float(linkage.get("operated_area_mu") or 0.0))
            meta = {"village_code": str(linkage.get("village_code") or "missing_or_unusable"), "village_name": str(linkage.get("village_name") or "missing_or_unusable"), "order_grain_eligible": linkage_truthy(linkage.get("order_grain_eligible")), "service_access_level": str(linkage.get("service_access_level") or "missing_or_unusable"), "insurance_exposure_level": str(linkage.get("insurance_exposure_level") or "medium_risk_diagnostic_from_insurance_claim_summary"), "land_rent_pressure_level": str(linkage.get("land_rent_pressure_level") or "missing_or_unusable"), "synthetic_linkage_status": str(linkage.get("evidence_level") or SYNTHETIC_LINKAGE_STATUS)}
        else:
            farmer_type, profile = farmer_type_profile(index, params)
            area = land_area(farmer, index, profile)
            meta = {"village_code": "missing_or_unusable", "village_name": "missing_or_unusable", "order_grain_eligible": False, "service_access_level": "missing_or_unusable_no_synthetic_linkage", "insurance_exposure_level": "medium_risk_diagnostic_from_insurance_claim_summary", "land_rent_pressure_level": "missing_or_unusable", "synthetic_linkage_status": "synthetic_linkage_missing_for_this_farmer"}
        states.append({"farmer_id": farmer_id, "farmer_type": farmer_type, "profile": profile, "operated_area_mu": area, "cash_income": 0.0, "risk_expectation": _float(params, "disaster_loss_rate") * profile_float(profile, "risk_aversion_multiplier", 1.0), "service_used": False, "insured": False, "land_transfer_status": "initial_sample_state", **meta})
    return states
