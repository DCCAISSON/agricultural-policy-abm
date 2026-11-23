from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.quzhou_game_payoff import _float, action_flag
from core.quzhou_land_closure import _round6, _safe_float, coordination_metrics

SERVICE_LEDGER_FIELDS = ["year", "village_code", "service_demand_area_mu", "service_capacity_mu", "matched_service_area_mu", "unmet_service_area_mu", "service_fee_yuan_per_mu", "service_provider_margin_proxy", "service_matching_friction", "claim_boundary"]
ORDER_GRAIN_LEDGER_FIELDS = ["year", "farmer_id", "eligible_area_mu", "expected_order_grain_output_kg", "purchase_price_yuan_per_jin", "extra_subsidy_yuan_per_jin", "order_grain_revenue", "eligibility_basis", "claim_boundary"]
INSURANCE_RISK_LEDGER_FIELDS = ["year", "farmer_id", "insured", "risk_exposure_level", "expected_loss_yuan", "expected_payout_yuan", "premium_yuan", "risk_buffer_effect", "claim_boundary"]


def service_matching_ledger_rows(year: int, agent_state_rows: Sequence[Mapping[str, object]], params: Mapping[str, object], enterprises: int) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in agent_state_rows:
        code = str(row.get("village_code") or "missing_or_unusable")
        item = grouped.setdefault(code, {"demand": 0.0})
        if row.get("service_used") in (True, "True", "true"):
            item["demand"] += _safe_float(row.get("early_rice_area_mu"))
    total_demand = sum(float(item["demand"]) for item in grouped.values())
    capacity = max(0.0, float(enterprises) * _float(params, "service_capacity_mu_per_enterprise"))
    friction = float(coordination_metrics(params)["service_matching_friction"])
    fee = _float(params, "service_fee_yuan_per_mu")
    rows: list[dict[str, object]] = []
    for code in sorted(grouped):
        demand = float(grouped[code]["demand"])
        capacity_share = 0.0 if total_demand <= 0.0 else capacity * demand / total_demand
        matched = min(demand * (1.0 - friction), capacity_share)
        rows.append({"year": year, "village_code": code, "service_demand_area_mu": _round6(demand), "service_capacity_mu": _round6(capacity_share), "matched_service_area_mu": _round6(matched), "unmet_service_area_mu": _round6(max(0.0, demand - matched)), "service_fee_yuan_per_mu": _round6(fee), "service_provider_margin_proxy": _round6(matched * fee * 0.35), "service_matching_friction": _round6(friction), "claim_boundary": "service_matching_synthetic_ledger_from_contract_fee_and_mechanism_assumption_not_real_contract_plot_match"})
    return rows or [{"year": year, "village_code": "ALL", "service_demand_area_mu": 0.0, "service_capacity_mu": _round6(capacity), "matched_service_area_mu": 0.0, "unmet_service_area_mu": 0.0, "service_fee_yuan_per_mu": _round6(fee), "service_provider_margin_proxy": 0.0, "service_matching_friction": _round6(friction), "claim_boundary": "service_matching_synthetic_ledger_from_contract_fee_and_mechanism_assumption_not_real_contract_plot_match"}]


def order_grain_ledger_rows(year: int, agent_state_rows: Sequence[Mapping[str, object]], params: Mapping[str, object]) -> list[dict[str, object]]:
    price = _float(params, "purchase_price_yuan_per_jin")
    extra = _float(params, "extra_subsidy_yuan_per_jin")
    rows: list[dict[str, object]] = []
    for row in agent_state_rows:
        eligible = row.get("order_grain_eligible") in (True, "True", "true")
        order_action = action_flag(row.get("selected_action"), "order")
        eligible_area = _safe_float(row.get("early_rice_area_mu")) if eligible and order_action else 0.0
        output = _safe_float(row.get("early_rice_output_kg")) if eligible_area > 0.0 else 0.0
        rows.append({"year": year, "farmer_id": row.get("farmer_id", "missing_or_unusable"), "eligible_area_mu": _round6(eligible_area), "expected_order_grain_output_kg": _round6(output), "purchase_price_yuan_per_jin": _round6(price), "extra_subsidy_yuan_per_jin": _round6(extra), "order_grain_revenue": _round6(output * 2.0 * (price + extra)), "eligibility_basis": "assumed_eligible_from_synthetic_linkage_mechanism" if eligible else "not_eligible_in_synthetic_linkage_sample_mechanism", "claim_boundary": "order_grain_synthetic_eligibility_ledger_not_real_purchase_contract_or_policy_entitlement"})
    return rows


def insurance_risk_ledger_rows(year: int, agent_state_rows: Sequence[Mapping[str, object]], params: Mapping[str, object]) -> list[dict[str, object]]:
    price = _float(params, "purchase_price_yuan_per_jin") + _float(params, "extra_subsidy_yuan_per_jin")
    loss_rate = _float(params, "disaster_loss_rate")
    coverage = _float(params, "indemnity_coverage_rate")
    premium_per_mu = _float(params, "insurance_premium_yuan_per_mu")
    rows: list[dict[str, object]] = []
    for row in agent_state_rows:
        insured = row.get("insured") in (True, "True", "true")
        early_area = _safe_float(row.get("early_rice_area_mu"))
        expected_loss = _safe_float(row.get("early_rice_output_kg")) * 2.0 * price * loss_rate
        payout = expected_loss * coverage if insured else 0.0
        premium = early_area * premium_per_mu if insured else 0.0
        rows.append({"year": year, "farmer_id": row.get("farmer_id", "missing_or_unusable"), "insured": insured, "risk_exposure_level": row.get("insurance_exposure_level", "medium_risk_diagnostic_from_insurance_claim_summary"), "expected_loss_yuan": _round6(expected_loss), "expected_payout_yuan": _round6(payout), "premium_yuan": _round6(premium), "risk_buffer_effect": _round6(payout - premium), "claim_boundary": "insurance_risk_buffer_only_not_claim_anomaly_detection_no_claim_loss_plot_time_subject_network"})
    return rows
