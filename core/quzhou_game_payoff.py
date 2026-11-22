from __future__ import annotations

from collections.abc import Mapping, Sequence

ACTION_SET = {
    "plant_early_rice_with_service_order_insurance": {"service": True, "insurance": True, "order": True},
    "plant_early_rice_self_order_no_insurance": {"service": False, "insurance": False, "order": True},
    "keep_area_self_production_no_insurance": {"service": False, "insurance": False, "order": False},
    "rent_out_or_reduce_area": {"service": False, "insurance": False, "rent_out": True},
    "rent_in_with_service_if_profitable": {"service": True, "insurance": True, "order": True, "rent_in": True},
}
CROP_ACTIONS = tuple(name for name in ACTION_SET if name != "rent_out_or_reduce_area")
DECISION_FIELDS = [
    "year", "agent_type", "agent_id", "farmer_type", "land_area_mu", "state_summary", "available_actions",
    "selected_action", "selected_action_net_return", "expected_net_return", "best_early_rice_net_return",
    "rent_out_net_return", "early_rice_output_kg", "grain_price_yuan_per_jin", "order_extra_subsidy_yuan_per_jin",
    "expected_price_with_order_yuan_per_jin", "expected_price_without_order_yuan_per_jin", "expected_yield_kg_per_mu",
    "order_grain_revenue", "order_grain_extra_subsidy_component", "scale_subsidy", "early_rice_cost_component",
    "land_rent_component", "nonfarm_opportunity_component", "service_fee_component", "insurance_premium",
    "insurance_payout_expected", "risk_penalty", "constraint_summary", "information_source", "feedback_path", "claim_boundary",
]
MARKET_FIELDS = [
    "year", "coordination_level", "friction_reduction", "farmer_count", "plant_early_rice_count", "service_demand_area_mu",
    "service_capacity_mu", "fulfilled_service_area_mu", "service_capacity_utilization", "average_expected_net_return",
    "total_order_grain_revenue", "total_insurance_payout_expected", "runtime_land_rent", "runtime_service_fee",
]
AGENT_STATE_FIELDS = [
    "year", "farmer_id", "farmer_type", "village_code", "village_name", "operated_area_mu", "early_rice_area_mu",
    "early_rice_output_kg", "selected_action", "cash_income", "expected_net_return", "service_used", "insured",
    "order_grain_eligible", "service_access_level", "insurance_exposure_level", "land_rent_pressure_level",
    "synthetic_linkage_status", "insurance_payout_expected", "risk_expectation", "next_year_area_mu", "claim_boundary",
]
SYSTEM_STATE_FIELDS = [
    "year", "total_operated_area_mu", "total_early_rice_area_mu", "total_early_rice_output_kg", "plant_early_rice_count",
    "rent_out_or_reduce_count", "service_uptake_count", "average_cash_income", "average_expected_net_return",
    "total_order_grain_revenue", "total_service_fee", "total_insurance_payout_expected", "average_risk_expectation",
    "service_capacity_utilization", "village_coordination_result", "claim_boundary",
]
SCENARIO_FIELDS = [
    "scenario_id", "early_rice_area_mu", "early_rice_output_kg", "average_expected_net_return", "plant_early_rice_count",
    "rent_out_or_reduce_count", "service_uptake_count", "insurance_expected_payout", "fiscal_cost_proxy", "trajectory_years",
    "final_early_rice_area_mu", "final_early_rice_output_kg", "final_average_cash_income", "final_average_risk_expectation",
    "cumulative_early_rice_output_kg", "final_service_capacity_utilization", "coordination_level", "service_matching_friction",
    "service_unmet_area_mu_proxy", "interpretation_boundary",
]


def _float(mapping: Mapping[str, object], key: str, default: float | None = None) -> float:
    value = mapping.get(key, default)
    if value is None:
        raise KeyError(key)
    return float(value)


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def action_flag(action: object, flag: str) -> bool:
    return bool(ACTION_SET.get(str(action), {}).get(flag, False))


def farmer_type_profile(index: int, params: Mapping[str, object]) -> tuple[str, Mapping[str, object]]:
    profiles = params.get("farmer_type_profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        return "untyped_sample_farmer", {}
    sequence = params.get("farmer_type_sequence")
    names = [str(name) for name in sequence] if isinstance(sequence, list | tuple) else list(profiles)
    name = names[index % len(names)]
    profile = profiles.get(name, {})
    return name, profile if isinstance(profile, Mapping) else {}


def profile_for_type(name: str, params: Mapping[str, object]) -> Mapping[str, object]:
    profiles = params.get("farmer_type_profiles")
    if isinstance(profiles, Mapping) and isinstance(profiles.get(name, {}), Mapping):
        return profiles.get(name, {})
    return {}


def land_area(agent: object, index: int, profile: Mapping[str, object] | None = None) -> float:
    if profile and profile.get("area_mu") not in (None, ""):
        return max(1.0, float(profile["area_mu"]))
    for attr in ("operated_area", "land_area", "total_land_area", "farmland_area"):
        value = getattr(agent, attr, None)
        if value not in (None, 0):
            try:
                return max(1.0, float(value))
            except (TypeError, ValueError):
                pass
    return 5.0 + (index % 4)


def profile_float(profile: Mapping[str, object], key: str, default: float) -> float:
    try:
        return float(profile.get(key, default))
    except (TypeError, ValueError):
        return default


def risk_penalty(expected_loss: float, insurance_payout: float, params: Mapping[str, object], profile: Mapping[str, object] | None = None) -> float:
    return (expected_loss - insurance_payout) * _float(params, "risk_aversion_weight") * profile_float(profile or {}, "risk_aversion_multiplier", 1.0)


def farmer_payoff(area: float, params: Mapping[str, object], *, service: bool, insurance: bool, order: bool = False, rent_in: bool = False, rent_out: bool = False, profile: Mapping[str, object] | None = None) -> dict[str, float]:
    profile = profile or {}
    rent = area * _float(params, "rent_yuan_per_mu")
    nonfarm = area * _float(params, "nonfarm_opportunity_income_yuan_per_mu") * profile_float(profile, "nonfarm_opportunity_factor", 0.0)
    if rent_out:
        rent_cash = area * _float(params, "rent_out_income_yuan_per_mu") if "rent_out_income_yuan_per_mu" in params else rent * profile_float(profile, "rent_out_access_factor", 1.0)
        return {"net": rent_cash + nonfarm, "risk_penalty": 0.0, "order_revenue": 0.0, "base_grain_revenue": 0.0, "extra_subsidy": 0.0, "scale_subsidy": 0.0, "rent_component": rent_cash, "land_rent_paid": 0.0, "exit_rent_out_income": rent_cash, "nonfarm_component": nonfarm, "service_fee": 0.0, "insurance_premium": 0.0, "insurance_payout": 0.0, "cost_component": 0.0, "production_cost_component": 0.0, "output_kg": 0.0, "yield_kg_per_mu": 0.0, "price_yuan_per_jin": 0.0}
    yield_key = "total_yield_kg_per_mu" if "total_yield_kg_per_mu" in params and _float(params, "total_yield_kg_per_mu", 0.0) > 0.0 else "yield_kg_per_mu"
    yield_per_mu = _float(params, yield_key) * profile_float(profile, "yield_multiplier", 1.0)
    output_kg = area * yield_per_mu
    purchase_price = _float(params, "purchase_price_yuan_per_jin")
    extra = _float(params, "extra_subsidy_yuan_per_jin") if order else 0.0
    base_revenue = output_kg * 2.0 * purchase_price
    delivered_order_kg = _bounded(_float(params, "order_delivery_kg", 0.0), 0.0, output_kg) if order and "order_delivery_kg" in params else output_kg
    extra_subsidy = delivered_order_kg * 2.0 * extra
    revenue = base_revenue + extra_subsidy
    cost_key = "production_cost_excluding_rent_service_yuan_per_mu" if "production_cost_excluding_rent_service_yuan_per_mu" in params else "cost_yuan_per_mu"
    cost = area * _float(params, cost_key) * profile_float(profile, "cost_multiplier", 1.0)
    if service:
        cost *= 1.0 - profile_float(profile, "service_cost_saving_rate", 0.0)
    service_fee = area * _float(params, "service_fee_yuan_per_mu") * profile_float(profile, "service_fee_multiplier", 1.0) if service else 0.0
    if "land_rent_paid_yuan_per_mu" in params:
        land_rent_paid = area * _float(params, "land_rent_paid_yuan_per_mu")
    else:
        land_rent_paid = rent * profile_float(profile, "rent_in_cost_factor", 1.0) if rent_in else 0.0
    rent_component = -land_rent_paid
    expected_loss = revenue * _float(params, "disaster_loss_rate")
    premium = area * _float(params, "insurance_premium_yuan_per_mu") if insurance else 0.0
    payout = area * _float(params, "expected_insurance_payout_yuan_per_mu") if insurance and "expected_insurance_payout_yuan_per_mu" in params else expected_loss * _float(params, "indemnity_coverage_rate") if insurance else 0.0
    penalty = risk_penalty(expected_loss, payout, params, profile)
    subsidy = area * _float(params, "scale_subsidy_yuan_per_mu")
    net = revenue + subsidy + payout + rent_component - cost - service_fee - premium - penalty
    return {"net": net, "risk_penalty": penalty, "order_revenue": revenue, "base_grain_revenue": base_revenue, "extra_subsidy": extra_subsidy, "scale_subsidy": subsidy, "rent_component": rent_component, "land_rent_paid": land_rent_paid, "exit_rent_out_income": 0.0, "nonfarm_component": 0.0, "service_fee": service_fee, "insurance_premium": premium, "insurance_payout": payout, "cost_component": cost, "production_cost_component": cost, "output_kg": output_kg, "yield_kg_per_mu": yield_per_mu, "price_yuan_per_jin": purchase_price + extra}


def canonical_discrete_choice(area: float, params: Mapping[str, object], action_flags: Mapping[str, Mapping[str, bool]], *, feasible_actions: Sequence[str] | None = None, crop_actions: Sequence[str] | None = None, exit_action: str = "rent_out_or_reduce_area", threshold_enabled: bool = False, threshold_key: str = "crop_continuation_threshold_yuan_per_mu", profile: Mapping[str, object] | None = None) -> dict[str, object]:
    feasible_set = set(feasible_actions) if feasible_actions is not None else set(action_flags)
    feasible = [name for name in action_flags if name in feasible_set]
    if not feasible:
        raise ValueError("canonical_discrete_choice requires at least one feasible action")
    payoffs = {name: farmer_payoff(area, params, profile=profile, **action_flags[name]) for name in feasible}
    selected = max(payoffs, key=lambda name: payoffs[name]["net"])
    crop_set = set(crop_actions) if crop_actions is not None else {name for name, flags in action_flags.items() if not flags.get("rent_out", False)}
    feasible_crop = [name for name in feasible if name in crop_set]
    best_crop = max(feasible_crop, key=lambda name: payoffs[name]["net"]) if feasible_crop else ""
    exit_candidates = [exit_action] if exit_action in payoffs else [name for name in feasible if action_flags[name].get("rent_out", False)]
    exit_selected = exit_candidates[0] if exit_candidates else ""
    threshold = _float(params, threshold_key, _float(params, "satisfaction_threshold_yuan_per_mu", 0.0)) if threshold_enabled else 0.0
    crop_net_per_mu = payoffs[best_crop]["net"] / max(area, 1.0) if best_crop else 0.0
    exit_net_per_mu = payoffs[exit_selected]["net"] / max(area, 1.0) if exit_selected else 0.0
    threshold_gap = crop_net_per_mu - threshold if threshold_enabled else 0.0
    return {"selected_action": selected, "best_crop_action": best_crop, "rent_out_or_exit_action": exit_selected, "payoffs": payoffs, "feasible_actions": feasible, "threshold_status": "enabled_crop_continuation_threshold" if threshold_enabled else "disabled_diagnostic_not_claimable", "threshold_key": threshold_key, "crop_continuation_threshold_yuan_per_mu": threshold, "best_crop_net_yuan_per_mu": crop_net_per_mu, "rent_out_or_exit_net_yuan_per_mu": exit_net_per_mu, "threshold_gap_yuan_per_mu": threshold_gap, "selected_action_net_yuan_per_mu": payoffs[selected]["net"] / max(area, 1.0)}


def select_action(area: float, params: Mapping[str, object], profile: Mapping[str, object] | None = None) -> tuple[str, dict[str, dict[str, float]]]:
    payoffs = {name: farmer_payoff(area, params, profile=profile, **flags) for name, flags in ACTION_SET.items()}
    selected = max(payoffs, key=lambda name: payoffs[name]["net"])
    if payoffs[selected]["net"] / max(area, 1.0) < _float(params, "satisfaction_threshold_yuan_per_mu"):
        selected = "keep_area_self_production_no_insurance"
    return selected, payoffs
