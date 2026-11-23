from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.quzhou_game_payoff import (
    ACTION_SET,
    AGENT_STATE_FIELDS,
    CROP_ACTIONS,
    DECISION_FIELDS,
    MARKET_FIELDS,
    SCENARIO_FIELDS,
    SYSTEM_STATE_FIELDS,
    _bounded,
    _float,
    action_flag,
    farmer_payoff,
    farmer_type_profile,
    land_area,
    profile_float,
    profile_for_type,
    risk_penalty,
    select_action,
)
from core.quzhou_interaction_ledgers import (
    INSURANCE_RISK_LEDGER_FIELDS,
    ORDER_GRAIN_LEDGER_FIELDS,
    SERVICE_LEDGER_FIELDS,
    insurance_risk_ledger_rows,
    order_grain_ledger_rows,
    service_matching_ledger_rows,
)
from core.quzhou_land_closure import (
    LAND_BALANCE_FIELDS,
    LAND_CONSERVATION_FIELDS,
    LAND_FATE_FIELDS,
    LAND_TRANSFER_MARKET_FIELDS,
    PLOT_STATE_FIELDS,
    VILLAGE_SPATIAL_FIELDS,
    coordination_metrics,
    initialize_plot_states,
    land_stock_closure_rows,
    land_transfer_market_rows,
    plot_state_trajectory_rows,
    village_spatial_summary_rows,
)
from core.quzhou_synthetic_linkage import (
    SYNTHETIC_LINKAGE_FIELDS,
    SYNTHETIC_LINKAGE_STATUS,
    build_synthetic_linkage_rows,
    initialize_farmer_states,
    linkage_truthy,
)


def _params_with_state_risk(params: Mapping[str, object], state: Mapping[str, object]) -> dict[str, object]:
    adjusted = dict(params)
    adjusted["disaster_loss_rate"] = _bounded(float(state.get("risk_expectation") or params["disaster_loss_rate"]), 0.02, 0.50)
    return adjusted


def decision_row_from_state(state: Mapping[str, object], year: int, params: Mapping[str, object]) -> dict[str, object]:
    profile = state.get("profile") if isinstance(state.get("profile"), Mapping) else {}
    area = max(1.0, float(state["operated_area_mu"]))
    annual_params = _params_with_state_risk(params, state)
    selected, payoffs = select_action(area, annual_params, profile)
    chosen = payoffs[selected]
    inspected = payoffs["plant_early_rice_with_service_order_insurance"]
    best_crop = max((payoffs[name] for name in CROP_ACTIONS), key=lambda item: item["net"])
    return {"year": year, "agent_type": "farmer", "agent_id": state["farmer_id"], "farmer_type": state["farmer_type"], "land_area_mu": round(area, 6), "state_summary": f"land_area_mu={area:.3f};farmer_type={state['farmer_type']};risk={annual_params['disaster_loss_rate']:.4f};rent={params['rent_yuan_per_mu']}", "available_actions": ";".join(ACTION_SET), "selected_action": selected, "selected_action_net_return": round(chosen["net"], 6), "expected_net_return": round(chosen["net"], 6), "best_early_rice_net_return": round(best_crop["net"], 6), "rent_out_net_return": round(payoffs["rent_out_or_reduce_area"]["net"], 6), "early_rice_output_kg": round(inspected["output_kg"], 6), "grain_price_yuan_per_jin": params["purchase_price_yuan_per_jin"], "order_extra_subsidy_yuan_per_jin": params["extra_subsidy_yuan_per_jin"], "expected_price_with_order_yuan_per_jin": round(float(params["purchase_price_yuan_per_jin"]) + float(params["extra_subsidy_yuan_per_jin"]), 6), "expected_price_without_order_yuan_per_jin": params["purchase_price_yuan_per_jin"], "expected_yield_kg_per_mu": round(inspected["yield_kg_per_mu"], 6), "order_grain_revenue": round(inspected["order_revenue"], 6), "order_grain_extra_subsidy_component": round(inspected["extra_subsidy"], 6), "scale_subsidy": round(inspected["scale_subsidy"], 6), "early_rice_cost_component": round(inspected["cost_component"], 6), "land_rent_component": round(chosen["rent_component"], 6), "nonfarm_opportunity_component": round(chosen["nonfarm_component"], 6), "service_fee_component": round(inspected["service_fee"], 6), "insurance_premium": round(inspected["insurance_premium"], 6), "insurance_payout_expected": round(inspected["insurance_payout"], 6), "risk_penalty": round(inspected["risk_penalty"], 6), "constraint_summary": "finite_action_set;typed_sample_exposure;annual_state_evolution;copied_config_only", "information_source": "quzhou_v1_game_scope;local_moments;copied_runtime_config;prior_year_state", "feedback_path": "service_demand;land_transfer_intent;order_grain_revenue;risk_expectation;next_year_area", "claim_boundary": "sample_data_internal_annual_state_diagnostic_only"}


def farmer_decision_rows_from_states(states: Sequence[Mapping[str, object]], year: int, params: Mapping[str, object]) -> list[dict[str, object]]:
    return [decision_row_from_state(state, year, params) for state in states]


def farmer_decision_rows(model: object, year: int, params: Mapping[str, object]) -> list[dict[str, object]]:
    return farmer_decision_rows_from_states(initialize_farmer_states(model, params), year, params)


def update_farmer_state(state: Mapping[str, object], row: Mapping[str, object], params: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    profile = state.get("profile") if isinstance(state.get("profile"), Mapping) else {}
    action = str(row["selected_action"])
    area = float(row["land_area_mu"])
    crop = action != "rent_out_or_reduce_area"
    service_used = action_flag(action, "service")
    insured = action_flag(action, "insurance")
    cash_income = float(row["expected_net_return"])
    rent_alt_per_mu = float(row["rent_out_net_return"]) / max(area, 1.0)
    margin_per_mu = cash_income / max(area, 1.0)
    if crop:
        next_area = area * (1.08 if "rent_in" in action else 1.0)
        if margin_per_mu > rent_alt_per_mu + 30.0:
            next_area *= 1.03
        elif margin_per_mu < rent_alt_per_mu:
            next_area *= 0.97
        if service_used and margin_per_mu > 0:
            next_area *= 1.01
        land_status = "continued_or_expanded_early_rice"
    else:
        next_area = max(1.0, area * (1.0 - profile_float(profile, "rent_out_area_reduction_rate", 0.25)))
        land_status = "reduced_or_rented_out_area"
    prior_risk = float(state.get("risk_expectation") or params["disaster_loss_rate"])
    payout = float(row["insurance_payout_expected"] or 0.0) if insured else 0.0
    next_risk = prior_risk * 0.60 + _float(params, "disaster_loss_rate") * 0.40
    if insured and payout > 0:
        next_risk *= 0.85
    if not crop:
        next_risk *= 0.95
    next_state = dict(state) | {"operated_area_mu": round(next_area, 6), "cash_income": cash_income, "risk_expectation": round(_bounded(next_risk, 0.02, 0.50), 6), "service_used": service_used, "insured": insured, "land_transfer_status": land_status}
    early_area = area if crop else 0.0
    trajectory = {"year": row["year"], "farmer_id": row["agent_id"], "farmer_type": row["farmer_type"], "village_code": state.get("village_code", "missing_or_unusable"), "village_name": state.get("village_name", "missing_or_unusable"), "operated_area_mu": round(area, 6), "early_rice_area_mu": round(early_area, 6), "early_rice_output_kg": round(float(row["early_rice_output_kg"] or 0.0) if crop else 0.0, 6), "selected_action": action, "cash_income": round(cash_income, 6), "expected_net_return": row["expected_net_return"], "service_used": service_used, "insured": insured, "order_grain_eligible": bool(state.get("order_grain_eligible", False)), "service_access_level": state.get("service_access_level", "missing_or_unusable"), "insurance_exposure_level": state.get("insurance_exposure_level", "medium_risk_diagnostic_from_insurance_claim_summary"), "land_rent_pressure_level": state.get("land_rent_pressure_level", "missing_or_unusable"), "synthetic_linkage_status": state.get("synthetic_linkage_status", "synthetic_linkage_missing_for_this_farmer"), "insurance_payout_expected": round(payout, 6), "risk_expectation": round(float(next_state["risk_expectation"]), 6), "next_year_area_mu": round(next_area, 6), "claim_boundary": "annual_state_transition_sample_diagnostic_only_not_policy_recommendation"}
    return next_state, trajectory


def evolve_farmer_states(states: Sequence[Mapping[str, object]], farmer_rows: Sequence[Mapping[str, object]], params: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    next_states: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    state_by_id = {str(state["farmer_id"]): state for state in states}
    for row in farmer_rows:
        next_state, trajectory = update_farmer_state(state_by_id[str(row["agent_id"])], row, params)
        next_states.append(next_state)
        trajectories.append(trajectory)
    return next_states, trajectories


def _area_from_row(row: Mapping[str, object]) -> float:
    return float(row.get("land_area_mu") or row.get("operated_area_mu") or 0.0)


def actor_row(year: int, agent_type: str, agent_id: str, actions: str, selected: str, payoff: float, constraint: str, feedback: str) -> dict[str, object]:
    return {field: "" for field in DECISION_FIELDS} | {"year": year, "agent_type": agent_type, "agent_id": agent_id, "state_summary": constraint, "available_actions": actions, "selected_action": selected, "expected_net_return": round(payoff, 6), "selected_action_net_return": round(payoff, 6), "order_grain_revenue": round(payoff, 6) if agent_type == "grain_reserve_bureau" else 0.0, "constraint_summary": constraint, "information_source": "quzhou_v1_game_scope;copied_runtime_config", "feedback_path": feedback, "claim_boundary": "rule_based_local_game_diagnostic_only"}


def nonfarmer_decision_rows(year: int, farmer_rows: Sequence[Mapping[str, object]], params: Mapping[str, object], enterprises: int) -> list[dict[str, object]]:
    service_area = sum(_area_from_row(row) for row in farmer_rows if action_flag(row["selected_action"], "service"))
    capacity = max(0, enterprises) * _float(params, "service_capacity_mu_per_enterprise")
    fulfilled = min(service_area, capacity)
    return [actor_row(year, "service_provider", "service_market", "accept_orders;village_coordinated_ordering", "village_coordinated_ordering", fulfilled * (_float(params, "service_fee_yuan_per_mu") * 0.35), f"capacity_mu={capacity};demand_mu={service_area}", "service demand constrains fulfilled service area"), actor_row(year, "village_collective", "village_collective", "no_coordination;low_coordination;high_coordination", f"{params['coordination_level']}_coordination", 0.0, "not_full_fiscal_actor;sample_data", "coordination lowers transaction friction and service matching frictions"), actor_row(year, "government", "fixed_policy", "announce_order_grain;announce_subsidy;announce_insurance_support", "fixed_scenario_parameters", 0.0, "no_learning;no_policy_recommendation", "policy terms enter farmer payoff only"), actor_row(year, "grain_reserve_bureau", "order_grain_channel", "apply_rule_based_purchase_price;record_eligibility", "rule_based_order_grain_revenue", sum(float(row["order_grain_revenue"] or 0) for row in farmer_rows if action_flag(row["selected_action"], "order")), "not_national_crop_price", "order-grain revenue changes farmer payoff")]


def market_feedback_row(year: int, model: object, farmer_rows: Sequence[Mapping[str, object]], params: Mapping[str, object], enterprises: int) -> dict[str, object]:
    service_demand = sum(_area_from_row(row) for row in farmer_rows if action_flag(row["selected_action"], "service"))
    capacity = enterprises * _float(params, "service_capacity_mu_per_enterprise")
    fulfilled = min(service_demand, capacity)
    reductions = params.get("coordination_friction_reduction") if isinstance(params, Mapping) else {}
    reduction = float(reductions.get(params["coordination_level"], 0.0)) if isinstance(reductions, Mapping) else 0.0
    return {"year": year, "coordination_level": params["coordination_level"], "friction_reduction": reduction, "farmer_count": len(farmer_rows), "plant_early_rice_count": sum("plant_early_rice" in str(row["selected_action"]) for row in farmer_rows), "service_demand_area_mu": service_demand, "service_capacity_mu": capacity, "fulfilled_service_area_mu": fulfilled, "service_capacity_utilization": 0.0 if capacity <= 0 else fulfilled / capacity, "average_expected_net_return": sum(float(row["expected_net_return"] or 0) for row in farmer_rows) / max(1, len(farmer_rows)), "total_order_grain_revenue": sum(float(row["order_grain_revenue"] or 0) for row in farmer_rows if action_flag(row["selected_action"], "order")), "total_insurance_payout_expected": sum(float(row["insurance_payout_expected"] or 0) for row in farmer_rows if action_flag(row["selected_action"], "insurance")), "runtime_land_rent": getattr(getattr(model, "environment", None), "land_rental_price", None), "runtime_service_fee": params["service_fee_yuan_per_mu"]}


def system_trajectory_row(year: int, trajectory_rows: Sequence[Mapping[str, object]], farmer_rows: Sequence[Mapping[str, object]], params: Mapping[str, object], enterprises: int) -> dict[str, object]:
    capacity = enterprises * _float(params, "service_capacity_mu_per_enterprise")
    service_area = sum(float(row["early_rice_area_mu"] or 0.0) for row in trajectory_rows if row["service_used"] in (True, "True"))
    total_service_fee = sum(float(row["service_fee_component"] or 0.0) for row in farmer_rows if action_flag(row["selected_action"], "service"))
    coordination = coordination_metrics(params)
    return {"year": year, "total_operated_area_mu": round(sum(float(row["operated_area_mu"] or 0.0) for row in trajectory_rows), 6), "total_early_rice_area_mu": round(sum(float(row["early_rice_area_mu"] or 0.0) for row in trajectory_rows), 6), "total_early_rice_output_kg": round(sum(float(row["early_rice_output_kg"] or 0.0) for row in trajectory_rows), 6), "plant_early_rice_count": sum("plant_early_rice" in str(row["selected_action"]) for row in trajectory_rows), "rent_out_or_reduce_count": sum(row["selected_action"] == "rent_out_or_reduce_area" for row in trajectory_rows), "service_uptake_count": sum(row["service_used"] in (True, "True") for row in trajectory_rows), "average_cash_income": round(sum(float(row["cash_income"] or 0.0) for row in trajectory_rows) / max(1, len(trajectory_rows)), 6), "average_expected_net_return": round(sum(float(row["expected_net_return"] or 0.0) for row in trajectory_rows) / max(1, len(trajectory_rows)), 6), "total_order_grain_revenue": round(sum(float(row["order_grain_revenue"] or 0.0) for row in farmer_rows if action_flag(row["selected_action"], "order")), 6), "total_service_fee": round(total_service_fee, 6), "total_insurance_payout_expected": round(sum(float(row["insurance_payout_expected"] or 0.0) for row in trajectory_rows), 6), "average_risk_expectation": round(sum(float(row["risk_expectation"] or 0.0) for row in trajectory_rows) / max(1, len(trajectory_rows)), 6), "service_capacity_utilization": 0.0 if capacity <= 0 else round(min(service_area, capacity) / capacity, 6), "village_coordination_result": f"{coordination['coordination_level']}_coordination;land_matching_friction={coordination['land_matching_friction']};service_matching_friction={coordination['service_matching_friction']};village_land_bank_capture_rate={coordination['village_land_bank_capture_rate']}", "claim_boundary": "annual_system_trajectory_sample_diagnostic_only_not_policy_recommendation"}


def run_state_evolution(states: Sequence[Mapping[str, object]], params: Mapping[str, object], years: int, enterprises: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    current_states = [dict(state) for state in states]
    all_farmer_rows: list[dict[str, object]] = []
    all_agent_state_rows: list[dict[str, object]] = []
    all_system_rows: list[dict[str, object]] = []
    for year in range(1, years + 1):
        farmer_rows = farmer_decision_rows_from_states(current_states, year, params)
        current_states, trajectory_rows = evolve_farmer_states(current_states, farmer_rows, params)
        all_farmer_rows.extend(farmer_rows); all_agent_state_rows.extend(trajectory_rows)
        all_system_rows.append(system_trajectory_row(year, trajectory_rows, farmer_rows, params, enterprises))
    return all_farmer_rows, all_agent_state_rows, all_system_rows


def scenario_sanity_row(scenario_id: str, farmer_rows: Sequence[Mapping[str, object]], system_rows: Sequence[Mapping[str, object]] | None = None, params: Mapping[str, object] | None = None) -> dict[str, object]:
    crop_rows = [row for row in farmer_rows if row["selected_action"] != "rent_out_or_reduce_area"]
    system_rows = list(system_rows or [])
    final = system_rows[-1] if system_rows else {}
    coordination = coordination_metrics(params)
    service_demand = sum(_area_from_row(row) for row in farmer_rows if action_flag(row["selected_action"], "service"))
    return {"scenario_id": scenario_id, "early_rice_area_mu": round(sum(_area_from_row(row) for row in crop_rows), 6), "early_rice_output_kg": round(sum(float(row["early_rice_output_kg"] or 0) for row in crop_rows), 6), "average_expected_net_return": round(sum(float(row["expected_net_return"] or 0) for row in farmer_rows) / max(1, len(farmer_rows)), 6), "plant_early_rice_count": sum("plant_early_rice" in str(row["selected_action"]) for row in farmer_rows), "rent_out_or_reduce_count": sum(row["selected_action"] == "rent_out_or_reduce_area" for row in farmer_rows), "service_uptake_count": sum(action_flag(row["selected_action"], "service") for row in farmer_rows), "insurance_expected_payout": round(sum(float(row["insurance_payout_expected"] or 0) for row in farmer_rows), 6), "fiscal_cost_proxy": round(sum(float(row["scale_subsidy"] or 0) + float(row["order_grain_extra_subsidy_component"] or 0) + float(row["insurance_payout_expected"] or 0) for row in crop_rows), 6), "trajectory_years": len(system_rows), "final_early_rice_area_mu": final.get("total_early_rice_area_mu", ""), "final_early_rice_output_kg": final.get("total_early_rice_output_kg", ""), "final_average_cash_income": final.get("average_cash_income", ""), "final_average_risk_expectation": final.get("average_risk_expectation", ""), "cumulative_early_rice_output_kg": round(sum(float(row.get("total_early_rice_output_kg") or 0.0) for row in system_rows), 6), "final_service_capacity_utilization": final.get("service_capacity_utilization", ""), "coordination_level": coordination["coordination_level"], "service_matching_friction": coordination["service_matching_friction"], "service_unmet_area_mu_proxy": round(service_demand * float(coordination["service_matching_friction"]), 6), "interpretation_boundary": "local_sanity_direction_only_not_policy_recommendation"}
