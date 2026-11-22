from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.quzhou_game_payoff import action_flag

PLOT_STATE_FIELDS = ["year", "plot_id", "village_code", "village_name", "land_class", "plot_area_mu", "is_paddy_field", "assigned_farmer_id", "current_use", "rent_pressure_factor", "service_access_factor", "risk_factor", "selected_action", "synthetic_assignment_status", "claim_boundary"]
VILLAGE_SPATIAL_FIELDS = ["year", "village_code", "village_name", "plot_count", "plot_area_mu", "paddy_plot_count", "paddy_area_mu", "early_rice_plot_count", "early_rice_area_mu", "rent_out_or_idle_plot_count", "service_access_mean", "risk_factor_mean", "synthetic_assignment_status", "claim_boundary"]
LAND_FATE_FIELDS = ["year", "village_code", "village_name", "total_area_mu", "self_early_rice_area_mu", "same_village_other_farmer_acceptance_area_mu", "service_or_scale_trusteeship_area_mu", "village_land_bank_area_mu", "idle_or_unassigned_area_mu", "non_early_rice_use_area_mu", "synthetic_unknown_area_mu", "synthetic_assignment_status", "claim_boundary"]
LAND_CONSERVATION_FIELDS = ["year", "scope", "village_code", "village_name", "total_area_mu", "allocated_area_mu", "conservation_error_mu", "abs_conservation_error_mu", "within_tolerance", "tolerance_mu", "synthetic_assignment_status", "claim_boundary"]
LAND_BALANCE_FIELDS = ["year", "village_code", "village_name", "released_land_pool_mu", "land_demand_pool_mu", "matched_land_demand_mu", "unmet_land_demand_mu", "remaining_land_pool_mu", "village_land_bank_area_mu", "idle_or_unassigned_area_mu", "land_matching_friction", "village_land_bank_capture_rate", "synthetic_assignment_status", "claim_boundary"]
LAND_TRANSFER_MARKET_FIELDS = ["year", "village_code", "village_name", "transfer_out_candidate_count", "transfer_in_candidate_count", "transfer_out_willingness_mean", "transfer_in_willingness_mean", "released_land_supply_mu", "expansion_land_demand_mu", "matched_land_mu", "service_constrained_matched_mu", "same_village_neighbor_matched_mu", "unmet_land_demand_mu", "village_land_bank_area_mu", "idle_or_unassigned_area_mu", "counterparty_relation", "neighbor_relation_rule", "contract_term_signal", "annual_feedback_fields", "matching_rule", "synthetic_assignment_status", "claim_boundary"]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _bool_value(value: object) -> bool:
    return value in (True, "True", "true", "1", 1)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return default if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return default


def _round6(value: float) -> float:
    return round(float(value), 6)


def coordination_metrics(params: Mapping[str, object] | None = None) -> dict[str, float | str]:
    if params is None:
        return {"coordination_level": "legacy_unconstrained", "land_matching_friction": 0.0, "service_matching_friction": 0.0, "village_land_bank_capture_rate": 0.60}
    level = str(params.get("coordination_level") or "none")
    reductions = params.get("coordination_friction_reduction")
    if not isinstance(reductions, Mapping):
        reductions = {"none": 0.0, "low": 0.08, "high": 0.15}
    reduction = float(reductions.get(level, reductions.get("none", 0.0)))
    return {"coordination_level": level, "land_matching_friction": round(max(0.0, 0.20 - reduction), 6), "service_matching_friction": round(max(0.0, 0.20 - reduction), 6), "village_land_bank_capture_rate": round(min(0.90, 0.55 + reduction), 6)}


def initialize_plot_states(plot_rows: Sequence[Mapping[str, object]], farmer_states: Sequence[Mapping[str, object]], max_plots: int | None = None) -> list[dict[str, object]]:
    states: list[dict[str, object]] = []
    farmer_ids = [str(state["farmer_id"]) for state in farmer_states] or ["missing_or_unusable"]
    farmers_by_village: dict[str, list[str]] = {}
    for state in farmer_states:
        code = str(state.get("village_code") or "missing_or_unusable")
        if code != "missing_or_unusable":
            farmers_by_village.setdefault(code, []).append(str(state["farmer_id"]))
    village_offsets: dict[str, int] = {}
    for index, row in enumerate(plot_rows[:max_plots] if max_plots else plot_rows):
        village_code = str(row.get("village_code") or "missing_or_unusable")
        same_village_farmers = farmers_by_village.get(village_code, [])
        if same_village_farmers:
            offset = village_offsets.get(village_code, 0)
            assigned = same_village_farmers[offset % len(same_village_farmers)]
            village_offsets[village_code] = offset + 1
        else:
            assigned = farmer_ids[index % len(farmer_ids)]
        area = max(0.0001, float(row.get("plot_area_mu") or 0.0))
        is_paddy = _truthy(row.get("is_paddy_field"))
        states.append({"plot_id": row.get("plot_id") or f"plot_{index}", "village_code": row.get("village_code") or "missing_or_unusable", "village_name": row.get("village_name") or "missing_or_unusable", "land_class": row.get("land_class_code") or row.get("land_class_name") or "missing_or_unusable", "plot_area_mu": area, "is_paddy_field": is_paddy, "assigned_farmer_id": assigned, "current_use": "paddy_candidate" if is_paddy else "non_paddy_reference", "rent_pressure_factor": round(1.0 + min(area / 50.0, 1.0), 6), "service_access_factor": 1.0 if is_paddy else 0.5, "risk_factor": 1.0, "synthetic_assignment_status": "synthetic_assignment_for_spatial_smoke_not_calibration"})
    return states


def plot_state_trajectory_rows(year: int, plot_states: Sequence[Mapping[str, object]], agent_state_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    farmer_rows = {str(row["farmer_id"]): row for row in agent_state_rows}
    rows: list[dict[str, object]] = []
    for state in plot_states:
        farmer = farmer_rows.get(str(state["assigned_farmer_id"]), {})
        action = str(farmer.get("selected_action", "missing_or_unusable"))
        is_paddy = bool(state.get("is_paddy_field"))
        early_rice = is_paddy and action not in {"rent_out_or_reduce_area", "missing_or_unusable"}
        service = action_flag(action, "service")
        risk = float(farmer.get("risk_expectation") or 0.0) if farmer else float(state.get("risk_factor") or 1.0)
        rows.append({"year": year, "plot_id": state["plot_id"], "village_code": state["village_code"], "village_name": state["village_name"], "land_class": state["land_class"], "plot_area_mu": round(float(state["plot_area_mu"]), 6), "is_paddy_field": is_paddy, "assigned_farmer_id": state["assigned_farmer_id"], "current_use": "early_rice" if early_rice else ("rent_out_or_idle" if is_paddy else "non_paddy_reference"), "rent_pressure_factor": state["rent_pressure_factor"], "service_access_factor": round(float(state["service_access_factor"]) * (1.05 if service else 1.0), 6), "risk_factor": round(risk, 6), "selected_action": action, "synthetic_assignment_status": state["synthetic_assignment_status"], "claim_boundary": "plot_state_spatial_smoke_only_not_plot_calibration"})
    return rows


def village_spatial_summary_rows(year: int, plot_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in plot_rows:
        grouped.setdefault(str(row["village_code"]), {"village_name": row["village_name"], "rows": []})["rows"].append(row)
    summaries: list[dict[str, object]] = []
    for code in sorted(grouped):
        rows = grouped[code]["rows"]
        paddy = [row for row in rows if row["is_paddy_field"] in (True, "True", "true")]
        early = [row for row in rows if row["current_use"] == "early_rice"]
        idle = [row for row in rows if row["current_use"] == "rent_out_or_idle"]
        summaries.append({"year": year, "village_code": code, "village_name": grouped[code]["village_name"], "plot_count": len(rows), "plot_area_mu": round(sum(float(row["plot_area_mu"] or 0.0) for row in rows), 6), "paddy_plot_count": len(paddy), "paddy_area_mu": round(sum(float(row["plot_area_mu"] or 0.0) for row in paddy), 6), "early_rice_plot_count": len(early), "early_rice_area_mu": round(sum(float(row["plot_area_mu"] or 0.0) for row in early), 6), "rent_out_or_idle_plot_count": len(idle), "service_access_mean": round(sum(float(row["service_access_factor"] or 0.0) for row in rows) / max(1, len(rows)), 6), "risk_factor_mean": round(sum(float(row["risk_factor"] or 0.0) for row in rows) / max(1, len(rows)), 6), "synthetic_assignment_status": "synthetic_assignment_for_spatial_smoke_not_calibration", "claim_boundary": "village_plot_summary_spatial_smoke_only_not_spatial_calibration"})
    return summaries


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _transfer_out_willingness(row: Mapping[str, object]) -> float:
    action = str(row.get("selected_action") or "")
    if action == "rent_out_or_reduce_area":
        return 1.0
    operated = _safe_float(row.get("operated_area_mu"))
    next_area = _safe_float(row.get("next_year_area_mu"), operated)
    risk = _safe_float(row.get("risk_expectation"))
    base = 0.25 if next_area < operated else 0.05
    return _round6(min(1.0, max(0.0, base + risk * 0.5)))


def _transfer_in_willingness(row: Mapping[str, object]) -> float:
    operated = _safe_float(row.get("operated_area_mu"))
    next_area = _safe_float(row.get("next_year_area_mu"), operated)
    action = str(row.get("selected_action") or "")
    base = 1.0 if next_area > operated else 0.0
    if action == "rent_in_with_service_if_profitable":
        base = max(base, 0.85)
    elif _bool_value(row.get("service_used")):
        base = max(base, 0.35)
    return _round6(min(1.0, max(0.0, base)))


def land_transfer_market_rows(year: int, plot_rows: Sequence[Mapping[str, object]], agent_state_rows: Sequence[Mapping[str, object]], params: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    """Build a privacy-safe village-level land-transfer market ledger.

    The ledger makes the transfer market explicit for the Quzhou v1 prototype while
    preserving the existing no-land-creation and same-village-first constraints.
    Farmers remain the only decision主体: service organizations, village coordination,
    and government enter as constraints or fixed environment unless their own state,
    action and objective functions are fully specified elsewhere.
    """
    fate_rows, _conservation_rows, balance_rows = land_stock_closure_rows(year, plot_rows, agent_state_rows, params)
    fate_by_village = {str(row["village_code"]): row for row in fate_rows}
    balance_by_village = {str(row["village_code"]): row for row in balance_rows}
    farmers_by_village: dict[str, list[Mapping[str, object]]] = {}
    for row in agent_state_rows:
        code = str(row.get("village_code") or "missing_or_unusable")
        farmers_by_village.setdefault(code, []).append(row)
    out: list[dict[str, object]] = []
    for code in sorted(balance_by_village):
        balance = balance_by_village[code]
        fate = fate_by_village.get(code, {})
        farmers = farmers_by_village.get(code, [])
        out_w = [_transfer_out_willingness(row) for row in farmers]
        in_w = [_transfer_in_willingness(row) for row in farmers]
        transfer_out_count = sum(value > 0.5 for value in out_w)
        transfer_in_count = sum(value > 0.5 for value in in_w)
        matched = _safe_float(balance.get("matched_land_demand_mu"))
        service_match = min(matched, max(0.0, _safe_float(fate.get("service_or_scale_trusteeship_area_mu"))))
        same_village_match = max(0.0, matched - service_match)
        out.append({
            "year": year,
            "village_code": code,
            "village_name": balance.get("village_name", "missing_or_unusable"),
            "transfer_out_candidate_count": int(transfer_out_count),
            "transfer_in_candidate_count": int(transfer_in_count),
            "transfer_out_willingness_mean": _round6(_mean(out_w)),
            "transfer_in_willingness_mean": _round6(_mean(in_w)),
            "released_land_supply_mu": balance.get("released_land_pool_mu", 0.0),
            "expansion_land_demand_mu": balance.get("land_demand_pool_mu", 0.0),
            "matched_land_mu": balance.get("matched_land_demand_mu", 0.0),
            "service_constrained_matched_mu": _round6(service_match),
            "same_village_neighbor_matched_mu": _round6(same_village_match),
            "unmet_land_demand_mu": balance.get("unmet_land_demand_mu", 0.0),
            "village_land_bank_area_mu": balance.get("village_land_bank_area_mu", 0.0),
            "idle_or_unassigned_area_mu": balance.get("idle_or_unassigned_area_mu", 0.0),
            "counterparty_relation": "same_village_other_farmer_or_service_scale_operator",
            "neighbor_relation_rule": "same_village_first;explicit_plot_adjacency_not_observed;NetLogo_neighbor_rule_recorded_as_gap",
            "contract_term_signal": "contract_duration_available_only_in_2025_large_operator_extract_not_joined_to_plot",
            "annual_feedback_fields": "next_year_area_mu;land_destination;service_used;land_pool_signal;non_early_rice_use",
            "matching_rule": "same_village_first;service_constraint_first;no_land_creation;unmatched_supply_to_land_bank_or_idle",
            "synthetic_assignment_status": "synthetic_assignment_for_spatial_smoke_not_calibration",
            "claim_boundary": "farmer_decision_subject_land_transfer_market_diagnostic_only_not_real_land_transfer_market_calibration",
        })
    return out


def land_stock_closure_rows(year: int, plot_rows: Sequence[Mapping[str, object]], agent_state_rows: Sequence[Mapping[str, object]], params: Mapping[str, object] | None = None, *, tolerance_mu: float = 1e-6) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    farmer_rows = {str(row["farmer_id"]): row for row in agent_state_rows}
    coordination = coordination_metrics(params)
    land_matching_friction = float(coordination["land_matching_friction"])
    land_bank_capture_rate = float(coordination["village_land_bank_capture_rate"])
    grouped: dict[str, dict[str, object]] = {}
    for row in plot_rows:
        grouped.setdefault(str(row["village_code"]), {"village_name": row["village_name"], "rows": []})["rows"].append(row)
    fate_rows: list[dict[str, object]] = []
    conservation_rows: list[dict[str, object]] = []
    balance_rows: list[dict[str, object]] = []
    for code in sorted(grouped):
        fate, conservation, balance = _village_land_rows(year, code, grouped[code], farmer_rows, land_matching_friction, land_bank_capture_rate, tolerance_mu)
        fate_rows.append(fate); conservation_rows.append(conservation); balance_rows.append(balance)
    total = sum(float(row["total_area_mu"]) for row in fate_rows)
    allocated_total = sum(sum(float(row[field]) for field in LAND_FATE_FIELDS[4:11]) for row in fate_rows)
    error = total - allocated_total
    conservation_rows.append({"year": year, "scope": "global", "village_code": "ALL", "village_name": "ALL", "total_area_mu": _round6(total), "allocated_area_mu": _round6(allocated_total), "conservation_error_mu": _round6(error), "abs_conservation_error_mu": _round6(abs(error)), "within_tolerance": abs(error) <= max(tolerance_mu, 1e-5), "tolerance_mu": max(tolerance_mu, 1e-5), "synthetic_assignment_status": "synthetic_assignment_for_spatial_smoke_not_calibration", "claim_boundary": "global_land_conservation_internal_check_only_not_real_land_market_calibration"})
    return fate_rows, conservation_rows, balance_rows


def _village_land_rows(year: int, code: str, group: Mapping[str, object], farmer_rows: Mapping[str, Mapping[str, object]], land_matching_friction: float, land_bank_capture_rate: float, tolerance_mu: float) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    rows = group["rows"]
    total_area = sum(_safe_float(row.get("plot_area_mu")) for row in rows)
    non_early = sum(_safe_float(row.get("plot_area_mu")) for row in rows if not _bool_value(row.get("is_paddy_field")))
    early_rows = [row for row in rows if row.get("current_use") == "early_rice"]
    service_existing = sum(_safe_float(row.get("plot_area_mu")) for row in early_rows if action_flag(row.get("selected_action"), "service"))
    self_early = sum(_safe_float(row.get("plot_area_mu")) for row in early_rows if not action_flag(row.get("selected_action"), "service"))
    released = sum(_safe_float(row.get("plot_area_mu")) for row in rows if _bool_value(row.get("is_paddy_field")) and row.get("current_use") == "rent_out_or_idle")
    expansion_demand = 0.0; service_demand = service_existing
    for farmer_id in {str(row.get("assigned_farmer_id")) for row in rows}:
        farmer = farmer_rows.get(farmer_id)
        if farmer:
            expansion = max(0.0, _safe_float(farmer.get("next_year_area_mu")) - _safe_float(farmer.get("operated_area_mu")))
            expansion_demand += expansion
            service_demand += expansion if _bool_value(farmer.get("service_used")) else 0.0
    demand = self_early + service_existing + expansion_demand
    matched = min(released * (1.0 - land_matching_friction), demand)
    service_match = min(matched, max(0.0, service_demand - service_existing))
    other_match = max(0.0, matched - service_match)
    remaining = max(0.0, released - matched)
    land_bank = remaining * land_bank_capture_rate
    idle = remaining - land_bank
    service_total = service_existing + service_match
    allocated = self_early + other_match + service_total + land_bank + idle + non_early
    error = total_area - allocated
    fate = {"year": year, "village_code": code, "village_name": group["village_name"], "total_area_mu": _round6(total_area), "self_early_rice_area_mu": _round6(self_early), "same_village_other_farmer_acceptance_area_mu": _round6(other_match), "service_or_scale_trusteeship_area_mu": _round6(service_total), "village_land_bank_area_mu": _round6(land_bank), "idle_or_unassigned_area_mu": _round6(idle), "non_early_rice_use_area_mu": _round6(non_early), "synthetic_unknown_area_mu": 0.0, "synthetic_assignment_status": "synthetic_assignment_for_spatial_smoke_not_calibration", "claim_boundary": "land_fate_internal_conservation_smoke_only_not_land_market_calibration"}
    balance = {"year": year, "village_code": code, "village_name": group["village_name"], "released_land_pool_mu": _round6(released), "land_demand_pool_mu": _round6(demand), "matched_land_demand_mu": _round6(matched), "unmet_land_demand_mu": _round6(max(0.0, demand - matched)), "remaining_land_pool_mu": _round6(remaining), "village_land_bank_area_mu": _round6(land_bank), "idle_or_unassigned_area_mu": _round6(idle), "land_matching_friction": _round6(land_matching_friction), "village_land_bank_capture_rate": _round6(land_bank_capture_rate), "synthetic_assignment_status": "synthetic_assignment_for_spatial_smoke_not_calibration", "claim_boundary": "land_demand_supply_balance_internal_smoke_only_not_calibration"}
    conservation = {"year": year, "scope": "village", "village_code": code, "village_name": group["village_name"], "total_area_mu": _round6(total_area), "allocated_area_mu": _round6(allocated), "conservation_error_mu": _round6(error), "abs_conservation_error_mu": _round6(abs(error)), "within_tolerance": abs(error) <= tolerance_mu, "tolerance_mu": tolerance_mu, "synthetic_assignment_status": "synthetic_assignment_for_spatial_smoke_not_calibration", "claim_boundary": "land_conservation_internal_check_only_not_real_land_market_calibration"}
    return fate, conservation, balance
