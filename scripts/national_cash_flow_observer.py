# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python -m pytest -q tests/test_national_cash_flow_observer.py
"""Read settled national accounts before demographic turnover; never settle them."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, Protocol, TypeAlias

Scalar: TypeAlias = str | int | float | bool | None
Row: TypeAlias = dict[str, Scalar]


class AccountActor(Protocol):
    unique_id: str


class AccountModel(Protocol):
    simulation_base_year: int
    step_count: int
    farmers: Sequence[AccountActor]
    agricultural_enterprises: Sequence[AccountActor]


ACTOR_FIELDS: Final = (
    "subsidy_income", "scale_subsidy_income", "subsidy_revenue",
    "scale_subsidy_revenue", "land_rental_income", "land_rental_expense",
    "land_rental_cost", "service_fee_expense",
)
PAYMENT_FIELDS: Final = (
    "recipient", "claim_id", "claim_step", "verified_step", "payable_step",
    "amount", "basic_cash", "scale_cash", "in_kind_value", "planned_amount",
    "scaling_factor", "arrears_amount", "type", "verification_path",
)


def _number(value: Scalar) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        message = "nonfinite settled monetary observation"
        raise ValueError(message)
    return number


def _total(rows: Iterable[Mapping[str, Scalar]], field: str) -> float | None:
    values = [_number(row.get(field)) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _difference(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left - right


def observe_national_cash_flows(
    model: AccountModel, scenario_id: str, runtime_seed: int,
) -> Row:
    """Return one CSV-ready world/year row at the completed-production hook.

    Monetary fields already cover the represented patches: do not multiply by q.
    Nested records are JSON strings. Missing accounts remain null, never zero.
    """
    step = int(model.step_count)
    actor_rows: list[Row] = []
    receipt_rows: list[Row] = []
    missing_histories = 0
    for actor_type, actors in (
        ("farmer", model.farmers), ("enterprise", model.agricultural_enterprises),
    ):
        for actor in actors:
            actor_rows.append({
                "actor_id": str(actor.unique_id), "actor_type": actor_type,
                "fixed_rule_exit_pending": bool(getattr(actor, "fixed_rule_exit_pending", False)),
                "has_cash_attribute": hasattr(actor, "cash"),
                **{field: _number(getattr(actor, field, None)) for field in ACTOR_FIELDS},
            })
            history = getattr(actor, "subsidy_payment_history", None)
            if history is None:
                missing_histories += 1
                continue
            for payment in history:
                if payment.get("step") == step:
                    receipt_rows.append({
                        "actor_id": str(actor.unique_id), "actor_type": actor_type,
                        **{key: payment.get(key) for key in (
                            "step", "claim_id", "cash_amount", "basic_cash",
                            "scale_cash", "in_kind_value",
                        )},
                    })
    farmers = [row for row in actor_rows if row["actor_type"] == "farmer"]
    enterprises = [row for row in actor_rows if row["actor_type"] == "enterprise"]
    government = getattr(model, "government", None)
    raw_records = getattr(government, "subsidy_records", None)
    payments = [
        {field: record.get(field) for field in PAYMENT_FIELDS}
        for record in (raw_records or ())
    ]
    government_cash = _total(payments, "amount") if raw_records is not None else None
    government_in_kind = _total(payments, "in_kind_value") if raw_records is not None else None
    receipts_cash = _total(receipt_rows, "cash_amount") if missing_histories == 0 else None
    receipts_in_kind = _total(receipt_rows, "in_kind_value") if missing_histories == 0 else None
    household_cash = _total(farmers, "subsidy_income")
    enterprise_cash = _total(enterprises, "subsidy_revenue")
    enterprise_scale = _total(enterprises, "scale_subsidy_revenue")
    receipt_stock = None if household_cash is None or enterprise_cash is None else household_cash + enterprise_cash
    legacy_fiscal = None if receipt_stock is None or enterprise_scale is None else receipt_stock + enterprise_scale
    arrears = getattr(government, "arrears_paid_this_year", None)
    arrears_paid = None if arrears is None else _total(({"amount": value} for value in arrears.values()), "amount")
    budget = _number(getattr(government, "current_annual_subsidy_budget", None))
    remaining = _number(getattr(government, "remaining_budget", None))
    paid_total = _number(getattr(government, "total_subsidy_paid", None))
    budget_residual = _difference(_difference(_difference(budget, remaining), paid_total), arrears_paid)
    queue = getattr(government, "verified_subsidy_queue", None)
    service_stats = getattr(model, "service_market_stats", {})
    providers = getattr(model, "service_providers", ())
    villages = getattr(model, "village_collectives", ())
    village_rows = [
        {field: _number(getattr(getattr(village, "economy_state", None), field, None))
         for field in ("land_rental_income", "annual_service_fee")}
        for village in villages
    ]
    actor_ids = {row["actor_id"] for row in actor_rows}
    pending_ids = {row["actor_id"] for row in actor_rows if row["fixed_rule_exit_pending"]}
    matched_cash = _total((row for row in payments if str(row["recipient"]) in actor_ids), "amount")
    absent_recipient_cash = _total((row for row in payments if str(row["recipient"]) not in actor_ids), "amount")
    missing: list[str] = []
    if missing_histories:
        missing.append(f"payment_history_missing_for_{missing_histories}_actors")
    if raw_records is None:
        missing.append("government_subsidy_records_missing")
    result: Row = {
        "scenario_id": scenario_id, "runtime_seed": runtime_seed,
        "simulation_year": int(model.simulation_base_year + step), "step": step,
        "snapshot_timing": "pre_turnover_completed_production_year",
        "monetary_weighting": "represented_patch_amounts_no_additional_q",
        "farmer_count_pre_turnover": len(farmers), "enterprise_count_pre_turnover": len(enterprises),
        "government_payment_record_count": None if raw_records is None else len(payments),
        "government_cash_paid_yuan": government_cash,
        "government_basic_cash_paid_yuan": _total(payments, "basic_cash") if raw_records is not None else None,
        "government_scale_cash_paid_yuan": _total(payments, "scale_cash") if raw_records is not None else None,
        "government_in_kind_paid_value_yuan": government_in_kind,
        "government_reported_total_paid_yuan": paid_total,
        "farmer_subsidy_receipts_stock_yuan": household_cash,
        "farmer_scale_receipts_subset_yuan": _total(farmers, "scale_subsidy_income"),
        "enterprise_subsidy_receipts_stock_yuan": enterprise_cash,
        "enterprise_scale_receipts_subset_yuan": enterprise_scale,
        "actor_payment_history_cash_yuan": receipts_cash,
        "actor_payment_history_in_kind_value_yuan": receipts_in_kind,
        "government_minus_actor_history_cash_yuan": _difference(government_cash, receipts_cash),
        "government_minus_actor_history_in_kind_yuan": _difference(government_in_kind, receipts_in_kind),
        "government_minus_receipt_stock_yuan": _difference(government_cash, receipt_stock),
        "legacy_fiscal_reconstructed_yuan": legacy_fiscal,
        "legacy_enterprise_scale_double_count_yuan": _difference(legacy_fiscal, receipt_stock),
        "current_annual_subsidy_budget_yuan": budget, "remaining_budget_yuan": remaining,
        "arrears_clearance_booked_yuan": arrears_paid,
        "new_arrears_booked_yuan": _total(({"amount": v} for v in getattr(government, "current_year_arrears", {}).values()), "amount") if government is not None else None,
        "arrears_outstanding_yuan": _number(getattr(government, "total_arrears_outstanding", None)),
        "queued_planned_amount_yuan": _total(queue, "planned_amount") if queue is not None else None,
        "budget_bookkeeping_residual_yuan": budget_residual,
        "cash_to_present_actors_yuan": matched_cash if raw_records is not None else None,
        "cash_to_absent_actors_yuan": absent_recipient_cash if raw_records is not None else None,
        "cash_to_pending_exit_actors_yuan": _total((row for row in payments if str(row["recipient"]) in pending_ids), "amount") if raw_records is not None else None,
        "farmer_rent_expense_accrued_yuan": _total(farmers, "land_rental_expense"),
        "enterprise_rent_cost_accrued_yuan": _total(enterprises, "land_rental_cost"),
        "farmer_rent_income_accrued_yuan": _total(farmers, "land_rental_income"),
        "village_rent_income_state_yuan": _total(village_rows, "land_rental_income"),
        "village_service_fee_state_yuan": _total(village_rows, "annual_service_fee"),
        "farmer_service_expense_accrued_yuan": _total(farmers, "service_fee_expense"),
        "service_contract_fee_recorded_yuan": _number(service_stats.get("total_service_fee")),
        "provider_annual_revenue_state_yuan": _total(({"revenue": getattr(p, "annual_revenue", None)} for p in providers), "revenue"),
        "rent_cash_paid_yuan": None, "rent_cash_received_yuan": None,
        "service_cash_paid_yuan": None, "service_cash_received_yuan": None,
        "arrears_cash_received_yuan": None,
        "rent_observation_status": "accrual_only_no_linked_cash_settlement_ledger",
        "service_observation_status": "fee_accrual_only_provider_revenue_may_be_reset_or_cumulative",
        "arrears_observation_status": "clearance_booked_marker_not_verified_capital_receipt",
        "excluded_counter_status": "environment_annual_raw_subsidy_and_capitalization_count_expected_calls_not_payments",
        "observation_missing_json": json.dumps(missing, ensure_ascii=True),
        "government_payments_json": json.dumps(payments, ensure_ascii=True, sort_keys=True),
        "actor_receipts_json": json.dumps(receipt_rows, ensure_ascii=True, sort_keys=True),
        "actor_accounts_json": json.dumps(actor_rows, ensure_ascii=True, sort_keys=True),
    }
    return result
