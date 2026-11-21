# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# --- How to run ---
# python -m pytest -q tests/test_paper_b_flagship_counterfactual.py

from __future__ import annotations

import hashlib
import json
from typing import Protocol, TypeAlias

from modules.production_succession_accounting import (
    FinalUseObservation,
    FinalUsePatch,
    ProductionSuccessionAccount,
)
from scripts.paper_b_flagship_scenarios import Config, RuntimeEnvironment


Scalar: TypeAlias = str | int | float | bool | None
Row: TypeAlias = dict[str, Scalar]


class RuntimeCrop(Protocol):
    value: str


class RuntimeOperator(Protocol):
    unique_id: str
    representation_weight_q: float
    grain_area: float
    cash_crop_area: float
    grain_production: float
    agricultural_income: float
    subsidy_income: float
    subsidy_revenue: float
    scale_subsidy_revenue: float
    cash_production_cost: float
    service_fee_expense: float
    land_rental_expense: float
    imputed_family_labor_cost: float
    profit: float
    planned_crop: RuntimeCrop | None


class RuntimePatch(Protocol):
    area: float
    owner: RuntimeOperator | None
    leased_to: RuntimeOperator | None
    current_operator: RuntimeOperator | None
    current_crop: RuntimeCrop | None
    production_succession_release_event_id: str

    def get_patch_id(self) -> str: ...


class RuntimeLandBank(Protocol):
    production_succession_account: ProductionSuccessionAccount


class RuntimePopulationManager(Protocol):
    land_bank: RuntimeLandBank
    weighted_allocation_annual_stats: list[Row]


class FlagshipModel(Protocol):
    config: Config
    environment: RuntimeEnvironment
    farmers: list[RuntimeOperator]
    agricultural_enterprises: list[RuntimeOperator]
    land_patches: list[RuntimePatch]
    population_manager: RuntimePopulationManager
    simulation_base_year: int
    step_count: int
    v4_8_fixed_rule_trace: list[Row]

    def step(self) -> None: ...


def stable_hash(config: Config) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def initial_state_hash(model: FlagshipModel) -> str:
    payload = [
        {
            "id": str(farmer.unique_id),
            "weight": float(farmer.representation_weight_q),
            "area": active_area(farmer, model.land_patches),
        }
        for farmer in model.farmers
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def observe_final_use(
    model: FlagshipModel,
    account: ProductionSuccessionAccount,
    observation_year: int,
) -> None:
    observed = {event.release_event_id for event in account.final_uses}
    for allocation in account.allocations:
        if (
            allocation.release_event_id in observed
            or observation_year <= allocation.allocation_year
        ):
            continue
        matched = [
            patch
            for patch in model.land_patches
            if str(
                getattr(patch, "production_succession_release_event_id", "") or ""
            )
            == allocation.release_event_id
        ]
        if not matched:
            continue
        final_patches: list[FinalUsePatch] = []
        for patch in matched:
            operator = patch.current_operator
            crop = patch.current_crop
            if crop is None and operator is not None:
                crop = operator.planned_crop
            final_patches.append(
                FinalUsePatch(
                    patch_id=str(patch.get_patch_id()),
                    area_mu=float(patch.area),
                    crop_label=(
                        "unoperated"
                        if operator is None
                        else None
                        if crop is None
                        else str(crop.value)
                    ),
                    final_operator_id=(
                        "unknown" if operator is None else str(operator.unique_id)
                    ),
                )
            )
        account.observe_final_use(
            FinalUseObservation(
                release_event_id=allocation.release_event_id,
                observation_year=observation_year,
                patches=tuple(final_patches),
            )
        )


def operator_snapshot(
    model: FlagshipModel, account: ProductionSuccessionAccount | None = None
) -> Row:
    active: list[tuple[float, float]] = []
    for operator in model.farmers:
        if active_area(operator, model.land_patches) > 0.0:
            active.append((household_operating_profit(operator), weight(operator)))
    for operator in model.agricultural_enterprises:
        if active_area(operator, model.land_patches) > 0.0:
            active.append((float(operator.profit), 1.0))
    all_operators = [*model.farmers, *model.agricultural_enterprises]
    fiscal = sum(float(operator.subsidy_income) for operator in model.farmers)
    fiscal += sum(
        float(operator.subsidy_revenue)
        for operator in model.agricultural_enterprises
    )
    weight_sum = sum(item_weight for _, item_weight in active)
    household_grain_area = sum(
        float(operator.grain_area) for operator in model.farmers
    )
    professional_grain_area = sum(
        float(operator.grain_area) for operator in model.agricultural_enterprises
    )
    household_grain_output = sum(
        float(operator.grain_production) for operator in model.farmers
    )
    professional_grain_output = sum(
        float(operator.grain_production)
        for operator in model.agricultural_enterprises
    )
    new_successor_ids = _new_successor_ids(model, account)
    new_successors = [
        operator
        for operator in all_operators
        if str(operator.unique_id) in new_successor_ids
    ]
    new_successor_area = sum(active_area(operator, model.land_patches) for operator in new_successors)
    new_successor_grain_output = sum(
        float(operator.grain_production) for operator in new_successors
    )
    action_counts = _current_action_counts(model)
    action_total = sum(action_counts.values())
    return {
        "grain_area_mu": sum(float(operator.grain_area) for operator in all_operators),
        "cash_crop_area_mu": sum(
            float(operator.cash_crop_area) for operator in all_operators
        ),
        "grain_output_kg": sum(
            float(operator.grain_production) for operator in all_operators
        ),
        "operator_profit_weighted_sum_yuan": sum(
            profit * item_weight for profit, item_weight in active
        ),
        "operator_weight_sum": weight_sum,
        "operator_nonnegative_weight_sum": sum(
            item_weight for profit, item_weight in active if profit >= 0.0
        ),
        "operator_profit_weighted_median_yuan": weighted_median(active),
        "fiscal_spending_yuan": fiscal,
        "land_rent_yuan_per_mu": float(model.environment.land_rental_price),
        "household_grain_area_mu": household_grain_area,
        "professional_grain_area_mu": professional_grain_area,
        "household_grain_output_kg": household_grain_output,
        "professional_grain_output_kg": professional_grain_output,
        "new_successor_area_mu": new_successor_area,
        "new_successor_grain_output_kg": new_successor_grain_output,
        "new_successor_area_share": _ratio(new_successor_area, sum(active_area(operator, model.land_patches) for operator in all_operators)),
        "new_successor_output_share": _ratio(
            new_successor_grain_output,
            sum(float(operator.grain_production) for operator in all_operators),
        ),
        "grain_yield_growth_rate": _grain_yield_growth_rate(model),
        "grain_yield_multiplier": _grain_yield_multiplier(model),
        "action_total_count": action_total,
        **{
            f"action_{action}_count": count for action, count in action_counts.items()
        },
        **{
            f"action_{action}_share": _ratio(count, action_total) for action, count in action_counts.items()
        },
    }


def _current_action_counts(model: FlagshipModel) -> dict[str, int]:
    trace = model.v4_8_fixed_rule_trace
    actions = ("maintain_grain", "expand_cash", "contract", "exit")
    if not trace:
        return dict.fromkeys(actions, 0)
    if model.config.get("runtime_fixed_rule", {}).get("household_crop_portfolio", False):
        actions += ("portfolio_keep", "portfolio_down", "portfolio_up")
    latest_year = max(int(row.get("year", 0) or 0) for row in trace)
    return {
        action: sum(
            1
            for row in trace
            if int(row.get("year", 0) or 0) == latest_year
            and str(row.get("action", "")) == action
        )
        for action in actions
    }


def _new_successor_ids(
    model: FlagshipModel, account: ProductionSuccessionAccount | None
) -> set[str]:
    if account is None:
        return set()
    current_year = model.simulation_base_year + model.step_count
    return {
        event.destination_operator_id
        for event in account.allocations
        if event.succession_role == "new_entrant"
        and event.allocation_year <= current_year
    }


def _grain_yield_growth_rate(model: FlagshipModel) -> float:
    production = model.config.get("production", {})
    if not isinstance(production, dict):
        return 0.0
    technology = production.get("technology_progress", {})
    if not isinstance(technology, dict):
        return 0.0
    value = technology.get("grain_annual_yield_growth_rate", 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _grain_yield_multiplier(model: FlagshipModel) -> float:
    return (1.0 + _grain_yield_growth_rate(model)) ** model.step_count


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def household_operating_profit(operator: RuntimeOperator) -> float:
    return (
        float(operator.agricultural_income)
        + float(operator.subsidy_income)
        - float(operator.cash_production_cost)
        - float(operator.service_fee_expense)
        - float(operator.land_rental_expense)
        - float(operator.imputed_family_labor_cost)
    )


def operator_weights(operators: list[RuntimeOperator]) -> dict[str, float]:
    return {str(operator.unique_id): weight(operator) for operator in operators}


def weight(operator: RuntimeOperator) -> float:
    return max(0.0, float(operator.representation_weight_q or 1.0))


def active_area(operator: RuntimeOperator, patches: list[RuntimePatch]) -> float:
    return sum(
        float(patch.area)
        for patch in patches
        if patch.current_operator is operator
        or patch.leased_to is operator
        or (
            patch.owner is operator
            and patch.leased_to in (None, operator)
            and patch.current_operator in (None, operator)
        )
    )


def weighted_median(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = sum(item_weight for _, item_weight in ordered) / 2.0
    cumulative = 0.0
    for value, item_weight in ordered:
        cumulative += item_weight
        if cumulative >= midpoint:
            return value
    return ordered[-1][0]
