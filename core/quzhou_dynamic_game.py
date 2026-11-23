from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Mapping

from core.quzhou_game_payoff import ACTION_SET, canonical_discrete_choice


class GovernmentAction(StrEnum):
    MAINTAIN = "maintain"
    ORDER_SUPPORT = "order_support"
    SERVICE_COORDINATION = "service_coordination"
    INTEGRATED_REPAIR = "integrated_repair"


class VillageAction(StrEnum):
    LOW_COORDINATION = "low_coordination"
    HIGH_COORDINATION = "high_coordination"


@dataclass(frozen=True, slots=True)
class GameState:
    year: int
    operated_area_mu: float
    rent_pressure: float
    service_gap: float
    risk_expectation: float


@dataclass(frozen=True, slots=True)
class Beliefs:
    risk: float
    order_reliability: float
    service_reliability: float


@dataclass(frozen=True, slots=True)
class OperatorResponse:
    action: str
    net_return: float
    output_kg: float
    service_required: bool
    order_participation: bool
    insured: bool
    rent_in: bool
    rent_out: bool
    observes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ServiceResponse:
    accepted_area_mu: float
    capacity_mu: float
    service_margin: float


@dataclass(frozen=True, slots=True)
class EquilibriumResult:
    government_action: GovernmentAction
    village_action: VillageAction
    operator_action: str
    operator_best_response: str
    service_response: ServiceResponse
    village_payoff: float
    operator_payoff: float
    operator_output_kg: float
    government_payoff: float
    system_score: float
    beliefs: Beliefs
    next_beliefs: Beliefs
    next_state: GameState
    move_order: tuple[str, ...]
    information_sets: Mapping[str, tuple[str, ...]]
    equilibrium_concept: str


MOVE_ORDER: Final = ("government", "village", "operator", "service_provider", "market")
INFORMATION_SETS: Final = {
    "government": ("lagged_system_outcomes", "fiscal_capacity", "policy_state"),
    "village": ("local_land_release", "service_gap", "policy_signal"),
    "operator": ("own_area", "policy", "beliefs", "village_signal"),
    "service_provider": ("service_demand", "capacity", "village_action"),
}


def update_beliefs(previous: Beliefs, observed_loss: float, observed_order: bool, observed_service: bool, learning_rate: float = 0.25) -> Beliefs:
    rate = min(1.0, max(0.0, float(learning_rate)))
    risk_signal = min(0.5, max(0.0, float(observed_loss)))
    order_signal = 1.0 if observed_order else 0.0
    service_signal = 1.0 if observed_service else 0.0
    return Beliefs(
        risk=(1.0 - rate) * previous.risk + rate * risk_signal,
        order_reliability=(1.0 - rate) * previous.order_reliability + rate * order_signal,
        service_reliability=(1.0 - rate) * previous.service_reliability + rate * service_signal,
    )


def _policy_delta(action: GovernmentAction) -> tuple[float, float, float]:
    match action:
        case GovernmentAction.MAINTAIN:
            return 0.0, 0.0, 0.0
        case GovernmentAction.ORDER_SUPPORT:
            return 0.05, 0.10, 0.0
        case GovernmentAction.SERVICE_COORDINATION:
            return 0.0, 0.35, 0.15
        case GovernmentAction.INTEGRATED_REPAIR:
            return 0.05, 0.35, 0.20
        case unreachable:
            raise AssertionError(f"unhandled government action: {unreachable}")


def _policy_params(base: Mapping[str, float], action: GovernmentAction, beliefs: Beliefs) -> dict[str, float]:
    bonus_delta, coordination, insurance_delta = _policy_delta(action)
    params = dict(base)
    promised_bonus = float(base.get("extra_subsidy_yuan_per_jin", 0.0)) + bonus_delta
    params["extra_subsidy_yuan_per_jin"] = promised_bonus * beliefs.order_reliability
    service_reliability_cost = 2.0 - beliefs.service_reliability
    params["service_fee_yuan_per_mu"] = max(
        0.0,
        float(base.get("service_fee_yuan_per_mu", 0.0))
        * (1.0 - 0.25 * coordination)
        * service_reliability_cost,
    )
    if "expected_insurance_payout_yuan_per_mu" in base or insurance_delta != 0.0:
        params["expected_insurance_payout_yuan_per_mu"] = float(base.get("expected_insurance_payout_yuan_per_mu", 0.0)) + insurance_delta * 100.0
    params["disaster_loss_rate"] = beliefs.risk
    return params


def _effective_action_flags(village_action: VillageAction) -> dict[str, dict[str, bool]]:
    flags = {name: dict(values) for name, values in ACTION_SET.items()}
    if village_action is VillageAction.LOW_COORDINATION:
        flags["plant_early_rice_with_service_order_insurance"]["service"] = False
        flags["rent_in_with_service_if_profitable"]["service"] = False
    return flags


def operator_best_response(area_mu: float, base: Mapping[str, float], beliefs: Beliefs, government_action: GovernmentAction, village_action: VillageAction) -> OperatorResponse:
    params = _policy_params(base, government_action, beliefs)
    flags = _effective_action_flags(village_action)
    decision = canonical_discrete_choice(max(1.0, area_mu), params, flags, threshold_enabled=False)
    selected = str(decision["selected_action"])
    payoffs = decision["payoffs"]
    selected_payoff = payoffs[selected]
    selected_flags = flags[selected]
    return OperatorResponse(
        action=selected,
        net_return=float(selected_payoff["net"]),
        output_kg=float(selected_payoff["output_kg"]),
        service_required=bool(selected_flags.get("service", False)),
        order_participation=bool(selected_flags.get("order", False)),
        insured=bool(selected_flags.get("insurance", False)),
        rent_in=bool(selected_flags.get("rent_in", False)),
        rent_out=bool(selected_flags.get("rent_out", False)),
        observes=INFORMATION_SETS["operator"],
    )


def village_payoff(state: GameState, government_action: GovernmentAction, action: VillageAction) -> float:
    _, coordination, _ = _policy_delta(government_action)
    if action is VillageAction.LOW_COORDINATION:
        return 0.0
    return 1.2 * state.service_gap + 0.4 * state.rent_pressure + coordination - 0.45


def village_best_response(state: GameState, government_action: GovernmentAction) -> VillageAction:
    return max(VillageAction, key=lambda action: village_payoff(state, government_action, action))


def service_best_response(state: GameState, operator: OperatorResponse, village_action: VillageAction, base: Mapping[str, float]) -> ServiceResponse:
    demand = state.operated_area_mu if operator.service_required else 0.0
    multiplier = 1.0 if village_action is VillageAction.HIGH_COORDINATION else 0.6
    capacity = max(0.0, state.operated_area_mu * (1.0 - state.service_gap) * multiplier)
    accepted = min(demand, capacity)
    fee = float(base.get("service_fee_yuan_per_mu", 0.0))
    return ServiceResponse(accepted, capacity, accepted * fee * 0.35)


def _downstream(state: GameState, base: Mapping[str, float], beliefs: Beliefs, action: GovernmentAction) -> tuple[VillageAction, OperatorResponse, ServiceResponse, float, float]:
    village = village_best_response(state, action)
    operator = operator_best_response(state.operated_area_mu, base, beliefs, action, village)
    service = service_best_response(state, operator, village, base)
    fiscal_cost = state.operated_area_mu * (sum(_policy_delta(action)[::2]) * 100.0)
    score = operator.output_kg + 0.5 * operator.net_return + service.accepted_area_mu - fiscal_cost
    return village, operator, service, village_payoff(state, action, village), score


def rollout_with_government_action(state: GameState, base: Mapping[str, float], action: GovernmentAction, beliefs: Beliefs | None = None) -> EquilibriumResult:
    prior = beliefs or Beliefs(state.risk_expectation, 0.5, 0.5)
    village, operator, service, village_value, score = _downstream(state, base, prior, action)
    next_beliefs = update_beliefs(
        prior,
        observed_loss=state.risk_expectation,
        observed_order=operator.order_participation,
        observed_service=service.accepted_area_mu > 0.0,
    )
    next_state = GameState(
        year=state.year + 1,
        operated_area_mu=max(1.0, state.operated_area_mu * (1.02 if action is GovernmentAction.INTEGRATED_REPAIR else 0.99)),
        rent_pressure=min(1.0, max(0.0, state.rent_pressure + 0.02 * (1 if action is GovernmentAction.ORDER_SUPPORT else -1))),
        service_gap=min(1.0, max(0.0, state.service_gap - 0.1 if village is VillageAction.HIGH_COORDINATION else state.service_gap + 0.03)),
        risk_expectation=next_beliefs.risk,
    )
    return EquilibriumResult(
        action,
        village,
        operator.action,
        operator.action,
        service,
        village_value,
        operator.net_return,
        operator.output_kg,
        score,
        score,
        prior,
        next_beliefs,
        next_state,
        MOVE_ORDER,
        INFORMATION_SETS,
        "pure_strategy_sequential_equilibrium_candidate",
    )


def solve_sequential_equilibrium(state: GameState, base: Mapping[str, float], beliefs: Beliefs | None = None) -> EquilibriumResult:
    prior = beliefs or Beliefs(state.risk_expectation, 0.5, 0.5)
    candidates = [rollout_with_government_action(state, base, action, prior) for action in GovernmentAction]
    return max(candidates, key=lambda candidate: candidate.system_score)
