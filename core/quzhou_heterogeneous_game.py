from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Final, Literal, Mapping, Sequence

from core.quzhou_dynamic_game import (
    Beliefs,
    GameState,
    GovernmentAction,
    VillageAction,
    _effective_action_flags,
    _policy_delta,
    _policy_params,
    village_best_response,
)
from core.quzhou_game_payoff import farmer_payoff
from core.quzhou_annual_season_accounts import AnnualAccount, AnnualSettlement, SeasonAccount, settle_year

MAX_EXACT_OPERATOR_COUNT: Final = 6


@dataclass(frozen=True, slots=True)
class OperatorProfile:
    operator_id: str
    area_mu: float
    parameter_overrides: Mapping[str, float]
    characteristics: Mapping[str, float]
    annual_account: AnnualAccount | None = None


@dataclass(frozen=True, slots=True)
class HeterogeneousOperatorGameResult:
    actions: Mapping[str, str]
    payoffs: Mapping[str, float]
    outputs_kg: Mapping[str, float]
    service_fill_rate: float
    land_fill_rate: float
    system_score: float
    pure_nash: bool
    equilibrium_concept: str


@dataclass(frozen=True, slots=True)
class OperatorGameContext:
    state: GameState
    base: Mapping[str, float]
    government_action: GovernmentAction
    village_action: VillageAction
    beliefs: Beliefs
    service_capacity_mu: float
    available_land_mu: float
    expansion_demand_mu: float = 0.0
    matching_capacity_mu: float | None = None


@dataclass(frozen=True, slots=True)
class HeterogeneousSequentialGameResult:
    government_action: GovernmentAction
    village_action: VillageAction
    operator_game: HeterogeneousOperatorGameResult
    government_payoff: float
    system_score: float
    equilibrium_concept: str


def _evaluate_operator_profile(
    actions: Sequence[str],
    profiles: Sequence[OperatorProfile],
    context: OperatorGameContext,
) -> tuple[dict[str, float], dict[str, float], float, float, float]:
    _, allocated, service_fill, land_fill = _operator_allocation(actions, profiles, context)
    payoffs: dict[str, float] = {}
    outputs: dict[str, float] = {}
    for profile, action in zip(profiles, actions, strict=True):
        original, expansion = _operator_payoff_parts(profile, action, context, allocated[profile.operator_id], service_fill)
        capital = original["required_capital_yuan"] + expansion["required_capital_yuan"]
        labor = original["required_labor_days"] + expansion["required_labor_days"]
        feasible = capital <= profile.characteristics.get("working_capital_yuan", float("inf")) + 1e-8 and labor <= profile.characteristics.get("available_labor_days", float("inf")) + 1e-8
        payoffs[profile.operator_id] = original["net"] + expansion["net"] if feasible else -float("inf")
        outputs[profile.operator_id] = original["output_kg"] + expansion["output_kg"]
    return payoffs, outputs, service_fill, land_fill, sum(payoffs.values())


def _operator_allocation(
    actions: Sequence[str], profiles: Sequence[OperatorProfile], context: OperatorGameContext,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    flags = _effective_action_flags(context.village_action)
    applicant_area = sum(
        profile.area_mu
        for profile, action in zip(profiles, actions, strict=True)
        if flags[action].get("rent_in", False)
    )
    service_base = sum(p.area_mu for p, a in zip(profiles, actions, strict=True) if flags[a].get("service", False))
    service_upper = service_base + max(0.0, context.available_land_mu)
    assured_fill = min(1.0, context.service_capacity_mu / service_upper) if service_upper > 0.0 else 1.0
    demands: dict[str, float] = {}
    for profile, action in zip(profiles, actions, strict=True):
        demand = context.expansion_demand_mu * profile.area_mu / applicant_area if flags[action].get("rent_in", False) and applicant_area > 0.0 else 0.0
        if flags[action].get("rent_in", False) and "working_capital_yuan" in profile.characteristics:
            char = profile.characteristics
            original, unit = _operator_payoff_parts(profile, action, context, 1.0, 1.0)
            limited, limited_unit = _operator_payoff_parts(profile, action, context, 1.0, assured_fill)
            demand = max(0.0, min(char["expansion_limit_mu"],
                (char["working_capital_yuan"] - max(original["required_capital_yuan"], limited["required_capital_yuan"])) / max(unit["required_capital_yuan"], limited_unit["required_capital_yuan"]),
                (char["available_labor_days"] - limited["required_labor_days"]) / limited_unit["required_labor_days"]))
            _, expansion = _operator_payoff_parts(profile, action, context, demand, assured_fill)
            if expansion["net"] <= 0.0:
                demand = 0.0
        demands[profile.operator_id] = demand
    land_demand = sum(demands.values())
    matching_limit = context.available_land_mu if context.matching_capacity_mu is None else min(context.available_land_mu, context.matching_capacity_mu)
    land_fill = 1.0 if land_demand <= 0.0 else min(1.0, max(0.0, matching_limit) / land_demand)
    allocated = {
        profile.operator_id: (
            demands[profile.operator_id] * land_fill
            if flags[action].get("rent_in", False) and applicant_area > 0.0
            else 0.0
        )
        for profile, action in zip(profiles, actions, strict=True)
    }
    service_demand = sum(
        profile.area_mu + allocated[profile.operator_id]
        for profile, action in zip(profiles, actions, strict=True)
        if flags[action].get("service", False)
    )
    service_fill = 1.0 if service_demand <= 0.0 else min(1.0, max(0.0, context.service_capacity_mu) / service_demand)
    return demands, allocated, service_fill, land_fill


def _operator_payoff_parts(
    profile: OperatorProfile, action: str, context: OperatorGameContext, expansion_area: float, service_fill: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if profile.annual_account is not None:
        annual = annual_operator_settlement(profile, action, context, expansion_area, service_fill)
        return dict(annual.original), dict(annual.expansion)
    params = _policy_params(dict(context.base) | dict(profile.parameter_overrides), context.government_action, context.beliefs)
    selected_flags = _effective_action_flags(context.village_action)[action]
    total_area = profile.area_mu + expansion_area
    if total_area < profile.characteristics.get("scale_subsidy_threshold_mu", 0.0):
        params["scale_subsidy_yuan_per_mu"] = 0.0
    original = _service_settled_payoff(profile.area_mu, params, profile.characteristics, selected_flags | {"rent_in": False}, service_fill)
    expansion_params = dict(params)
    if "working_capital_yuan" in profile.characteristics and "order_delivery_kg" in params:
        expansion_params["order_delivery_kg"] = 0.0
    expansion = _service_settled_payoff(expansion_area, expansion_params, profile.characteristics, selected_flags, service_fill) if selected_flags.get("rent_in", False) and expansion_area > 0.0 else {key: 0.0 for key in original}
    for area, values in ((profile.area_mu, original), (expansion_area, expansion)):
        values["required_capital_yuan"] = sum(values.get(key, 0.0) for key in ("cost_component", "service_fee", "insurance_premium", "land_rent_paid"))
        saving = profile.characteristics.get("service_labor_saving_rate", 0.0) * service_fill if selected_flags.get("service", False) else 0.0
        values["required_labor_days"] = 0.0 if selected_flags.get("rent_out", False) else area * profile.characteristics.get("self_labor_days_per_mu", 0.0) * (1.0 - saving)
    return original, expansion


def annual_operator_settlement(profile: OperatorProfile, action: str, context: OperatorGameContext,
                               expansion_area: float, service_fill: float) -> AnnualSettlement:
    account = profile.annual_account
    assert account is not None
    raw = dict(context.base) | dict(profile.parameter_overrides)
    effective = _policy_params(raw, context.government_action, context.beliefs)
    effective["extra_subsidy_yuan_per_jin"] = raw["extra_subsidy_yuan_per_jin"]
    effective["risk_aversion_weight"] *= profile.characteristics.get("risk_aversion_multiplier", 1.0)
    production = effective | {"scale_subsidy_yuan_per_mu": 0.0, "extra_subsidy_yuan_per_jin": 0.0,
        "land_rent_paid_yuan_per_mu": 0.0, "rent_out_income_yuan_per_mu": 0.0}
    flags = _effective_action_flags(context.village_action)[action]
    if flags.get("rent_out", False):
        account = replace(account, released_at_year_end_mu=profile.area_mu, year_end_reason="exit")
    seasons = list(account.other_seasons)
    if account.completed_early is not None:
        seasons.append(account.completed_early)
    annual_areas: tuple[tuple[Literal["original", "expansion"], float], ...] = (("original", profile.area_mu), ("expansion", expansion_area))
    for part, area in annual_areas:
        if flags.get("rent_out", False) or area <= 0 or (part == "original" and account.completed_early is not None):
            continue
        if part == "expansion" and not flags.get("rent_in", False):
            continue
        values = _service_settled_payoff(area, production, profile.characteristics, flags, service_fill)
        saving = profile.characteristics.get("service_labor_saving_rate", 0.0) * service_fill if flags.get("service", False) else 0.0
        verified = account.verified_order_kg if part == "original" else None
        share = account.conditional_order_share if part == "original" else 0.0
        sold = account.early_sold_kg if part == "original" and account.early_sold_kg is not None else values["output_kg"]
        quantity_basis = "explicit_sales_quantity" if part == "original" and account.early_sold_kg is not None else "conditional_all_harvest_sold"
        seasons.append(SeasonAccount("early_rice", area, values["output_kg"], sold,
            raw["purchase_price_yuan_per_jin"] * 2, values["cost_component"], values["service_fee"],
            area * profile.characteristics.get("self_labor_days_per_mu", 0.0) * (1 - saving), values["insurance_premium"],
            flags.get("insurance", False), flags.get("order", False), verified, share, False,
            f"{quantity_basis};verified_order_separate_or_explicit_share", part))
    result = settle_year(account, tuple(seasons), profile.area_mu, expansion_area, effective)
    assert result.original["unresolved_support_count"] + result.expansion["unresolved_support_count"] == 0, "unresolved annual support cannot enter QNEW payoff; supply explicit amounts or conditions"
    return result


def _service_settled_payoff(
    area_mu: float,
    params: Mapping[str, float],
    characteristics: Mapping[str, float],
    selected_flags: Mapping[str, bool],
    service_fill: float,
) -> dict[str, float]:
    if not selected_flags.get("service", False):
        return farmer_payoff(area_mu, params, profile=characteristics, **selected_flags)
    without_flags = dict(selected_flags)
    without_flags["service"] = False
    without = farmer_payoff(area_mu, params, profile=characteristics, **without_flags)
    full = farmer_payoff(area_mu, params, profile=characteristics, **selected_flags)
    settled = dict(full)
    settled["net"] = without["net"] + service_fill * (full["net"] - without["net"])
    settled["service_fee"] = full["service_fee"] * service_fill
    settled["cost_component"] = without["cost_component"] + service_fill * (full["cost_component"] - without["cost_component"])
    settled["production_cost_component"] = settled["cost_component"]
    return settled


def solve_heterogeneous_operator_game(
    state: GameState,
    base: Mapping[str, float],
    profiles: Sequence[OperatorProfile],
    government_action: GovernmentAction,
    village_action: VillageAction,
    beliefs: Beliefs,
    *,
    service_capacity_mu: float,
    available_land_mu: float,
) -> HeterogeneousOperatorGameResult:
    if len(profiles) > MAX_EXACT_OPERATOR_COUNT:
        raise ValueError(f"exact operator subgame supports at most {MAX_EXACT_OPERATOR_COUNT} representative profiles")
    action_names = tuple(_effective_action_flags(village_action))
    context = OperatorGameContext(
        state,
        base,
        government_action,
        village_action,
        beliefs,
        service_capacity_mu,
        available_land_mu,
    )
    best_pure: HeterogeneousOperatorGameResult | None = None
    best_any: HeterogeneousOperatorGameResult | None = None
    for actions in product(action_names, repeat=len(profiles)):
        payoffs, outputs, service_fill, land_fill, score = _evaluate_operator_profile(actions, profiles, context)
        pure_nash = True
        for index, profile in enumerate(profiles):
            for alternative in action_names:
                if alternative == actions[index]:
                    continue
                deviation = list(actions)
                deviation[index] = alternative
                deviation_payoffs, _, _, _, _ = _evaluate_operator_profile(deviation, profiles, context)
                if deviation_payoffs[profile.operator_id] > payoffs[profile.operator_id] + 1e-9:
                    pure_nash = False
                    break
            if not pure_nash:
                break
        concept = "exact_pure_strategy_nash_operator_subgame" if pure_nash else "no_pure_strategy_nash_profile"
        result = HeterogeneousOperatorGameResult(
            dict(zip((profile.operator_id for profile in profiles), actions, strict=True)),
            payoffs,
            outputs,
            service_fill,
            land_fill,
            score,
            pure_nash,
            concept,
        )
        if best_any is None or result.system_score > best_any.system_score:
            best_any = result
        if pure_nash and (best_pure is None or result.system_score > best_pure.system_score):
            best_pure = result
    if best_pure is not None:
        return best_pure
    if best_any is None:
        return HeterogeneousOperatorGameResult({}, {}, {}, 1.0, 1.0, 0.0, True, "exact_pure_strategy_nash_operator_subgame")
    return best_any


def solve_heterogeneous_sequential_game(
    state: GameState,
    base: Mapping[str, float],
    profiles: Sequence[OperatorProfile],
    beliefs: Beliefs,
    *,
    service_capacity_mu: float,
    available_land_mu: float,
) -> HeterogeneousSequentialGameResult:
    best: HeterogeneousSequentialGameResult | None = None
    for government_action in GovernmentAction:
        village_action = village_best_response(state, government_action)
        operator_game = solve_heterogeneous_operator_game(
            state,
            base,
            profiles,
            government_action,
            village_action,
            beliefs,
            service_capacity_mu=service_capacity_mu,
            available_land_mu=available_land_mu,
        )
        fiscal_cost = state.operated_area_mu * sum(_policy_delta(government_action)[::2]) * 100.0
        system_score = operator_game.system_score - fiscal_cost
        concept = (
            "finite_stackelberg_candidate_with_exact_operator_subgame"
            if operator_game.pure_nash
            else "finite_stackelberg_candidate_without_pure_operator_subgame"
        )
        candidate = HeterogeneousSequentialGameResult(
            government_action,
            village_action,
            operator_game,
            system_score,
            system_score,
            concept,
        )
        if best is None or candidate.system_score > best.system_score:
            best = candidate
    if best is None:
        raise AssertionError("government action set must not be empty")
    return best
