from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Final, TypeAlias

from core.quzhou_dynamic_game import GameState, GovernmentAction, rollout_with_government_action


RLState: TypeAlias = tuple[int, int, int, int]
Scalar: TypeAlias = str | int | float | bool | None
Row: TypeAlias = dict[str, Scalar]
ACTION_ORDER: Final = tuple(GovernmentAction)


@dataclass(slots=True)
class GovernmentQLearner:
    seed: int = 42
    learning_rate: float = 0.2
    discount_factor: float = 0.9
    epsilon: float = 0.1
    q_table: dict[RLState, list[float]] = field(default_factory=dict)
    sar_rows: list[Row] = field(default_factory=list)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def _values(self, state: RLState) -> list[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(ACTION_ORDER)
        return self.q_table[state]

    def select_action(self, state: RLState) -> int:
        values = self._values(state)
        if self._rng.random() < max(0.0, min(1.0, self.epsilon)):
            return self._rng.randrange(len(ACTION_ORDER))
        return max(range(len(values)), key=lambda index: values[index])

    def update(self, state: RLState, action: int, reward: float, next_state: RLState, done: bool) -> None:
        values = self._values(state)
        future = 0.0 if done else max(self._values(next_state))
        values[action] += self.learning_rate * (float(reward) + self.discount_factor * future - values[action])
        self.sar_rows.append({"state": state, "action": action, "reward": float(reward), "next_state": next_state, "done": done})

    def q_value(self, state: RLState, action: int) -> float:
        return self._values(state)[action]


def discretize_state(state: GameState) -> RLState:
    rent = int(min(2, max(0, state.rent_pressure * 3)))
    service = int(min(2, max(0, state.service_gap * 3)))
    risk = int(min(2, max(0, state.risk_expectation * 10 // 1)))
    year = int(state.year % 3)
    return rent, service, risk, year


def reward_from_outcome(outcome_score: float, area_mu: float) -> float:
    return float(outcome_score) / max(1.0, float(area_mu))


@dataclass(frozen=True, slots=True)
class PolicyComparison:
    rows: list[Row]
    q_tables: dict[int, dict[RLState, list[float]]]
    evaluation_rows: list[Row]


@dataclass(frozen=True, slots=True)
class FixedGovernmentPolicy:
    action_index: int = 0

    def select_action(self, state: RLState) -> int:
        return self.action_index


@dataclass(frozen=True, slots=True)
class FrozenGovernmentQPolicy:
    q_table: dict[RLState, list[float]]

    def select_action(self, state: RLState) -> int:
        values = self.q_table.get(state, [0.0] * len(ACTION_ORDER))
        return max(range(len(values)), key=lambda index: values[index])


def _run_worldline(
    base: dict[str, float],
    initial: GameState,
    action_selector: GovernmentQLearner | FixedGovernmentPolicy | FrozenGovernmentQPolicy,
    policy_mode: str,
    years: int,
    seed: int,
    *,
    learn: bool = False,
) -> list[Row]:
    state = initial
    rows: list[Row] = []
    for step in range(years):
        current = discretize_state(state)
        action_index = action_selector.select_action(current)
        action = ACTION_ORDER[action_index]
        outcome = rollout_with_government_action(state, base, action)
        next_state = discretize_state(outcome.next_state)
        rows.append(
            {
                "seed": seed,
                "step": step,
                "policy_mode": policy_mode,
                "state": current,
                "action": action.value,
                "village_action": outcome.village_action.value,
                "operator_action": outcome.operator_action,
                "reward": reward_from_outcome(outcome.system_score, state.operated_area_mu),
                "system_score": outcome.system_score,
                "operator_payoff": outcome.operator_payoff,
                "output_kg": outcome.operator_output_kg,
                "operated_area_mu": state.operated_area_mu,
                "next_state": next_state,
                "done": step == years - 1,
            }
        )
        if learn and isinstance(action_selector, GovernmentQLearner):
            action_selector.update(current, action_index, float(rows[-1]["reward"]), next_state, step == years - 1)
        state = outcome.next_state
    return rows


def run_policy_comparison(base: dict[str, float], initial: GameState, seeds: tuple[int, ...] = (42, 43, 44), years: int = 10, episodes: int = 10) -> PolicyComparison:
    rows: list[Row] = []
    evaluation_rows: list[Row] = []
    q_tables: dict[int, dict[RLState, list[float]]] = {}
    for seed in seeds:
        fixed = FixedGovernmentPolicy()
        rows.extend(_run_worldline(base, initial, fixed, "fixed", years, seed))
        learner = GovernmentQLearner(seed=seed, epsilon=0.2)
        for episode in range(max(1, episodes)):
            rows.extend(_run_worldline(base, initial, learner, "government_q", years, seed * 1000 + episode, learn=True))
        q_tables[seed] = {state: list(values) for state, values in learner.q_table.items()}
        evaluation_rows.extend(_run_worldline(base, initial, fixed, "fixed_evaluation", years, seed))
        evaluation_rows.extend(
            _run_worldline(
                base,
                initial,
                FrozenGovernmentQPolicy(q_tables[seed]),
                "government_q_evaluation",
                years,
                seed,
            )
        )
    return PolicyComparison(rows, q_tables, evaluation_rows)
