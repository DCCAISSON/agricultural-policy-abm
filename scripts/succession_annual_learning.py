"""Shared openings and finite-horizon fitted policy tables."""

from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
from math import ceil, fsum
from statistics import mean

import numpy as np

from core.land_succession import CROPS, Succession
from scripts.run_succession_shared_budget import JointBudget, JointMove, OFF, setup

Worlds = tuple[Succession, Succession]
State = tuple[int, int, int, int]
Branch = tuple[float, State, float]
Opening = tuple[int, State, list[Branch]]


def roots(seed: int, design: dict, cash_reset: bool = False) -> Worlds:
    _, (a, b) = setup(design)
    if cash_reset:
        a, b = replace(a, carry_cash=False), replace(b, carry_cash=False)
    return Succession(2 * seed, a), Succession(2 * seed + 1, b)


def observation(worlds: Worlds, design: dict) -> State:
    a, b = worlds
    return (
        design["active_years"] - a.year,
        grain_bin(a.observed["grain_kg"], a.land, design),
        grain_bin(b.observed["grain_kg"], b.land, design),
        design["budget_yuan"],
    )


def grain_bin(grain: float, land: float, design: dict) -> int:
    return bisect_right(design["grain_bin_edges"], grain / (land * CROPS["grain"][0]))


def branch_opening(
    worlds: Worlds, budget: JointBudget, moves: tuple[JointMove, ...], design: dict,
) -> tuple[list[Branch], list[Worlds]]:
    """Factor the menu across independent economies after fixing reservations."""
    local = ({}, {})
    last = worlds[0].year + 1 == design["active_years"]
    for move in moves:
        if all(option in lookup for option, lookup in zip(move, local)):
            continue
        a, b = worlds
        next_worlds = a.fork(), b.fork()
        rows = budget.advance(next_worlds, move)
        tail = [0.0, 0.0]
        if last:
            a1, b1 = next_worlds
            terminal = a1.fork(), b1.fork()
            for _ in range(design["horizon"] - design["active_years"]):
                for i, row in enumerate(budget.advance(terminal, (OFF, OFF))):
                    tail[i] += row["grain_kg"]
        for i, option in enumerate(move):
            local[i][option] = (next_worlds[i], rows[i]["grain_kg"], tail[i])
    rewards, successors = [], []
    for a, b in moves:
        wa, ya, ta = local[0][a]
        wb, yb, tb = local[1][b]
        successor = wa, wb
        rewards.append((ya + yb, observation(successor, design), ta + tb))
        successors.append(successor)
    return rewards, successors


def explore(seed: int, design: dict) -> list[Opening]:
    budget, _ = setup(design)
    moves = budget.feasible_moves()
    worlds = roots(seed, design)
    draws = np.random.default_rng([seed, 7753]).integers(len(moves), size=design["active_years"])
    rows = []
    for index in draws:
        state = observation(worlds, design)
        branches, successors = branch_opening(worlds, budget, moves, design)
        rows.append((seed, state, branches))
        worlds = successors[index]
    return rows


def run_path(
    seed: int, design: dict, strategy: str, *, fixed: int = 0,
    table: dict | None = None, cash_reset: bool = False,
) -> list[dict]:
    from scripts.run_succession_shared_budget import METRICS

    budget, _ = setup(design)
    moves = budget.feasible_moves()
    worlds = roots(seed, design, cash_reset)
    rows = []
    for year in range(1, design["horizon"] + 1):
        state = observation(worlds, design)
        entry = table.get(state_key(state)) if table is not None else None
        if state[0] <= 0:
            action, reason = 0, "withdrawal"
        elif table is None:
            action, reason = fixed, "fixed"
        elif entry is None:
            action, reason = 0, "unseen"
        elif not entry["supported"]:
            action, reason = 0, "sparse"
        else:
            action, reason = entry[strategy.split("_")[0]], "supported"
        for region, row in zip(design["regions"], budget.advance(worlds, moves[action])):
            rows.append(dict(
                seed=seed, strategy=strategy, cash_reset=int(cash_reset), year=year, region=region["id"],
                observation=state_key(state), action=action, reason=reason,
                **{k: row[k] for k in ("policy", "reservation", "guarantee_mu", *METRICS)},
            ))
    return rows


def state_key(state: State) -> str:
    return ":".join(map(str, state))


def fit(openings: list[Opening], design: dict, pool: dict[int, float]) -> tuple[dict, dict]:
    grouped = defaultdict(list)
    for _, state, branches in openings:
        grouped[state].append(branches)
    table, value = {}, {}
    by_year = defaultdict(int)
    for _, state, _ in openings:
        by_year[state[0]] += 1
    minimum = design["min_openings_per_state_action"]
    common, supported = 0, 0
    for state in sorted(grouped):
        remaining = state[0]
        rows = grouped[state]
        current, q = [], []
        for branches in zip(*rows, strict=True):
            current.append(mean(r for r, _, _ in branches))
            q.append(mean(
                r + (tail if remaining == 1 else value.get(next_state, pool[design["active_years"] - remaining + 2]))
                for r, next_state, tail in branches
            ))
        enough = len(rows) >= minimum
        q_choice = max(range(len(q)), key=q.__getitem__) if enough else 0
        myopic_choice = max(range(len(current)), key=current.__getitem__) if enough else 0
        value[state] = q[q_choice]
        frequent = len(rows) >= ceil(design["common_state_fraction_per_year"] * by_year[remaining])
        common += frequent * len(q)
        supported += (frequent and enough) * len(q)
        table[state_key(state)] = dict(
            openings=len(rows), common=frequent, supported=enough,
            q=q_choice, myopic=myopic_choice, q_values=q, current_grain=current,
        )
    return table, dict(
        distinct_states=len(grouped), opening_visits=len(openings),
        common_state_actions=common, supported_common_state_actions=supported,
        coverage=supported / common,
    )


def p0_tail(paths: list[list[dict]], horizon: int) -> dict[int, float]:
    return {start: mean(fsum(r["grain_kg"] for r in path if r["year"] >= start) for path in paths)
            for start in range(1, horizon + 1)}
