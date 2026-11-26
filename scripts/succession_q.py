"""Finite-horizon government learning using only lagged regional accounts."""

from collections import Counter
from dataclasses import dataclass
from math import ceil
from types import SimpleNamespace

import numpy as np

from core.land_succession import CROPS, POLICIES, Scenario, Succession

YEARS = 8
WEIGHTS = {
    "base": (1., .18, .14),
    "continuity_high": (1., .36, .14),
    "fiscal_high": (1., .18, .28),
}
STATE_NAMES = ("remaining", "low_grain", "unmet_release", "strict_continuity", "high_spending")
THRESHOLDS = (.55, .35, .05, .7)
MIN_ACTION_VISITS = 5
MIN_COVERAGE = .95
COMMON_STATE_SHARE = .01
LEARNING_RATE = .14
DISCOUNT = .92
NO_SUPPORT, GRAIN_PRE, GRAIN_POST, TAKEOVER_PRE, TAKEOVER_POST = (
    POLICIES.index(code) for code in ("none", "grain_pre", "grain_post", "takeover_pre", "takeover_post")
)
TRANSPARENT_RULE = (
    "low grain and high unmet -> grain_post",
    "low grain and low unmet -> grain_pre",
    "not low grain and strict continuity and high spending -> takeover_post",
    "not low grain and strict continuity and not high spending -> takeover_pre",
    "otherwise -> none",
)
State = tuple[int, ...]
Cell = str | int | float | bool
SAR_FIELDS = ("strategy", "stage", "episode", "seed", "condition", "year", "state",
              "next_state", "policy", "reward", "epsilon", "explored", "grain_kg",
              "continuity_area", "spending")


def observation(world: Succession) -> State:
    r = world.observed
    low, unmet, continuity, high = THRESHOLDS
    return (
        YEARS - world.year,
        int(r["grain_kg"] < low * world.land * CROPS["grain"][0]),
        int(r["offered_area"] > 0 and r["unfilled_area"] / r["offered_area"] > unmet),
        int(r["continuity_area"] >= continuity * world.land),
        int(world.scenario.budget > 0 and r["spending"] > high * world.scenario.budget),
    )


def transparent(state: State) -> int:
    _, low_grain, unmet, strong_continuity, high_spend = state
    if low_grain:
        return GRAIN_POST if unmet else GRAIN_PRE
    if strong_continuity:
        return TAKEOVER_POST if high_spend else TAKEOVER_PRE
    return NO_SUPPORT


def reward(world: Succession, result: dict[str, float], profile: str = "base") -> float:
    a, b, c = WEIGHTS[profile]
    fiscal = result["spending"] / world.scenario.budget if world.scenario.budget else 0.
    return (a * result["grain_kg"] / (world.land * CROPS["grain"][0])
            + b * result["continuity_area"] / world.land - c * fiscal)


@dataclass(frozen=True, slots=True)
class LearnedPolicy:
    strategy: str
    profile: str
    aggregated: bool
    table: dict[State, tuple[float, ...]]
    visits: dict[State, tuple[int, ...]]
    coverage_passed: bool

    def choose(self, observation: State) -> tuple[int, str]:
        state = observation[:-1] if self.aggregated else observation
        if state not in self.visits:
            return transparent(observation), "unseen_state"
        eligible = [a for a, count in enumerate(self.visits[state]) if count >= MIN_ACTION_VISITS]
        if not eligible:
            return transparent(observation), "no_supported_action"
        return max(eligible, key=self.table[state].__getitem__), ""


@dataclass(frozen=True, slots=True)
class TrainingJob:
    profile: str
    repeat: int
    seed_start: int
    scenarios: tuple[tuple[str, Scenario], ...]
    initial_paths: int

    @property
    def strategy(self) -> str:
        return f"q_{self.profile}_{self.repeat:02d}"


@dataclass(frozen=True, slots=True)
class TrainingResult:
    policy: LearnedPolicy
    sar: list[tuple[Cell, ...]]
    checkpoints: list[dict[str, Cell]]
    visits: list[dict[str, Cell]]


def _coverage(counts: dict[State, tuple[int, ...]], episodes: int) -> tuple[float, int, int]:
    common_min = max(1, ceil(COMMON_STATE_SHARE * episodes))
    common = [row for row in counts.values() if sum(row) >= common_min]
    covered = sum(sum(n >= MIN_ACTION_VISITS for n in row) for row in common)
    total = len(common) * len(POLICIES)
    return covered / total, covered, total


def train(job: TrainingJob) -> TrainingResult:
    from agents.government_agent import GovernmentAgent

    gov = SimpleNamespace(q_table={}, q_table_frozen=False, learning_rate=LEARNING_RATE,
                          discount_factor=DISCOUNT, last_state=None, last_action=None)
    counts: dict[State, Counter[int]] = {}
    sar: list[tuple[Cell, ...]] = []
    checkpoints: list[dict[str, Cell]] = []
    visits: list[dict[str, Cell]] = []
    train_rng = np.random.default_rng([job.seed_start, 283])

    def phase(stage: str, first: int, last: int, aggregated: bool, seed_offset: int):
        snapshot = {s: tuple(map(float, q)) for s, q in gov.q_table.items()}
        interval = min(256, job.initial_paths)
        for episode in range(first, last):
            seed = job.seed_start + seed_offset + episode
            condition, scenario = job.scenarios[episode % len(job.scenarios)]
            world = Succession(seed, scenario)
            epsilon = (max(.08, .55 * (1 - episode / job.initial_paths))
                       if stage == "initial" else
                       .08 if stage == "extended" else max(.08, .55 * (1 - episode / last)))
            for year in range(1, YEARS + 1):
                full = observation(world)
                state = full[:-1] if aggregated else full
                table = gov.q_table.setdefault(state, [0.] * len(POLICIES))
                explore = bool(train_rng.random() < epsilon)
                action = int(train_rng.integers(len(POLICIES))) if explore else int(np.argmax(table))
                gov.last_state, gov.last_action = state, action
                counts.setdefault(state, Counter())[action] += 1
                out = world.step(POLICIES[action])
                full_next = observation(world)
                nxt = full_next[:-1] if aggregated else full_next
                score = reward(world, out, job.profile)
                GovernmentAgent._update_q_table(gov, score, nxt, done=year == YEARS)
                sar.append((job.strategy, stage, episode + 1, seed, condition, year, str(state),
                            str(nxt), POLICIES[action], score, epsilon, explore, out["grain_kg"],
                            out["continuity_area"], out["spending"]))
            if (episode + 1) % interval == 0 or episode + 1 == last:
                table_now = {s: tuple(map(float, q)) for s, q in gov.q_table.items()}
                values = sorted(table_now.keys() & snapshot.keys())
                diffs = [abs(x - y) for s in values for x, y in zip(table_now[s], snapshot[s])]
                counts_now = {s: tuple(a[i] for i in range(len(POLICIES))) for s, a in counts.items()}
                coverage, covered, total = _coverage(counts_now, episode + 1)
                checkpoints.append(dict(
                    strategy=job.strategy, stage=stage, episodes=episode + 1, epsilon=epsilon,
                    states=len(counts_now), common_action_coverage=coverage,
                    covered_common_actions=covered, common_actions=total,
                    compared_states=len(values),
                    greedy_changes=sum(int(np.argmax(table_now[s]) != np.argmax(snapshot[s]))
                                       for s in values),
                    mean_q_change=float(np.mean(diffs)) if diffs else "",
                    max_q_change=max(diffs, default=""),
                ))
                snapshot = table_now
        table = {s: tuple(map(float, q)) for s, q in gov.q_table.items()}
        states = {s: tuple(a[i] for i in range(len(POLICIES))) for s, a in counts.items()}
        common_min = max(1, ceil(COMMON_STATE_SHARE * last))
        for state, row in sorted(states.items()):
            visits.extend(dict(strategy=job.strategy, stage=stage, state=str(state),
                               state_visits=sum(row), common=sum(row) >= common_min,
                               policy=policy, action_visits=row[a], q_value=table[state][a])
                          for a, policy in enumerate(POLICIES))
        coverage, _, _ = _coverage(states, last)
        print(f"{job.strategy} {stage}: {last} paths, common coverage={coverage:.3f}", flush=True)
        return LearnedPolicy(job.strategy, job.profile, aggregated, table, states,
                             coverage >= MIN_COVERAGE)

    n = job.initial_paths
    learned = phase("initial", 0, n, False, 0)
    if not learned.coverage_passed:
        learned = phase("extended", n, 2*n, False, 0)
    if not learned.coverage_passed:
        gov.q_table.clear()
        counts.clear()
        train_rng = np.random.default_rng([job.seed_start, 283, 1])
        learned = phase("aggregated", 0, 2*n, True, 5000)
    return TrainingResult(learned, sar, checkpoints, visits)
