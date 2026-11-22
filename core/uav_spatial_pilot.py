"""Synthetic pest/service ABM on Mesa 3.x; no calibrated biological parameters."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mesa import Agent, Model


@dataclass(frozen=True)
class UAVParameters:
    seasons: int = 5
    weeks: int = 10
    entry: bool = True
    spread: float = 0.25
    capacity: float = 3.0  # Weekly work hours per provider, including travel.
    ordering: str = "booking"
    travel_hours: float = 0.08
    growth: float = 0.3
    manual_effect: float = 0.55
    drone_effect: float = 0.75
    fee: float = 55.0
    job_cost: float = 28.0
    fixed_cost: float = 20.0
    travel_cost: float = 8.0
    subsidy: float = 0.0
    budget: float = 600.0
    coordination_cost: float = 0.0
    learning: float = 0.4
    yield_kg: float = 500.0
    price: float = 2.4
    damage: float = 0.65
    other_cost: float = 350.0
    decision_rule: str = "legacy"
    work_rate: float = 1.0  # Mu per hour; every pilot plot has area one mu.
    preparation_hours: float = 0.0
    early_confirmation: bool = False
    fallback_cost: float = 0.0
    min_interval: int = 1  # Weeks between actual treatments, not requests.
    initial_belief: float = 0.8
    social_weight: float = 0.0


def mix_pests(pressure: np.ndarray, edges: list[tuple[int, int]], spread: float) -> np.ndarray:
    """Symmetric edge flows conserve pest mass, with at most four neighbors."""
    result = pressure.copy()
    for left, right in edges:
        flow = spread * (pressure[right] - pressure[left]) / 4.0
        result[left] += flow
        result[right] -= flow
    return result


class UAVFarmer(Agent):
    def __init__(self, model: UAVModel, farmer_id: int, plots: list[int], manual_cost: float):
        super().__init__(model)
        self.farmer_id = farmer_id
        self.plots = plots
        self.manual_cost = manual_cost
        self.completion_belief = model.p.initial_belief

    def avoided_loss(self, pressure: float, effect: float, weeks_left: int) -> float:
        p = self.model.p
        horizon = min(3, weeks_left)
        if p.decision_rule == "legacy":
            return p.yield_kg * p.price * p.damage * pressure * horizon / p.weeks * effect
        untreated, treated = pressure, pressure * (1 - effect)
        reduction = 0.0
        # shortcut: private three-week forecast, without future shocks or neighbor actions.
        for week in range(horizon):
            if week:
                untreated = min(1.0, untreated + p.growth * untreated * (1 - untreated) + 0.009)
                treated = min(1.0, treated + p.growth * treated * (1 - treated) + 0.009)
            reduction += untreated - treated
        return p.yield_kg * p.price * p.damage * reduction / p.weeks

    def choose(self, pressure: float, weeks_left: int) -> tuple[str, float, float]:
        p = self.model.p
        manual = self.avoided_loss(pressure, p.manual_effect, weeks_left) - self.manual_cost
        service = self.completion_belief * (
            self.avoided_loss(pressure, p.drone_effect, weeks_left)
            - p.fee + p.subsidy * self.model.payment_belief
        )
        if p.early_confirmation:
            service += (1 - self.completion_belief) * max(0.0, manual - p.fallback_cost)
        choices = [(0.0, "none"), (manual, "manual")]
        if p.entry:
            choices.append((service, "drone"))
        return max(choices, key=lambda x: x[0])[1], manual, service


class UAVProvider(Agent):
    def __init__(self, model: UAVModel, village: int):
        super().__init__(model)
        self.village = village
        self.depot = (2 + village * 5, 2)

    def dispatch(self, orders: list[int], priority: np.ndarray,
                 pressure: np.ndarray) -> tuple[list[int], float, float]:
        p = self.model.p
        remaining = orders.copy()
        position = self.depot
        used, travel = 0.0, 0.0
        done = []

        def distance(j: int) -> int:
            return abs(j % 10 - position[0]) + abs(j // 10 - position[1])

        while remaining:
            if p.ordering == "cluster":
                job = min(remaining, key=lambda j: (distance(j), -pressure[j], priority[j]))
            elif p.ordering == "pressure":
                job = min(remaining, key=lambda j: (-pressure[j], priority[j]))
            else:
                job = min(remaining, key=lambda j: priority[j])
            remaining.remove(job)
            trip = distance(job) * p.travel_hours
            work = p.preparation_hours + 1.0 / p.work_rate
            if used + trip + work > p.capacity + 1e-10:
                continue
            used += trip + work
            travel += trip
            done.append(job)
            position = (job % 10, job // 10)
        return done, used, travel


class UAVModel(Model):
    def __init__(self, parameters: UAVParameters | None = None, *, seed: int = 0,
                 fixed_actions: dict[tuple[int, int, int], str] | None = None):
        super().__init__(seed=seed)
        self.p = parameters or UAVParameters()
        p = self.p
        if not (p.seasons > 0 and p.weeks > 0 and p.capacity >= 0 and
                0 <= p.spread <= 1 and 0 <= p.learning <= 1 and
                0 <= p.social_weight <= 1 and
                0 <= p.manual_effect <= 1 and 0 <= p.drone_effect <= 1 and
                0 <= p.damage <= 1 and 0 <= p.growth <= 1 and
                min(p.fee, p.subsidy, p.budget, p.travel_hours, p.coordination_cost,
                    p.job_cost, p.fixed_cost, p.travel_cost, p.other_cost,
                    p.preparation_hours, p.fallback_cost) >= 0 and
                p.work_rate > 0 and p.min_interval >= 1 and isinstance(p.min_interval, int) and
                0 <= p.initial_belief <= 1 and p.decision_rule in {"legacy", "forecast"} and
                p.ordering in {"booking", "cluster", "pressure"}):
            raise ValueError("Invalid UAV parameters")
        self.year = 0
        self.fixed_actions = fixed_actions or {}
        if any(action not in {"none", "manual"} for action in self.fixed_actions.values()):
            raise ValueError("Fixed focal actions must be none/manual")
        self.owner = np.zeros(60, dtype=int)
        self.farmers = []
        for village in range(2):
            plots = [j for j in range(60) if (j % 10 >= 5) == bool(village)]
            self.random.shuffle(plots)
            for local in range(15):
                farmer_id = village * 15 + local
                holdings = plots[2 * local:2 * local + 2]
                self.owner[holdings] = farmer_id
                self.farmers.append(UAVFarmer(self, farmer_id, holdings, self.random.uniform(45, 90)))
        self.providers = [UAVProvider(self, v) for v in range(2)]
        self.edges = [(j, k) for j in range(60) for k in (j + 1, j + 10)
                      if k < 60 and (k == j + 10 or j // 10 == k // 10)]
        self.initial_pressure = np.array([self.random.uniform(0.25, 0.6) for _ in range(60)])
        self.pressure = self.initial_pressure.copy()
        # Every exogenous draw is made before any policy-dependent decision.
        self.weather = np.array([self.random.uniform(0.8, 1.2) for _ in range(p.seasons * p.weeks)])
        self.shocks = np.array([[self.random.uniform(0, 0.018) for _ in range(60)]
                                for _ in self.weather])
        self.priorities = np.array([self.random.sample(range(60), 60) for _ in self.weather])
        self.payment_belief = 1.0
        self.plot_rows: list[dict] = []
        self.farmer_rows: list[dict] = []
        self.provider_rows: list[dict] = []
        self.summary_rows: list[dict] = []
        # Stipulated acquaintance rings; no extra draws disturb paired environments.
        self.social_edges = [(15 * v + i, 15 * v + (i + 1) % 15)
                             for v in range(2) for i in range(15)]
        self.social_rows: list[dict] = []
        self.week_detail: list[dict] = []

    def input_rows(self) -> list[dict]:
        return [{"plot": j, "farmer": int(self.owner[j]), "x": j % 10, "y": j // 10,
                 "village": int(j % 10 >= 5), "area_mu": 1,
                 "initial_pest": float(self.initial_pressure[j]),
                 "manual_cost": self.farmers[self.owner[j]].manual_cost} for j in range(60)]

    def step(self) -> None:
        for _ in self.weekly_steps():
            pass

    def weekly_steps(self):
        """Yield after each real weekly update; settle accounts before the final yield."""
        p = self.p
        if self.year >= p.seasons:
            raise RuntimeError("Pilot horizon completed")
        self.year += 1
        if self.year > 1:
            self.pressure = 0.5 * self.pressure + 0.5 * self.initial_pressure
        cumulative = np.zeros(60)
        costs = np.zeros(60)
        done_count = np.zeros(60, dtype=int)
        orders_count = np.zeros(60, dtype=int)
        manual_count = np.zeros(60, dtype=int)
        fallback_count = np.zeros(60, dtype=int)
        manual_spending = np.zeros(60)
        last_treated = np.full(60, -p.min_interval, dtype=int)
        weekly_start = len(self.plot_rows)
        provider_start = len(self.provider_rows)
        for week in range(1, p.weeks + 1):
            tick = (self.year - 1) * p.weeks + week - 1
            before = self.pressure.copy()
            expected = np.clip(before + p.growth * before * (1 - before), 0, 1)
            if p.decision_rule == "forecast":
                expected = np.clip(expected + 0.009, 0, 1)
            actions, manuals, services, eligible = [], [], [], []
            for j in range(60):
                action, manual, service = self.farmers[self.owner[j]].choose(expected[j], p.weeks - week + 1)
                action = self.fixed_actions.get((self.year, week, j), action)
                eligible.append(bool(week - last_treated[j] >= p.min_interval))
                if week - last_treated[j] < p.min_interval:
                    action = "none"
                actions.append(action)
                manuals.append(manual)
                services.append(service)
            requests = [j for j, action in enumerate(actions) if action == "drone"]
            done = set()
            for provider in self.providers:
                jobs = [j for j in requests if int(j % 10 >= 5) == provider.village]
                completed, used, travel = provider.dispatch(jobs, self.priorities[tick], before)
                done.update(completed)
                fixed = p.fixed_cost if p.entry else 0.0
                revenue = len(completed) * p.fee
                resource_cost = len(completed) * p.job_cost + travel * p.travel_cost + fixed
                self.provider_rows.append({"year": self.year, "week": week,
                    "provider": provider.village, "orders": len(jobs), "completed": len(completed),
                    "used_hours": used, "capacity_hours": p.capacity if p.entry else 0.0,
                    "travel_hours": travel,
                    "preparation_hours": len(completed) * p.preparation_hours,
                    "spray_hours": len(completed) / p.work_rate,
                    "revenue": revenue, "cost": resource_cost,
                    "profit": revenue - resource_cost})
            effectiveness = np.zeros(60)
            actual_actions = ["none"] * 60
            manual_paid = np.zeros(60)
            for j, action in enumerate(actions):
                orders_count[j] += action == "drone"
                if j in done:
                    effectiveness[j] = p.drone_effect
                    done_count[j] += 1
                    costs[j] += p.fee
                    actual_actions[j] = "drone"
                elif action == "manual" or (action == "drone" and p.early_confirmation
                                             and manuals[j] > p.fallback_cost):
                    effectiveness[j] = p.manual_effect
                    manual_count[j] += 1
                    fallback = action == "drone"
                    fallback_count[j] += fallback
                    manual_paid[j] = self.farmers[self.owner[j]].manual_cost + p.fallback_cost * fallback
                    costs[j] += manual_paid[j]
                    manual_spending[j] += manual_paid[j]
                    actual_actions[j] = "manual"
                if actual_actions[j] != "none":
                    assert week - last_treated[j] >= p.min_interval
                    last_treated[j] = week
            grown = np.clip(before + p.growth * self.weather[tick] * before * (1 - before)
                            + self.shocks[tick], 0, 1)
            treated = grown * (1 - effectiveness)
            self.pressure = mix_pests(treated, self.edges, p.spread)
            assert np.isclose(self.pressure.sum(), treated.sum())
            assert np.all((self.pressure >= -1e-10) & (self.pressure <= 1 + 1e-10))
            cumulative += self.pressure
            self.week_detail = [{"plot": j, "eligible": eligible[j],
                "grown": float(grown[j]), "treated": float(treated[j]),
                "net_spread": float(self.pressure[j] - treated[j])} for j in range(60)]
            for j in range(60):
                self.plot_rows.append({"year": self.year, "week": week, "plot": j,
                    "farmer": int(self.owner[j]), "action": actions[j], "completed": int(j in done),
                    "actual_action": actual_actions[j], "manual_cost": float(manual_paid[j]),
                    "fallback": int(actions[j] == "drone" and actual_actions[j] == "manual"),
                    "effectiveness": float(effectiveness[j]),
                    "manual_gain": manuals[j], "drone_gain": services[j],
                    "completion_belief": self.farmers[self.owner[j]].completion_belief,
                    "pest_before": float(before[j]), "pest_after": float(self.pressure[j]),
                    "fee": p.fee if j in done else 0.0, "grant": 0.0})
            observed = {}
            for farmer in self.farmers:
                count = sum(j in requests for j in farmer.plots)
                observed[farmer.farmer_id] = (sum(j in done for j in farmer.plots) / count
                                            if count else None)
            for farmer in self.farmers:
                i = farmer.farmer_id
                friends = [b if a == i else a for a, b in self.social_edges if i in (a, b)]
                reports = [observed[j] for j in friends if observed[j] is not None]
                peer = float(np.mean(reports)) if reports else None
                own = observed[i]
                signal = own
                if p.social_weight > 0 and peer is not None:
                    signal = peer if own is None else (1 - p.social_weight) * own + p.social_weight * peer
                before_belief = farmer.completion_belief
                if signal is not None:
                    farmer.completion_belief += p.learning * (signal - farmer.completion_belief)
                self.social_rows.append({"year": self.year, "week": week, "farmer": i,
                    "friends": friends, "reports": [{"farmer": j, "completion": observed[j]}
                                                       for j in friends if observed[j] is not None],
                    "own": own, "peer": peer, "signal": signal, "before": before_belief,
                    "after": farmer.completion_belief, "weight": p.social_weight})
            if week < p.weeks:
                yield self.year, week
        output = p.yield_kg * (1 - p.damage * cumulative / p.weeks)
        nominal = float(done_count.sum() * p.subsidy)
        ratio = min(1.0, p.budget / nominal) if nominal else 1.0
        grant_per_job = p.subsidy * ratio
        self.payment_belief += p.learning * (ratio - self.payment_belief)
        for row in self.plot_rows[weekly_start:]:
            row["grant"] = row["completed"] * grant_per_job
        farmer_total = 0.0
        for farmer in self.farmers:
            js = farmer.plots
            grant = float(done_count[js].sum() * grant_per_job)
            revenue = float(output[js].sum() * p.price)
            spending = float(costs[js].sum() + len(js) * p.other_cost)
            net = revenue - spending + grant
            farmer_total += net
            self.farmer_rows.append({"year": self.year, "farmer": farmer.farmer_id,
                "output_kg": float(output[js].sum()), "orders": int(orders_count[js].sum()),
                "completed": int(done_count[js].sum()), "manual": int(manual_count[js].sum()),
                "fallback": int(fallback_count[js].sum()),
                "revenue": revenue, "cost": spending, "grant": grant, "net": net})
        providers = self.provider_rows[provider_start:]
        fiscal = nominal * ratio + p.coordination_cost
        provider_profit = sum(r["profit"] for r in providers)
        resource_net = (float(output.sum() * p.price) - 60 * p.other_cost
                        - float(manual_spending.sum())
                        - sum(r["cost"] for r in providers) - p.coordination_cost)
        assert np.isclose(farmer_total + provider_profit - fiscal, resource_net)
        assert nominal * ratio <= p.budget + 1e-8
        self.summary_rows.append({"year": self.year, "output_kg": float(output.sum()),
            "orders": int(orders_count.sum()), "completed": int(done_count.sum()),
            "unserved": int(orders_count.sum() - done_count.sum()),
            "manual": int(manual_count.sum()), "fallback": int(fallback_count.sum()),
            "buyers": int(sum(done_count[f.plots].sum() > 0 for f in self.farmers)),
            "pesticide_dose_units": int(done_count.sum() + manual_count.sum()),
            "farmer_net": farmer_total, "provider_profit": provider_profit,
            "fiscal": fiscal, "resource_net": resource_net,
            "pest": float(cumulative.mean() / p.weeks)})
        yield self.year, p.weeks

    def run(self) -> UAVModel:
        for _ in range(self.p.seasons):
            self.step()
        return self
