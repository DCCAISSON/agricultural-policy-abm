"""Decision revision reusing the unchanged first-round market and settlement."""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from core.service_land_mechanism import Farmer, Parameters, ServiceLandModel


@dataclass(frozen=True)
class RevisionParameters(Parameters):
    seasons: int = 6
    capacity_aware: bool = True
    management_gamma: float = 4.0
    other_demand_multiplier: float = 1.0


class RevisedFarmer(Farmer):
    def __init__(self, unique_id, model, cost, capacity):
        super().__init__(unique_id, model, cost, capacity)
        self.other_orders = 0
        self.plan_cache = {}

    def service_expectation(self, orders: int) -> tuple[float, float]:
        p = self.model.p
        if not p.capacity_aware:
            return self.completion, self.quality
        if not orders:
            return 1.0, 1.0
        total = orders + self.other_orders
        served = min(total, p.daily_capacity * p.last_day)
        quality_sum = self.model.slot_quality_prefix[served]
        return served / total, quality_sum / served if served else 1.0

    def plan(self, holding: set[int]) -> tuple[float, dict[int, str]]:
        p = self.model.p
        if not p.capacity_aware and p.management_gamma == 0:
            return super().plan(holding)
        key = tuple(sorted(holding))
        if key in self.plan_cache:
            return self.plan_cache[key]
        ranked = sorted(holding, key=lambda i: (-self.model.plots[i].potential_kg, i))
        n = len(ranked)
        revenues = [0.0]
        for i in ranked:
            revenues.append(revenues[-1] + self.model.plots[i].potential_kg * p.grain_price)
        best_value, best_counts = n * p.outside_return, (0, 0)
        expectations = [self.service_expectation(m) for m in range(n + 1)]
        for own in range(min(n, self.capacity) + 1):
            own_net = revenues[own] - (self.cost + p.self_cost) * own
            for external in range(n - own + 1):
                probability, quality = expectations[external]
                committed = own + external
                external_net = probability * (
                    quality * (revenues[committed] - revenues[own])
                    + external * (p.subsidy - self.model.provider.fee - (1 - p.sunk_share) * self.cost)
                ) - p.sunk_share * self.cost * external
                value = (own_net + external_net + (n - committed) * p.outside_return
                         - self.model.management_cost(committed))
                if value > best_value + 1e-9:
                    best_value, best_counts = value, (own, external)
        own, external = best_counts
        choices = {i: "self" if j < own else "service" if j < own + external else "outside"
                   for j, i in enumerate(ranked)}
        result = best_value, choices
        self.plan_cache[key] = result
        return result


class RevisedServiceLandModel(ServiceLandModel):
    def __init__(self, parameters: RevisionParameters | None = None, *, seed: int = 0):
        super().__init__(parameters or RevisionParameters(), seed=seed)
        if self.p.management_gamma < 0 or self.p.other_demand_multiplier < 0:
            raise ValueError("Negative management cost or demand multiplier")
        replacements = []
        for old in self.farmers:
            old.remove()
            replacements.append(RevisedFarmer(old.unique_id, self, old.cost, old.capacity))
        self.farmers = replacements
        self.slot_quality_prefix = [0.0]
        for slot in range(self.p.daily_capacity * self.p.last_day):
            day = slot // self.p.daily_capacity + 1
            q = max(0.0, 1 - self.p.delay_loss * max(0, day - self.p.timely_days))
            self.slot_quality_prefix.append(self.slot_quality_prefix[-1] + q)

    def management_cost(self, committed: int) -> float:
        return self.p.management_gamma * committed * (committed - 1) / 2

    def step(self) -> None:
        if self.year == 0:
            intents = {}
            for f in self.farmers:
                owned = {p.plot_id for p in self.plots if p.owner == f.unique_id}
                _, modes = Farmer.plan(f, owned)
                intents[f.unique_id] = sum(mode == "service" for mode in modes.values())
        else:
            intents = {r["farmer_id"]: r["service_orders_mu"] for r in self.farmer_rows
                       if r["year"] == self.year}
        total = sum(intents.values())
        for f in self.farmers:
            f.other_orders = round(self.p.other_demand_multiplier * (total - intents[f.unique_id]))
            f.plan_cache.clear()
        super().step()
        current_plots = [r for r in self.plot_rows if r["year"] == self.year]
        current_farmers = [r for r in self.farmer_rows if r["year"] == self.year]
        total_cost = 0.0
        for f, row in zip(self.farmers, current_farmers):
            committed = row["self_mu"] + row["service_orders_mu"]
            cost = self.management_cost(committed)
            probability, quality = f.service_expectation(row["service_orders_mu"])
            # For the legacy branch, the recorded forecast is the pre-settlement belief.
            if not self.p.capacity_aware:
                probability, quality = row["expected_completion_before"], row["expected_quality_before"]
            row.update(management_cost=cost, planned_rice_mu=committed,
                       forecast_other_orders=f.other_orders, planned_completion=probability,
                       planned_quality=quality)
            row["agricultural_net"] -= cost
            for plot in current_plots:
                if plot["operator"] == f.unique_id:
                    charge = cost / committed if plot["mode"] != "outside" and committed else 0.0
                    plot["management_cost"] = charge
                    plot["operator_net"] -= charge
            total_cost += cost
        summary = self.summary_rows[-1]
        summary["management_cost_yuan"] = total_cost
        summary["households_net_yuan"] -= total_cost
        summary["resource_net_yuan"] -= total_cost
        summary["largest_operator_mu"] = max(r["operating_mu"] for r in current_farmers)
        summary["unused_self_capacity_mu"] = sum(f.capacity - r["self_mu"] for f, r in zip(self.farmers, current_farmers))
        summary["planned_rice_mu"] = sum(r["planned_rice_mu"] for r in current_farmers)
        assert isclose(sum(r["management_cost"] for r in current_plots), total_cost, abs_tol=1e-7)
        assert isclose(sum(r["agricultural_net"] for r in current_farmers), summary["households_net_yuan"], abs_tol=1e-7)
        error = (summary["households_net_yuan"] + summary["provider_net_yuan"]
                 - summary["fiscal_yuan"] - summary["resource_net_yuan"])
        summary["accounting_error"] = error
        assert isclose(error, 0, abs_tol=1e-7)
