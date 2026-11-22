"""Two-season, synthetic service/land mechanism experiment for Mesa 2.3.2.

All numerical defaults are stipulated, not estimated Quzhou parameters.
One indivisible plot is one mu and requires one effective service hour.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from statistics import mean

from mesa import Agent, Model


@dataclass(frozen=True)
class Parameters:
    families: int = 24
    plots_per_family: int = 4
    seasons: int = 2
    yield_kg: float = 400.0
    grain_price: float = 2.6
    outside_return: float = 60.0
    self_cost: float = 100.0
    service_fee: float = 240.0
    provider_cost: float = 100.0
    subsidy: float = 0.0
    daily_capacity: int = 4
    timely_days: int = 5
    last_day: int = 8
    delay_loss: float = 0.06
    sunk_share: float = 0.25
    trade: bool = False
    transaction_cost: float = 10.0
    seller_share: float = 0.5
    own_capacity_scale: int = 1
    reverse_queue: bool = False
    fee_response: float = 0.0
    initial_completion: float = 1.0
    initial_quality: float = 1.0


@dataclass(frozen=True)
class Plot:
    plot_id: int
    owner: int
    potential_kg: float


class Farmer(Agent):
    def __init__(self, unique_id: int, model: ServiceLandModel, cost: float, capacity: int):
        super().__init__(unique_id, model)
        self.cost = cost
        self.capacity = capacity
        self.completion = model.p.initial_completion
        self.quality = model.p.initial_quality
        self.holding: set[int] = set()

    def plan(self, holding: set[int]) -> tuple[float, dict[int, str]]:
        p = self.model.p
        choices: dict[int, str] = {}
        gains: list[tuple[float, int]] = []
        total = 0.0
        for plot_id in sorted(holding):
            revenue = self.model.plots[plot_id].potential_kg * p.grain_price
            external = self.completion * (
                revenue * self.quality + p.subsidy - self.model.provider.fee
                - (1 - p.sunk_share) * self.cost
            ) - p.sunk_share * self.cost
            best = max(p.outside_return, external)
            choices[plot_id] = "service" if external > p.outside_return else "outside"
            total += best
            gains.append((revenue - self.cost - p.self_cost - best, plot_id))
        for gain, plot_id in sorted(gains, key=lambda x: (-x[0], x[1]))[: self.capacity]:
            if gain > 0:
                total += gain
                choices[plot_id] = "self"
        return total, choices


class ServiceProvider(Agent):
    def __init__(self, unique_id: int, model: ServiceLandModel):
        super().__init__(unique_id, model)
        self.fee = model.p.service_fee
        self.previous_orders = 0

    def quote(self) -> None:
        p = self.model.p
        timely_capacity = p.daily_capacity * p.timely_days
        overload = max(0.0, self.previous_orders / max(1, timely_capacity) - 1)
        self.fee = p.service_fee * (1 + p.fee_response * overload)

    def serve(self, orders: list[int], priority: dict[int, int]) -> dict[int, int]:
        p = self.model.p
        queue = sorted(orders, key=priority.__getitem__, reverse=p.reverse_queue)
        completed: dict[int, int] = {}
        for day in range(1, p.last_day + 1):
            jobs = queue[(day - 1) * p.daily_capacity : day * p.daily_capacity]
            for plot_id in jobs:
                completed[plot_id] = day
            self.model.daily_rows.append({
                "year": self.model.year, "day": day,
                "capacity_hours": p.daily_capacity, "used_hours": len(jobs),
                "completed_mu": len(jobs),
            })
        self.previous_orders = len(orders)
        return completed


class ServiceLandModel(Model):
    def __init__(self, parameters: Parameters | None = None, *, seed: int = 0):
        super().__init__(seed=seed)
        self.p = parameters or Parameters()
        p = self.p
        if not (p.families > 0 and p.plots_per_family > 0 and p.seasons > 0
                and p.daily_capacity >= 0 and 0 < p.timely_days <= p.last_day
                and 0 <= p.seller_share <= 1 and 0 <= p.sunk_share <= 1
                and 0 <= p.initial_completion <= 1 and 0 <= p.initial_quality <= 1
                and p.own_capacity_scale >= 0 and p.subsidy >= 0 and p.delay_loss >= 0):
            raise ValueError("Invalid experiment parameters")
        self.year = 0
        self.farmers = [
            Farmer(i, self, self.random.uniform(650, 1000),
                   self.random.choice((0, 0, 1, 2, 4, 6)) * p.own_capacity_scale)
            for i in range(p.families)
        ]
        self.provider = ServiceProvider(p.families, self)
        self.plots = [
            Plot(i, i // p.plots_per_family, p.yield_kg * self.random.uniform(0.9, 1.1))
            for i in range(p.families * p.plots_per_family)
        ]
        # Pre-draw all exogenous ordering, so counterfactual branches cannot shift RNG draws.
        self.orders = []
        for _ in range(p.seasons):
            land = self.random.sample(range(len(self.plots)), len(self.plots))
            queue = self.random.sample(range(len(self.plots)), len(self.plots))
            bidders = self.random.sample(range(p.families), p.families)
            self.orders.append((land, {v: k for k, v in enumerate(queue)}, bidders))
        self.plot_rows: list[dict] = []
        self.farmer_rows: list[dict] = []
        self.daily_rows: list[dict] = []
        self.trade_rows: list[dict] = []
        self.summary_rows: list[dict] = []

    def input_rows(self) -> list[dict]:
        return [{"plot_id": plot.plot_id, "owner": plot.owner,
                 "area_mu": 1, "potential_kg": plot.potential_kg,
                 "owner_cost_yuan_mu": self.farmers[plot.owner].cost,
                 "owner_self_capacity_mu": self.farmers[plot.owner].capacity}
                for plot in self.plots]

    def _trade(self, land_order: list[int], bidders: list[int]) -> tuple[dict, dict]:
        p = self.p
        rents = {plot.plot_id: 0.0 for plot in self.plots}
        tenants = {plot.plot_id: plot.owner for plot in self.plots}
        if not p.trade:
            return rents, tenants
        for plot_id in land_order:
            owner = self.farmers[self.plots[plot_id].owner]
            before, _ = owner.plan(owner.holding)
            after, _ = owner.plan(owner.holding - {plot_id})
            ask = before - after
            best_bid, winner = float("-inf"), None
            for bidder_id in bidders:
                if bidder_id == owner.unique_id:
                    continue
                bidder = self.farmers[bidder_id]
                current, _ = bidder.plan(bidder.holding)
                expanded, _ = bidder.plan(bidder.holding | {plot_id})
                bid = expanded - current - p.transaction_cost
                if bid > best_bid:
                    best_bid, winner = bid, bidder
            traded = winner is not None and best_bid > ask + 1e-9
            rent = ask + p.seller_share * (best_bid - ask) if traded else 0.0
            if traded:
                owner.holding.remove(plot_id)
                winner.holding.add(plot_id)
                tenants[plot_id] = winner.unique_id
                rents[plot_id] = rent
            self.trade_rows.append({
                "year": self.year, "plot_id": plot_id, "owner": owner.unique_id,
                "best_bidder": winner.unique_id if winner is not None else -1,
                "ask": ask, "bid_net_transaction_cost": best_bid if winner else 0.0,
                "traded": int(traded), "rent": rent,
                "seller_expected_gain": rent - ask if traded else 0.0,
                "buyer_expected_gain": best_bid - rent if traded else 0.0,
            })
        return rents, tenants

    def step(self) -> None:
        if self.year >= self.p.seasons:
            raise RuntimeError("Experiment horizon already completed")
        p = self.p
        land_order, queue, bidders = self.orders[self.year]
        self.year += 1
        self.provider.quote()
        for farmer in self.farmers:
            farmer.holding = {plot.plot_id for plot in self.plots if plot.owner == farmer.unique_id}
        rents, tenants = self._trade(land_order, bidders)
        modes: dict[int, str] = {}
        for farmer in self.farmers:
            _, choice = farmer.plan(farmer.holding)
            modes.update(choice)
        orders = [plot_id for plot_id, mode in modes.items() if mode == "service"]
        completed = self.provider.serve(orders, queue)
        period_plots = []
        for plot in self.plots:
            farmer = self.farmers[tenants[plot.plot_id]]
            mode = modes[plot.plot_id]
            day = 0 if mode == "self" else completed.get(plot.plot_id, -1)
            planted = mode == "self" or (mode == "service" and day >= 1)
            late = max(0, day - p.timely_days) if planted else 0
            quality = max(0.0, 1 - p.delay_loss * late) if planted else 0.0
            output = plot.potential_kg * quality
            subsidy = p.subsidy if mode == "service" and planted else 0.0
            service_paid = self.provider.fee if mode == "service" and planted else 0.0
            production_cost = farmer.cost if planted else (p.sunk_share * farmer.cost if mode == "service" else 0.0)
            self_cost = p.self_cost if mode == "self" else 0.0
            outside = p.outside_return if mode == "outside" else 0.0
            transaction = p.transaction_cost if tenants[plot.plot_id] != plot.owner else 0.0
            row = {
                "year": self.year, "plot_id": plot.plot_id, "owner": plot.owner,
                "operator": tenants[plot.plot_id], "area_mu": 1, "mode": mode,
                "service_day": day, "late_days": late, "quality": quality,
                "planted_mu": int(planted), "output_kg": output,
                "grain_revenue": output * p.grain_price, "outside_return": outside,
                "production_cost": production_cost, "self_cost": self_cost,
                "service_paid": service_paid, "subsidy_paid": subsidy,
                "rent": rents[plot.plot_id], "transaction_cost": transaction,
                "operator_net": output * p.grain_price + outside + subsidy
                - production_cost - self_cost - service_paid - rents[plot.plot_id] - transaction,
            }
            period_plots.append(row)
        self.plot_rows.extend(period_plots)
        service_rows = [r for r in period_plots if r["mode"] == "service"]
        period_farmers = []
        for farmer in self.farmers:
            operated = [r for r in period_plots if r["operator"] == farmer.unique_id]
            received = sum(r["rent"] for r in period_plots if r["owner"] == farmer.unique_id)
            own_orders = [r for r in operated if r["mode"] == "service"]
            observations = own_orders or service_rows
            old_completion, old_quality = farmer.completion, farmer.quality
            if observations:
                done = [r for r in observations if r["planted_mu"]]
                farmer.completion = len(done) / len(observations)
                farmer.quality = mean(r["quality"] for r in done) if done else old_quality
            row = {
                "year": self.year, "farmer_id": farmer.unique_id,
                "operating_mu": len(operated), "early_rice_mu": sum(r["planted_mu"] for r in operated),
                "self_mu": sum(r["mode"] == "self" for r in operated),
                "service_orders_mu": len(own_orders), "rent_received": received,
                "rent_paid": sum(r["rent"] for r in operated),
                "service_paid": sum(r["service_paid"] for r in operated),
                "subsidy_paid": sum(r["subsidy_paid"] for r in operated),
                "agricultural_net": sum(r["operator_net"] for r in operated) + received,
                "expected_completion_before": old_completion, "expected_quality_before": old_quality,
                "expected_completion_after": farmer.completion, "expected_quality_after": farmer.quality,
            }
            period_farmers.append(row)
        self.farmer_rows.extend(period_farmers)
        provider_revenue = sum(r["service_paid"] for r in period_plots)
        provider_cost = p.provider_cost * len(completed)
        fiscal = sum(r["subsidy_paid"] for r in period_plots)
        households_net = sum(r["agricultural_net"] for r in period_farmers)
        resource_net = sum(r["grain_revenue"] + r["outside_return"] - r["production_cost"]
                           - r["self_cost"] - r["transaction_cost"] for r in period_plots) - provider_cost
        accounting_error = households_net + provider_revenue - provider_cost - fiscal - resource_net
        summary = {
            "year": self.year, "land_mu": len(self.plots),
            "early_rice_mu": sum(r["planted_mu"] for r in period_plots),
            "output_kg": sum(r["output_kg"] for r in period_plots),
            "self_mu": sum(r["mode"] == "self" for r in period_plots),
            "service_orders_mu": len(orders), "service_completed_mu": len(completed),
            "late_service_mu": sum(r["late_days"] > 0 for r in period_plots),
            "unserved_mu": len(orders) - len(completed),
            "land_transferred_mu": sum(tenants[x.plot_id] != x.owner for x in self.plots),
            "original_operator_rice_mu": sum(r["planted_mu"] for r in period_plots if r["owner"] == r["operator"]),
            "active_farmers": sum(r["early_rice_mu"] > 0 for r in period_farmers),
            "fiscal_yuan": fiscal, "households_net_yuan": households_net,
            "provider_net_yuan": provider_revenue - provider_cost,
            "resource_net_yuan": resource_net, "service_fee_yuan_mu": self.provider.fee,
            "rent_total_yuan": sum(rents.values()), "accounting_error": accounting_error,
        }
        assert len(modes) == len(self.plots) == sum(len(f.holding) for f in self.farmers)
        assert sum(row["used_hours"] for row in self.daily_rows if row["year"] == self.year) == len(completed)
        assert all(row["used_hours"] <= p.daily_capacity for row in self.daily_rows)
        assert isclose(sum(r["rent_received"] for r in period_farmers), sum(r["rent_paid"] for r in period_farmers), abs_tol=1e-8)
        assert isclose(fiscal, p.subsidy * len(completed), abs_tol=1e-8)
        assert isclose(accounting_error, 0, abs_tol=1e-7)
        assert all(r["self_mu"] <= f.capacity for r, f in zip(period_farmers, self.farmers))
        assert all(r["seller_expected_gain"] >= -1e-8 and r["buyer_expected_gain"] >= -1e-8 for r in self.trade_rows)
        self.summary_rows.append(summary)
        self.running = self.year < p.seasons

    def run_model(self) -> None:
        while self.running:
            self.step()
