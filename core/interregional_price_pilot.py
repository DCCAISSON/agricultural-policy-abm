"""Two-region production and trade pilot, separate from the eligibility-audit model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mesa import Agent, Model
from mesa.space import MultiGrid


@dataclass(frozen=True)
class PriceParameters:
    years: int = 20
    policy_year: int = 4
    premium: float = 0.30
    purchase_price: float = 2.40
    trade: bool = True
    transport_multiplier: float = 1.0
    route_capacity_kg: float | None = None
    demand_intercept: float = 3.20
    demand_slope: float = 0.000015
    learning: float = 0.5
    adjustment_cost: float = 250.0
    yield_noise: float = 0.04
    handling_cost: float = 0.04
    inventory_reference_price: float = 2.40


def clear_exports(quantity: np.ndarray, costs: np.ndarray, destination_price: float,
                  intercept: float, slope: float, capacity: float) -> tuple[np.ndarray, float]:
    """Competitive direct shipment with linear domestic inverse demand.

    Sell the lowest-transport-cost grain abroad until its net receipt equals
    the rising domestic price, or the physical route capacity binds.
    """
    exported = np.zeros(len(quantity))
    local_price = intercept - slope * float(quantity.sum())
    if local_price <= 0:
        raise ValueError("Domestic demand does not absorb harvest at a positive price")
    remaining = capacity
    for i in np.argsort(costs, kind="stable"):
        net_export_price = destination_price - costs[i]
        increment = min(float(quantity[i]), remaining,
                        max(0.0, (net_export_price - local_price) / slope))
        exported[i] = increment
        local_price += slope * increment
        remaining -= increment
    return exported, float(local_price)


class RiceFarmer(Agent):
    def __init__(self, model: PriceTransmissionModel, index: int):
        super().__init__(model)
        self.index = index
        self.region = index // 24
        col, row = index % 6, (index % 24) // 6
        model.grid.place_agent(self, (col + self.region * 8, row))
        r = model.rng
        self.land = float(r.uniform(8, 12))
        self.yield_per_mu = float(r.uniform(450, 550))
        self.unit_cost = float(r.uniform(680, 820))
        self.alternative_margin = float(r.uniform(180, 280))
        self.convex_cost = float(r.uniform(300, 420))
        self.distance = 20.0 + 15.0 * (5 - col) + 3.0 * abs(row - 1.5)
        self.transport_cost = (0.04 + 0.0025 * self.distance) * model.p.transport_multiplier
        self.area = self.land * 0.55
        self.expected_price = model.p.purchase_price
        self.last_price = model.p.purchase_price

    def plan(self) -> float:
        p = self.model.p
        margin = self.yield_per_mu * self.expected_price - self.unit_cost - self.alternative_margin
        return float(np.clip((self.land * margin + p.adjustment_cost * self.area)
                             / (self.convex_cost + p.adjustment_cost), 0, self.land))


class RegionalBuyer(Agent):
    def settle(self, quantity: float, price: float) -> dict:
        # Origin is deliberately absent from the buyer's decision inputs.
        p = self.model.p
        gross = price * quantity
        value = p.inventory_reference_price * quantity
        handling = p.handling_cost * quantity
        return {"procurement_kg": quantity, "procurement_cash": gross,
                "inventory_reference_value": value, "handling_cost": handling,
                "support_cost_proxy": gross + handling - value}


class PriceTransmissionModel(Model):
    def __init__(self, parameters: PriceParameters | None = None, *, seed: int = 0,
                 fixed_areas: np.ndarray | None = None):
        super().__init__(seed=seed)
        self.p = parameters or PriceParameters()
        p = self.p
        if not (p.years > 0 and 1 <= p.policy_year <= p.years and p.premium >= 0
                and p.purchase_price > 0 and p.demand_intercept > 0 and p.demand_slope > 0
                and 0 <= p.learning <= 1 and 0 <= p.yield_noise < 1
                and p.adjustment_cost >= 0 and p.transport_multiplier >= 0
                and p.handling_cost >= 0 and p.inventory_reference_price >= 0
                and (p.route_capacity_kg is None or p.route_capacity_kg >= 0)):
            raise ValueError("Invalid production/trade parameters")
        self.grid = MultiGrid(14, 4, torus=False)
        self.farmers = [RiceFarmer(self, i) for i in range(48)]
        self.buyer = RegionalBuyer(self)
        self.shocks = self.rng.uniform(1 - p.yield_noise, 1 + p.yield_noise, (p.years, 48))
        if fixed_areas is not None:
            fixed_areas = np.array(fixed_areas, dtype=float, copy=True)
            if (fixed_areas.shape != (p.years, 48) or not np.isfinite(fixed_areas).all()
                    or (fixed_areas < 0).any()
                    or (fixed_areas > np.array([f.land for f in self.farmers]) + 1e-9).any()):
                raise ValueError("Fixed plans must cover every farmer-year within land limits")
        self.fixed_areas = fixed_areas
        self.year = 0
        self.rows: list[dict] = []
        self.annual: list[dict] = []
        self.frames: list[dict] = []

    def step(self):
        if self.year >= self.p.years:
            raise RuntimeError("Experiment horizon completed")
        p = self.p
        t = self.year
        self.year += 1
        previous = np.array([f.area for f in self.farmers])
        plans = np.array([f.plan() for f in self.farmers])
        if self.fixed_areas is not None:
            plans = self.fixed_areas[t].copy()
        harvest = plans * np.array([f.yield_per_mu for f in self.farmers]) * self.shocks[t]
        # The price change occurs after this season's planting. No foresight of draws.
        b_price = p.purchase_price + (p.premium if self.year >= p.policy_year else 0)
        costs = np.array([f.transport_cost for f in self.farmers[:24]])
        capacity = float(harvest[:24].sum()) if p.route_capacity_kg is None else p.route_capacity_kg
        exported, a_price = clear_exports(harvest[:24], costs, b_price,
                                         p.demand_intercept, p.demand_slope,
                                         capacity if p.trade else 0.0)
        rows = []
        for i, f in enumerate(self.farmers):
            q, area = float(harvest[i]), float(plans[i])
            cross = float(exported[i]) if f.region == 0 else 0.0
            domestic = q - cross if f.region == 0 else 0.0
            buyer_kg = cross if f.region == 0 else q
            domestic_receipt = domestic * a_price
            buyer_receipt = buyer_kg * b_price
            transport = cross * f.transport_cost
            net_receipt = domestic_receipt + buyer_receipt - transport
            price = net_receipt / q if q > 0 else (
                max(a_price, b_price - f.transport_cost) if f.region == 0 and p.trade
                and p.route_capacity_kg is None else (a_price if f.region == 0 else b_price))
            production_cost = f.unit_cost * area + f.convex_cost * area ** 2 / (2 * f.land)
            adjustment = p.adjustment_cost * (area - previous[i]) ** 2 / (2 * f.land)
            alternative_income = f.alternative_margin * (f.land - area)
            row = {"year": self.year, "farmer": i, "region": f.region,
                   "x": f.pos[0], "y": f.pos[1], "land_mu": f.land,
                   "distance_km": f.distance if f.region == 0 else 0,
                   "transport_per_kg": f.transport_cost if f.region == 0 else 0,
                   "expected_price": f.expected_price, "previous_area_mu": float(previous[i]),
                   "rice_area_mu": area, "alternative_area_mu": f.land - area,
                   "harvest_kg": q, "local_kg": domestic, "cross_kg": cross,
                   "buyer_kg": buyer_kg, "local_receipt": domestic_receipt,
                   "buyer_receipt": buyer_receipt, "transport_cost": transport,
                   "realized_net_price": price, "production_cost": production_cost,
                   "adjustment_cost": adjustment, "alternative_income": alternative_income,
                   "net_income": net_receipt - production_cost - adjustment + alternative_income,
                   "choice_frozen": self.fixed_areas is not None}
            f.area = area
            f.last_price = price
            f.expected_price += p.learning * (price - f.expected_price)
            row["next_expected_price"] = f.expected_price
            rows.append(row)
        buy = self.buyer.settle(float(harvest[24:].sum() + exported.sum()), b_price)
        a_quantity = float(harvest[:24].sum())
        summary = {"year": self.year, "a_price": a_price, "b_price": b_price,
                   "cross_kg": float(exported.sum()), "a_output_kg": a_quantity,
                   "b_output_kg": float(harvest[24:].sum()), "total_output_kg": float(harvest.sum()),
                   "a_area_mu": float(plans[:24].sum()), "b_area_mu": float(plans[24:].sum()),
                   "a_income": sum(r["net_income"] for r in rows[:24]),
                   "b_income": sum(r["net_income"] for r in rows[24:]),
                   "transport_cost": sum(r["transport_cost"] for r in rows),
                   "a_local_kg": a_quantity - float(exported.sum()),
                   "foreign_support_proxy": (b_price - p.inventory_reference_price + p.handling_cost)
                                            * float(exported.sum()), **buy}
        self.rows.extend(rows)
        self.annual.append(summary)
        self.frames.append({"summary": dict(summary), "farmers": rows})

    def run(self):
        while self.year < self.p.years:
            self.step()
        return self
