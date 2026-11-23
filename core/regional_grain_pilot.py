"""Three synthetic regions, lawful trade, eligibility and budget-limited audit."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mesa import Agent, Model


@dataclass(frozen=True)
class GrainParameters:
    seasons: int = 5
    per_region: int = 30
    rates: tuple[float, ...] = (0.03, 0.25, 0.08)
    prices: tuple[float, ...] = (2.4, 2.55, 2.42)
    grant_budget: float = 6000.0
    audit_budget: float = 600.0
    audit_cost: float = 40.0
    shared: bool = False
    shared_regions: tuple[int, ...] = (0, 1, 2)
    sharing_cost: float = 120.0
    local_accuracy: float = 0.25
    shared_accuracy: float = 0.9
    local_false_reject: float = 0.04
    shared_false_reject: float = 0.01
    full_information: bool = False
    allow_ineligible: bool = True
    trade: bool = True
    penalty: float = 1.0
    compliance_cost: float = 15.0
    learning: float = 0.5
    production_cost: float = 1.8


class GrainSeller(Agent):
    def __init__(self, model: GrainModel, seller_id: int, origin: int,
                 quantity: float, application_cost: float):
        super().__init__(model)
        self.seller_id = seller_id
        self.origin = origin
        self.quantity = quantity
        self.application_cost = application_cost

    def expected_net(self, quantity: float, prices: np.ndarray, destination: int, apply: bool) -> float:
        """Risk-neutral reporting choice under imperfect audit and payment rationing."""
        m, p = self.model, self.model.p
        transport = 0.06 * abs(self.origin - destination) * quantity
        commercial = (prices[destination] - p.production_cost) * quantity - transport
        if not apply:
            return float(commercial)
        eligible = self.origin == destination
        nominal = quantity * p.rates[destination]
        accuracy, false_reject, _ = m.audit_rules(destination)
        check = 1.0 if p.full_information else m.audit_belief[destination]
        reject = check * (false_reject if eligible else accuracy)
        compliance = p.compliance_cost * (1 + check)
        expected = commercial + (1 - reject) * nominal * m.payment_belief[destination] - compliance
        if not eligible:
            expected -= self.application_cost + reject * p.penalty * nominal
        return float(expected)

    def choose(self, quantity: float, prices: np.ndarray) -> tuple[int, bool, float]:
        p = self.model.p
        options = []
        for destination in range(3) if p.trade else (self.origin,):
            options.append((self.expected_net(quantity, prices, destination, False), destination, False))
            eligible = self.origin == destination
            if not eligible and not p.allow_ineligible:
                continue
            nominal = quantity * p.rates[destination]
            if nominal == 0:
                continue
            options.append((self.expected_net(quantity, prices, destination, True), destination, True))
        value, destination, apply = max(options, key=lambda x: x[0])
        return destination, apply, float(value)


class RegionalAuditor(Agent):
    def __init__(self, model: GrainModel, region: int):
        super().__init__(model)
        self.region = region

    def process(self, rows: list[dict], priority: np.ndarray, signal: np.ndarray) -> dict:
        m, p = self.model, self.model.p
        accuracy, false_reject, overhead = m.audit_rules(self.region)
        applications = [r for r in rows if r["apply"]]
        slots = int(max(0.0, p.audit_budget - overhead) // p.audit_cost)
        if p.full_information:
            slots = len(applications)
        selected = sorted(applications, key=lambda r: priority[r["seller"], self.region])[:slots]
        selected_ids = {r["seller"] for r in selected}
        for row in applications:
            row["checked"] = int(row["seller"] in selected_ids)
            # Truth affects the measurement error, not selection or the government's information set.
            test_reject = signal[row["seller"], self.region] < (
                false_reject if row["eligible"] else accuracy)
            row["observed_reject"] = int(row["checked"] and test_reject)
            row["approved"] = 1 - row["observed_reject"]
            row["compliance"] = p.compliance_cost * (1 + row["checked"])
            if row["observed_reject"] and not row["eligible"]:
                row["fine"] = p.penalty * row["claim"]
        approved = sum(r["claim"] for r in applications if r["approved"])
        ratio = min(1.0, p.grant_budget / approved) if approved else 1.0
        for row in applications:
            row["payment"] = row["claim"] * row["approved"] * ratio
        m.payment_belief[self.region] += p.learning * (ratio - m.payment_belief[self.region])
        observed_check = len(selected) / len(applications) if applications else 1.0
        m.audit_belief[self.region] += p.learning * (observed_check - m.audit_belief[self.region])
        cost = overhead + len(selected) * p.audit_cost
        payment = sum(r["payment"] for r in rows)
        assert payment <= p.grant_budget + 1e-8
        if not p.full_information:
            assert cost <= p.audit_budget + 1e-8
        return {"year": m.year, "region": self.region, "applications": len(applications),
            "ineligible_applications": sum(not r["eligible"] for r in applications),
            "checked": len(selected), "payment": payment,
            "eligible_payment": sum(r["payment"] for r in rows if r["eligible"]),
            "ineligible_payment": sum(r["payment"] for r in rows if not r["eligible"]),
            "false_rejects": sum(r["observed_reject"] for r in rows if r["eligible"]),
            "fine": sum(r["fine"] for r in rows), "recovered": 0.0,
            "audit_cost": cost, "budget_unused": p.grant_budget - payment,
            "crossborder_kg": sum(r["quantity"] for r in rows if r["origin"] != self.region),
            "payment_ratio": ratio, "next_audit_belief": float(m.audit_belief[self.region])}


class GrainModel(Model):
    def __init__(self, parameters: GrainParameters | None = None, *, seed: int = 0,
                 fixed_decisions: dict[tuple[int, int], tuple[int, bool]] | None = None):
        super().__init__(seed=seed)
        self.p = parameters or GrainParameters()
        p = self.p
        if not (p.seasons > 0 and p.per_region > 0 and len(p.rates) == len(p.prices) == 3
                and min(p.rates) >= 0 and min(p.prices) > 0 and p.audit_cost > 0
                and min(p.audit_budget, p.grant_budget, p.sharing_cost, p.penalty, p.compliance_cost) >= 0
                and all(0 <= x <= 1 for x in (p.local_accuracy, p.shared_accuracy,
                    p.local_false_reject, p.shared_false_reject, p.learning))
                and set(p.shared_regions) <= {0, 1, 2}
                and (not p.shared or p.sharing_cost <= p.audit_budget)):
            raise ValueError("Invalid grain parameters")
        self.year = 0
        n = p.per_region * 3
        self.fixed_decisions = fixed_decisions
        if fixed_decisions is not None:
            required = {(year, seller) for year in range(1, p.seasons + 1) for seller in range(n)}
            if set(fixed_decisions) != required:
                raise ValueError("Replay must contain every seller-year exactly once")
            for (_, seller), (destination, apply) in fixed_decisions.items():
                origin = seller // p.per_region
                if (destination not in range(3) or apply not in (True, False)
                        or (not p.trade and destination != origin)
                        or (apply and not p.allow_ineligible and destination != origin)):
                    raise ValueError("Replay action violates scenario rules")
        self.sellers = [GrainSeller(self, i, i // p.per_region,
                        self.random.uniform(1200, 2500), self.random.uniform(30, 180)) for i in range(n)]
        self.auditors = [RegionalAuditor(self, r) for r in range(3)]
        self.quantity_shocks = np.array([[self.random.uniform(0.9, 1.1) for _ in range(n)]
                                         for _ in range(p.seasons)])
        self.price_shocks = np.array([[self.random.uniform(-0.04, 0.04) for _ in range(3)]
                                     for _ in range(p.seasons)])
        self.audit_draws = np.array([[[self.random.random() for _ in range(3)] for _ in range(n)]
                                    for _ in range(p.seasons)])
        self.priority_draws = np.array([[[self.random.random() for _ in range(3)] for _ in range(n)]
                                       for _ in range(p.seasons)])
        self.payment_belief = np.ones(3)
        self.audit_belief = np.array([min(1, int(max(0, p.audit_budget - self.audit_rules(r)[2])
                                              // p.audit_cost) / p.per_region) for r in range(3)])
        self.ledger: list[dict] = []
        self.region_rows: list[dict] = []
        self.summary_rows: list[dict] = []

    def audit_rules(self, region: int) -> tuple[float, float, float]:
        p = self.p
        if p.full_information:
            return 1.0, 0.0, 0.0
        if p.shared and region in p.shared_regions:
            return p.shared_accuracy, p.shared_false_reject, p.sharing_cost
        return p.local_accuracy, p.local_false_reject, 0.0

    def input_rows(self) -> list[dict]:
        return [{"seller": s.seller_id, "origin": s.origin, "base_quantity_kg": s.quantity,
                 "ineligible_application_cost": s.application_cost} for s in self.sellers]

    def step(self) -> None:
        p = self.p
        if self.year >= p.seasons:
            raise RuntimeError("Pilot horizon completed")
        tick = self.year
        self.year += 1
        rows = []
        prices = np.array(p.prices) + self.price_shocks[tick]
        for seller in self.sellers:
            i = seller.seller_id
            quantity = seller.quantity * self.quantity_shocks[tick, i]
            destination, apply, expected = seller.choose(quantity, prices)
            best_expected = expected
            if self.fixed_decisions is not None:
                destination, apply = self.fixed_decisions[self.year, i]
                expected = seller.expected_net(quantity, prices, destination, apply)
            eligible = destination == seller.origin
            rows.append({"year": self.year, "seller": i, "batch": f"{self.year}-{i}",
                "origin": seller.origin, "destination": destination, "quantity": float(quantity),
                "eligible": int(eligible), "apply": int(apply), "expected_net": expected,
                "best_expected_net": best_expected,
                "decision_mode": "fixed" if self.fixed_decisions is not None else "adaptive",
                "payment_belief": float(self.payment_belief[destination]),
                "audit_belief": float(self.audit_belief[destination]),
                "reported_origin": destination if apply else seller.origin,
                "claim": float(quantity * p.rates[destination]) if apply else 0.0,
                "sale_revenue": float(quantity * prices[destination]),
                "transport": float(quantity * 0.06 * abs(destination - seller.origin)),
                "production_cost": float(quantity * p.production_cost),
                "application_cost": seller.application_cost if apply and not eligible else 0.0,
                "compliance": 0.0, "checked": 0, "observed_reject": 0,
                "approved": 0, "payment": 0.0, "fine": 0.0})
        region_start = len(self.region_rows)
        for auditor in self.auditors:
            self.region_rows.append(auditor.process([r for r in rows if r["destination"] == auditor.region],
                                                   self.priority_draws[tick], self.audit_draws[tick]))
        for row in rows:
            row["commercial_net"] = (row["sale_revenue"] - row["production_cost"] - row["transport"]
                                     - row["application_cost"] - row["compliance"])
            row["recipient_net"] = row["commercial_net"] + row["payment"] - row["fine"]
        self.ledger.extend(rows)
        regions = self.region_rows[region_start:]
        total = {key: sum(r[key] for r in regions) for key in (
            "applications", "ineligible_applications", "checked", "payment", "eligible_payment",
            "ineligible_payment", "false_rejects", "fine", "recovered", "audit_cost", "crossborder_kg")}
        total.update({"year": self.year, "quantity": sum(r["quantity"] for r in rows),
            "recipient_net": sum(r["recipient_net"] for r in rows),
            "commercial_net": sum(r["commercial_net"] for r in rows),
            "transport": sum(r["transport"] for r in rows),
            "compliance": sum(r["compliance"] for r in rows)})
        total["fiscal_net"] = total["payment"] + total["audit_cost"] - total["fine"]
        total["resource_net"] = total["commercial_net"] - total["audit_cost"]
        assert np.isclose(total["recipient_net"] - total["fiscal_net"],
                          total["commercial_net"] - total["audit_cost"])
        produced = sum(s.quantity * self.quantity_shocks[tick, s.seller_id] for s in self.sellers)
        assert np.isclose(total["quantity"], produced)
        assert len({r["batch"] for r in rows}) == len(self.sellers)
        self.summary_rows.append(total)

    def run(self) -> GrainModel:
        for _ in range(self.p.seasons):
            self.step()
        return self
