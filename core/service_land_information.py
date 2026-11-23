"""Structural information audit; preserve both previous implementations."""
from dataclasses import dataclass

from core.service_land_revision import RevisionParameters, RevisedFarmer, RevisedServiceLandModel


@dataclass(frozen=True)
class InformationParameters(RevisionParameters):
    seasons: int = 12
    forecast_weight: float = 1.0
    current_information: bool = False
    cubic_management: bool = False


class InformationFarmer(RevisedFarmer):
    def plan(self, holding):
        if self.unique_id in self.model.submitted and holding == self.holding:
            return self.model.submitted[self.unique_id]
        return super().plan(holding)


class InformationModel(RevisedServiceLandModel):
    def __init__(self, parameters=None, *, seed=0):
        super().__init__(parameters or InformationParameters(), seed=seed)
        if not 0 < self.p.forecast_weight <= 1:
            raise ValueError("Forecast weight must be in (0, 1]")
        self.submitted = {}
        self.previous_forecast = {}
        self.coordination_rows = []
        replacements = []
        for old in self.farmers:
            old.remove()
            replacements.append(InformationFarmer(old.unique_id, self, old.cost, old.capacity))
        self.farmers = replacements

    def management_cost(self, committed):
        if self.p.cubic_management:
            return self.p.management_gamma * committed * (committed - 1) * max(committed - 2, 0) / 12
        return super().management_cost(committed)

    def _trade(self, land_order, bidders):
        for f in self.farmers:
            observed = f.other_orders
            if self.year > 1:
                w = self.p.forecast_weight
                f.other_orders = round(w * observed + (1 - w) * self.previous_forecast[f.unique_id])
            self.previous_forecast[f.unique_id] = f.other_orders
            f.plan_cache.clear()
        rents, tenants = super()._trade(land_order, bidders)
        self.coordination_status, self.coordination_rounds = "off", 0
        if self.p.current_information:
            plans = {f.unique_id: f.plan(f.holding) for f in self.farmers}
            counts = {i: sum(m == "service" for m in plan[1].values()) for i, plan in plans.items()}
            seen = set()
            for sweep in range(1, 51):
                signature = tuple(tuple(sorted(plans[i][1].items())) for i in sorted(plans))
                if signature in seen:
                    self.coordination_status = "cycle"
                    break
                seen.add(signature)
                changed = 0
                for i in bidders:
                    f = self.farmers[i]
                    f.other_orders = sum(counts.values()) - counts[i]
                    f.plan_cache.clear()
                    candidate = f.plan(f.holding)
                    changed += candidate[1] != plans[i][1]
                    plans[i] = candidate
                    counts[i] = sum(m == "service" for m in candidate[1].values())
                self.coordination_rounds = sweep
                self.coordination_rows.append(dict(year=self.year, sweep=sweep, changed=changed,
                                                   total_orders=sum(counts.values())))
                if not changed:
                    self.coordination_status = "stable"
                    break
            else:
                self.coordination_status = "limit"
            # Execute the final submitted plans even if sequential updating cycles.
            self.submitted = plans
        return rents, tenants

    def step(self):
        self.submitted = {}
        super().step()
        rows = [r for r in self.farmer_rows if r["year"] == self.year]
        total = sum(r["service_orders_mu"] for r in rows)
        for r in rows:
            actual = total - r["service_orders_mu"]
            r["actual_other_orders"] = actual
            r["forecast_error_mu"] = r["forecast_other_orders"] - actual
        self.summary_rows[-1].update(
            forecast_mae_mu=sum(abs(r["forecast_error_mu"]) for r in rows) / len(rows),
            zero_orders=int(total == 0), overloaded=int(total > self.p.daily_capacity * self.p.last_day),
            coordination_status=self.coordination_status, coordination_rounds=self.coordination_rounds)
