"""Single-season shock extension; deposition is an accounting layer, not efficacy."""
from dataclasses import asdict

import numpy as np

from core.uav_spatial_pilot import UAVModel, UAVParameters

FOCAL = (22, 23, 32, 33)
WIND = {"east": (1, 0), "west": (-1, 0), "north": (0, -1), "south": (0, 1)}
GROUP_NAMES = {"focal": "冲击地块", "upwind": "紧邻上风向", "downwind": "紧邻下风向", "other": "其他地块"}


def neighbor(plot: int, wind: str) -> int | None:
    dx, dy = WIND[wind]
    x, y = plot % 10 + dx, plot // 10 + dy
    return y * 10 + x if 0 <= x < 10 and 0 <= y < 6 else None


def plot_groups(wind: str) -> dict[str, list[int]]:
    opposite = {"east": "west", "west": "east", "north": "south", "south": "north"}
    groups = {"focal": list(FOCAL)}
    for name, direction in (("upwind", opposite[wind]), ("downwind", wind)):
        groups[name] = sorted({neighbor(j, direction) for j in FOCAL} - set(FOCAL) - {None})
    assigned = {j for js in groups.values() for j in js}
    groups["other"] = sorted(set(range(60)) - assigned)
    return groups


def deposition_rows(actions: list[str], owners, week: int, wind: str, drift: bool) -> list[dict]:
    """The user's illustrative 100 = 90 + 3 + 7 ledger, identical for both methods."""
    rows = []
    for j, action in enumerate(actions):
        if action == "none":
            continue
        recipient = neighbor(j, wind)
        transferred = 3.0 if drift and recipient is not None else 0.0
        rows.append(dict(week=week, source=j, farmer=int(owners[j]), method=action,
            potential_recipient=recipient, recipient=recipient if transferred else None,
            recipient_farmer=int(owners[recipient]) if transferred else None,
            applied=100.0, target=90.0, neighbor=transferred,
            outside_or_ineffective=10.0 - transferred,
            baseline_other=7.0, boundary_loss=3.0 if drift and recipient is None else 0.0,
            diverted_by_shutdown=0.0 if drift else 3.0))
    return rows


class SpilloverModel(UAVModel):
    """Reuse the original Mesa decisions and ecology; inject one observed shock."""
    shock_week = 4
    shock_size = 0.35
    service_fee = 8.0
    chemical_cost = 10.0

    def __init__(self, *, seed=0, shock=True, spread=True, drift=True, wind="east"):
        if type(seed) is not int or not 0 <= seed <= 999:
            raise ValueError("随机环境编号须为0至999的整数")
        if any(type(v) is not bool for v in (shock, spread, drift)) or wind not in WIND:
            raise ValueError("冲击、传播、转移或风向设置不正确")
        self.settings = dict(seed=seed, shock=shock, spread=spread, drift=drift, wind=wind)
        # Supply sufficiency is a stipulated benchmark, not measured weekly capacity.
        p = UAVParameters(seasons=1, weeks=10, decision_rule="forecast", min_interval=2,
            work_rate=30, capacity=2, travel_hours=0, preparation_hours=0,
            initial_belief=1, learning=0, social_weight=0, spread=.25 if spread else 0,
            fee=self.service_fee + self.chemical_cost, job_cost=4 + self.chemical_cost,
            manual_effect=.65, drone_effect=.65)
        super().__init__(p, seed=seed)
        for f in self.farmers:
            f.manual_cost = 15 + (f.manual_cost - 45) * 10 / 45 + self.chemical_cost
        self.social_edges = []
        self.groups = plot_groups(wind)
        self.group_of = {j: group for group, js in self.groups.items() for j in js}
        self.deposition: list[dict] = []
        self.spatial_rows: list[dict] = []
        self.group_rows: list[dict] = []
        self.spatial_totals: dict = {}
        self.cumulative_pressure = np.zeros(60)
        self.cumulative_cost = np.zeros(60)
        self.cumulative_deposition = np.zeros(60)

    def weekly_steps(self):
        base = super().weekly_steps()
        for week in range(1, self.p.weeks + 1):
            prior = self.pressure.copy()
            if self.settings["shock"] and week == self.shock_week:
                self.pressure[list(FOCAL)] = np.minimum(1, self.pressure[list(FOCAL)] + self.shock_size)
            impulse = self.pressure - prior
            time = next(base)
            plots = self.plot_rows[-60:]
            assert all(r["action"] != "drone" or r["completed"] for r in plots)
            deposits = deposition_rows([r["actual_action"] for r in plots], self.owner,
                                       week, self.settings["wind"], self.settings["drift"])
            self.deposition.extend(deposits)
            applied, target, incoming, other_owner, outgoing, outside = (np.zeros(60) for _ in range(6))
            for r in deposits:
                j = r["source"]
                applied[j], target[j] = r["applied"], r["target"]
                outgoing[j], outside[j] = r["neighbor"], r["outside_or_ineffective"]
                if r["recipient"] is not None:
                    incoming[r["recipient"]] += r["neighbor"]
                    if r["farmer"] != r["recipient_farmer"]:
                        other_owner[r["recipient"]] += r["neighbor"]
            assert np.isclose(applied.sum(), target.sum() + incoming.sum() + outside.sum())
            self.cumulative_pressure += self.pressure
            self.cumulative_deposition += incoming
            for r, detail in zip(plots, self.week_detail):
                j = r["plot"]
                cost = r["fee"] + r["manual_cost"]
                self.cumulative_cost[j] += cost
                loss = self.p.yield_kg * self.p.damage * self.cumulative_pressure[j] / self.p.weeks
                final_output = self.p.yield_kg - loss if week == self.p.weeks else None
                self.spatial_rows.append(dict(r, **{k: v for k, v in detail.items() if k != "plot"},
                    group=self.group_of[j], pest_before_shock=float(prior[j]), shock_added=float(impulse[j]),
                    applied=float(applied[j]), target_deposition=float(target[j]),
                    incoming=float(incoming[j]), incoming_other_farmer=float(other_owner[j]),
                    outgoing=float(outgoing[j]), outside_or_ineffective=float(outside[j]),
                    cumulative_incoming=float(self.cumulative_deposition[j]),
                    treatment_cost=float(cost), chemical_cost=self.chemical_cost if applied[j] else 0.0,
                    operation_cost=float(cost - self.chemical_cost) if applied[j] else 0.0,
                    cumulative_cost=float(self.cumulative_cost[j]), cumulative_loss_kg=float(loss),
                    output_kg=final_output,
                    farmer_net=(final_output * self.p.price - self.p.other_cost - self.cumulative_cost[j]
                                if final_output is not None else None)))
            current = self.spatial_rows[-60:]
            for group, js in self.groups.items():
                rs = [current[j] for j in js]
                self.group_rows.append(dict(week=week, group=group, area_mu=len(js),
                    pest=float(np.mean([r["pest_after"] for r in rs])),
                    cumulative_loss_kg=sum(r["cumulative_loss_kg"] for r in rs),
                    cumulative_cost=sum(r["cumulative_cost"] for r in rs),
                    applied=sum(r["applied"] for r in rs), incoming=sum(r["incoming"] for r in rs),
                    cumulative_incoming=sum(r["cumulative_incoming"] for r in rs),
                    output_kg=sum(r["output_kg"] for r in rs) if week == self.p.weeks else None,
                    farmer_net=sum(r["farmer_net"] for r in rs) if week == self.p.weeks else None))
            totals = {k: sum(r[k] for r in self.deposition) for k in
                      ("applied", "target", "neighbor", "outside_or_ineffective")}
            self.spatial_totals = dict(totals,
                cumulative_loss_kg=float(sum(r["cumulative_loss_kg"] for r in current)),
                cumulative_cost=float(self.cumulative_cost.sum()),
                incoming_other_farmer=sum(r["incoming_other_farmer"] for r in self.spatial_rows))
            yield time

    def initial_inputs(self):
        return [dict(r, group=self.group_of[r["plot"]],
                     manual_operation_cost=r["manual_cost"] - self.chemical_cost)
                for r in self.input_rows()]

    def assumptions(self):
        return dict(model=asdict(self.p), shock_week=self.shock_week, shock_size=self.shock_size,
            focal_plots=FOCAL, service_fee=self.service_fee, chemical_cost=self.chemical_cost,
            manual_operation_cost_range=[15, 25], provider_operation_cost=4,
            dose_unit="standardized active-ingredient-equivalent; not grams or liquid volume",
            applied_per_treatment=100, target_per_treatment=90, transferable_per_treatment=3,
            baseline_other_per_treatment=7, deposition_efficacy_feedback=False,
            phytotoxicity=False, residue_model=False, empirical_validation=False)
