"""One physical-season activity budget for own farming, bidding and execution."""
from dataclasses import dataclass
from math import fsum, isfinite
from warnings import catch_warnings, filterwarnings

import numpy as np
from scipy.optimize import OptimizeWarning, linprog

from core.succession_allocation import CROPS, EPS, CropOffer


@dataclass(frozen=True, slots=True)
class Activity:
    name: str
    grain: bool
    kg: float
    unit_value: float
    inputs: float
    days: float
    sold: float = 1.
    markets: tuple[int,...] = ()

    def __post_init__(self):
        if (not all(isfinite(v) and v >= 0 for v in (self.kg, self.unit_value, self.inputs, self.days))
                or self.kg <= 0 or self.days <= 0 or not 0 <= self.sold <= 1):
            raise ValueError(f"Invalid activity budget: {self.name}")


@dataclass(frozen=True, slots=True)
class EconomicCore:
    activities: tuple[Activity, ...]
    hired_wage: float = 0.
    hired_limit: float = 0.

    def __post_init__(self):
        if (not self.activities or len({a.name for a in self.activities}) != len(self.activities)
                or not any(a.grain for a in self.activities)
                or not all(isfinite(v) and v >= 0 for v in (self.hired_wage, self.hired_limit))
                or self.hired_limit > 0 and self.hired_wage == 0):
            raise ValueError("Invalid activity menu or hired labor terms")

    @property
    def grant_cap(self):
        return min(a.inputs for a in self.activities if a.grain)


LEGACY_ACTIVITIES = EconomicCore(tuple(Activity(k, k == "grain", *v) for k, v in CROPS.items()))


@dataclass(frozen=True, slots=True)
class Plan:
    area: dict[str, tuple[tuple[Activity, float], ...]]
    cash: float
    days: float
    hired: float
    job: float
    value: float
    cash_shadow: float
    time_shadow: float


def solve(economics, offers, cash, labor, wage, job_days, market=0):
    """Maximize gross farm value plus paid outside work, net of paid inputs.

    The last two columns are paid outside work and hired farm days. Hire can
    cover farm work only. Only pre-plant credit reduces the planting cash bill.
    """
    activities = tuple(a for a in economics.activities if not a.markets or market in a.markets)
    if not offers or not activities:
        job = min(labor, job_days)
        return Plan({}, 0., 0., 0., job, wage*job, 0., 0.)
    k = len(activities)
    n = len(offers)*k
    money, days, value = (np.zeros(n+2) for _ in range(3))
    land = np.zeros((len(offers), n+2))
    for i, p in enumerate(offers):
        for j, a in enumerate(activities):
            col = i*k+j
            land[i, col] = 1.
            money[col] = a.inputs - (p.credit/p.area if a.grain else 0.)
            days[col] = a.days
            value[col] = a.kg*a.unit_value*p.quality - a.inputs + (p.grant/p.area if a.grain else 0.)
    money[-1] = economics.hired_wage
    value[-1] = -economics.hired_wage
    value[-2] = wage
    time = days.copy()
    time[-2:] = (1., -1.)
    hire_only = -days
    hire_only[-1] = 1.
    # Paths, not these small programs, are parallelized. SciPy passes through
    # this native HiGHS option even though it is not in the linprog signature.
    with catch_warnings():
        filterwarnings("ignore",message="Unrecognized options detected:.*threads",category=OptimizeWarning)
        optimum = linprog(-value, A_ub=np.vstack((land, money, time, hire_only)),
                          b_ub=[*(p.area for p in offers), max(0.,cash), labor, 0.],
                          bounds=[(0., None)]*n+[(0., job_days if wage > 0 else 0.), (0., economics.hired_limit)],
                          method="highs",options={"threads":1})
    if not optimum.success:
        raise RuntimeError(f"Production allocation failed: {optimum.message}")
    x = optimum.x
    return Plan({p.id: rows for i, p in enumerate(offers)
                 if (rows := tuple((a, float(x[i*k+j])) for j, a in enumerate(activities) if x[i*k+j] > EPS))},
                float(money@x), float(days@x), float(x[-1]), float(x[-2]), float(-optimum.fun),
                max(0.,-float(optimum.ineqlin.marginals[-3])),
                max(0.,-float(optimum.ineqlin.marginals[-2])))


def offer(plot, policy, rate, *, takeover=False):
    g = rate*plot.area if policy.startswith("grain_") or (takeover and policy.startswith("takeover_")) else 0.
    return CropOffer(plot.id, plot.quality, g, g if policy.endswith("_pre") else 0., area=plot.area)


def gain_bound(base, activities, incoming, expense):
    """Extend the old LP dual by a land multiplier for each incoming lot."""
    def land(p):
        return p.area*max(0.,max((
            a.kg*a.unit_value*p.quality-a.inputs*(1.+base.cash_shadow)-a.days*base.time_shadow+
            ((p.grant+base.cash_shadow*p.credit)/p.area if a.grain else 0.)
            for a in activities
        ),default=0.))
    return fsum(land(p) for p in incoming)-(1.+base.cash_shadow)*expense


def clear_packages(world, listed, kept, opening, labor, wage, old, held, policy, rate):
    """Rank feasible incremental packages at the posted rent, within provinces.

    The first acceptance pays the annual fixed charge once. Each accepted lot
    pays a transaction charge. Ranking only compares evaluated packages; it
    does not claim the global allocation or a marginal uniform auction price.
    """
    from core.land_succession import Contract

    s = world.scenario
    available = opening.copy()
    plans = {key: tuple(rows) for key, rows in kept.items()}
    active = {c.buyer for c in held.values()}
    plot_lookup = {p.id:p for p in world.plots}
    own = {k:tuple(offer(plot_lookup[p.id],"none",0.) for p in rows if p.id not in held)
           for k,rows in kept.items()}
    ongoing = {k:[] for k in world.operators}
    for pid,c in held.items():
        ongoing[c.buyer].append((c.end,c.rent,offer(plot_lookup[pid],"none",0.)))
    pending = {}
    for p in listed:
        pending.setdefault(world.operators[p.owner].market, []).append(p)
    new, demand = {}, {}
    for region, lots in sorted(pending.items()):
        buyers = [k for k, a in sorted(world.operators.items())
                  if a.market == region and k not in world.closed and available[k] > 0 and labor[k] > 0]
        activities = tuple(a for a in s.economics.activities if not a.markets or region in a.markets)
        lots.sort(key=lambda p: (-p.quality, -p.area, p.id))
        base_cache = {}
        forecast_cache = {}
        while lots:
            first = lots.pop(0)
            group = [first, *lots[:3]]
            feasible = []
            for k in buyers:
                if k == first.owner:
                    continue
                fixed = 0. if k in active else s.start_cost
                base = None
                additions = []
                rent = 0.
                for p in group:
                    if p.owner == k:
                        continue
                    rent += s.ask_per_mu*p.area
                    additions.append(p)
                    expense = rent + fixed + s.fee*len(additions)
                    if expense > available[k]+EPS:
                        break
                    def upper_value(p, instrument, support):
                        terms = offer(p,instrument,support,takeover=old.get(p.id)!=k)
                        return p.area*max((max(0., a.kg*a.unit_value*p.quality-a.inputs+
                                               (terms.grant/p.area if a.grain else 0.))
                                           for a in activities), default=0.)
                    future_value = fsum(upper_value(p,"none",0.) for p in additions)-rent
                    future_bound = fsum(.95**h*(future_value-(0. if any(end>=world.year+h for end,_,_
                                                                       in ongoing[k]) else s.start_cost))
                                        for h in range(1,s.contract_years))
                    optimistic = fsum(upper_value(p,policy,rate) for p in additions)-expense+future_bound
                    if optimistic <= EPS:
                        continue
                    if base is None:
                        if k not in base_cache:
                            base_cache[k] = solve(s.economics, plans.get(k, ()), available[k], labor[k],
                                                  wage[k], world.operators[k].job_days,region)
                        base = base_cache[k]
                    incoming = tuple(offer(p, policy, rate, takeover=old.get(p.id) != k) for p in additions)
                    current_bound = gain_bound(base,activities,incoming,expense)
                    if s.economics.hired_limit==0. and current_bound+future_bound <= EPS:
                        continue
                    # Future feasibility uses stationary opening resources
                    # rather than realized cash, weather or a promised grant.
                    follow_on = tuple(offer(p,"none",0.) for p in additions)
                    future = []
                    for h in range(1,s.contract_years):
                        if (k,h) not in forecast_cache:
                            carry = [(r,p) for end,r,p in ongoing[k] if end>=world.year+h]
                            prior = fsum(r for r,p in carry) + (s.start_cost if carry else 0.)
                            existing = own.get(k,())+tuple(p for _,p in carry)
                            later = solve(s.economics,existing,world.cash[k]-prior,labor[k],wage[k],
                                          world.operators[k].job_days,region)
                            forecast_cache[k,h] = prior, bool(carry), existing, later
                        prior, carry, existing, later = forecast_cache[k,h]
                        due = prior+rent+(0. if carry else s.start_cost)
                        if due > world.cash[k]+EPS:
                            break
                        future.append((h,prior,due,existing,later))
                    if len(future)!=s.contract_years-1:
                        continue
                    if s.economics.hired_limit==0. and current_bound+fsum(
                        .95**h*gain_bound(later,activities,follow_on,due-prior)
                        for h,prior,due,_,later in future
                    ) <= EPS:
                        continue
                    trial = solve(s.economics, plans.get(k, ())+incoming, available[k]-expense,
                                  labor[k], wage[k], world.operators[k].job_days,region)
                    gain = trial.value-base.value-expense
                    if not all(p.id in trial.area for p in additions):
                        continue
                    for h,prior,due,existing,later in future:
                        follow = solve(s.economics,existing+follow_on,
                                       world.cash[k]-due,labor[k],wage[k],world.operators[k].job_days,region)
                        gain += .95**h*(follow.value-later.value-(due-prior))
                    if gain > EPS:
                        size = fsum(p.area for p in additions)
                        feasible.append((-gain/size, -gain, k, tuple(additions), expense, incoming))
                        count, prior = demand.get(k, (0, 0.))
                        demand[k] = (max(count,len(additions)), max(prior,s.ask_per_mu))
            if not feasible:
                continue
            _, _, winner, package, expense, incoming = min(feasible, key=lambda row: row[:3])
            active.add(winner)
            available[winner] -= expense
            plans[winner] = plans.get(winner, ())+incoming
            del base_cache[winner]
            for h in range(1,s.contract_years):
                forecast_cache.pop((winner,h),None)
            won = {p.id for p in package}
            lots = [p for p in lots if p.id not in won]
            for p in package:
                rent = s.ask_per_mu*p.area
                end = world.year+s.contract_years-1
                new[p.id] = Contract(winner, rent, rent, rent, world.year, end)
                ongoing[winner].append((end,rent,offer(p,"none",0.)))
    return new, demand
