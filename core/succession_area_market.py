"""Province-local 7a matching for survey-derived physical lots."""

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from core.succession_allocation import CROPS, EPS

if TYPE_CHECKING:
    from core.land_succession import Contract, Plot, Succession


def clear_area_market(
    world: "Succession", listed: list["Plot"], cash: dict[str, float],
    planned_days: dict[str, float], wage: dict[str, float], previous: dict[str, str],
    held_buyers: set[str], policy: str, rate: float,
) -> tuple[dict[str, "Contract"], dict[str, tuple[int, float]]]:
    """Match descending bids and settle at the local midpoint without overdraft."""
    from core.land_succession import Contract

    scenario = world.scenario
    fee, ask = scenario.fee, scenario.ask_per_mu
    # A single reservation-floor clearing price per province; maximum bids
    # and the production requirement are reserved separately for each lot.
    listings = defaultdict(list)
    for plot in listed:
        listings[world.operators[plot.owner].market].append(plot)
    bidders = defaultdict(list)
    for k, a in world.operators.items():
        if k not in world.closed and a.market in listings:
            bidders[a.market].append(a)
    contracts, demand = {}, {}
    for market, lots in listings.items():
        people = sorted(bidders[market], key=lambda a: a.id)
        if not people:
            continue
        area = np.array([p.area for p in lots])[None, :]
        quality = np.array([p.quality for p in lots])[None, :]
        rights = np.array([p.owner for p in lots])[None, :]
        old = np.array([previous.get(p.id, "") for p in lots])[None, :]
        keys = np.array([a.id for a in people])[:, None]
        startup = np.array([0. if a.id in held_buyers else scenario.start_cost for a in people])[:, None]
        budget = np.array([cash[a.id] for a in people])[:, None]-startup
        labor = np.array([max(0.,a.labor-planned_days[a.id]) for a in people])[:, None]
        job = np.array([a.job_days for a in people])[:, None]
        w = np.array([wage[a.id] for a in people])[:, None]
        eligible = keys!=rights
        grant = (rate * area if policy.startswith("grain_") else
                 rate * area * (keys!=old) if policy.startswith("takeover_") else 0.)
        bids, inputs, days = [], [], []
        for crop, (yield_, price, cost, work) in CROPS.items():
            payment = grant if crop=="grain" else 0.
            spent = area*cost-(payment if policy.endswith("_pre") else 0.)
            used = area*work
            opportunity = w*(np.minimum(labor,job)-np.minimum(np.maximum(0.,labor-used),job))
            willingness = (area*(yield_*price*quality-cost)+payment-opportunity-fee-startup
                           + np.zeros_like(budget))
            cap = np.minimum(willingness,budget-fee-spent)
            bids.append(np.where(eligible & (used<=labor+EPS) & (cap>=ask*area),cap/area,-np.inf))
            inputs.append(np.broadcast_to(spent,cap.shape))
            days.append(np.broadcast_to(used,cap.shape))
        chosen = bids[1]>bids[0]
        per_mu = np.maximum(*bids)
        cost = np.where(chosen,*reversed(inputs))
        time = np.where(chosen,*reversed(days))
        ii, jj = np.nonzero(np.isfinite(per_mu))
        order = np.lexsort((jj, ii, -per_mu[ii,jj]))
        remaining_money, remaining_days = budget[:,0].copy(), labor[:,0].copy()
        winners = {}
        for b in range(len(people)):
            valid = per_mu[b,np.isfinite(per_mu[b])]
            if len(valid):
                demand[people[b].id] = len(valid),float(valid.max())
        for i,j in zip(ii[order],jj[order]):
            if j in winners:
                continue
            bid = per_mu[i,j]
            reserved = bid*lots[j].area+fee+cost[i,j]
            if reserved<=remaining_money[i]+EPS and time[i,j]<=remaining_days[i]+EPS:
                remaining_money[i] -= reserved
                remaining_days[i] -= time[i,j]
                winners[j] = (people[i].id,bid)
        marginal = min((bid for _,bid in winners.values()),default=ask)
        for j,(buyer,bid) in winners.items():
            p = lots[j]
            contracts[p.id] = Contract(buyer,(ask+marginal)*p.area/2,bid*p.area,ask*p.area,
                                       world.year,world.year+scenario.contract_years-1)
    return contracts,demand
