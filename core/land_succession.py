"""Synthetic land succession and fiscal transmission, one annual economy."""
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from math import floor, isclose

import numpy as np

from core.succession_allocation import AREA, CROPS, EPS, CropOffer, joint_allocation

RATE = 85.
BUDGET = 7200.
SEARCH_FEE = 280.
TENANT_COST = 1000.
ASK_PER_MU = 130.
RECOVERY_COST = 120.
RECOVERY_DAYS = 2.
POLICIES = ("none", "grain_pre", "grain_post", "takeover_pre", "takeover_post")
TIMING_CONTROLS = ()


@dataclass(frozen=True, slots=True)
class Scenario:
    search_failure: float = .15
    cash: float = 1.
    labor: float = 1.
    fee: float = SEARCH_FEE
    owners: int = 24
    entrants: int = 8
    budget: float = BUDGET
    rate: float = RATE
    outside_wage: float = 1.
    labor_realization: float | None = None
    cash_available: float | None = None
    harvest: float | None = None
    contract_years: int = 1
    recovery: bool = False
    reentry: bool = False
    carry_cash: bool = True
    carry_exit: bool = True
    ask_per_mu: float = ASK_PER_MU
    start_cost: float = TENANT_COST
    feedback: bool = True


@dataclass(frozen=True, slots=True)
class Operator:
    id: str
    tenant: bool
    cash: float
    labor: float
    wage: float
    job_days: float


@dataclass(frozen=True, slots=True)
class Plot:
    id: str
    owner: str
    quality: float
    initial_crop: str


@dataclass(frozen=True, slots=True)
class Contract:
    buyer: str
    rent: float
    bid: float
    ask: float
    start: int
    end: int


def choose_crop(plot, cash, labor, wage, job_days, expected_support=0., pre_credit=0.):
    plan = joint_allocation([CropOffer(plot.id, plot.quality, expected_support, pre_credit)],
                            cash, labor, wage, job_days)
    return plan.crops.get(plot.id), plan.value


class Succession:
    def __init__(self, seed: int, scenario: Scenario):
        self.seed, self.scenario = seed, scenario
        rng = np.random.default_rng([seed, 0, 91])
        self.operators: dict[str, Operator] = {}
        self.plots: list[Plot] = []
        for h in range(scenario.owners):
            a = Operator(f"H{h:02}", False, rng.uniform(3900, 10000), rng.uniform(22, 45),
                         rng.uniform(0, 185), rng.uniform(8, 38))
            self.operators[a.id] = a
            for k in range(2):
                self.plots.append(Plot(f"L{2*h+k:03}", a.id, rng.uniform(.83, 1.17),
                                       "grain" if rng.random() < .8 else "cash"))
        for j in range(scenario.entrants):
            a = Operator(f"T{j:02}", True, scenario.cash*rng.uniform(4200, 15500),
                         scenario.labor*rng.uniform(16, 50), 0., 0.)
            self.operators[a.id] = a
        self.cash = {a.id:a.cash for a in self.operators.values()}
        self.operator = {p.id:p.owner for p in self.plots}
        self.crop = {p.id:p.initial_crop for p in self.plots}
        self.closed: dict[str,int] = {}
        self.contracts: dict[str,Contract] = {}
        self.last_leases = {}
        self.year = 0
        self.land = AREA*len(self.plots)
        self.observed = dict(grain_area=AREA*sum(p.initial_crop=="grain" for p in self.plots),
                             grain_kg=AREA*CROPS["grain"][0]*sum(
                                 p.quality for p in self.plots if p.initial_crop=="grain"),
                             offered_area=0., unfilled_area=0., successor_grain=0.,
                             continuity_area=0., spending=0.)

    def fork(self):
        """Copy mutable stocks for a conditionally fixed-contract comparison."""
        other = copy(self)
        for name in ("cash", "operator", "crop", "closed", "contracts", "last_leases"):
            setattr(other, name, getattr(self, name).copy())
        other.plots = self.plots.copy()
        other.operators = self.operators.copy()
        return other

    def _market(self, listed, plan_cash, plan_days, wage, old, held_buyers, policy, amount):
        s = self.scenario
        ask = AREA*s.ask_per_mu
        bids = []
        quotas = {}
        demand = {}
        for k, a in sorted(self.operators.items()):
            if k in self.closed or not listed:
                continue
            available = max(0., plan_cash[k]-(0. if k in held_buyers else s.start_cost))
            fee = s.fee
            work_days = max(0., a.labor-plan_days[k])
            if available < ask+fee or work_days < AREA*CROPS["grain"][3]:
                continue
            # Capacity uses the least expensive quoted crop, with no reliance on rent income.
            working = AREA*CROPS["grain"][2]
            slots = min(len(listed), floor(work_days/(AREA*CROPS["grain"][3])),
                        floor(available/(ask+fee+working)))
            if not slots and policy.endswith("_pre"):
                slots = min(len(listed), floor(work_days/(AREA*CROPS["grain"][3])),
                            floor(available/(ask+fee+working-amount)))
            if not slots:
                continue
            quotas[k] = slots
            cap = available/slots-fee
            startup = (0. if k in held_buyers else s.start_cost)/slots
            eligible_listings = 0
            max_bid = 0.
            for p in listed:
                if p.owner==k:
                    continue
                qualified = policy.startswith("grain_") or (
                    policy.startswith("takeover_") and old.get(p.id)!=k)
                grant = amount if qualified else 0.
                values = []
                for crop,(y,price,cost,days) in CROPS.items():
                    if AREA*days > work_days:
                        continue
                    g = grant if crop=="grain" else 0.
                    prospective_cost = AREA*cost-(g if policy.endswith("_pre") else 0.)
                    opportunity = wage[k]*(min(work_days,a.job_days)-
                                            min(max(0.,work_days-AREA*days),a.job_days))
                    values.append(min(AREA*(y*price*p.quality-cost)+g-opportunity-startup-fee,
                                      cap-prospective_cost))
                bid = max(values, default=-1.)
                if bid >= ask:
                    bids.append((-bid, k, -p.quality, p.id))
                    max_bid = max(max_bid,bid)
                    eligible_listings += 1
            demand[k] = (min(slots,eligible_listings),max_bid)
        winners = {}
        for negative, k, _, p in sorted(bids):
            if quotas[k] and p not in winners:
                winners[p] = (k,-negative)
                quotas[k] -= 1
        marginal_bid = min((bid for _,bid in winners.values()), default=ask)
        rent = (ask+marginal_bid)/2.
        contracts = {p:Contract(k,rent,bid,ask,self.year,self.year+s.contract_years-1)
                     for p,(k,bid) in winners.items()}
        return contracts, demand

    def step(self, policy, *, ledger=False, fixed=None):
        """Advance one year; fixed is the same-opening-state P0 contract ledger."""
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy: {policy}")
        self.year += 1
        s = self.scenario
        if not s.feedback or not s.carry_cash:
            self.cash = {a.id:a.cash for a in self.operators.values()}
        if not s.feedback or not s.carry_exit:
            self.closed.clear()
        keys = sorted(self.operators)
        plots = sorted(self.plots, key=lambda p:p.id)
        by_id = {p.id:p for p in plots}
        old, previous_crop = self.operator, self.crop
        opening = self.cash.copy()
        account = {k:dict(rent_paid=0.,rent_income=0.,fees=0.,setup=0.,inputs=0.,
                          sales=0.,pre_support=0.,post_support=0.,job_income=0.,
                          consumption=0.,farm_days=0.,job_days=0.,restart_capital=0.,
                          recovery_cost=0.,recovery_days=0.) for k in keys}
        for k, year in tuple(self.closed.items()):
            if s.reentry and self.year >= year+2:
                injection = max(0.,self.operators[k].cash-self.cash[k])
                self.cash[k] += injection
                account[k]["restart_capital"] = injection
                del self.closed[k]
        # Draws keyed by stable sorted IDs and by independent streams.
        stream = lambda n: np.random.default_rng([self.seed,self.year,n])
        job_move = dict(zip(keys,stream(91).uniform(.70,1.38,len(keys))))
        labor_shock = dict(zip(keys,stream(92).uniform(.45,1.07,len(keys))))
        liquid_shock = dict(zip(keys,stream(93).uniform(.68,1.,len(keys))))
        search = dict(zip(by_id,stream(94).random(len(plots))))
        shock = s.harvest if s.harvest is not None else stream(95).choice(
            [.35,.65,1.,1.14],p=[.07,.13,.59,.21])
        wage = {k:self.operators[k].wage*s.outside_wage*job_move[k] for k in keys}
        pre = policy.endswith("_pre")
        ask = AREA*s.ask_per_mu
        # A fixed guarantee announced before orders; unused reservations never raise the rate.
        amount = min(s.rate*AREA,s.budget/max(1,len(plots)),AREA*CROPS["grain"][2]) if policy!="none" else 0.

        held = {}
        defaulted = set()
        portfolios = defaultdict(dict)
        for p,c in self.contracts.items():
            if c.end>=self.year:
                portfolios[c.buyer][p] = c
        for k, holdings in portfolios.items():
            if k in self.closed or self.cash[k]+EPS < sum(c.rent for c in holdings.values())+s.start_cost:
                defaulted.update(holdings)
            else:
                held.update(holdings)
        held_buyers = {c.buyer for c in held.values()}
        plan_cash = self.cash.copy()
        for p,c in held.items():
            plan_cash[c.buyer] -= c.rent
        for k in held_buyers:
            plan_cash[k] = max(0.,plan_cash[k]-s.start_cost)
        retained = set()
        released = set()
        posted = set()
        self.operator = {p:c.buyer for p,c in held.items()}
        plan_spend = defaultdict(float)
        plan_days = defaultdict(float)
        planned_crop = {}
        demand = {}
        if fixed is None:
            available = defaultdict(list)
            for p in plots:
                if p.id not in held:
                    available[p.owner].append(p)
            protected = defaultdict(list)
            for pid,c in held.items():
                protected[c.buyer].append(CropOffer(pid,by_id[pid].quality,
                    amount if policy.startswith("grain_") else 0.,
                    amount if pre and policy.startswith("grain_") else 0.))
            for k in keys:
                a = self.operators[k]
                grant = amount if policy.startswith("grain_") else 0.
                own = [CropOffer(p.id,p.quality,grant,grant if pre else 0.,
                                 ask*(1.-s.search_failure)) for p in available[k]]
                plan = joint_allocation(own+protected[k],plan_cash[k],a.labor,wage[k],a.job_days)
                plan_spend[k],plan_days[k] = plan.cash,plan.days
                planned_crop.update(plan.crops)
                for p in available[k]:
                    if p.id in plan.crops:
                        retained.add(p.id)
                        self.operator[p.id] = k
                    else:
                        released.add(p.id)
            posted = {p for p in released if search[p]>=s.search_failure}
            free_cash = {k:max(0.,plan_cash[k]-plan_spend[k]) for k in keys}
            new, demand = self._market([p for p in plots if p.id in posted],free_cash,plan_days,wage,
                                       old,held_buyers,policy,amount)
            if s.recovery:
                for pid in released-new.keys():
                    self.operator[pid] = by_id[pid].owner
        else:
            # P0 fixes trading, reservation rents and pre-auction outcomes, not crops.
            held, new = {}, {}
            self.operator = {}
            for f in fixed:
                pid = f["plot"]
                if f["assigned_operator"] is not None:
                    self.operator[pid] = f["assigned_operator"]
                if f["released"]:
                    released.add(pid)
                if f["posted"]:
                    posted.add(pid)
                if f["defaulted"]:
                    defaulted.add(pid)
                if f["lease"]:
                    c = Contract(f["operator"],f["rent"],f["bid"],f["ask"],f["contract_start"],f["contract_end"])
                    (new if f["matched"] else held)[pid] = c
                elif f["retained_plan"]:
                    retained.add(pid)
        self.contracts = held|new
        for p,c in self.contracts.items():
            self.operator[p] = c.buyer
            account[c.buyer]["rent_paid"] += c.rent
            account[by_id[p].owner]["rent_income"] += c.rent
            account[c.buyer]["fees"] += s.fee if p in new else 0.
            account[c.buyer]["setup"] = s.start_cost
        for k,a in account.items():
            self.cash[k] += a["rent_income"]-a["rent_paid"]-a["fees"]-a["setup"]
            assert self.cash[k]>=-1e-7, (self.year,k,self.cash[k])
        liquid = {k:max(0.,self.cash[k])*(s.cash_available if s.cash_available is not None else liquid_shock[k])
                  for k in keys}
        paid_contract_cash = self.cash.copy()
        labor = {k:self.operators[k].labor*(s.labor_realization if s.labor_realization is not None else labor_shock[k])
                 for k in keys}
        allocated = defaultdict(list)
        qualification = {}
        for p in plots:
            k = self.operator.get(p.id)
            if k is None:
                continue
            qualified = policy.startswith("grain_") or (
                policy.startswith("takeover_") and p.id in new and old.get(p.id)!=k)
            qualification[p.id] = qualified
            recovery = p.id in released and p.id not in self.contracts
            g = amount if qualified else 0.
            allocated[k].append(CropOffer(p.id,p.quality,g,g if pre else 0.,
                                           restart_cash=RECOVERY_COST if recovery else 0.,
                                           restart_days=RECOVERY_DAYS if recovery else 0.))
        result, awards, failure = {}, {}, {}
        for k in keys:
            op = self.operators[k]
            plan = joint_allocation(allocated[k],liquid[k],labor[k],wage[k],op.job_days)
            a = account[k]
            for p in allocated[k]:
                crop = plan.crops.get(p.id)
                if crop is None:
                    min_cash = AREA*CROPS["grain"][2]+p.restart_cash-p.credit
                    min_labor = AREA*CROPS["grain"][3]+p.restart_days
                    cash_short = min_cash>liquid[k]-plan.cash+EPS
                    labor_short = min_labor>labor[k]-plan.days+EPS
                    failure[p.id] = ("cash_and_labor" if cash_short and labor_short else
                                     "cash" if cash_short else "labor" if labor_short else "no_viable_plan")
                    continue
                y,price,inputs,days = CROPS[crop]
                qty = AREA*y*p.quality*shock
                payment = p.grant if crop=="grain" else 0.
                awards[p.id] = payment
                a["pre_support" if pre else "post_support"] += payment
                a["inputs"] += AREA*inputs
                a["sales"] += qty*price
                a["farm_days"] += AREA*days
                a["recovery_cost"] += p.restart_cash
                a["recovery_days"] += p.restart_days
                result[p.id] = (crop,qty)
            assert a["inputs"]+a["recovery_cost"]-a["pre_support"]<=liquid[k]+1e-7
            a["job_days"] = min(op.job_days,max(0.,labor[k]-plan.days))
            a["job_income"] = a["job_days"]*wage[k]
            a["foregone_job_income"] = (min(op.job_days,labor[k])-a["job_days"])*wage[k]

        actors = []
        planted_count = defaultdict(int)
        for p in result:
            planted_count[self.operator[p]] += 1
        leases = defaultdict(int)
        for c in self.contracts.values():
            leases[c.buyer] += 1
        for k,a in account.items():
            op = self.operators[k]
            a["support"] = a["pre_support"]+a["post_support"]
            profit = a["sales"]-a["inputs"]-a["rent_paid"]-a["fees"]-a["setup"]-a["recovery_cost"]
            income = profit+a["support"]+a["rent_income"]+a["job_income"]
            balance = opening[k]+a["restart_capital"]+income
            assert balance>=-1e-7
            a["consumption"] = min(max(0.,balance),800. if op.tenant else 1900.)
            self.cash[k] = max(0.,balance-a["consumption"])
            exit_now = op.tenant and leases[k]>0 and self.cash[k]<s.start_cost+ask+s.fee
            if exit_now:
                self.closed[k] = self.year
            assert a["farm_days"]+a["recovery_days"]+a["job_days"]<=labor[k]+1e-7
            slots, bid = demand.get(k,(0,0.))
            actors.append(dict(id=k,kind="tenant" if op.tenant else "owner",opening=opening[k],
                profit=profit,income=income,closing=self.cash[k],labor_capacity=labor[k],liquidity=liquid[k],
                post_rent_cash=paid_contract_cash[k],off_farm_wage=wage[k],weather=shock,
                planned_cash=plan_spend[k],planned_days=plan_days[k],leases=leases[k],
                quoted_plots=slots,highest_bid_per_mu=bid/AREA,
                expanded=leases[k]>self.last_leases.get(k,0),shrunk=leases[k]<self.last_leases.get(k,0),
                operation_area=AREA*planted_count[k],exit=exit_now,**a))
        self.last_leases = dict(leases)
        fields = []
        trading = []
        for p in plots:
            pid = p.id
            c = self.contracts.get(pid)
            crop,qty = result.get(pid,("idle",0.))
            assigned,prev = self.operator.get(pid),old.get(pid)
            who = assigned
            if who is not None and not c and crop=="idle":
                # No tenant contract binds this owner to an actually unplanted plot.
                del self.operator[pid]
                who = None
            qualified = qualification.get(pid,False)
            changed = who is not None and prev not in (None,who)
            unchanged = who is not None and who==prev
            recovery = who==p.owner and pid in released and c is None
            reason = failure.get(pid,"" if crop!="idle" else ("search" if pid not in posted else "no_match"))
            rent = c.rent if c else 0.
            fee = s.fee if pid in new else 0.
            fields.append(dict(plot=pid,owner=p.owner,operator=who,assigned_operator=assigned,previous_operator=prev,
                previous_crop=previous_crop[pid],crop=crop,planned_crop=planned_crop.get(pid),
                quality=p.quality,physical_area=AREA,planted_area=AREA if crop!="idle" else 0.,
                harvest=qty,lease=bool(c),released=pid in released,posted=pid in posted,matched=pid in new,
                rent=rent,rent_per_mu=rent/AREA,bid=c.bid if c else 0.,ask=c.ask if c else ask,
                contract_start=c.start if c else None,contract_end=c.end if c else None,fee=fee,
                defaulted=pid in defaulted,recovered=recovery,retained_plan=pid in retained,
                eligible=qualified,expected_support=amount if qualified else 0.,
                commitment=amount if qualified else 0.,support=awards.get(pid,0.),
                failure=reason,changed=changed,same_operator=unchanged,
                new_lease=pid in new and who!=prev,renewed=pid in new and who==prev,
                continued_lease=pid in held,leased_idle=bool(c) and crop=="idle",
                leased_cash=bool(c) and crop=="cash",retained_grain=unchanged and previous_crop[pid]==crop=="grain",
                switched_to_cash=unchanged and previous_crop[pid]=="grain" and crop=="cash",
                successor_grain=changed and crop=="grain",reactivated=prev is None and crop!="idle",
                continuation=changed and previous_crop[pid]==crop=="grain",next_operator=who,next_crop=crop))
            if ledger and (pid in released or c or pid in defaulted):
                trading.append(dict(plot=pid,owner=p.owner,area=AREA,ask=c.ask if c else ask,
                    search_open=pid in posted,posted=pid in posted,buyer=c.buyer if c else None,
                    bid=c.bid if c else None,rent=rent,rent_per_mu=rent/AREA,fee=fee,
                    start=c.start if c else None,end=c.end if c else None,
                    matched=pid in new,continued=pid in held,defaulted=pid in defaulted,
                    result="matched" if pid in new else "continued" if c else "no_match" if pid in posted else "search"))
        self.crop = {p["plot"]:p["crop"] for p in fields}
        economic = _annual(fields, actors, s, self.land)
        self.observed = economic
        return (economic,fields,actors,trading) if ledger else economic


def _annual(fields, actors, scenario, land):
    grain = [f for f in fields if f["crop"]=="grain"]
    leased = [f for f in fields if f["lease"]]
    spending = sum(f["support"] for f in grain)
    rents = sum(f["rent"] for f in leased)
    assert spending<=scenario.budget+1e-7
    assert isclose(spending,sum(a["support"] for a in actors),abs_tol=1e-7)
    assert isclose(rents,sum(a["rent_paid"] for a in actors),abs_tol=1e-7)
    assert isclose(rents,sum(a["rent_income"] for a in actors),abs_tol=1e-7)
    total_planted = sum(a["operation_area"] for a in actors)
    owners = [a for a in actors if a["kind"]=="owner"]
    tenants = [a for a in actors if a["kind"]=="tenant"]
    amount = lambda key: AREA*sum(f[key] for f in fields)
    return dict(grain_area=AREA*len(grain),grain_kg=sum(f["harvest"] for f in grain),
        offered_area=amount("released"),matched_area=amount("matched"),
        posted_area=amount("posted"),leased_area=AREA*len(leased),
        unfilled_area=AREA*sum(f["released"] and not f["matched"] for f in fields),
        idle_area=land-total_planted,cash_crop_area=total_planted-AREA*len(grain),
        successor_grain=amount("successor_grain"),continuity_area=amount("continuation"),
        retained_grain=amount("retained_grain"),reactivated_area=amount("reactivated"),
        renewed_area=amount("renewed"),held_area=amount("continued_lease"),default_area=amount("defaulted"),
        recovered_area=amount("recovered"),leased_idle=amount("leased_idle"),
        leased_cash=amount("leased_cash"),operator_change=amount("changed"),
        leased_grain=AREA*sum(f["crop"]=="grain" for f in leased),
        spending=spending,budget=scenario.budget,unused_budget=scenario.budget-spending,
        announced_rate=scenario.rate,committed=sum(f["commitment"] for f in fields),
        rent_total=rents,rent_mu=rents/(AREA*len(leased)) if leased else 0.,
        tenant_profit=sum(a["profit"] for a in tenants),owner_income=sum(a["income"] for a in owners),
        owner_rent_income=sum(a["rent_income"] for a in owners),
        owner_profit=sum(a["profit"] for a in owners),owner_nonfarm=sum(a["job_income"] for a in owners),
        owner_support=sum(a["support"] for a in owners),tenant_support=sum(a["support"] for a in tenants),
        owner_rent_paid=sum(a["rent_paid"] for a in owners),tenant_rent_paid=sum(a["rent_paid"] for a in tenants),
        tenant_income=sum(a["income"] for a in tenants),tenant_area=sum(a["operation_area"] for a in tenants),
        owner_area=sum(a["operation_area"] for a in owners),
        owner_opening_cash=sum(a["opening"] for a in owners),tenant_opening_cash=sum(a["opening"] for a in tenants),
        owner_closing_cash=sum(a["closing"] for a in owners),tenant_closing_cash=sum(a["closing"] for a in tenants),
        concentration=sum((a["operation_area"]/total_planted)**2 for a in actors) if total_planted else 0.,
        cash_unplanted=AREA*sum(f["failure"] in ("cash","cash_and_labor") for f in fields),
        labor_unplanted=AREA*sum(f["failure"] in ("labor","cash_and_labor") for f in fields),
        expanded=sum(a["expanded"] for a in tenants),shrunk=sum(a["shrunk"] for a in tenants),
        exits=sum(a["exit"] for a in tenants))
