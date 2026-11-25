"""A closed, synthetic annual land market. No survey records or F1 policy environment."""
from dataclasses import dataclass
from math import floor, isclose

import numpy as np

from modules.land_clearing_house import LandClearingHouse

AREA = 4.0
# Per mu: physical output, price, preplant cost, peak-season days.
CROPS = {"grain": (450., 2.5, 620., 2.5), "cash": (630., 2.5, 940., 5.)}
RATE = 85.
BUDGET = 7200.
SEARCH_FEE = 280.
TENANT_COST = 1000.
ASK_PER_MU = 130.
POLICIES = ("none", "grain_post", "takeover_pre")
TIMING_CONTROLS = ("takeover_post", "grain_pre")


@dataclass(frozen=True)
class Scenario:
    search_failure: float = .15
    cash: float = 1.
    labor: float = 1.
    feedback: bool = True
    fee: float = SEARCH_FEE
    owners: int = 24
    entrants: int = 8
    budget: float = BUDGET
    rate: float = RATE
    outside_wage: float = 1.
    labor_realization: float | None = None
    cash_available: float | None = None
    harvest: float | None = None


@dataclass(frozen=True)
class Operator:
    id: str
    tenant: bool
    cash: float
    labor: float
    wage: float
    job_days: float


@dataclass(frozen=True)
class Plot:
    id: str
    owner: str
    quality: float
    initial_crop: str


def eligible(policy, tenant):
    return policy.startswith("grain_") or (tenant and policy.startswith("takeover_"))


def choose_crop(plot, cash, labor, wage, job_days, expected_support=0., pre_credit=0.):
    best = None
    value = 0.
    for crop, (yield_mu, price, inputs, days) in CROPS.items():
        cost = AREA * inputs
        time = AREA * days
        support = expected_support if crop == "grain" else 0.
        credit = pre_credit if crop == "grain" else 0.
        opportunity = wage * (min(labor, job_days) - min(max(0., labor-time), job_days))
        surplus = AREA * yield_mu * price * plot.quality - cost - opportunity + support
        if cost-credit <= cash+1e-8 and time <= labor+1e-8 and surplus > value:
            value = surplus
            best = crop
    return best, value


class Succession:
    def __init__(self, seed: int, scenario: Scenario):
        self.seed, self.scenario = seed, scenario
        rng = np.random.default_rng([seed, 0, 91])
        self.operators = {}
        self.plots = []
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
        self.cash = {a.id: a.cash for a in self.operators.values()}
        self.operator = {p.id:p.owner for p in self.plots}
        self.crop = {p.id:p.initial_crop for p in self.plots}
        self.closed = set()
        self.last_leases = {}
        self.year = 0
        self.land = AREA*len(self.plots)
        self.observed = dict(grain_area=AREA*sum(p.initial_crop=="grain" for p in self.plots),
                             grain_kg=AREA*CROPS["grain"][0]*sum(p.quality for p in self.plots if p.initial_crop=="grain"),
                             offered_area=0., unfilled_area=0., successor_grain=0., spending=0.)

    def step(self, policy, *, ledger=False):
        if policy not in (*POLICIES, *TIMING_CONTROLS):
            raise ValueError(policy)
        self.year += 1
        s = self.scenario
        if not s.feedback:
            self.cash = {a.id:a.cash for a in self.operators.values()}
            self.closed.clear()
        rng = np.random.default_rng([self.seed, self.year, 91])
        keys = sorted(self.operators)
        job_move = dict(zip(keys, rng.uniform(.70, 1.38, len(keys))))
        labor_shock = dict(zip(keys, rng.uniform(.45, 1.07, len(keys))))
        liquid_shock = dict(zip(keys, rng.uniform(.68, 1., len(keys))))
        search = dict(zip((p.id for p in self.plots), rng.random(len(self.plots))))
        shock = s.harvest if s.harvest is not None else rng.choice([.35, .65, 1., 1.14],p=[.07,.13,.59,.21])
        opening = self.cash.copy()
        wage = {k:self.operators[k].wage*s.outside_wage*job_move[k] for k in keys}
        support = min(s.rate*AREA, s.budget/max(1,len(self.plots))) if policy!="none" else 0.
        pre = policy.endswith("_pre")
        expected_fill = 1.-s.search_failure
        fee = s.fee
        ask = ASK_PER_MU*AREA
        old, previous_crop = self.operator.copy(), self.crop.copy()
        self.operator = {}
        orders, released = [], []
        plan_cash = opening.copy()
        plan_labor = {k:self.operators[k].labor for k in keys}
        for p in self.plots:
            a = self.operators[p.owner]
            grant = support if eligible(policy, False) else 0.
            crop, value = choose_crop(p,plan_cash[a.id],plan_labor[a.id],wage[a.id],a.job_days,
                                      grant,grant if pre else 0.)
            if crop is None or value <= ask*expected_fill:
                released.append(p)
                if search[p.id] >= s.search_failure:
                    orders.append(dict(id=p.id,region=1,side="sell",price=ask,quantity=1.))
            else:
                self.operator[p.id]=a.id
                plan_cash[a.id] -= AREA*CROPS[crop][2]-(grant if crop=="grain" and pre else 0.)
                plan_labor[a.id] -= AREA*CROPS[crop][3]
        # Equal indivisible plots: one quantity unit in the inherited market is four mu.
        for a in self.operators.values():
            if not a.tenant or a.id in self.closed:
                continue
            grant = support if eligible(policy, True) else 0.
            available = max(0.,opening[a.id]-TENANT_COST)
            options = []
            for crop,(yield_mu,price,inputs,days) in CROPS.items():
                g = grant if crop=="grain" else 0.
                working = inputs*AREA - (g if pre else 0.)
                slots = min(floor(a.labor/(days*AREA)),floor(available/(ask+fee+working)))
                if slots:
                    bid = min(yield_mu*price*AREA-inputs*AREA+g-TENANT_COST/slots,
                              available/slots-working)-fee
                    if bid>=ask:
                        options.append((slots*(bid-ask),bid,slots))
            if options:
                _,bid,slots=max(options)
                orders.append(dict(id=a.id,region=1,side="buy",price=bid,quantity=float(slots)))
        contracts = LandClearingHouse.clear_area_orders(orders,self.year,1)
        deals = {c["seller"]:c for c in contracts}
        assert len(deals)==len(contracts) and all(c["area"]==1. for c in contracts)
        assert set(deals).issubset({p.id for p in released})
        plot_index = {p.id:p for p in self.plots}
        account = {k:dict(rent_paid=0.,rent_income=0.,fees=0.,setup=0.,inputs=0.,sales=0.,
                          pre_support=0.,post_support=0.,job_income=0.,consumption=0.,farm_days=0.,job_days=0.) for k in keys}
        for p,c in deals.items():
            k = c["buyer"]
            a = account[k]
            a["rent_paid"] += c["rent"]
            account[plot_index[p].owner]["rent_income"] += c["rent"]
            a["fees"] += fee
            a["setup"] = TENANT_COST
            self.operator[p] = k
        for k,a in account.items():
            self.cash[k] += a["rent_income"]-a["rent_paid"]-a["fees"]-a["setup"]
            assert self.cash[k]>=-1e-7
        liquid = {k:max(0.,self.cash[k])*(s.cash_available if s.cash_available is not None else liquid_shock[k]) for k in keys}
        labor = {k:self.operators[k].labor*(s.labor_realization if s.labor_realization is not None else labor_shock[k]) for k in keys}
        remaining = labor.copy()
        candidates = [(p,self.operator[p.id]) for p in self.plots if p.id in self.operator]
        eligible_count = sum(eligible(policy,self.operators[k].tenant) for _,k in candidates)
        advance = min(s.rate*AREA,s.budget/max(1,eligible_count)) if pre else 0.
        result, awards = {}, {}
        for p,k in candidates:
            a = self.operators[k]
            grant = advance if eligible(policy,a.tenant) else 0.
            post = support if not pre and eligible(policy,a.tenant) else 0.
            crop,_ = choose_crop(p,liquid[k],remaining[k],wage[k],a.job_days,grant+post,grant)
            if crop is None:
                timed_out = remaining[k]<min(x[3] for x in CROPS.values())*AREA
                cash_short = liquid[k]<min(CROPS["grain"][2]*AREA-grant,CROPS["cash"][2]*AREA)
                result[p.id] = ("idle",0.,"labor" if timed_out else "cash" if cash_short else "no_viable_plan")
                continue
            y,price,inputs,days = CROPS[crop]
            payment = grant if crop=="grain" else 0.
            qty = AREA*y*p.quality*shock
            account[k]["pre_support"] += payment
            awards[p.id] = payment
            account[k]["inputs"] += AREA*inputs
            account[k]["sales"] += qty*price
            account[k]["farm_days"] += AREA*days
            self.cash[k] += payment-AREA*inputs
            liquid[k] += payment-AREA*inputs
            remaining[k] -= AREA*days
            assert self.cash[k]>=-1e-7 and liquid[k]>=-1e-7 and remaining[k]>=-1e-7
            result[p.id] = (crop,qty,"")
        grain = [(p,k) for p,k in candidates if result[p.id][0]=="grain"]
        post_claims = [(p,k) for p,k in grain if not pre and eligible(policy,self.operators[k].tenant)]
        for p,k in post_claims:
            payment=min(s.rate*AREA,s.budget/len(post_claims))
            account[k]["post_support"] += payment
            awards[p.id] = payment
        spending = sum(awards.values())
        assert spending<=s.budget+1e-7
        leases = {k:0 for k in keys}
        for c in contracts:
            leases[c["buyer"]] += 1
        actors=[]
        for k,a in account.items():
            op=self.operators[k]
            a["job_days"]=max(0.,min(op.job_days,remaining[k]))
            a["job_income"]=a["job_days"]*wage[k]
            a["support"]=a["pre_support"]+a["post_support"]
            profit=a["sales"]-a["inputs"]-a["rent_paid"]-a["fees"]-a["setup"]
            income=profit+a["support"]+a["rent_income"]+a["job_income"]
            closing=opening[k]+income
            assert closing>=-1e-7
            a["consumption"]=min(max(0.,closing),800. if op.tenant else 1900.)
            self.cash[k]=max(0.,closing-a["consumption"])
            exit_now=op.tenant and leases[k]>0 and self.cash[k]<TENANT_COST+ask+fee
            self.closed.update([k] if exit_now else [])
            actors.append(dict(id=k,kind="tenant" if op.tenant else "owner",opening=opening[k],
                profit=profit,income=income,closing=self.cash[k],labor_capacity=labor[k],leases=leases[k],
                expanded=leases[k]>self.last_leases.get(k,0),shrunk=leases[k]<self.last_leases.get(k,0),
                exit=exit_now,**a))
        self.last_leases=leases
        assert isclose(sum(a["rent_paid"] for a in actors),sum(a["rent_income"] for a in actors),abs_tol=1e-7)
        fields=[]
        for p in self.plots:
            c=deals.get(p.id)
            crop,qty,reason=result.get(p.id,("idle",0.,"search" if search[p.id]<s.search_failure else "no_match"))
            who=self.operator.get(p.id)
            fields.append(dict(plot=p.id,owner=p.owner,operator=who,previous_operator=old.get(p.id),
                previous_crop=previous_crop[p.id],crop=crop,physical_area=AREA,planted_area=AREA if crop!="idle" else 0.,
                harvest=qty,lease=bool(c),rent=c["rent"] if c else 0.,support=awards.get(p.id,0.),failure=reason,
                changed=who is not None and old.get(p.id) not in (None,who),
                successor_grain=bool(c) and crop=="grain",
                continuation=bool(c) and old.get(p.id) not in (None,who)
                and previous_crop[p.id]=="grain" and crop=="grain"))
        self.crop={p["plot"]:p["crop"] for p in fields}
        assert isclose(spending,sum(a["support"] for a in actors),abs_tol=1e-7)
        economic = dict(grain_area=AREA*len(grain),grain_kg=sum(result[p.id][1] for p,_ in grain),
            offered_area=AREA*len(released),leased_area=AREA*len(deals),unfilled_area=AREA*(len(released)-len(deals)),
            idle_area=sum(AREA for p in fields if p["crop"]=="idle"),cash_crop_area=sum(AREA for p in fields if p["crop"]=="cash"),
            successor_grain=sum(AREA for p in fields if p["successor_grain"]),
            continuity_area=sum(AREA for p in fields if p["continuation"]),
            leased_idle=sum(AREA for p in fields if p["lease"] and p["crop"]=="idle"),
            leased_cash=sum(AREA for p in fields if p["lease"] and p["crop"]=="cash"),
            operator_change=sum(AREA for p in fields if p["changed"]),spending=spending,
            tenant_profit=sum(a["profit"] for a in actors if a["kind"]=="tenant"),
            owner_income=sum(a["income"] for a in actors if a["kind"]=="owner"),
            expanded=sum(a["kind"]=="tenant" and a["expanded"] for a in actors),
            shrunk=sum(a["kind"]=="tenant" and a["shrunk"] for a in actors),exits=sum(a["exit"] for a in actors))
        assert isclose(economic["grain_area"]+economic["cash_crop_area"]+economic["idle_area"],self.land)
        self.observed=economic
        if ledger:
            trading=[]
            offers={o["id"]:o for o in orders if o["side"]=="sell"}
            for p in released:
                c=deals.get(p.id)
                trading.append(dict(plot=p.id,owner=p.owner,area=AREA,ask=ask,search_open=p.id in offers,
                    buyer=c["buyer"] if c else None,bid=c["bid"]+fee if c else None,rent=c["rent"] if c else 0.,
                    fee=fee if c else 0.,start=self.year,end=self.year,
                    result="matched" if c else "no_match" if p.id in offers else "search"))
            return economic,fields,actors,trading
        return economic
