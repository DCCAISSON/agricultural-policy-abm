"""Exact joint allocation for two crops on equal-area, indivisible plots."""
from dataclasses import dataclass
from itertools import product
import numpy as np

AREA = 4.0
# Per mu: physical output, price, preplant cost, peak-season days.
CROPS = {"grain": (450., 2.5, 620., 2.5), "cash": (630., 2.5, 940., 5.)}
EPS = 1e-8


@dataclass(frozen=True, slots=True)
class CropOffer:
    id: str
    quality: float
    grant: float = 0.
    credit: float = 0.
    forgone_rent: float = 0.
    restart_cash: float = 0.
    restart_days: float = 0.
    area: float = AREA


@dataclass(frozen=True, slots=True)
class Allocation:
    crops: dict[str, str]
    cash: float
    days: float
    value: float


def variable_allocation(
    offers: list[CropOffer], cash: float, labor: float, wage: float, job_days: float,
) -> Allocation:
    """Apply the 7a cash/labor/surplus objective to unequal-area lots."""
    outside = min(labor, job_days)

    def opportunity(days: float) -> float:
        return wage * (outside - min(max(0., labor-days), job_days))

    candidates = []
    for p in offers:
        for crop, (yield_, price, inputs, work) in CROPS.items():
            grant, credit = (p.grant, p.credit) if crop == "grain" else (0., 0.)
            cost = p.area * inputs + p.restart_cash
            days = p.area * work + p.restart_days
            net = p.area * yield_ * price * p.quality - cost + grant - p.forgone_rent
            if cost-credit <= cash+EPS and days <= labor+EPS and net-opportunity(days)>EPS:
                candidates.append((p.id, crop, cost-credit, days, net))
    if len(offers) <= 4:
        by_plot = {p.id: [(p.id, "idle", 0., 0., 0.)] for p in offers}
        for row in candidates:
            by_plot[row[0]].append(row)
        best = Allocation({}, 0., 0., 0.)
        for items in product(*by_plot.values()):
            used_cash = sum(x[2] for x in items)
            used_days = sum(x[3] for x in items)
            if used_cash <= cash+EPS and used_days <= labor+EPS:
                value = sum(x[4] for x in items) - opportunity(used_days)
                if value > best.value+EPS:
                    best = Allocation({x[0]: x[1] for x in items if x[1] != "idle"}, used_cash, used_days, value)
        return best
    best = Allocation({}, 0., 0., 0.)
    for weight in (0., 0.5, 1.):
        def priority(row: tuple) -> float:
            money, days, net = row[2:]
            scarcity = weight*money/max(cash,EPS) + (1.-weight)*days/max(labor,EPS)
            return (net-opportunity(days))/max(scarcity,EPS)
        employed, used_cash, used_days, value = {}, 0., 0., 0.
        for pid, crop, money, days, net in sorted(candidates, key=priority, reverse=True):
            incremental = net-(opportunity(used_days+days)-opportunity(used_days))
            if pid not in employed and used_cash+money<=cash+EPS and used_days+days<=labor+EPS and incremental>EPS:
                employed[pid] = crop
                used_cash += money
                used_days += days
                value += incremental
        if value > best.value+EPS:
            best = Allocation(employed, used_cash, used_days, value)
    return best


def refine_mixed_plan(offers: list[CropOffer], base: Allocation, cash: float,
                      labor: float, wage: float, job_days: float) -> Allocation:
    """Compare strictly profitable, feasible mixed neighbors of the 7a plan."""
    if len(offers)<=4 or not base.crops or len(set(base.crops.values()))==2:
        return base
    rows=[("", "", 0., 0., 0., 0, 0)]
    counts={k:sum(c==k for c in base.crops.values()) for k in CROPS}
    baseline_net=0.
    for plot in offers:
        def terms(crop: str) -> tuple[float,float,float]:
            if not crop:
                return 0.,0.,0.
            y,p,c,d=CROPS[crop]
            grain=crop=="grain"
            cost=plot.area*c+plot.restart_cash
            return cost-plot.credit*grain,plot.area*d+plot.restart_days,(
                plot.area*y*p*plot.quality-cost+plot.grant*grain-plot.forgone_rent)
        old=base.crops.get(plot.id,"")
        m,t,net=terms(old)
        baseline_net+=net
        for new in ("","grain","cash"):
            if new==old:
                continue
            x,y,g=terms(new)
            rows.append((plot.id,new,x-m,y-t,g-net,
                         int(new=="grain")-int(old=="grain"),int(new=="cash")-int(old=="cash")))
    delta=np.array([x[2:] for x in rows])
    money=base.cash+delta[:,0,None]+delta[None,:,0]
    days=base.days+delta[:,1,None]+delta[None,:,1]
    ids=np.array([x[0] for x in rows])
    feasible=(ids[:,None]!=ids[None,:]) & (money<=cash+EPS)&(days<=labor+EPS)
    for i,crop in enumerate(("grain","cash"),3):
        feasible &= counts[crop]+delta[:,i,None]+delta[None,:,i]>0
    net=baseline_net+delta[:,2,None]+delta[None,:,2]-wage*np.maximum(0.,days-max(0.,labor-job_days))
    score=np.where(feasible,net,-np.inf)
    i,j=np.unravel_index(int(np.argmax(score)),score.shape)
    if score[i,j]<=base.value+EPS:
        return base
    result=base.crops.copy()
    for k in (i,j):
        pid,crop,*_=rows[k]
        if pid:
            if crop:
                result[pid]=crop
            else:
                result.pop(pid,None)
    return Allocation(result,float(money[i,j]),float(days[i,j]),float(score[i,j]))


def joint_allocation(
    offers: list[CropOffer], cash: float, labor: float, wage: float, job_days: float,
) -> Allocation:
    """Maximize expected surplus over the feasible joint crop and idle choices.

    Within each equal-cost group, yield revenue increases with quality, and
    cash crops have the larger quality slope. Thus for given counts the best
    assignment is cash, then grain, on the quality-ranked plots.
    """
    if not offers:
        return Allocation({}, 0., 0., 0.)
    groups: dict[tuple[float, ...], list[CropOffer]] = {}
    for p in sorted(offers, key=lambda p: (-p.quality, p.id)):
        key = (p.grant, p.credit, p.forgone_rent, p.restart_cash, p.restart_days)
        groups.setdefault(key, []).append(p)
    choices = []
    gy, gp, gi, gd = CROPS["grain"]
    cy, cp, ci, cd = CROPS["cash"]
    for (grant, credit, rent, startup, delay), plots in groups.items():
        gc, cc = AREA*gi+startup-credit, AREA*ci+startup
        gt, ct = AREA*gd+delay, AREA*cd+delay
        prefix = [0.]
        for p in plots:
            prefix.append(prefix[-1]+p.quality)
        options = []
        for c in range(min(len(plots), int((cash+EPS)/cc), int((labor+EPS)/ct))+1):
            money, time = c*cc, c*ct
            affordable = int((cash-money+EPS)/gc) if gc>EPS else len(plots)-c
            for g in range(min(len(plots)-c, affordable, int((labor-time+EPS)/gt))+1):
                gross = AREA*(cy*cp*prefix[c]+gy*gp*(prefix[c+g]-prefix[c]))
                net = gross-c*(cc+rent)-g*(AREA*gi+startup+rent-grant)
                options.append((money+g*gc, time+g*gt, net, c, g))
        choices.append(options)
    value = spent = days = 0.
    selected = ()
    paid_before = min(labor, job_days)
    for opt in product(*choices):
        c = sum(x[0] for x in opt)
        t = sum(x[1] for x in opt)
        if c > cash+EPS or t > labor+EPS:
            continue
        net = sum(x[2] for x in opt)-wage*(paid_before-min(max(0., labor-t),job_days))
        if net > value+EPS:
            value, spent, days, selected = net, c, t, opt
    result = {}
    for plots, (_, _, _, c, g) in zip(groups.values(), selected):
        result.update((p.id, "cash") for p in plots[:c])
        result.update((p.id, "grain") for p in plots[c:c+g])
    return Allocation(result, spent, days, value)
