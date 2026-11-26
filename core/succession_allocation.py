"""Exact joint allocation for two crops on equal-area, indivisible plots."""
from dataclasses import dataclass
from itertools import product

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


@dataclass(frozen=True, slots=True)
class Allocation:
    crops: dict[str, str]
    cash: float
    days: float
    value: float


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
