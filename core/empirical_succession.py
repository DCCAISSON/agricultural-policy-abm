"""Read-only synthetic land ledger for the household area-economy scheduler."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import fsum, isclose


EPS = 1e-7


@dataclass(frozen=True)
class Slice:
    start: float
    end: float
    operator: str
    crop: str


def equal(actual: float, expected: float, name: str) -> None:
    if not isclose(actual, expected, abs_tol=1e-5, rel_tol=1e-10):
        raise ValueError(f"{name}: {actual} != {expected}")


class SyntheticLandLedger:
    """Each surveyed baseline operation has a coordinate-free unit [0, physical mu].

    Subdivide units at actual modeled contract boundaries and allocate harvested
    composition on surviving intervals. Unknown external origin has no temporal
    identity. A synthetic right-holder is labeled only on model-rentable land;
    the baseline operator and right-holder are not otherwise interchangeable.
    """

    def __init__(self, baseline: list[dict]):
        self.base = {r["actor"]: r for r in baseline}
        self.previous = {}
        self.initial = {}
        self._previous_operators: dict[str, list[tuple[float, float, str]]] = {}
        self._previous_planting: dict[str, tuple[float, float]] = {}
        for identity, r in self.base.items():
            grain, cash = r["sow_food"], r["sow_cash"]
            self._previous_operators[identity] = [(0., r["area_e"], identity)]
            if not (grain >= 0 and cash >= 0 and grain + cash > 0):
                portions = ((r["area_e"], "unknown"),)
                if grain == 0. and cash == 0.:
                    self._previous_planting[identity] = (r["area_e"],0.)
            else:
                portions = ((r["area_e"]*grain/(grain+cash), "grain"),
                            (r["area_e"]*cash/(grain+cash), "cash"))
                self._previous_planting[identity] = (r["area_e"], portions[0][0])
            start = 0.
            states = []
            for area, crop in portions:
                if area > EPS:
                    states.append(Slice(start,start+area,identity,crop))
                    start += area
            self.previous[identity] = tuple(states)
            self.initial[identity] = tuple(states)
        self.year = 2010

    def _advance_succession_bounds(
        self, stock: list[tuple[str, float, float, str]],
        planting: dict[str, tuple[float, float]],
    ) -> dict[str, float]:
        """Bound crop overlap using physical assignments and operator totals only."""
        current_operators: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        overlap: dict[str, dict[str | None, list[float]]] = defaultdict(lambda: defaultdict(list))
        for unit, start, end, buyer in stock:
            if unit in self.base:
                current_operators[unit].append((start, end, buyer))
            if buyer not in planting:
                continue
            if unit not in self.base:
                overlap[buyer][None].append(end-start)
                continue
            for lo, hi, seller in self._previous_operators.get(unit, ()):
                area = min(hi, end)-max(lo, start)
                if area > 0.:
                    overlap[buyer][seller].append(area)

        strict_lower, strict_upper, unknown_lower, unknown_upper = [], [], [], []
        unknown_stock, previous_pool_stock = [], []
        for buyer, (area, grain) in planting.items():
            prior_lower, prior_upper, unknown = [], [], []
            for seller, pieces in overlap[buyer].items():
                shared = fsum(pieces)
                prior = self._previous_planting.get(seller) if seller is not None else None
                if prior is None:
                    unknown.append(shared)
                    if seller is not None and seller.startswith("_area_pool_"):
                        previous_pool_stock.append(shared)
                elif seller != buyer:
                    old_area, old_grain = prior
                    # Grain in the changed area can be displaced onto the seller's
                    # other previous land, including land from the transient pool.
                    prior_lower.append(max(0., old_grain-(old_area-shared)))
                    prior_upper.append(min(old_grain, shared))
            unknown_area = fsum(unknown)
            unknown_stock.append(unknown_area)
            strict_lower.append(max(0., grain-(area-fsum(prior_lower))))
            strict_upper.append(min(grain, fsum(prior_upper)))
            unknown_lower.append(max(0., grain-(area-unknown_area)))
            unknown_upper.append(min(grain, unknown_area))

        self._previous_operators = current_operators
        self._previous_planting = planting
        return dict(
            strict_succession_lower_mu=fsum(strict_lower),
            strict_succession_upper_mu=fsum(strict_upper),
            prior_crop_unknown_mu=fsum(unknown_stock),
            prior_external_pool_unknown_mu=fsum(previous_pool_stock),
            grain_on_prior_unknown_lower_mu=fsum(unknown_lower),
            grain_on_prior_unknown_upper_mu=fsum(unknown_upper),
        )

    def advance(self, year: int, actors: list[dict], contracts: list[dict],
                external: list[dict], fiscal_budget: float) -> tuple[list[dict], dict]:
        if year != self.year+1:
            raise ValueError("Missing or repeated simulated year")
        rows = {r["id"]: r for r in actors}
        if len(rows) != len(actors):
            raise ValueError("Duplicate actor account")
        regional = {r["region"]: r for r in external}
        sent, received, rent_paid, rent_received = [defaultdict(float) for _ in range(4)]
        by_unit = defaultdict(list)
        entrants = []
        for d in contracts:
            if not d["start"] <= year <= d["end"]:
                raise ValueError("Contract is outside its active window")
            seller, buyer, area = d["seller"], d["buyer"], d["area"]
            if seller == buyer or area <= 0 or not d["ask"]-EPS <= d["rent"] <= d["bid"]+EPS:
                raise ValueError("Invalid lease quantity, counterparty or rent")
            sent[seller] += area
            received[buyer] += area
            rent_paid[buyer] += area*d["rent"]
            rent_received[seller] += area*d["rent"]
            if seller in rows and rows[seller]["region"] != d["region"]:
                raise ValueError("Seller crosses regional boundary")
            if buyer in rows and rows[buyer]["region"] != d["region"]:
                raise ValueError("Buyer crosses regional boundary")
            if d.get("channel") == "external_supply":
                entrants.append((buyer, area))
            elif d.get("channel") == "external_demand":
                # Static source slice chosen from unsold rights, never from bought land.
                by_unit[seller].append((None,None,buyer,area))
            else:
                if d["plot"] != seller:
                    raise ValueError("Internal benchmark leases require one baseline unit per household")
                equal(d["slice_end"]-d["slice_start"],area,"lease slice")
                by_unit[seller].append((d["slice_start"],d["slice_end"],buyer,area))
        stock = []
        for ident, r in self.base.items():
            limit = r["area_e"]
            rights = r["verified_right"]
            issued = sorted((a,b,k) for a,b,k,_ in by_unit[ident] if a is not None)
            tail = issued[-1][1] if issued else 0.
            for _,_,k,q in (x for x in by_unit[ident] if x[0] is None):
                issued.append((tail,tail+q,k))
                tail += q
            cursor = 0.
            for a,b,k in issued:
                if a < cursor-EPS or b > rights+EPS:
                    raise ValueError("Overlapping slices or lease outside transferable rights")
                if a > cursor+EPS:
                    stock.append((ident,cursor,a,ident))
                stock.append((ident,a,b,k))
                cursor = b
            if limit > cursor+EPS:
                stock.append((ident,cursor,limit,ident))
        for sequence,(buyer,area) in enumerate(entrants):
            stock.append((f"external:{year}:{sequence}",0.,area,buyer))
        goals, measured = {}, defaultdict(float)
        for k,r in rows.items():
            for account, title in ((sent,"rented_out"),(received,"rented_in"),
                                   (rent_paid,"rent_paid"),(rent_received,"rent_received")):
                equal(account[k],r[title],title)
            equal(r["opening"]+r["net"]-r["consumption"],r["closing"],"cash")
            if r["claim"]+EPS < r["paid"] or r["paid"] < -EPS:
                raise ValueError("Invalid fiscal claim or payment")
            if r["cultivated"] > r["operated"]+EPS or r["input_hire_cost"] > r["preplant_funds"]+1e-5:
                raise ValueError("Post-contract production exceeds land or cash")
            if (r["grain"] > EPS) != (r["grain_kg"] > EPS):
                raise ValueError("Harvest is inconsistent with grain sowing")
            # Grain/cash areas are annual sowing mu-times; a physical allocation
            # apportions cultivated stock in proportion rather than claiming plots.
            total = r["grain"]+r["crop"]
            g = r["cultivated"]*r["grain"]/total if total > EPS else 0.
            goals[k] = [g, r["cultivated"]-g]
        for z,r in regional.items():
            k = f"_area_pool_{z}"
            for account,title in ((sent,"rented_out"),(received,"rented_in"),
                                  (rent_paid,"rent_paid"),(rent_received,"rent_received")):
                equal(account[k],r[title],f"external {title}")
            equal(r["opening_area"]-sent[k]+received[k],r["operated"],"external stock")
        paid = sum(r["paid"] for r in rows.values())
        if paid > fiscal_budget+EPS:
            raise ValueError("Fiscal budget exceeded")
        planting = {k: (r["operated"], goals[k][0]) for k, r in rows.items()}
        # Preserve the sorted allocation and its points as legacy diagnostics.
        current = defaultdict(list)
        for unit,a,b,k in sorted(stock,key=lambda v:(v[0].startswith("external:"),v[0],v[1])):
            measured[k] += b-a
            if k not in goals:
                current[unit].append(Slice(a,b,k,"external_unknown"))
                continue
            cursor = a
            for i,crop in enumerate(("grain","cash")):
                q = min(goals[k][i],b-cursor)
                goals[k][i] -= q
                if q > EPS:
                    current[unit].append(Slice(cursor,cursor+q,k,crop))
                cursor += q
            if b > cursor+EPS:
                current[unit].append(Slice(cursor,b,k,"idle"))
        for k,r in rows.items():
            equal(measured[k],r["operated"],"physical operating stock")
            equal(sum(goals[k]),0.,"allocated cultivated stock")
        counts = Counter(dict.fromkeys((
            "contracted_mu","contract_rent_yuan","contracted_not_sown_mu","contracted_non_grain_mu",
            "retained_grain_mu","strict_synthetic_succession_mu","other_grain_recovery_mu",
            "leased_not_sown_mu","leased_non_grain_mu","unidentified_origin_or_use_mu","other_mu",
            "fixed_initial_grain_mu","grain_on_fixed_initial_base_mu","original_operator_on_fixed_initial_grain_mu",
        ),0.))
        for d in contracts:
            counts["contracted_mu"] += d["area"]
            counts["contract_rent_yuan"] += d["area"]*d["rent"]
        internal_rent_mu = counts["contract_rent_yuan"]/counts["contracted_mu"] if counts["contracted_mu"] else None
        counts.update(annual_grain_mu_times=sum(r["grain"] for r in rows.values()),
                      grain_kg=sum(r["grain_kg"] for r in rows.values()),
                      paid_yuan=paid,claimed_yuan=sum(r["claim"] for r in rows.values()),
                      physical_operated_mu=sum(r["operated"] for r in rows.values()),
                      seller_rent_yuan=sum(r["rent_received"] for r in rows.values()),
                      producer_income_yuan=sum(r["net"] for r in rows.values()),
                      owner_offered_unmatched_mu=sum(max(0.,r["offer"]-sent[k]) for k,r in rows.items()))
        records = []
        for unit, parts in current.items():
            prior = self.previous.get(unit, ())
            initial = self.initial.get(unit, ())
            rights = self.base[unit]["verified_right"] if unit in self.base else 0.
            for p in parts:
                boundaries = sorted({p.start,p.end,*(
                    x for r in (*prior,*initial) for x in (r.start,r.end) if p.start+EPS<x<p.end-EPS
                ),*(x for x in (rights,) if p.start+EPS<x<p.end-EPS)})
                for a,b in zip(boundaries,boundaries[1:]):
                    mid = (a+b)/2
                    last = next((s for s in prior if s.start-EPS<=mid<s.end+EPS),None)
                    orig = next((s for s in initial if s.start-EPS<=mid<s.end+EPS),None)
                    status = "other"
                    assigned = not p.operator.startswith("_area_pool_")
                    known = last is not None and last.crop not in ("unknown","external_unknown")
                    if not known or not assigned:
                        status = "unidentified_origin_or_use"
                    elif last.crop == p.crop == "grain":
                        status = "retained_grain" if last.operator == p.operator else "strict_synthetic_succession"
                    elif p.crop == "grain":
                        status = "other_grain_recovery"
                    elif p.crop == "idle" and p.operator != unit:
                        status = "leased_not_sown"
                    elif p.crop == "cash" and p.operator != unit:
                        status = "leased_non_grain"
                    # Outcomes that do not require a known prior crop remain identifiable.
                    if assigned and p.operator != unit:
                        if p.crop == "idle":
                            counts["contracted_not_sown_mu"] += b-a
                        elif p.crop == "cash":
                            counts["contracted_non_grain_mu"] += b-a
                    counts[status+"_mu"] += b-a
                    if orig and orig.crop == "grain":
                        counts["fixed_initial_grain_mu"] += b-a
                        if p.crop == "grain":
                            counts["grain_on_fixed_initial_base_mu"] += b-a
                            if p.operator == orig.operator:
                                counts["original_operator_on_fixed_initial_grain_mu"] += b-a
                    records.append(dict(year=year,unit=unit,lower_mu=a,upper_mu=b,
                        synthetic_right_holder=unit if unit in self.base and mid<rights+EPS else None,
                        assignment=p.operator,lease_signer=p.operator if assigned and p.operator!=unit else
                            (p.operator if not assigned else None),
                        producer=p.operator if p.crop in ("grain","cash") else None,
                        previous_assignment=last.operator if last else None,previous_crop=last.crop if last else None,
                        initial_crop=orig.crop if orig else None,crop=p.crop,status=status))
        counts.update(self._advance_succession_bounds(stock, planting))
        self.previous = {k:tuple(v) for k,v in current.items() if k in self.base}
        self.year = year
        return records,dict(counts)|{"rent_yuan_per_mu_if_traded":internal_rent_mu,"budget_yuan":fiscal_budget}
