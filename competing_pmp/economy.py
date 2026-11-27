"""Annual Regional AHM-PMP Economy with Sequential Shadow-Price Land Succession Market."""

from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from math import fsum, isclose

import numpy as np

from competing_pmp.calibration_2010 import CalibratedRegion, HouseholdAgent, Parcel
from competing_pmp.config import CompetingDesign
from competing_pmp.pmp_solver import marginal_land_bid_per_mu, solve_household_pmp

POLICIES = ("none", "grain_pre", "takeover_pre", "grain_post", "takeover_post")


@dataclass(frozen=True, slots=True)
class LeaseContract:
    parcel_id: str
    lessor: str
    lessee: str
    rent_yuan: float
    bid_per_mu: float
    ask_per_mu: float
    start_year: int
    end_year: int


def verify_annual_ledger(world: "RegionalPMPEconomy", summary: dict, parcel_rows: list[dict], actor_rows: list[dict]) -> None:
    """Strict double-entry conservation audit across parcels, households, entrants, and fiscal accounts."""
    req = lambda a, b: isclose(a, b, rel_tol=1e-10, abs_tol=1e-6)
    assert len(parcel_rows) == len(world.region.parcels)
    assert req(fsum(p["physical_area"] for p in parcel_rows), world.land_mu)
    assert req(fsum(p["planted_physical_mu"] for p in parcel_rows), summary["planted_physical_mu"])
    assert req(fsum(p["idle_physical_mu"] for p in parcel_rows), summary["idle_area"])
    assert req(summary["planted_physical_mu"] + summary["idle_area"], world.land_mu)
    assert req(fsum(p["harvest_grain_kg"] for p in parcel_rows), summary["grain_kg"])
    assert req(fsum(p["support_yuan"] for p in parcel_rows), summary["spending"])
    assert req(fsum(a["support_yuan"] for a in actor_rows), summary["spending"])
    assert 0.0 <= summary["spending"] <= summary["committed"] + 1e-6 <= summary["reservation"] + 2e-6
    assert req(fsum(p["rent_yuan"] for p in parcel_rows), fsum(a["rent_paid"] for a in actor_rows))
    assert req(fsum(p["rent_yuan"] for p in parcel_rows), fsum(a["rent_income"] for a in actor_rows))
    for a in actor_rows:
        assert a["working_cap_end"] >= -1e-6
        assert a["wealth_end"] >= -1e-6
        assert a["net_upfront_outlay"] <= a["working_cap_start"] + 1e-5


class RegionalPMPEconomy:
    """One policy region operating under Non-Separable AHM + Concave PMP + Shadow-Price Bidding."""

    def __init__(self, seed: int, region: CalibratedRegion, design: CompetingDesign):
        self.seed = seed
        self.region = region
        self.design = design
        self.year = 0
        self.land_mu = region.physical_area

        self.actors: dict[str, HouseholdAgent] = {
            a.id: a for a in (*region.households, *region.entrants)
        }
        self.parcels: tuple[Parcel, ...] = tuple(sorted(region.parcels, key=lambda p: p.id))
        self.parcel_map: dict[str, Parcel] = {p.id: p for p in self.parcels}

        self.working_cap: dict[str, float] = {a.id: a.working_capital_init for a in self.actors.values()}
        self.wealth: dict[str, float] = {a.id: a.savings_stock for a in self.actors.values()}
        self.ref_share: dict[str, float] = {a.id: a.ref_grain_share for a in self.actors.values()}
        self.contracts: dict[str, LeaseContract] = {}
        self.last_operator: dict[str, str] = {p.id: p.owner for p in self.parcels}

        init_grain_phys = fsum(p.area for p in self.parcels if p.initial_crop == "grain")
        self.observed: dict[str, float] = {
            "grain_kg": region.initial_grain_kg,
            "grain_area": init_grain_phys,
            "grain_cult_rate": init_grain_phys / max(1.0, self.land_mu),
            "unabsorbed_rate": 0.0,
            "leased_rate": 0.0,
            "tight_liquidity_share": 0.25,
            "spending": 0.0,
        }

    def fork(self) -> "RegionalPMPEconomy":
        other = copy(self)
        other.working_cap = self.working_cap.copy()
        other.wealth = self.wealth.copy()
        other.ref_share = self.ref_share.copy()
        other.contracts = self.contracts.copy()
        other.last_operator = self.last_operator.copy()
        other.observed = self.observed.copy()
        return other

    def announced_rate_per_phys_mu(self, policy: str, budget_cap: float) -> float:
        """Compute announced subsidy per eligible physical grain mu with realistic eligible pool denominator."""
        if policy == "none" or budget_cap <= 0.0:
            return 0.0
        tech = self.region.tech
        if policy.startswith("grain_"):
            # Universal grain base: regional physical land weighted by baseline grain share
            elig_pool = max(1.0, self.land_mu * max(0.55, tech.baseline_grain_share))
        else:
            # Takeover base: regional tradable land weighted by pre-2011 calibrated turnover/lease pool
            elig_pool = max(1.0, self.region.verified_tradable_area * max(0.16, tech.turnover_pool_share))
        return min(self.design.nominal_rate_yuan_per_mu, budget_cap / elig_pool, tech.grain_input_yuan_sown_mu)

    def step(
        self,
        policy: str,
        budget_cap: float,
        *,
        ledger: bool = False,
        village_exchange: bool = False,
    ) -> tuple[dict, list[dict], list[dict]]:
        """Advance one simulation year and return regional metrics (+ optional micro ledgers)."""
        if policy not in POLICIES:
            raise ValueError(f"Unsupported policy: {policy}")
        self.year += 1
        d = self.design
        tech = self.region.tech
        actor_ids = sorted(self.actors)

        # Independent reproducible RNG streams keyed by (seed, year, channel)
        rng_stream = lambda ch: np.random.default_rng([self.seed, self.year, ch])
        wage_shock = dict(zip(actor_ids, rng_stream(11).uniform(0.85, 1.22, len(actor_ids))))
        labor_shock = dict(zip(actor_ids, rng_stream(12).uniform(0.72, 1.08, len(actor_ids))))
        wc_shock = dict(zip(actor_ids, rng_stream(13).uniform(0.82, 1.02, len(actor_ids))))
        search_draw = dict(zip((p.id for p in self.parcels), rng_stream(14).random(len(self.parcels))))
        weather_mult = float(rng_stream(15).choice([0.55, 0.78, 1.0, 1.12], p=[0.06, 0.14, 0.58, 0.22]))

        drift = (1.0 + d.real_offfarm_wage_drift) ** self.year
        opp_wage = {
            aid: self.actors[aid].offfarm_wage_day * drift * wage_shock[aid] for aid in actor_ids
        }
        fam_labor = {
            aid: self.actors[aid].ag_days_year * labor_shock[aid] for aid in actor_ids
        }
        wc_open = {
            aid: max(0.0, self.working_cap[aid] * wc_shock[aid]) for aid in actor_ids
        }
        wealth_open = self.wealth.copy()

        rate_announced = self.announced_rate_per_phys_mu(policy, budget_cap)
        is_pre = policy.endswith("_pre")
        is_univ = policy.startswith("grain_")
        is_takeover = policy.startswith("takeover_")

        # 1. Enforce or default multi-year in-force contracts
        held_contracts: dict[str, LeaseContract] = {}
        defaulted_pids: set[str] = set()
        by_lessee: dict[str, list[LeaseContract]] = defaultdict(list)
        for pid, c in self.contracts.items():
            if c.end_year >= self.year:
                by_lessee[c.lessee].append(c)

        wc_after_held = wc_open.copy()
        rent_paid = defaultdict(float)
        rent_income = defaultdict(float)
        fees_paid = defaultdict(float)

        for aid, clist in by_lessee.items():
            tot_rent = fsum(c.rent_yuan for c in clist)
            if wc_after_held[aid] + 1e-7 < tot_rent:
                for c in clist:
                    defaulted_pids.add(c.parcel_id)
            else:
                wc_after_held[aid] -= tot_rent
                rent_paid[aid] += tot_rent
                for c in clist:
                    held_contracts[c.parcel_id] = c
                    rent_income[c.lessor] += c.rent_yuan

        active_lessees = {c.lessee for c in held_contracts.values()}

        # 2. Pre-market household retention vs. release unified AHM-PMP decision
        own_free_by_hh: dict[str, list[Parcel]] = defaultdict(list)
        held_by_actor: dict[str, list[Parcel]] = defaultdict(list)
        for p in self.parcels:
            if p.id in held_contracts:
                held_by_actor[held_contracts[p.id].lessee].append(p)
            else:
                own_free_by_hh[p.owner].append(p)

        retained_pids: set[str] = set()
        released_pids: set[str] = set()
        planned_wc_reserve: dict[str, float] = defaultdict(float)
        planned_fam_reserve: dict[str, float] = defaultdict(float)

        expected_rent_mu = tech.ask_rent_yuan_phys_mu * (1.0 - d.search_friction_prob)

        for aid in actor_ids:
            a = self.actors[aid]
            if a.is_entrant:
                held_mu = fsum(p.area for p in held_by_actor[aid])
                if held_mu > 0:
                    # Entrant reserves cash & labor for existing held leases
                    sub_mu = rate_announced if (is_univ or is_takeover) else 0.0
                    hp = solve_household_pmp(
                        held_mu, a, tech, d,
                        family_labor_avail=fam_labor[aid],
                        opp_wage_day=opp_wage[aid],
                        working_cap_avail=wc_after_held[aid],
                        pre_sub_per_phys_grain=sub_mu if is_pre else 0.0,
                        post_sub_per_phys_grain=0.0 if is_pre else sub_mu,
                        ref_share=self.ref_share[aid],
                    )
                    planned_wc_reserve[aid] = max(
                        0.0, hp.material_inputs_yuan + d.upfront_hired_cash_share * hp.hired_cost_yuan - hp.pre_subsidy_yuan
                    )
                    planned_fam_reserve[aid] = hp.family_days_used
                continue

            held_mu = fsum(p.area for p in held_by_actor[aid])
            own_trad = [p for p in own_free_by_hh[aid] if p.tradable]
            own_nontrad = [p for p in own_free_by_hh[aid] if not p.tradable]
            trad_mu = fsum(p.area for p in own_trad)
            nontrad_mu = fsum(p.area for p in own_nontrad)

            sub_own = rate_announced if is_univ else 0.0
            sub_held = rate_announced if (is_univ or is_takeover) else 0.0
            full_mu = held_mu + trad_mu + nontrad_mu
            w_sub_full = (sub_own * (trad_mu + nontrad_mu) + sub_held * held_mu) / max(1e-9, full_mu)

            # Option A: Retain all unencumbered own land
            plan_keep_all = solve_household_pmp(
                full_mu, a, tech, d,
                family_labor_avail=fam_labor[aid],
                opp_wage_day=opp_wage[aid],
                working_cap_avail=wc_after_held[aid],
                pre_sub_per_phys_grain=w_sub_full if is_pre else 0.0,
                post_sub_per_phys_grain=0.0 if is_pre else w_sub_full,
                ref_share=self.ref_share[aid],
            )
            val_keep_all = plan_keep_all.net_economic_surplus_yuan

            # Option B: Release all tradable own plots to rental market, cultivate held + nontradable (or exit)
            rem_mu = held_mu + nontrad_mu
            w_sub_rem = (sub_own * nontrad_mu + sub_held * held_mu) / max(1e-9, rem_mu) if rem_mu > 0 else 0.0
            plan_rel_all = solve_household_pmp(
                rem_mu, a, tech, d,
                family_labor_avail=fam_labor[aid],
                opp_wage_day=opp_wage[aid],
                working_cap_avail=wc_after_held[aid],
                pre_sub_per_phys_grain=w_sub_rem if is_pre else 0.0,
                post_sub_per_phys_grain=0.0 if is_pre else w_sub_rem,
                ref_share=self.ref_share[aid],
            )
            val_rel_all = plan_rel_all.net_economic_surplus_yuan + expected_rent_mu * trad_mu

            # Option C: Partial release if household has >=2 tradable plots and family labor < full land demand
            best_keep_trad = own_trad
            best_plan = plan_keep_all
            best_val = val_keep_all

            if trad_mu > 1e-9 and val_rel_all > best_val + 1e-6:
                best_keep_trad = []
                best_plan = plan_rel_all
                best_val = val_rel_all

            if len(own_trad) >= 2:
                half_n = len(own_trad) // 2
                keep_sub = own_trad[:half_n]
                rel_sub = own_trad[half_n:]
                k_mu = held_mu + nontrad_mu + fsum(p.area for p in keep_sub)
                r_mu = fsum(p.area for p in rel_sub)
                w_sub_k = (sub_own * (k_mu - held_mu) + sub_held * held_mu) / max(1e-9, k_mu)
                plan_part = solve_household_pmp(
                    k_mu, a, tech, d,
                    family_labor_avail=fam_labor[aid],
                    opp_wage_day=opp_wage[aid],
                    working_cap_avail=wc_after_held[aid],
                    pre_sub_per_phys_grain=w_sub_k if is_pre else 0.0,
                    post_sub_per_phys_grain=0.0 if is_pre else w_sub_k,
                    ref_share=self.ref_share[aid],
                )
                val_part = plan_part.net_economic_surplus_yuan + expected_rent_mu * r_mu
                if val_part > best_val + 1e-6:
                    best_keep_trad = keep_sub
                    best_plan = plan_part
                    best_val = val_part

            keep_ids = {p.id for p in own_nontrad} | {p.id for p in best_keep_trad}
            for p in own_free_by_hh[aid]:
                if p.id in keep_ids:
                    retained_pids.add(p.id)
                else:
                    released_pids.add(p.id)

            planned_wc_reserve[aid] = max(
                0.0, best_plan.material_inputs_yuan + d.upfront_hired_cash_share * best_plan.hired_cost_yuan - best_plan.pre_subsidy_yuan
            )
            planned_fam_reserve[aid] = best_plan.family_days_used

        # 3. Sequential Shadow-Price Land Succession Market within submarkets
        posted_pids = {pid for pid in released_pids if search_draw[pid] >= d.search_friction_prob}
        listed_by_submkt: dict[int, list[Parcel]] = defaultdict(list)
        for pid in sorted(posted_pids):
            p = self.parcel_map[pid]
            listed_by_submkt[p.submarket].append(p)

        # Track dynamic state of each bidder during sequential market clearing
        cur_area: dict[str, float] = {
            aid: fsum(p.area for p in held_by_actor[aid])
            + fsum(p.area for p in own_free_by_hh[aid] if p.id in retained_pids)
            for aid in actor_ids
        }
        cur_wc = wc_after_held.copy()
        new_contracts: dict[str, LeaseContract] = {}

        releasing_owners = {self.parcel_map[rpid].owner for rpid in released_pids}

        for sm, sm_parcels in sorted(listed_by_submkt.items()):
            # Sort larger parcels first so consolidators match efficiently
            sm_parcels_sorted = sorted(sm_parcels, key=lambda x: (-x.area, x.id))
            bidders = [aid for aid in actor_ids if self.actors[aid].submarket == sm and aid not in releasing_owners]
            if not bidders:
                continue

            # Cache base_plan and (area_rounded, qual_lot) bids per bidder; invalidate only when bidder wins a parcel
            bid_cache: dict[str, dict[tuple[float, bool], float]] = {aid: {} for aid in bidders}
            base_plan_cache: dict[str, object] = {}

            for p in sm_parcels_sorted:
                best_bidder = None
                best_bid_mu = tech.ask_rent_yuan_phys_mu - 1e-9
                second_bid_mu = tech.ask_rent_yuan_phys_mu
                area_key = round(p.area, 5)

                for aid in bidders:
                    if aid == p.owner:
                        continue
                    a = self.actors[aid]
                    qual_lot = is_univ or (is_takeover and self.last_operator.get(p.id) != aid)
                    ckey = (area_key, qual_lot)
                    cached_val = bid_cache[aid].get(ckey)
                    if cached_val is not None:
                        bid_mu = cached_val
                    else:
                        lot_sub = rate_announced if qual_lot else 0.0
                        base_sub = rate_announced if is_univ else (rate_announced if (is_takeover and aid in active_lessees and a.is_entrant) else 0.0)
                        bp = base_plan_cache.get(aid)
                        if bp is None:
                            bp = solve_household_pmp(
                                cur_area[aid], a, tech, d,
                                family_labor_avail=fam_labor[aid],
                                opp_wage_day=opp_wage[aid],
                                working_cap_avail=cur_wc[aid],
                                pre_sub_per_phys_grain=base_sub if is_pre else 0.0,
                                post_sub_per_phys_grain=0.0 if is_pre else base_sub,
                                ref_share=self.ref_share[aid],
                            )
                            base_plan_cache[aid] = bp
                        bid_mu = marginal_land_bid_per_mu(
                            base_area_mu=cur_area[aid],
                            lot_area_mu=p.area,
                            agent=a,
                            tech=tech,
                            design=d,
                            family_labor_avail=fam_labor[aid],
                            opp_wage_day=opp_wage[aid],
                            working_cap_avail=cur_wc[aid],
                            base_pre_sub=base_sub if is_pre else 0.0,
                            base_post_sub=0.0 if is_pre else base_sub,
                            lot_pre_sub=lot_sub if is_pre else 0.0,
                            lot_post_sub=0.0 if is_pre else lot_sub,
                            already_active_lessee=(aid in active_lessees),
                            base_plan=bp,
                        )
                        bid_cache[aid][ckey] = bid_mu

                    if bid_mu > best_bid_mu + 1e-9:
                        if best_bidder is not None:
                            second_bid_mu = max(second_bid_mu, best_bid_mu)
                        best_bid_mu = bid_mu
                        best_bidder = aid
                    elif bid_mu >= tech.ask_rent_yuan_phys_mu:
                        second_bid_mu = max(second_bid_mu, bid_mu)

                if best_bidder is not None and best_bid_mu >= tech.ask_rent_yuan_phys_mu:
                    # Midpoint clearing rent between ask/second-bid anchor and winning shadow-price bid
                    anchor = 0.5 * (tech.ask_rent_yuan_phys_mu + min(best_bid_mu, second_bid_mu))
                    clear_rent_mu = 0.5 * (anchor + best_bid_mu)
                    rent_tot = clear_rent_mu * p.area
                    fee_now = 0.0 if best_bidder in active_lessees else d.lot_transaction_fee_yuan
                    if cur_wc[best_bidder] + 1e-7 >= rent_tot + fee_now:
                        cur_wc[best_bidder] -= (rent_tot + fee_now)
                        cur_area[best_bidder] += p.area
                        active_lessees.add(best_bidder)
                        bid_cache[best_bidder].clear()
                        base_plan_cache.pop(best_bidder, None)
                        rent_paid[best_bidder] += rent_tot
                        rent_income[p.owner] += rent_tot
                        fees_paid[best_bidder] += fee_now
                        new_contracts[p.id] = LeaseContract(
                            parcel_id=p.id,
                            lessor=p.owner,
                            lessee=best_bidder,
                            rent_yuan=rent_tot,
                            bid_per_mu=best_bid_mu,
                            ask_per_mu=tech.ask_rent_yuan_phys_mu,
                            start_year=self.year,
                            end_year=self.year + d.contract_years - 1,
                        )

        all_active_contracts = {**held_contracts, **new_contracts}
        self.contracts = all_active_contracts

        # 4. Assign parcels to post-market operators; unmatched released parcels return to owner for fallback check
        assigned_parcels: dict[str, list[Parcel]] = defaultdict(list)
        unmatched_pids = released_pids - set(new_contracts)
        for p in self.parcels:
            if p.id in all_active_contracts:
                assigned_parcels[all_active_contracts[p.id].lessee].append(p)
            elif p.id in retained_pids:
                assigned_parcels[p.owner].append(p)
            else:
                # Unmatched released parcel: owner can cultivate if zero-rent cultivation is profitable
                assigned_parcels[p.owner].append(p)

        # Optional pre-2011 calibrated village net inflow land for 2011-2015 cohort historical check ONLY
        ext_inflow_mu_by_hh: dict[str, float] = defaultdict(float)
        if village_exchange and d.enable_historical_village_pool:
            pool_cap = tech.village_net_inflow_rate * self.land_mu * min(2.2, 0.75 + 0.30 * self.year)
            # Allocate external village rental supply to high-margin cohort expanders who did not release land
            expanders = [
                h.id for h in self.region.households
                if not any(op.id in released_pids for op in own_free_by_hh[h.id])
                and cur_wc[h.id] > tech.ask_rent_yuan_phys_mu * 3.0
            ]
            if expanders and pool_cap > 0:
                per_exp = min(8.0, pool_cap / len(expanders))
                for hid in expanders:
                    a = self.actors[hid]
                    bid_ext = marginal_land_bid_per_mu(
                        base_area_mu=fsum(p.area for p in assigned_parcels[hid]),
                        lot_area_mu=per_exp,
                        agent=a,
                        tech=tech,
                        design=d,
                        family_labor_avail=fam_labor[hid],
                        opp_wage_day=opp_wage[hid],
                        working_cap_avail=cur_wc[hid],
                        base_pre_sub=rate_announced if (is_univ and is_pre) else 0.0,
                        base_post_sub=rate_announced if (is_univ and not is_pre) else 0.0,
                        lot_pre_sub=rate_announced if ((is_univ or is_takeover) and is_pre) else 0.0,
                        lot_post_sub=rate_announced if ((is_univ or is_takeover) and not is_pre) else 0.0,
                        already_active_lessee=(hid in active_lessees),
                    )
                    if bid_ext >= tech.ask_rent_yuan_phys_mu and pool_cap > 1e-6:
                        take_mu = min(per_exp, pool_cap)
                        ext_rent = tech.ask_rent_yuan_phys_mu * take_mu
                        if cur_wc[hid] >= ext_rent + 100.0:
                            cur_wc[hid] -= ext_rent
                            ext_inflow_mu_by_hh[hid] = take_mu
                            pool_cap -= take_mu

        # 5. Execute post-market household & entrant PMP production and calculate raw subsidy claims
        actor_plans = {}
        elig_area_share: dict[str, float] = {}
        raw_claim_by_parcel: dict[str, float] = defaultdict(float)
        parcel_plan_alloc: dict[str, tuple[str, float, float, float, float, bool]] = {}

        for aid in actor_ids:
            plist = assigned_parcels[aid]
            p_area = fsum(p.area for p in plist) + ext_inflow_mu_by_hh[aid]
            if p_area <= 1e-9:
                actor_plans[aid] = solve_household_pmp(
                    0.0, self.actors[aid], tech, d,
                    family_labor_avail=fam_labor[aid],
                    opp_wage_day=opp_wage[aid],
                    working_cap_avail=cur_wc[aid],
                )
                elig_area_share[aid] = 0.0
                continue

            # Determine share of operated physical area eligible for current policy
            elig_mu = 0.0
            for p in plist:
                is_qual = is_univ or (is_takeover and (p.id in all_active_contracts or self.last_operator.get(p.id) != aid))
                if is_qual:
                    elig_mu += p.area
            if is_univ or is_takeover:
                elig_mu += ext_inflow_mu_by_hh[aid]
            e_frac = elig_mu / p_area
            elig_area_share[aid] = e_frac

            eff_sub = rate_announced * e_frac
            plan = solve_household_pmp(
                p_area,
                self.actors[aid],
                tech,
                d,
                family_labor_avail=fam_labor[aid],
                opp_wage_day=opp_wage[aid],
                working_cap_avail=cur_wc[aid],
                pre_sub_per_phys_grain=eff_sub if is_pre else 0.0,
                post_sub_per_phys_grain=0.0 if is_pre else eff_sub,
                ref_share=self.ref_share[aid],
            )
            actor_plans[aid] = plan

            util_ratio = plan.planted_physical_mu / p_area if plan.cultivating and p_area > 0 else 0.0
            g_sh = plan.grain_share if plan.cultivating else 0.0
            mci_a = self.actors[aid].mci

            for p in plist:
                p_planted = p.area * util_ratio
                p_idle = p.area - p_planted
                p_sown_g = p_planted * mci_a * g_sh
                p_sown_c = p_planted * mci_a * (1.0 - g_sh)
                is_qual = is_univ or (is_takeover and (p.id in all_active_contracts or self.last_operator.get(p.id) != aid))
                if is_qual and p_planted * g_sh > 1e-9 and policy != "none":
                    raw_claim_by_parcel[p.id] = rate_announced * (p_planted * g_sh)
                parcel_plan_alloc[p.id] = (aid, p_planted, p_idle, p_sown_g, p_sown_c, is_qual)

        # 6. Fiscal pro-rata hard-cap settlement (guarantees 0 <= spending <= committed <= budget_cap)
        total_raw_claim = fsum(raw_claim_by_parcel.values())
        if total_raw_claim <= 1e-9 or budget_cap <= 0.0 or policy == "none":
            scale_factor = 0.0
            committed = 0.0
        elif total_raw_claim <= budget_cap:
            scale_factor = 1.0
            committed = total_raw_claim
        else:
            scale_factor = budget_cap / total_raw_claim
            committed = budget_cap

        # If pre-planting subsidy was scaled down by pro-rata cap, ensure no actor violates upfront cash
        support_by_actor: dict[str, float] = defaultdict(float)
        support_by_parcel: dict[str, float] = {}
        for p in self.parcels:
            sup = raw_claim_by_parcel.get(p.id, 0.0) * scale_factor
            support_by_parcel[p.id] = sup
            op_id = parcel_plan_alloc[p.id][0]
            support_by_actor[op_id] += sup

        # Re-solve any actor whose scaled pre-subsidy caused net upfront cost to slightly exceed cur_wc
        for aid in actor_ids:
            plan = actor_plans[aid]
            if not plan.cultivating:
                continue
            actual_pre = support_by_actor[aid] if is_pre else 0.0
            if plan.material_inputs_yuan + d.upfront_hired_cash_share * plan.hired_cost_yuan - actual_pre > cur_wc[aid] + 1e-6:
                orig_actor_cap = actual_pre
                plist = assigned_parcels[aid]
                p_area = fsum(p.area for p in plist) + ext_inflow_mu_by_hh[aid]
                rate_scaled_eff = rate_announced * scale_factor * elig_area_share[aid]

                # Step A: Solve under exact pro-rata scaled marginal rate
                plan_a = solve_household_pmp(
                    p_area, self.actors[aid], tech, d,
                    family_labor_avail=fam_labor[aid],
                    opp_wage_day=opp_wage[aid],
                    working_cap_avail=cur_wc[aid],
                    pre_sub_per_phys_grain=rate_scaled_eff,
                    post_sub_per_phys_grain=0.0,
                    ref_share=self.ref_share[aid],
                )
                plan = plan_a
                for _iter in range(3):
                    util_ratio = plan.planted_physical_mu / p_area if plan.cultivating and p_area > 0 else 0.0
                    g_sh = plan.grain_share if plan.cultivating else 0.0
                    mci_a = self.actors[aid].mci
                    new_parcel_sups: dict[str, float] = {}
                    for p in plist:
                        p_planted = p.area * util_ratio
                        p_idle = p.area - p_planted
                        p_sown_g = p_planted * mci_a * g_sh
                        p_sown_c = p_planted * mci_a * (1.0 - g_sh)
                        is_qual = parcel_plan_alloc[p.id][5]
                        sup = (rate_announced * scale_factor * p_planted * g_sh) if (is_qual and policy != "none") else 0.0
                        new_parcel_sups[p.id] = sup
                        parcel_plan_alloc[p.id] = (aid, p_planted, p_idle, p_sown_g, p_sown_c, is_qual)
                    new_tot_aid = fsum(new_parcel_sups.values())
                    aid_cap_scale = (orig_actor_cap / new_tot_aid) if new_tot_aid > orig_actor_cap and new_tot_aid > 1e-12 else 1.0
                    realized_pre = new_tot_aid * aid_cap_scale
                    if plan.material_inputs_yuan + d.upfront_hired_cash_share * plan.hired_cost_yuan - realized_pre <= cur_wc[aid] + 1e-6:
                        break
                    orig_actor_cap = min(orig_actor_cap, realized_pre)
                    plan = solve_household_pmp(
                        p_area, self.actors[aid], tech, d,
                        family_labor_avail=fam_labor[aid],
                        opp_wage_day=opp_wage[aid],
                        working_cap_avail=cur_wc[aid] + orig_actor_cap,
                        pre_sub_per_phys_grain=0.0,
                        post_sub_per_phys_grain=0.0,
                        ref_share=self.ref_share[aid],
                    )

                actor_plans[aid] = plan
                support_by_actor[aid] = 0.0
                for pid_sub, sup_val in new_parcel_sups.items():
                    capped_sup = sup_val * aid_cap_scale
                    support_by_parcel[pid_sub] = capped_sup
                    support_by_actor[aid] += capped_sup

        spending = fsum(support_by_parcel.values())
        if spending > budget_cap + 1e-9:
            cap_fix = budget_cap / spending
            for pid_sub in support_by_parcel:
                support_by_parcel[pid_sub] *= cap_fix
            support_by_actor.clear()
            for p in self.parcels:
                support_by_actor[parcel_plan_alloc[p.id][0]] += support_by_parcel[p.id]
            spending = fsum(support_by_parcel.values())
        committed = min(budget_cap, max(committed, spending))

        # 7. Realize weather-shocked harvest, settle cash/working-capital/wealth, update reference shares
        parcel_rows: list[dict] = []
        actor_rows: list[dict] = []
        grain_by_actor: dict[str, float] = defaultdict(float)
        rev_by_actor: dict[str, float] = defaultdict(float)

        retained_grain_kg = 0.0
        successor_grain_kg = 0.0
        entrant_grain_kg = 0.0
        continuity_area = 0.0
        leased_idle_mu = 0.0

        for p in self.parcels:
            op_id, p_planted, p_idle, p_sown_g, p_sown_c, is_qual = parcel_plan_alloc[p.id]
            op_agent = self.actors[op_id]
            q_eff = 0.5 * (p.quality + op_agent.grain_yield_factor)
            g_kg = tech.grain_yield_kg_sown_mu * q_eff * weather_mult * p_sown_g
            g_rev = g_kg * tech.grain_price_yuan_kg
            c_rev = tech.cash_rev_yuan_sown_mu * (0.35 + 0.65 * weather_mult) * p_sown_c

            grain_by_actor[op_id] += g_kg
            rev_by_actor[op_id] += g_rev + c_rev

            is_leased = p.id in all_active_contracts
            if op_id == p.owner:
                retained_grain_kg += g_kg
            else:
                successor_grain_kg += g_kg
                if op_agent.is_entrant:
                    entrant_grain_kg += g_kg
            if p_sown_g > 1e-9 and self.last_operator.get(p.id) != op_id:
                continuity_area += p_planted * (p_sown_g / max(1e-9, p_sown_g + p_sown_c))
            if is_leased and p_idle > 1e-9:
                leased_idle_mu += p_idle

            if p_planted > 1e-9:
                self.last_operator[p.id] = op_id

            c_obj = all_active_contracts.get(p.id)
            parcel_rows.append({
                "parcel": p.id,
                "owner": p.owner,
                "operator": op_id,
                "is_entrant_operator": op_agent.is_entrant,
                "leased": is_leased,
                "released": p.id in released_pids,
                "posted": p.id in posted_pids,
                "defaulted": p.id in defaulted_pids,
                "eligible": is_qual,
                "physical_area": p.area,
                "planted_physical_mu": p_planted,
                "idle_physical_mu": p_idle,
                "sown_grain_mu": p_sown_g,
                "sown_cash_mu": p_sown_c,
                "harvest_grain_kg": g_kg,
                "rent_yuan": c_obj.rent_yuan if c_obj else 0.0,
                "bid_per_mu": c_obj.bid_per_mu if c_obj else 0.0,
                "support_yuan": support_by_parcel[p.id],
            })

        tight_count = 0
        hh_count = 0
        for aid in actor_ids:
            a = self.actors[aid]
            plan = actor_plans[aid]
            sup = support_by_actor[aid]
            pre_sup = sup if is_pre else 0.0
            post_sup = 0.0 if is_pre else sup
            net_upfront = max(0.0, plan.material_inputs_yuan + d.upfront_hired_cash_share * plan.hired_cost_yuan - pre_sup)

            # Check if working capital was >80% utilized
            if not a.is_entrant:
                hh_count += 1
                if net_upfront >= 0.80 * max(1.0, cur_wc[aid]):
                    tight_count += 1

            # Extra external village land output in historical cohort check (not counted in closed 5277.5-mu parcel ledger)
            ext_mu = ext_inflow_mu_by_hh[aid]
            ext_sown_g = 0.0
            ext_sown_c = 0.0
            ext_g_kg = 0.0
            if ext_mu > 0 and plan.cultivating:
                p_tot = fsum(p.area for p in assigned_parcels[aid]) + ext_mu
                u_rat = plan.planted_physical_mu / p_tot
                ext_sown_g = ext_mu * u_rat * a.mci * plan.grain_share
                ext_sown_c = ext_mu * u_rat * a.mci * (1.0 - plan.grain_share)
                ext_g_kg = tech.grain_yield_kg_sown_mu * a.grain_yield_factor * weather_mult * ext_sown_g

            sales_rev = rev_by_actor[aid] + ext_g_kg * tech.grain_price_yuan_kg + tech.cash_rev_yuan_sown_mu * weather_mult * ext_sown_c
            ag_net_cash = (
                sales_rev + pre_sup + post_sup
                - plan.material_inputs_yuan - plan.hired_cost_yuan
                - rent_paid[aid] - fees_paid[aid] + rent_income[aid]
            )

            # Non-farm income increases if family agricultural days drop below baseline ag_days_year
            freed_days = max(0.0, a.ag_days_year - plan.family_days_used)
            nonfarm_inc = a.offfarm_annual_base * drift * wage_shock[aid] + 0.55 * opp_wage[aid] * freed_days
            cons = a.consumption_base * (1.0 + 0.03 * self.year)

            # Update revolving working capital (principal recovery from harvest inflow + retained net operating surplus)
            upfront_principal_total = net_upfront + rent_paid[aid] + fees_paid[aid]
            harvest_inflow_net_autumn = max(
                0.0,
                sales_rev + post_sup - (1.0 - d.upfront_hired_cash_share) * plan.hired_cost_yuan + 0.5 * rent_income[aid],
            )
            recovered_principal = min(upfront_principal_total, harvest_inflow_net_autumn)
            net_operating_surplus = max(0.0, harvest_inflow_net_autumn - upfront_principal_total)
            wc_rem_after_sow = max(0.0, cur_wc[aid] - net_upfront)
            wc_end = max(
                500.0 if a.is_entrant else d.working_capital_floor_yuan * 0.75,
                wc_rem_after_sow + recovered_principal + d.surplus_retention_to_working_cap * net_operating_surplus,
            )
            wc_cap_limit = max(4500.0, 2.8 * (plan.material_inputs_yuan + plan.hired_cost_yuan + rent_paid[aid]) + 3000.0)
            wc_end = min(wc_end, wc_cap_limit)

            wealth_end = max(0.0, wealth_open[aid] + ag_net_cash + nonfarm_inc - cons)
            self.working_cap[aid] = wc_end
            self.wealth[aid] = wealth_end

            if plan.cultivating and (plan.sown_grain_mu + plan.sown_cash_mu) > 1e-6:
                self.ref_share[aid] = (
                    (1.0 - d.reference_share_adaptation) * self.ref_share[aid]
                    + d.reference_share_adaptation * plan.grain_share
                )

            p_list_a = assigned_parcels[aid]
            phys_op_closed = fsum(parcel_plan_alloc[p.id][1] + parcel_plan_alloc[p.id][2] for p in p_list_a)
            sown_g_closed = fsum(parcel_plan_alloc[p.id][3] for p in p_list_a)
            sown_c_closed = fsum(parcel_plan_alloc[p.id][4] for p in p_list_a)

            actor_rows.append({
                "id": aid,
                "is_entrant": a.is_entrant,
                "cultivating": plan.cultivating,
                "operated_physical_mu": phys_op_closed + ext_mu,
                "closed_operated_physical_mu": phys_op_closed,
                "planted_physical_mu": plan.planted_physical_mu,
                "sown_grain_mu": sown_g_closed + ext_sown_g,
                "sown_cash_mu": sown_c_closed + ext_sown_c,
                "grain_kg": grain_by_actor[aid] + ext_g_kg,
                "family_days": plan.family_days_used,
                "hired_days": plan.hired_days_used,
                "hired_cost": plan.hired_cost_yuan,
                "material_inputs": plan.material_inputs_yuan,
                "rent_paid": rent_paid[aid],
                "rent_income": rent_income[aid],
                "support_yuan": sup,
                "ag_net_cash": ag_net_cash,
                "nonfarm_income": nonfarm_inc,
                "working_cap_start": cur_wc[aid],
                "net_upfront_outlay": net_upfront,
                "working_cap_end": wc_end,
                "wealth_end": wealth_end,
            })

        # 8. Aggregate regional metrics (strictly on the conserved 5,277.5-mu parcel ledger!)
        tot_grain_kg = fsum(r["harvest_grain_kg"] for r in parcel_rows)
        planted_phys = fsum(r["planted_physical_mu"] for r in parcel_rows)
        idle_phys = fsum(r["idle_physical_mu"] for r in parcel_rows)
        sown_g_tot = fsum(r["sown_grain_mu"] for r in parcel_rows)
        sown_c_tot = fsum(r["sown_cash_mu"] for r in parcel_rows)
        grain_phys_equiv = fsum(
            r["planted_physical_mu"] * (r["sown_grain_mu"] / max(1e-9, r["sown_grain_mu"] + r["sown_cash_mu"]))
            for r in parcel_rows if (r["sown_grain_mu"] + r["sown_cash_mu"]) > 1e-9
        )
        cash_phys_equiv = max(0.0, planted_phys - grain_phys_equiv)

        offered_mu = fsum(self.parcel_map[pid].area for pid in released_pids)
        posted_mu = fsum(self.parcel_map[pid].area for pid in posted_pids)
        matched_mu = fsum(self.parcel_map[pid].area for pid in new_contracts)
        unfilled_mu = max(0.0, offered_mu - matched_mu)
        leased_mu = fsum(self.parcel_map[pid].area for pid in all_active_contracts)
        held_mu_tot = fsum(self.parcel_map[pid].area for pid in held_contracts)
        default_mu_tot = fsum(self.parcel_map[pid].area for pid in defaulted_pids)
        rent_tot = fsum(c.rent_yuan for c in all_active_contracts.values())
        avg_rent_mu = rent_tot / leased_mu if leased_mu > 1e-9 else 0.0

        hh_only = [r for r in actor_rows if not r["is_entrant"]]
        cult_hh_share = sum(1 for r in hh_only if r["cultivating"]) / max(1, len(hh_only))

        summary = {
            "policy": policy,
            "reservation": budget_cap,
            "rate_announced_mu": rate_announced,
            "effective_rate_scale": scale_factor,
            "committed": committed,
            "spending": spending,
            "grain_kg": tot_grain_kg,
            "grain_area": grain_phys_equiv,
            "cash_crop_area": cash_phys_equiv,
            "planted_physical_mu": planted_phys,
            "idle_area": idle_phys,
            "sown_grain_mu_seasons": sown_g_tot,
            "sown_cash_mu_seasons": sown_c_tot,
            "retained_grain": retained_grain_kg,
            "successor_grain": successor_grain_kg,
            "entrant_grain": entrant_grain_kg,
            "continuity_area": continuity_area,
            "offered_area": offered_mu,
            "posted_area": posted_mu,
            "matched_area": matched_mu,
            "unfilled_area": unfilled_mu,
            "leased_area": leased_mu,
            "held_area": held_mu_tot,
            "default_area": default_mu_tot,
            "leased_idle": leased_idle_mu,
            "rent_total": rent_tot,
            "avg_rent_per_leased_mu": avg_rent_mu,
            "hired_days_total": fsum(r["hired_days"] for r in actor_rows),
            "hired_cost_total": fsum(r["hired_cost"] for r in actor_rows),
            "smallholder_cultivating_share": cult_hh_share,
            "owner_income": fsum(r["ag_net_cash"] + r["nonfarm_income"] for r in hh_only),
            "owner_ag_cash": fsum(r["ag_net_cash"] for r in hh_only),
            "owner_rent_income": fsum(r["rent_income"] for r in hh_only),
            "owner_rent_paid": fsum(r["rent_paid"] for r in hh_only),
            "owner_support": fsum(r["support_yuan"] for r in hh_only),
            "weather_multiplier": weather_mult,
        }

        # Update public structural state (weather-normalized!) for two-level Government Planner
        self.observed = {
            "grain_kg": tot_grain_kg,
            "grain_area": grain_phys_equiv,
            "grain_cult_rate": grain_phys_equiv / max(1.0, self.land_mu),
            "unabsorbed_rate": (unfilled_mu + idle_phys) / max(1.0, self.land_mu),
            "leased_rate": leased_mu / max(1.0, self.region.verified_tradable_area),
            "tight_liquidity_share": tight_count / max(1, hh_count),
            "spending": spending,
        }

        if ledger:
            verify_annual_ledger(self, summary, parcel_rows, actor_rows)
            return summary, parcel_rows, actor_rows
        return summary, [], []
