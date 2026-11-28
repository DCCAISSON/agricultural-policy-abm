"""Unified Native Mesa 2.3.2 Model (`AgriSuccessionMesaModel`) for Four-Region Agricultural Economy & RL.

All batch experiments, RL training/evaluation, counterfactual fixed-contract forks, and interactive
browser visualizations run through this single `mesa.Model` implementation.
"""

from collections import defaultdict
from dataclasses import dataclass
from math import floor, fsum, isclose
import mesa
import numpy as np

from mesa_agri_abm.agents import (
    GovernmentMesaAgent,
    HouseholdMesaAgent,
    OperatorMesaAgent,
    ScaleOrgMesaAgent,
    build_operator_agent,
)
from mesa_agri_abm.calibration import CalibratedRegionV2, NAMED_GRAINS, ParcelBlueprint
from mesa_agri_abm.config import V2Design
from mesa_agri_abm.pmp_production import (
    PMPPlanV2,
    SubsidySchedule,
    eligible_subsidized_grain_phys_mu,
)

LocalAllocV2 = tuple[str, int]
JointActionV2 = tuple[LocalAllocV2, ...]
OFF_LOCAL: LocalAllocV2 = ("none", 0)


@dataclass(frozen=True, slots=True)
class LeaseContractV2:
    parcel_id: str
    offer_group_id: str
    lessor: str
    lessee: str
    rent_yuan: float
    rent_per_mu: float
    winning_bid_per_mu: float
    runner_up_bid_per_mu: float
    ask_per_mu: float
    start_year: int
    end_year: int


@dataclass(frozen=True, slots=True)
class FixedContractSnapshot:
    """Locks P0 contracts, assigned operators, and rent transfers for Experiment 1 Fixed-Contract fork."""

    year: int
    region_name: str
    assigned_operator: dict[str, str]
    active_contracts: dict[str, LeaseContractV2]
    new_takeover_pids: frozenset[str]
    rent_paid: dict[str, float]
    rent_income: dict[str, float]
    fees_paid: dict[str, float]
    wc_after_market: dict[str, float]
    leased_pids: frozenset[str]
    released_pids: frozenset[str]
    unmatched_pids: frozenset[str]
    recovered_pids: frozenset[str]


def build_policy_menu(regions: tuple[CalibratedRegionV2, ...], design: V2Design) -> tuple[JointActionV2, ...]:
    """Construct the 25-action shared-budget fiscal menu across 4 policy regions:
    - Action 0: P0 (all zero incremental spending)
    - Actions 1..5: National area-proportional allocation (producer_pre, producer_post,
                    scale_takeover_pre, scale_takeover_post, service_trusteeship)
    - Actions 6..21: Region-priority tilt (4 regions x 4 key instruments: 55% to priority region
                     with spillover + 45% foundational support across other 3 regions)
    - Actions 22..24: Institutional regionally tailored mixes (e.g., Hilly Service + Grain-Plains Scale)
    """
    n_reg = len(regions)
    areas = [r.physical_area for r in regions]
    tot_area = fsum(areas)
    budget = design.annual_common_budget_yuan

    def split_by_area(amount: int, idxs: tuple[int, ...]) -> dict[int, int]:
        if amount <= 0 or not idxs:
            return {i: 0 for i in idxs}
        sub_tot = fsum(areas[i] for i in idxs)
        raw = [amount * (areas[i] / sub_tot) for i in idxs]
        fl = [floor(x) for x in raw]
        rem = amount - sum(fl)
        order = sorted(range(len(idxs)), key=lambda k: (-(raw[k] - fl[k]), idxs[k]))
        for k in order[:rem]:
            fl[k] += 1
        return {idxs[k]: fl[k] for k in range(len(idxs))}

    def max_useful_for(r_idx: int, inst: str) -> int:
        reg = regions[r_idx]
        if inst.startswith("producer_"):
            return floor(design.producer_cap_yuan_per_mu * reg.physical_area)
        if inst.startswith("scale_takeover_"):
            pool = max(reg.verified_tradable_area * reg.tech.scale_pool_share, 0.30 * reg.physical_area)
            return floor(design.scale_takeover_cap_yuan_per_mu * pool)
        if inst == "service_trusteeship":
            return floor(design.service_cap_yuan_per_mu * reg.physical_area)
        if inst == "order_farming":
            return floor(design.order_cap_yuan_per_mu * reg.physical_area)
        return 0

    actions: list[JointActionV2] = [(OFF_LOCAL,) * n_reg]

    # Actions 1..5: Uniform instrument across all 4 regions, area-proportional caps
    prop_all = split_by_area(budget, tuple(range(n_reg)))
    for inst in ("producer_pre", "producer_post", "scale_takeover_pre", "scale_takeover_post", "service_trusteeship"):
        actions.append(tuple((inst, prop_all[i]) if prop_all[i] > 0 else OFF_LOCAL for i in range(n_reg)))

    # Actions 6..21: Region-priority tilt (4 regions x 4 instruments)
    focus_insts = ("producer_pre", "scale_takeover_pre", "service_trusteeship", "producer_post")
    for r_idx in range(n_reg):
        for inst in focus_insts:
            if budget <= 0:
                actions.append((OFF_LOCAL,) * n_reg)
                continue
            target_raw = floor(0.50 * budget)
            cap_max = max_useful_for(r_idx, inst)
            primary = min(target_raw, cap_max) if design.overflow_reallocation else target_raw
            rest_budget = budget - primary
            others = tuple(j for j in range(n_reg) if j != r_idx)
            spill = split_by_area(rest_budget, others)
            base_inst = "service_trusteeship" if inst == "service_trusteeship" else "producer_pre"
            row = tuple(
                (inst, primary) if j == r_idx else ((base_inst, spill[j]) if spill[j] > 0 else OFF_LOCAL)
                for j in range(n_reg)
            )
            actions.append(row)

    # Actions 22..24: Regionally tailored institutional mixes (East=0, Central=1, West=2, Northeast=3)
    tailored_specs = (
        ("service_trusteeship", "scale_takeover_pre", "service_trusteeship", "scale_takeover_pre"),  # 22: Hilly service + Plains scale
        ("producer_pre", "producer_pre", "service_trusteeship", "scale_takeover_pre"),               # 23: Producer + West service + NE scale
        ("producer_pre", "scale_takeover_pre", "producer_pre", "producer_pre"),                      # 24: Central scale + Producer safety net
    )
    for spec in tailored_specs:
        actions.append(tuple((spec[i], prop_all[i]) if prop_all[i] > 0 else OFF_LOCAL for i in range(n_reg)))

    if design.menu_mode == "v2_25":
        assert len(actions) == 25
    elif design.menu_mode == "v2_2_clean":
        # Actions 25..29: Cross-instrument regional mixes + Order Grain (zero heuristic verification suffixes)
        if budget <= 0:
            for _ in range(5):
                actions.append((OFF_LOCAL,) * n_reg)
        else:
            e_50 = min(floor(0.50 * budget), max_useful_for(0, "producer_post"))
            sp_50 = split_by_area(budget - e_50, (1, 2, 3))
            # 25: East tilt producer_post + Central/West/NE service_trusteeship
            actions.append((("producer_post", e_50), ("service_trusteeship", sp_50[1]), ("service_trusteeship", sp_50[2]), ("service_trusteeship", sp_50[3])))
            # 26: East tilt producer_pre + Central/West/NE service_trusteeship
            actions.append((("producer_pre", e_50), ("service_trusteeship", sp_50[1]), ("service_trusteeship", sp_50[2]), ("service_trusteeship", sp_50[3])))
            # 27: Area-proportional East producer_post + Central/West/NE service_trusteeship
            actions.append((("producer_post", prop_all[0]), ("service_trusteeship", prop_all[1]), ("service_trusteeship", prop_all[2]), ("service_trusteeship", prop_all[3])))
            # 28: National uniform order_farming
            actions.append(tuple(("order_farming", prop_all[i]) if prop_all[i] > 0 else OFF_LOCAL for i in range(n_reg)))
            # 29: Coastal deficit order_farming (East) + inland/northern service_trusteeship (Central/West/NE)
            actions.append((("order_farming", prop_all[0]), ("service_trusteeship", prop_all[1]), ("service_trusteeship", prop_all[2]), ("service_trusteeship", prop_all[3])))
        assert len(actions) == 30
    else:
        raise ValueError(f"Unsupported menu_mode: {design.menu_mode}")

    for act in actions:
        tot_c = sum(c for _, c in act)
        assert tot_c in (0, budget), f"Menu budget sum mismatch: {tot_c} vs {budget}"
    return tuple(actions)


def symmetric_rent_decomposition(
    area_p: float,
    rent_p: float,
    area_0: float,
    rent_0: float,
    parcel_rents_p: dict[str, tuple[float, float]],
    parcel_rents_0: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Exact symmetric price-area rent decomposition + common-leased parcel intersection differential."""
    total_diff = rent_p - rent_0
    if area_p > 1e-9 and area_0 > 1e-9:
        r_p = rent_p / area_p
        r_0 = rent_0 / area_0
        price_effect = 0.5 * (area_p + area_0) * (r_p - r_0)
        area_effect = 0.5 * (r_p + r_0) * (area_p - area_0)
        both_positive = 1.0
    else:
        r_p = rent_p / area_p if area_p > 1e-9 else 0.0
        r_0 = rent_0 / area_0 if area_0 > 1e-9 else 0.0
        price_effect = 0.0
        area_effect = total_diff
        both_positive = 0.0

    common_pids = set(parcel_rents_p).intersection(parcel_rents_0)
    common_area = fsum(parcel_rents_p[pid][0] for pid in common_pids)
    if common_area > 1e-9:
        common_rent_diff = fsum(parcel_rents_p[pid][1] - parcel_rents_0[pid][1] for pid in common_pids)
        common_unit_diff = common_rent_diff / common_area
    else:
        common_rent_diff = 0.0
        common_unit_diff = 0.0

    return {
        "total_rent_diff_yuan": total_diff,
        "unit_rent_p_yuan_mu": r_p,
        "unit_rent_0_yuan_mu": r_0,
        "price_effect_yuan": price_effect,
        "area_effect_yuan": area_effect,
        "both_sides_leased_flag": both_positive,
        "common_leased_mu": common_area,
        "common_rent_diff_yuan": common_rent_diff,
        "common_unit_rent_diff_yuan_mu": common_unit_diff,
    }


class RegionalSubEconomy:
    """Regional subsystem hosted inside `AgriSuccessionMesaModel`."""

    def __init__(self, model: "AgriSuccessionMesaModel", reg_seed: int, cal: CalibratedRegionV2):
        self.model = model
        self.seed = reg_seed
        self.cal = cal
        self.name = cal.name
        self.tech = cal.tech
        self.land_mu = cal.physical_area

        self.parcels: tuple[ParcelBlueprint, ...] = tuple(sorted(cal.parcels, key=lambda p: p.id))
        self.parcel_map: dict[str, ParcelBlueprint] = {p.id: p for p in self.parcels}

        self.households: dict[str, HouseholdMesaAgent] = {}
        self.organizations: dict[str, ScaleOrgMesaAgent] = {}
        self.operators: dict[str, OperatorMesaAgent] = {}

        for bp in cal.households:
            ag = build_operator_agent(bp.id, model, bp, cal.tech)
            assert isinstance(ag, HouseholdMesaAgent)
            self.households[bp.id] = ag
            self.operators[bp.id] = ag

        for bp in cal.organizations:
            ag = build_operator_agent(bp.id, model, bp, cal.tech)
            assert isinstance(ag, ScaleOrgMesaAgent)
            self.organizations[bp.id] = ag
            self.operators[bp.id] = ag

        self.contracts: dict[str, LeaseContractV2] = {}
        self.last_operator: dict[str, str] = {p.id: p.owner for p in self.parcels}
        self.last_was_grain: dict[str, bool] = {p.id: (p.initial_crop == "grain") for p in self.parcels}
        self.last_was_idle: dict[str, bool] = {p.id: (p.initial_crop == "idle") for p in self.parcels}
        self.ever_subsidized_parcels: set[str] = set()
        self.last_subsidized_operator: dict[str, str] = {}

        init_g_mu = fsum(p.area for p in self.parcels if p.initial_crop == "grain")
        self.observed: dict[str, float] = {
            "grain_kg": cal.initial_grain_kg,
            "grain_phys_mu": init_g_mu,
            "grain_cult_rate": init_g_mu / max(1.0, self.land_mu),
            "idle_rate": fsum(p.area for p in self.parcels if p.initial_crop == "idle") / max(1.0, self.land_mu),
            "unabsorbed_rate": 0.0,
            "leased_rate": 0.0,
            "tight_liquidity_share": 0.25,
            "fiscal_exec_rate": 1.0,
            "last_unit_rent": cal.tech.ask_rent_yuan_phys_mu,
        }
        self.last_parcel_rents: dict[str, tuple[float, float]] = {}
        self.last_parcel_rows: list[dict] = []
        self.last_actor_rows: list[dict] = []

    def public_feature_list(self, design: V2Design) -> list[float]:
        obs = self.observed
        if design.menu_mode == "v2_25":
            return [
                float(obs["grain_cult_rate"]),
                float(obs["idle_rate"]),
                float(obs["unabsorbed_rate"]),
                float(obs["leased_rate"]),
                float(obs["tight_liquidity_share"]),
            ]
        # Clean 17-dim public macro state (1 + 4 * 4 genuinely observable macro rates)
        return [
            float(obs["grain_cult_rate"]),
            float(obs["idle_rate"]),
            float(obs["leased_rate"]),
            float(obs["tight_liquidity_share"]),
        ]

    def clone_into(self, new_model: "AgriSuccessionMesaModel") -> "RegionalSubEconomy":
        other = RegionalSubEconomy.__new__(RegionalSubEconomy)
        other.model = new_model
        other.seed = self.seed
        other.cal = self.cal
        other.name = self.name
        other.tech = self.tech
        other.land_mu = self.land_mu
        other.parcels = self.parcels
        other.parcel_map = self.parcel_map
        other.households = {k: v.clone_into(new_model) for k, v in self.households.items()}
        other.organizations = {k: v.clone_into(new_model) for k, v in self.organizations.items()}
        other.operators = {**other.households, **other.organizations}
        other.contracts = self.contracts.copy()
        other.last_operator = self.last_operator.copy()
        other.last_was_grain = self.last_was_grain.copy()
        other.last_was_idle = self.last_was_idle.copy()
        other.ever_subsidized_parcels = set(self.ever_subsidized_parcels)
        other.last_subsidized_operator = self.last_subsidized_operator.copy()
        other.observed = self.observed.copy()
        other.last_parcel_rents = self.last_parcel_rents.copy()
        other.last_parcel_rows = []
        other.last_actor_rows = []
        return other

    def build_schedule(self, instrument: str, budget_cap: float, design: V2Design) -> SubsidySchedule:
        thresh = design.scale_threshold_for(self.name)
        v_sav = design.village_contiguous_mach_saving if design.enable_village_coordination else 0.0
        g_cov = design.village_grain_covenant_floor if design.enable_village_coordination else 0.0

        if instrument == "none" or budget_cap <= 0.0:
            return SubsidySchedule(
                instrument="none",
                rate_per_phys_mu=0.0,
                scale_threshold_mu=thresh,
                single_operator_cap_mu=design.single_operator_subsidy_cap_mu,
                enable_single_operator_cap=design.enable_single_operator_cap,
                service_discount_share=0.0,
                order_price_bonus_yuan_kg=0.0,
                village_pooled_saving=v_sav,
                grain_covenant_floor=g_cov,
            )
        order_bonus = 0.0
        if instrument.startswith("producer_"):
            elig_pool = max(1.0, self.land_mu * max(0.58, self.tech.baseline_grain_share))
            rate = min(design.producer_cap_yuan_per_mu, budget_cap / elig_pool)
            disc = 0.0
        elif instrument.startswith("scale_takeover_"):
            elig_pool = max(1.0, self.land_mu * max(0.22, self.tech.scale_pool_share))
            rate = min(design.scale_takeover_cap_yuan_per_mu, budget_cap / elig_pool)
            disc = 0.0
        elif instrument == "service_trusteeship":
            elig_pool = max(1.0, self.land_mu * max(0.55, self.tech.baseline_grain_share))
            rate = min(design.service_cap_yuan_per_mu, budget_cap / elig_pool)
            disc = design.service_discount_share
        elif instrument == "order_farming":
            elig_pool = max(1.0, self.land_mu * max(0.55, self.tech.baseline_grain_share))
            rate = min(design.order_cap_yuan_per_mu, budget_cap / elig_pool)
            disc = 0.0
            order_bonus = design.order_max_price_bonus_ratio * self.tech.grain_price_yuan_kg
        else:
            raise ValueError(f"Unsupported policy instrument: {instrument}")

        return SubsidySchedule(
            instrument=instrument,
            rate_per_phys_mu=rate,
            scale_threshold_mu=thresh,
            single_operator_cap_mu=design.single_operator_subsidy_cap_mu,
            enable_single_operator_cap=design.enable_single_operator_cap,
            service_discount_share=disc,
            order_price_bonus_yuan_kg=order_bonus,
            village_pooled_saving=v_sav,
            grain_covenant_floor=g_cov,
        )

    def step_region(
        self,
        year: int,
        instrument: str,
        budget_cap: float,
        design: V2Design,
        *,
        collect_micro_ledger: bool = False,
        village_exchange: bool = False,
        fixed_contract_ref: FixedContractSnapshot | None = None,
        record_contract_snapshot: bool = False,
    ) -> tuple[dict, FixedContractSnapshot | None]:
        actor_ids = sorted(self.operators)
        rng = lambda ch: np.random.default_rng([self.seed, year, ch])

        wage_shock = dict(zip(actor_ids, rng(11).uniform(0.85, 1.22, len(actor_ids))))
        labor_shock = dict(zip(actor_ids, rng(12).uniform(0.74, 1.08, len(actor_ids))))
        wc_shock = dict(zip(actor_ids, rng(13).uniform(0.84, 1.02, len(actor_ids))))
        search_draw = dict(zip((p.id for p in self.parcels), rng(14).random(len(self.parcels))))
        weather_mult = float(rng(15).choice([0.58, 0.80, 1.0, 1.12], p=[0.06, 0.14, 0.58, 0.22]))

        drift = (1.0 + design.real_offfarm_wage_drift) ** year

        # 0. Opening carry-over + Background Fertility Protection Subsidy (`fertility_decoupled`)
        fertility_received: dict[str, float] = defaultdict(float)
        fertility_forfeited_mu = 0.0
        if design.enable_background_fertility_subsidy:
            for p in self.parcels:
                if design.fertility_anti_abandonment_rule and self.last_was_idle[p.id]:
                    fertility_forfeited_mu += p.area
                else:
                    fertility_received[p.owner] += p.area * self.tech.fertility_base_yuan_phys_mu

        for aid in actor_ids:
            op = self.operators[aid]
            op.current_year_num = year
            op.current_opp_wage = op.bp.offfarm_wage_day * drift * wage_shock[aid]
            age_decay = (
                (1.0 - design.elderly_labor_decay_rate) ** max(0.0, (op.bp.head_age_2010 + year) - 58.0)
                if (design.enable_household_aging_and_exit_inertia and op.is_survey_household)
                else 1.0
            )
            op.current_fam_labor = op.bp.ag_days_year * age_decay * labor_shock[aid]
            # Background fertility cash enters household spring liquidity via "One-Card" transfer
            op.current_wc_open = max(0.0, op.working_cap * wc_shock[aid] + fertility_received[aid])

        sched_init = self.build_schedule(instrument, budget_cap, design)

        # 1 & 2 & 3. Either use Fixed-Contract Snapshot (Exp 1) OR run Endogenous Land Rental Market
        if fixed_contract_ref is not None:
            assigned_op = fixed_contract_ref.assigned_operator.copy()
            active_contracts = fixed_contract_ref.active_contracts.copy()
            new_takeover_pids = set(fixed_contract_ref.new_takeover_pids)
            rent_paid = defaultdict(float, fixed_contract_ref.rent_paid)
            rent_income = defaultdict(float, fixed_contract_ref.rent_income)
            fees_paid = defaultdict(float, fixed_contract_ref.fees_paid)
            wc_after_mkt = {
                aid: max(0.0, fixed_contract_ref.wc_after_market[aid]) for aid in actor_ids
            }
            leased_pids = set(fixed_contract_ref.leased_pids)
            released_pids = set(fixed_contract_ref.released_pids)
            unmatched_pids = set(fixed_contract_ref.unmatched_pids)
            recovered_pids = set(fixed_contract_ref.recovered_pids)
            defaulted_pids: set[str] = set()
            snap_out = None
        else:
            # 1a. Enforce or default multi-year in-force contracts
            held_contracts: dict[str, LeaseContractV2] = {}
            defaulted_pids = set()
            by_lessee: dict[str, list[LeaseContractV2]] = defaultdict(list)
            for pid, c in self.contracts.items():
                if c.end_year >= year:
                    by_lessee[c.lessee].append(c)

            wc_after_held = {aid: self.operators[aid].current_wc_open for aid in actor_ids}
            rent_paid = defaultdict(float)
            rent_income = defaultdict(float)
            fees_paid = defaultdict(float)

            for aid, clist in by_lessee.items():
                tot_rent = fsum(c.rent_yuan for c in clist)
                if not self.operators[aid].active_in_market or wc_after_held[aid] + 1e-7 < tot_rent:
                    for c in clist:
                        defaulted_pids.add(c.parcel_id)
                else:
                    wc_after_held[aid] -= tot_rent
                    rent_paid[aid] += tot_rent
                    for c in clist:
                        held_contracts[c.parcel_id] = c
                        rent_income[c.lessor] += c.rent_yuan

            # 1b. Household retention vs. release decisions
            own_free_by_hh: dict[str, list[ParcelBlueprint]] = defaultdict(list)
            held_by_actor: dict[str, list[ParcelBlueprint]] = defaultdict(list)
            for p in self.parcels:
                if p.id in held_contracts:
                    held_by_actor[held_contracts[p.id].lessee].append(p)
                else:
                    own_free_by_hh[p.owner].append(p)

            retained_pids: set[str] = set()
            released_pids = set()
            eff_friction = design.search_friction_prob * (
                design.village_friction_mult if design.enable_village_coordination else 1.0
            )
            eff_setup_fee = design.contract_setup_fee_yuan * (
                design.village_setup_fee_mult if design.enable_village_coordination else 1.0
            )
            if design.dynamic_expected_rent_update and self.observed.get("last_unit_rent", 0.0) > 10.0:
                base_exp_rent = 0.55 * self.tech.ask_rent_yuan_phys_mu + 0.45 * float(self.observed["last_unit_rent"])
            else:
                base_exp_rent = self.tech.ask_rent_yuan_phys_mu
            expected_rent_mu = base_exp_rent * (1.0 - eff_friction)
            # Anti-abandonment rule increases willingness to lease out rather than leave idle
            if design.enable_background_fertility_subsidy and design.fertility_anti_abandonment_rule:
                expected_rent_mu += 0.35 * self.tech.fertility_base_yuan_phys_mu

            for aid, hh in self.households.items():
                held_mu = fsum(p.area for p in held_by_actor[aid])
                held_new_tk = fsum(
                    p.area for p in held_by_actor[aid] if self.last_operator[p.id] != aid
                )
                own_trad = [p for p in own_free_by_hh[aid] if p.tradable]
                own_nontrad = [p for p in own_free_by_hh[aid] if not p.tradable]
                nontrad_mu = fsum(p.area for p in own_nontrad)

                keep_list, rel_list, chosen_plan = hh.decide_land_retention(
                    held_mu, held_new_tk, own_trad, nontrad_mu,
                    expected_rent_mu, design, sched_init, wc_after_held[aid],
                )
                if (
                    design.allow_exit_entrust_nontrad
                    and len(keep_list) == 0
                    and chosen_plan.planted_physical_mu <= 1e-6
                ):
                    for p in own_nontrad:
                        released_pids.add(p.id)
                else:
                    for p in own_nontrad:
                        retained_pids.add(p.id)
                for p in keep_list:
                    retained_pids.add(p.id)
                for p in rel_list:
                    released_pids.add(p.id)

            # 1c. Sequential Incremental Shadow-Price Land Clearing by Submarket
            # Offer-group fee rule: setup fee is paid at most once per (lessee, offer_group_id) in a year!
            cur_mu: dict[str, float] = {
                aid: fsum(p.area for p in held_by_actor[aid])
                + fsum(self.parcel_map[pid].area for pid in retained_pids if self.parcel_map[pid].owner == aid)
                for aid in actor_ids
            }
            cur_tk_mu: dict[str, float] = {
                aid: fsum(p.area for p in held_by_actor[aid] if self.last_operator[p.id] != aid)
                for aid in actor_ids
            }
            wc_rem = wc_after_held.copy()
            signed_groups_by_lessee: dict[str, set[str]] = defaultdict(set)
            new_contracts: dict[str, LeaseContractV2] = {}
            unmatched_pids = set()

            by_submkt: dict[int, list[ParcelBlueprint]] = defaultdict(list)
            for pid in sorted(released_pids):
                by_submkt[self.parcel_map[pid].submarket].append(self.parcel_map[pid])

            candidates_by_sm: dict[int, list[str]] = defaultdict(list)
            for aid in actor_ids:
                op = self.operators[aid]
                if op.active_in_market:
                    candidates_by_sm[op.submarket].append(aid)

            for sm, plist in sorted(by_submkt.items()):
                cands = candidates_by_sm[sm]
                releasing_owners = {p.owner for p in plist}

                base_plan_cache: dict[str, PMPPlanV2] = {}
                bid_cache: dict[tuple[str, float, bool], float] = {}

                def refresh_bidder(b_id: str) -> None:
                    base_plan_cache[b_id] = self.operators[b_id].evaluate_production(
                        cur_mu[b_id], cur_tk_mu[b_id], design, sched_init, wc_rem[b_id]
                    )

                for b_id in cands:
                    if b_id not in releasing_owners and wc_rem[b_id] > 100.0:
                        refresh_bidder(b_id)

                ordered_plots = sorted(plist, key=lambda p: (-p.area, p.id))
                for p in ordered_plots:
                    if search_draw[p.id] < eff_friction:
                        unmatched_pids.add(p.id)
                        continue
                    ask_mu = self.tech.ask_rent_yuan_phys_mu * p.quality
                    area_key = round(p.area, 4)
                    grp_id_eff = f"VPOOL_{sm}" if design.enable_village_coordination else p.offer_group_id
                    valid_bids: list[tuple[float, str]] = []

                    for b_id in cands:
                        if b_id == p.owner or b_id in releasing_owners or b_id not in base_plan_cache:
                            continue
                        is_new_grp = grp_id_eff not in signed_groups_by_lessee[b_id]
                        ckey = (b_id, area_key, is_new_grp)
                        if ckey not in bid_cache:
                            bid_cache[ckey] = self.operators[b_id].evaluate_land_bid(
                                cur_mu[b_id], cur_tk_mu[b_id], p.area, is_new_grp,
                                design, sched_init, wc_rem[b_id],
                                base_plan=base_plan_cache[b_id],
                                setup_fee_override=eff_setup_fee,
                            )
                        b_val = bid_cache[ckey]
                        if b_val + 1e-7 >= ask_mu:
                            valid_bids.append((b_val, b_id))

                    if not valid_bids:
                        unmatched_pids.add(p.id)
                        continue

                    # Sort highest bid first; deterministic tie-break on actor ID
                    valid_bids.sort(key=lambda x: (-x[0], x[1]))
                    win_bid_mu, winner = valid_bids[0]
                    second_bid_mu = valid_bids[1][0] if len(valid_bids) >= 2 else ask_mu

                    if design.pricing_rule == "second_price":
                        clear_mu = max(ask_mu, min(win_bid_mu, second_bid_mu))
                    else:
                        clear_mu = 0.5 * (win_bid_mu + max(ask_mu, second_bid_mu if len(valid_bids) >= 2 else ask_mu))

                    rent_y = clear_mu * p.area
                    is_new_grp = grp_id_eff not in signed_groups_by_lessee[winner]
                    fee_y = eff_setup_fee if is_new_grp else 0.0

                    if wc_rem[winner] + 1e-7 < rent_y + fee_y:
                        unmatched_pids.add(p.id)
                        continue

                    wc_rem[winner] -= (rent_y + fee_y)
                    rent_paid[winner] += rent_y
                    rent_income[p.owner] += rent_y
                    fees_paid[winner] += fee_y
                    cur_mu[winner] += p.area
                    if self.last_operator[p.id] != winner:
                        cur_tk_mu[winner] += p.area
                    signed_groups_by_lessee[winner].add(grp_id_eff)

                    lease_yrs = design.lease_years_for_category(self.operators[winner].actor_category)
                    new_contracts[p.id] = LeaseContractV2(
                        parcel_id=p.id,
                        offer_group_id=grp_id_eff,
                        lessor=p.owner,
                        lessee=winner,
                        rent_yuan=rent_y,
                        rent_per_mu=clear_mu,
                        winning_bid_per_mu=win_bid_mu,
                        runner_up_bid_per_mu=second_bid_mu,
                        ask_per_mu=ask_mu,
                        start_year=year,
                        end_year=year + max(1, lease_yrs) - 1,
                    )

                    # Immediately invalidate cached bids for winner and recompute post-win state (RegMAS rule!)
                    for k_del in [k for k in bid_cache if k[0] == winner]:
                        del bid_cache[k_del]
                    if wc_rem[winner] > 100.0:
                        refresh_bidder(winner)
                    else:
                        base_plan_cache.pop(winner, None)

            # 1d. Conditional owner recovery on unmatched released parcels
            recovered_pids = set()
            by_unm_owner: dict[str, list[ParcelBlueprint]] = defaultdict(list)
            for pid in sorted(unmatched_pids):
                by_unm_owner[self.parcel_map[pid].owner].append(self.parcel_map[pid])

            for oid, ulist in by_unm_owner.items():
                u_mu = fsum(p.area for p in ulist)
                rec_cost = design.unmatched_recovery_cost_per_mu * u_mu
                rec_days = design.unmatched_recovery_days_per_mu * u_mu
                hh = self.households[oid]
                if wc_rem[oid] <= rec_cost or hh.current_fam_labor <= rec_days:
                    continue
                p_without = hh.evaluate_production(cur_mu[oid], cur_tk_mu[oid], design, sched_init, wc_rem[oid])
                saved_lab = hh.current_fam_labor
                hh.current_fam_labor = max(0.0, saved_lab - rec_days)
                p_with = hh.evaluate_production(
                    cur_mu[oid] + u_mu, cur_tk_mu[oid], design, sched_init, wc_rem[oid] - rec_cost
                )
                if p_with.planted_physical_mu > p_without.planted_physical_mu + 0.25 and (
                    p_with.net_economic_surplus_yuan - rec_cost > p_without.net_economic_surplus_yuan
                ):
                    wc_rem[oid] -= rec_cost
                    fees_paid[oid] += rec_cost
                    cur_mu[oid] += u_mu
                    for p in ulist:
                        recovered_pids.add(p.id)
                else:
                    hh.current_fam_labor = saved_lab

            active_contracts = {**held_contracts, **new_contracts}
            leased_pids = set(active_contracts)
            assigned_op = {}
            new_takeover_pids = set()
            for p in self.parcels:
                if p.id in active_contracts:
                    op_id = active_contracts[p.id].lessee
                else:
                    op_id = p.owner
                assigned_op[p.id] = op_id
                if (p.id in active_contracts) and (self.last_operator[p.id] != op_id):
                    new_takeover_pids.add(p.id)

            wc_after_mkt = wc_rem.copy()
            snap_out = (
                FixedContractSnapshot(
                    year=year,
                    region_name=self.name,
                    assigned_operator=assigned_op.copy(),
                    active_contracts=active_contracts.copy(),
                    new_takeover_pids=frozenset(new_takeover_pids),
                    rent_paid=dict(rent_paid),
                    rent_income=dict(rent_income),
                    fees_paid=dict(fees_paid),
                    wc_after_market=wc_after_mkt.copy(),
                    leased_pids=frozenset(leased_pids),
                    released_pids=frozenset(released_pids),
                    unmatched_pids=frozenset(unmatched_pids),
                    recovered_pids=frozenset(recovered_pids),
                )
                if record_contract_snapshot
                else None
            )

        # Optional open village rental inflow for Stage 1 cohort validation ONLY
        ext_inflow_mu: dict[str, float] = defaultdict(float)
        if village_exchange and fixed_contract_ref is None:
            pool_rem = self.tech.village_net_inflow_rate * self.land_mu * min(year, 5)
            ask_v = self.tech.ask_rent_yuan_phys_mu
            op_mu_tmp = defaultdict(float)
            tk_mu_tmp = defaultdict(float)
            for p in self.parcels:
                oid = assigned_op[p.id]
                if p.id in leased_pids or p.id not in unmatched_pids or p.id in recovered_pids:
                    op_mu_tmp[oid] += p.area
                    if p.id in new_takeover_pids:
                        tk_mu_tmp[oid] += p.area
            for aid in sorted(self.households):
                if pool_rem <= 0.5:
                    break
                chunk = min(6.0, pool_rem)
                b_v = self.households[aid].evaluate_land_bid(
                    op_mu_tmp[aid], tk_mu_tmp[aid], chunk, False, design, sched_init, wc_after_mkt[aid]
                )
                if b_v >= ask_v and wc_after_mkt[aid] >= ask_v * chunk + 200.0:
                    wc_after_mkt[aid] -= ask_v * chunk
                    rent_paid[aid] += ask_v * chunk
                    ext_inflow_mu[aid] += chunk
                    pool_rem -= chunk

        # 4. Assemble operated parcel portfolios & solve PMP production with hard fiscal cap enforcement
        parcels_by_op: dict[str, list[ParcelBlueprint]] = defaultdict(list)
        for p in self.parcels:
            if (p.id in unmatched_pids) and (p.id not in recovered_pids) and (p.id not in leased_pids):
                continue
            parcels_by_op[assigned_op[p.id]].append(p)

        op_phys_mu: dict[str, float] = {
            aid: fsum(p.area for p in parcels_by_op[aid]) + ext_inflow_mu[aid] for aid in actor_ids
        }
        op_new_tk_mu: dict[str, float] = {
            aid: fsum(p.area for p in parcels_by_op[aid] if p.id in new_takeover_pids) + ext_inflow_mu[aid]
            for aid in actor_ids
        }

        sched_eff = sched_init
        plans: dict[str, PMPPlanV2] = {}
        sub_budget_cap = budget_cap

        def _claim_for_plan(pl: PMPPlanV2, sc: SubsidySchedule) -> float:
            if sc.instrument == "service_trusteeship":
                return pl.service_subsidy_yuan
            if sc.instrument == "order_farming":
                return pl.order_subsidy_yuan
            if sc.instrument != "none":
                return pl.pre_cash_subsidy_yuan + pl.post_cash_subsidy_yuan
            return 0.0

        def solve_all_operators(sc: SubsidySchedule) -> tuple[dict[str, PMPPlanV2], float]:
            pl_map: dict[str, PMPPlanV2] = {}
            tot_claim = 0.0
            for aid in actor_ids:
                pl = self.operators[aid].evaluate_production(
                    op_phys_mu[aid], op_new_tk_mu[aid], design, sc, wc_after_mkt[aid],
                    rent_offset_yuan=rent_income[aid],
                )
                pl_map[aid] = pl
                tot_claim += _claim_for_plan(pl, sc)
            return pl_map, tot_claim

        plans, raw_claim_total = solve_all_operators(sched_eff)

        # Fixed-point proportional rate scaling if total claims exceed sub_budget_cap
        if sched_eff.instrument != "none" and raw_claim_total > sub_budget_cap + 1e-7:
            for _ in range(5):
                if raw_claim_total <= sub_budget_cap + 1e-7:
                    break
                scale_f = max(0.0, (sub_budget_cap / raw_claim_total) * 0.999999)
                sched_eff = SubsidySchedule(
                    instrument=sched_eff.instrument,
                    rate_per_phys_mu=sched_eff.rate_per_phys_mu * scale_f,
                    scale_threshold_mu=sched_eff.scale_threshold_mu,
                    single_operator_cap_mu=sched_eff.single_operator_cap_mu,
                    enable_single_operator_cap=sched_eff.enable_single_operator_cap,
                    service_discount_share=sched_eff.service_discount_share * (
                        scale_f if sched_eff.instrument == "service_trusteeship" else 1.0
                    ),
                    order_price_bonus_yuan_kg=sched_eff.order_price_bonus_yuan_kg * (
                        scale_f if sched_eff.instrument == "order_farming" else 1.0
                    ),
                    village_pooled_saving=sched_eff.village_pooled_saving,
                    grain_covenant_floor=sched_eff.grain_covenant_floor,
                )
                plans, raw_claim_total = solve_all_operators(sched_eff)

        # 5. Harvest realization, 6-Category Parcel Transition Ledger & Settlement
        committed_yuan = min(budget_cap, raw_claim_total) if sched_eff.instrument != "none" else 0.0
        actual_incremental_spending = 0.0
        support_by_actor: dict[str, float] = defaultdict(float)

        for aid, pl in plans.items():
            sup = _claim_for_plan(pl, sched_eff)
            support_by_actor[aid] = sup
            actual_incremental_spending += sup

        if actual_incremental_spending > budget_cap + 1e-4:
            raise AssertionError(f"Regional spending {actual_incremental_spending} breached cap {budget_cap}")

        # Parcel-level accounting & exact 6-fate decomposition
        fate_incumbent_grain_mu = 0.0
        fate_strict_succession_grain_mu = 0.0
        fate_expanded_grain_mu = 0.0
        fate_idle_recovered_grain_mu = 0.0
        fate_leased_cash_mu = 0.0
        fate_own_cash_mu = 0.0
        fate_leased_unplanted_mu = 0.0
        fate_own_idle_mu = 0.0

        incumbent_grain_kg = 0.0
        strict_succession_grain_kg = 0.0
        expanded_grain_kg = 0.0

        # Beneficiary cohort tracking metrics (for parcels subsidized in earlier years <= withdrawal_after)
        cohort_tracked_mu = 0.0
        cohort_retained_same_op_mu = 0.0
        cohort_still_grain_mu = 0.0
        cohort_grain_kg = 0.0

        parcel_rows: list[dict] = []
        parcel_rents_now: dict[str, tuple[float, float]] = {}
        parcel_harvest_by_op: dict[str, float] = defaultdict(float)
        next_last_op: dict[str, str] = {}
        next_was_grain: dict[str, bool] = {}
        next_was_idle: dict[str, bool] = {}

        op_parcel_phys_sum: dict[str, float] = {
            aid: fsum(q.area for q in parcels_by_op[aid]) for aid in actor_ids
        }

        for p in self.parcels:
            is_leased = p.id in active_contracts
            rent_y = active_contracts[p.id].rent_yuan if is_leased else 0.0
            if is_leased:
                parcel_rents_now[p.id] = (p.area, rent_y)

            unmatched_idle = (p.id in unmatched_pids) and (p.id not in recovered_pids) and (not is_leased)
            op_id = assigned_op[p.id]
            prev_op = self.last_operator[p.id]
            prev_g = self.last_was_grain[p.id]
            prev_idle = self.last_was_idle[p.id]

            if unmatched_idle or op_phys_mu[op_id] <= 1e-9:
                pl_frac = 0.0
                g_sh = 0.0
                h_kg = 0.0
                sup_p = 0.0
            else:
                pl = plans[op_id]
                pl_frac = min(1.0, pl.planted_physical_mu / op_phys_mu[op_id])
                g_sh = pl.grain_share
                op_op = self.operators[op_id]
                sown_g_p = p.area * pl_frac * g_sh * op_op.bp.mci
                h_kg = sown_g_p * op_op.bp.grain_yield_kg_sown_mu * weather_mult
                parcel_harvest_by_op[op_id] += h_kg
                # Allocate operator's realized incremental support proportionally across operated/grain parcels
                if ext_inflow_mu[op_id] == 0.0 and op_parcel_phys_sum[op_id] > 1e-9:
                    sup_p = support_by_actor[op_id] * (p.area / op_parcel_phys_sum[op_id])
                else:
                    op_g_phys = max(1e-9, pl.planted_physical_mu * pl.grain_share)
                    sup_p = support_by_actor[op_id] * ((p.area * pl_frac * g_sh) / op_g_phys) if g_sh > 1e-9 and pl_frac > 1e-9 else 0.0

            g_phys = p.area * pl_frac * g_sh
            c_phys = p.area * pl_frac * (1.0 - g_sh)
            i_phys = max(0.0, p.area - g_phys - c_phys)

            # Partition grain area & kg into incumbent vs. strict operator-change succession vs. non-grain expansion
            if g_phys > 0.0:
                if op_id == prev_op and prev_g:
                    fate_incumbent_grain_mu += g_phys
                    incumbent_grain_kg += h_kg
                    p_fate = "incumbent_grain"
                elif op_id != prev_op and prev_g:
                    fate_strict_succession_grain_mu += g_phys
                    strict_succession_grain_kg += h_kg
                    p_fate = "strict_succession_grain"
                else:
                    fate_expanded_grain_mu += g_phys
                    expanded_grain_kg += h_kg
                    if prev_idle:
                        fate_idle_recovered_grain_mu += g_phys
                    p_fate = "idle_recovered_grain" if prev_idle else "converted_to_grain"
            elif c_phys > i_phys:
                p_fate = "leased_cash" if is_leased else "own_cash"
            else:
                p_fate = "leased_unplanted" if is_leased else "own_idle"

            if is_leased:
                fate_leased_cash_mu += c_phys
                fate_leased_unplanted_mu += i_phys
            else:
                fate_own_cash_mu += c_phys
                fate_own_idle_mu += i_phys

            # Track parcels previously subsidized during active policy window (<= withdrawal_after)
            if p.id in self.ever_subsidized_parcels:
                cohort_tracked_mu += p.area
                if op_id == self.last_subsidized_operator.get(p.id, ""):
                    cohort_retained_same_op_mu += p.area
                cohort_still_grain_mu += g_phys
                cohort_grain_kg += h_kg

            if sup_p > 1e-6 and year <= design.withdrawal_after:
                self.ever_subsidized_parcels.add(p.id)
                self.last_subsidized_operator[p.id] = op_id

            next_last_op[p.id] = op_id if pl_frac > 1e-6 else p.owner
            next_was_grain[p.id] = g_phys >= 0.25 * p.area
            next_was_idle[p.id] = pl_frac <= 1e-6

            if collect_micro_ledger:
                parcel_rows.append({
                    "year": year,
                    "region": self.name,
                    "parcel_id": p.id,
                    "province": p.province,
                    "submarket": p.submarket,
                    "grid_x": p.grid_x,
                    "grid_y": p.grid_y,
                    "is_synthetic_coord": p.is_synthetic_coord,
                    "owner": p.owner,
                    "prev_operator": prev_op,
                    "operator": op_id,
                    "operator_category": self.operators[op_id].actor_category,
                    "leased": int(is_leased),
                    "fate": p_fate,
                    "physical_area": round(p.area, 6),
                    "grain_physical_mu": round(g_phys, 6),
                    "cash_physical_mu": round(c_phys, 6),
                    "idle_physical_mu": round(i_phys, 6),
                    "harvest_grain_kg": round(h_kg, 4),
                    "rent_yuan": round(rent_y, 4),
                    "incremental_support_yuan": round(sup_p, 4),
                })

        # External village exchange grain (only when village_exchange=True in Stage 1 validation)
        ext_grain_kg_by_actor: dict[str, float] = defaultdict(float)
        for aid, ex_mu in ext_inflow_mu.items():
            if ex_mu > 0 and op_phys_mu[aid] > 0:
                pl = plans[aid]
                pl_f = min(1.0, pl.planted_physical_mu / op_phys_mu[aid])
                op_op = self.operators[aid]
                ext_grain_kg_by_actor[aid] = (
                    ex_mu * pl_f * pl.grain_share * op_op.bp.mci * op_op.bp.grain_yield_kg_sown_mu * weather_mult
                )

        # 6. Operator-level Harvest Income, Subsistence Split, Working-Capital Carry-over & Viability
        actor_rows: list[dict] = []
        survey_grain_kg = 0.0
        org_grain_kg_by_cat: dict[str, float] = defaultdict(float)
        named_grain_totals: dict[str, float] = defaultdict(float)

        survey_subsistence_kg = 0.0
        survey_marketed_kg = 0.0
        survey_grain_cash_sales_yuan = 0.0
        survey_subsistence_value_yuan = 0.0
        survey_net_income_list: list[float] = []
        survey_cultivating_count = 0
        survey_mixed_count = 0
        tight_liq_count = 0

        for aid in actor_ids:
            op = self.operators[aid]
            pl = plans[aid]
            sup_inc = support_by_actor[aid]
            fert_inc = fertility_received[aid]

            realized_grain_kg = parcel_harvest_by_op[aid] + ext_grain_kg_by_actor[aid]
            realized_cash_crop_rev = pl.expected_cash_rev_yuan * weather_mult

            # Named grain crop realization
            w_dict = dict(op.bp.grain_weights)
            tot_wy = max(1e-9, fsum(w_dict[k] * self.tech.named_grains[k].yield_kg_sown_mu for k in NAMED_GRAINS))
            for k in NAMED_GRAINS:
                named_grain_totals[k] += realized_grain_kg * (
                    w_dict[k] * self.tech.named_grains[k].yield_kg_sown_mu / tot_wy
                )

            # MP-MAS subsistence staple split vs. marketed cash sales
            if design.enable_subsistence_grain_split and op.is_survey_household:
                home_kg = min(realized_grain_kg, op.bp.subsistence_grain_kg)
                mkt_kg = max(0.0, realized_grain_kg - home_kg)
                deficit_kg = max(0.0, op.bp.subsistence_grain_kg - realized_grain_kg)
                grain_cash_sales = mkt_kg * op.bp.grain_price_yuan_kg
                in_kind_sub_val = home_kg * op.bp.grain_price_yuan_kg
                staple_purchase_cash = deficit_kg * (1.0 + design.staple_retail_wedge) * op.bp.grain_price_yuan_kg
                # In-kind staple consumption offsets cash food expenditure pound-for-pound at farmgate value
                cash_living_spend = max(0.0, op.bp.consumption_cash_base - in_kind_sub_val) + staple_purchase_cash
                survey_subsistence_kg += home_kg
                survey_marketed_kg += mkt_kg
                survey_grain_cash_sales_yuan += grain_cash_sales
                survey_subsistence_value_yuan += in_kind_sub_val
            else:
                home_kg = 0.0
                mkt_kg = realized_grain_kg
                grain_cash_sales = realized_grain_kg * op.bp.grain_price_yuan_kg
                in_kind_sub_val = 0.0
                cash_living_spend = op.bp.consumption_cash_base
                if op.is_survey_household:
                    survey_marketed_kg += mkt_kg
                    survey_grain_cash_sales_yuan += grain_cash_sales

            # Off-farm wage realization
            slack_used = min(pl.family_days_used, min(op.current_fam_labor, op.bp.slack_ag_days_year))
            comp_used = max(0.0, pl.family_days_used - slack_used)
            wage_earned = max(0.0, op.bp.offfarm_annual_base * drift * wage_shock[aid] - op.current_opp_wage * comp_used)

            pre_cash_sub_realized = sup_inc if sched_eff.instrument.endswith("_pre") else 0.0
            post_cash_sub_realized = (
                sup_inc if sched_eff.instrument in ("producer_post", "scale_takeover_post", "order_farming") else 0.0
            )

            upfront_prod_outlay = max(
                0.0,
                pl.non_mach_inputs_yuan + pl.net_mach_service_yuan + design.upfront_hired_cash_share * pl.hired_cost_yuan - pre_cash_sub_realized,
            )
            wc_mid = max(0.0, wc_after_mkt[aid] - upfront_prod_outlay)
            if op_phys_mu[aid] > 0.5 and pl.planted_physical_mu < 0.92 * op_phys_mu[aid]:
                tight_liq_count += 1

            autumn_hired_rem = (1.0 - design.upfront_hired_cash_share) * pl.hired_cost_yuan
            autumn_cash_inflow = (
                grain_cash_sales
                + realized_cash_crop_rev
                + post_cash_sub_realized
                + wage_earned
                - autumn_hired_rem
                - op.bp.admin_overhead_yuan
            )
            net_cash_after_living = autumn_cash_inflow - cash_living_spend

            # Full economic net income (including in-kind staple consumption value + net rent + subsidies)
            full_net_income = (
                grain_cash_sales
                + in_kind_sub_val
                + realized_cash_crop_rev
                + pre_cash_sub_realized
                + post_cash_sub_realized
                + fert_inc
                + wage_earned
                + rent_income[aid]
                - rent_paid[aid]
                - fees_paid[aid]
                - pl.non_mach_inputs_yuan
                - pl.net_mach_service_yuan
                - pl.hired_cost_yuan
                - op.bp.admin_overhead_yuan
            )

            # Recover spring working-capital principal, then allocate surplus into working_cap & wealth
            principal_spent = op.current_wc_open - wc_mid
            principal_rec = min(principal_spent, max(0.0, autumn_cash_inflow))
            rem_cash_surplus = net_cash_after_living - principal_rec

            if design.enable_cash_carryover:
                if rem_cash_surplus >= 0:
                    wc_next = wc_mid + principal_rec + design.surplus_to_working_cap_rate * rem_cash_surplus
                    wealth_next = op.wealth + (1.0 - design.surplus_to_working_cap_rate) * rem_cash_surplus
                else:
                    deficit = -rem_cash_surplus
                    draw_w = min(op.wealth, deficit)
                    wealth_next = op.wealth - draw_w
                    wc_next = max(design.working_cap_Min_buffer_yuan * 0.4, wc_mid + principal_rec - (deficit - draw_w))
            else:
                wc_next = op.bp.working_capital_init
                wealth_next = op.bp.savings_stock

            op.working_cap = wc_next
            op.wealth = wealth_next
            op.was_cultivating = (pl.sown_grain_mu + pl.sown_cash_mu) > 1e-6
            if pl.planted_physical_mu > 1e-6:
                op.ref_grain_share = 0.65 * op.ref_grain_share + 0.35 * pl.grain_share

            if isinstance(op, ScaleOrgMesaAgent):
                op.update_organizational_viability(full_net_income, op.bp.working_capital_init + op.bp.savings_stock, design)
                org_grain_kg_by_cat[op.actor_category] += realized_grain_kg
            else:
                survey_grain_kg += realized_grain_kg
                survey_net_income_list.append(full_net_income)
                if (pl.sown_grain_mu + pl.sown_cash_mu) > 1e-6:
                    survey_cultivating_count += 1
                    if pl.sown_grain_mu > 1e-6 and pl.sown_cash_mu > 1e-6:
                        survey_mixed_count += 1

            rec_summary = {
                "year": year,
                "region": self.name,
                "actor": aid,
                "actor_category": op.actor_category,
                "is_survey_household": int(op.is_survey_household),
                "active_in_market": int(op.active_in_market),
                "operated_physical_mu": round(op_phys_mu[aid], 4),
                "planted_physical_mu": round(pl.planted_physical_mu, 4),
                "sown_grain_mu": round(pl.sown_grain_mu, 4),
                "sown_cash_mu": round(pl.sown_cash_mu, 4),
                "grain_share": round(pl.grain_share, 4),
                "grain_kg": round(realized_grain_kg, 4),
                "subsistence_grain_kg": round(home_kg, 4),
                "marketed_grain_kg": round(mkt_kg, 4),
                "grain_cash_sales_yuan": round(grain_cash_sales, 2),
                "in_kind_subsistence_yuan": round(in_kind_sub_val, 2),
                "non_mach_inputs_yuan": round(pl.non_mach_inputs_yuan, 2),
                "net_mach_service_yuan": round(pl.net_mach_service_yuan, 2),
                "service_subsidy_yuan": round(pl.service_subsidy_yuan, 2),
                "hired_days": round(pl.hired_days_used, 2),
                "hired_cost_yuan": round(pl.hired_cost_yuan, 2),
                "rent_paid": round(rent_paid[aid], 4),
                "rent_income": round(rent_income[aid], 4),
                "incremental_support_yuan": round(sup_inc, 4),
                "fertility_support_yuan": round(fert_inc, 4),
                "full_net_income_yuan": round(full_net_income, 2),
                "working_cap_end": round(wc_next, 2),
                "wealth_end": round(wealth_next, 2),
            }
            op.last_year_summary = rec_summary
            if collect_micro_ledger:
                actor_rows.append(rec_summary)

        # Persist end-of-year parcel & contract states
        self.contracts = active_contracts
        self.last_operator = next_last_op
        self.last_was_grain = next_was_grain
        self.last_was_idle = next_was_idle
        self.last_parcel_rents = parcel_rents_now
        self.last_parcel_rows = parcel_rows
        self.last_actor_rows = actor_rows

        total_grain_phys_mu = fate_incumbent_grain_mu + fate_strict_succession_grain_mu + fate_expanded_grain_mu
        total_cash_phys_mu = fate_leased_cash_mu + fate_own_cash_mu
        total_idle_phys_mu = fate_leased_unplanted_mu + fate_own_idle_mu
        total_planted_phys_mu = total_grain_phys_mu + total_cash_phys_mu
        total_grain_kg = incumbent_grain_kg + strict_succession_grain_kg + expanded_grain_kg + fsum(ext_grain_kg_by_actor.values())

        leased_area_mu = fsum(self.parcel_map[pid].area for pid in leased_pids)
        released_area_mu = fsum(self.parcel_map[pid].area for pid in released_pids)
        unmatched_area_mu = fsum(self.parcel_map[pid].area for pid in unmatched_pids if pid not in recovered_pids and pid not in leased_pids)
        total_rent_yuan = fsum(c.rent_yuan for c in active_contracts.values())
        unit_rent_mu = total_rent_yuan / leased_area_mu if leased_area_mu > 1e-9 else 0.0

        # Inclusion Diagnostic Index (IDI): combines smallholder farming persistence & income equity
        n_hh = max(1, len(self.households))
        cult_share = survey_cultivating_count / n_hh
        inc_arr = np.maximum(0.0, np.asarray(survey_net_income_list, dtype=np.float64))
        gini = (
            float(np.abs(inc_arr[:, None] - inc_arr[None, :]).sum() / (2.0 * len(inc_arr) * max(1.0, inc_arr.sum())))
            if len(inc_arr) > 1 and inc_arr.sum() > 0
            else 0.0
        )
        idi_score = 0.55 * cult_share + 0.45 * (1.0 - gini)

        prev_u_rent = self.observed.get("last_unit_rent", self.tech.ask_rent_yuan_phys_mu)

        self.observed = {
            "grain_kg": total_grain_kg / max(0.5, weather_mult),  # Weather-normalized structural grain capacity
            "grain_phys_mu": total_grain_phys_mu,
            "grain_cult_rate": total_grain_phys_mu / max(1.0, self.land_mu),
            "idle_rate": total_idle_phys_mu / max(1.0, self.land_mu),
            "unabsorbed_rate": unmatched_area_mu / max(1.0, released_area_mu) if released_area_mu > 1.0 else 0.0,
            "leased_rate": leased_area_mu / max(1.0, self.land_mu),
            "tight_liquidity_share": tight_liq_count / max(1, len(actor_ids)),
            "fiscal_exec_rate": actual_incremental_spending / max(1.0, budget_cap) if budget_cap > 0 else 1.0,
            "last_unit_rent": unit_rent_mu if leased_area_mu > 1.0 else prev_u_rent,
        }

        reg_summary = {
            "year": year,
            "region": self.name,
            "policy": instrument,
            "reservation": round(budget_cap, 4),
            "announced_rate_mu": round(sched_eff.rate_per_phys_mu, 4),
            "committed": round(committed_yuan, 4),
            "spending": round(actual_incremental_spending, 4),
            "background_fertility_yuan": round(fsum(fertility_received.values()), 4),
            "fertility_forfeited_mu": round(fertility_forfeited_mu, 4),
            "grain_kg": round(total_grain_kg, 4),
            "survey_600_grain_kg": round(survey_grain_kg, 4),
            "family_farm_grain_kg": round(org_grain_kg_by_cat["family_farm"], 4),
            "cooperative_grain_kg": round(org_grain_kg_by_cat["cooperative"], 4),
            "agri_company_grain_kg": round(org_grain_kg_by_cat["agri_company"], 4),
            "wheat_kg": round(named_grain_totals["wheat"], 2),
            "rice_kg": round(named_grain_totals["rice"], 2),
            "maize_kg": round(named_grain_totals["maize"], 2),
            "soybean_kg": round(named_grain_totals["soybean"], 2),
            "other_grain_kg": round(named_grain_totals["other_grain"], 2),
            "subsistence_grain_kg": round(survey_subsistence_kg, 2),
            "marketed_grain_kg": round(survey_marketed_kg, 2),
            "grain_cash_sales_yuan": round(survey_grain_cash_sales_yuan, 2),
            "in_kind_subsistence_yuan": round(survey_subsistence_value_yuan, 2),
            "physical_area_mu": round(self.land_mu, 4),
            "planted_physical_mu": round(total_planted_phys_mu, 4),
            "grain_physical_mu": round(total_grain_phys_mu, 4),
            "cash_physical_mu": round(total_cash_phys_mu, 4),
            "idle_physical_mu": round(total_idle_phys_mu, 4),
            "fate_1_incumbent_grain_mu": round(fate_incumbent_grain_mu, 4),
            "fate_2_strict_succession_grain_mu": round(fate_strict_succession_grain_mu, 4),
            "fate_3_expanded_grain_mu": round(fate_expanded_grain_mu, 4),
            "fate_3a_idle_recovered_grain_mu": round(fate_idle_recovered_grain_mu, 4),
            "fate_4_leased_cash_mu": round(fate_leased_cash_mu, 4),
            "fate_5_leased_unplanted_mu": round(fate_leased_unplanted_mu, 4),
            "fate_6_own_idle_mu": round(fate_own_idle_mu, 4),
            "own_cash_mu": round(fate_own_cash_mu, 4),
            "incumbent_grain_kg": round(incumbent_grain_kg, 4),
            "strict_succession_grain_kg": round(strict_succession_grain_kg, 4),
            "expanded_grain_kg": round(expanded_grain_kg, 4),
            "leased_area_mu": round(leased_area_mu, 4),
            "released_area_mu": round(released_area_mu, 4),
            "unmatched_area_mu": round(unmatched_area_mu, 4),
            "total_rent_yuan": round(total_rent_yuan, 4),
            "unit_rent_yuan_mu": round(unit_rent_mu, 4),
            "survey_cultivating_hh": survey_cultivating_count,
            "survey_mixed_hh": survey_mixed_count,
            "survey_mean_net_income_yuan": round(float(np.mean(survey_net_income_list)) if survey_net_income_list else 0.0, 2),
            "active_scale_orgs": sum(1 for o in self.organizations.values() if o.active_in_market),
            "idi_score": round(idi_score, 4),
            "cohort_tracked_mu": round(cohort_tracked_mu, 4),
            "cohort_retained_same_op_mu": round(cohort_retained_same_op_mu, 4),
            "cohort_still_grain_mu": round(cohort_still_grain_mu, 4),
            "cohort_grain_kg": round(cohort_grain_kg, 4),
            "weather_mult": weather_mult,
        }

        self._verify_region_conservation(reg_summary, parcel_rows, actor_rows, ext_inflow_mu)
        return reg_summary, snap_out

    def _verify_region_conservation(
        self,
        s: dict,
        parcel_rows: list[dict],
        actor_rows: list[dict],
        ext_inflow_mu: dict[str, float],
    ) -> None:
        req = lambda a, b: isclose(a, b, rel_tol=1e-8, abs_tol=1e-2)
        # 1. Exact 6-fate physical area partition conservation
        six_sum = (
            s["fate_1_incumbent_grain_mu"]
            + s["fate_2_strict_succession_grain_mu"]
            + s["fate_3_expanded_grain_mu"]
            + s["fate_4_leased_cash_mu"]
            + s["own_cash_mu"]
            + s["fate_5_leased_unplanted_mu"]
            + s["fate_6_own_idle_mu"]
        )
        assert req(six_sum, self.land_mu), f"6-fate partition {six_sum} != {self.land_mu}"
        assert req(s["planted_physical_mu"] + s["idle_physical_mu"], self.land_mu)

        # 2. Survey + Scale Organizations grain output conservation
        org_sum = s["family_farm_grain_kg"] + s["cooperative_grain_kg"] + s["agri_company_grain_kg"]
        assert req(s["survey_600_grain_kg"] + org_sum, s["grain_kg"]), "Agent grain sum != regional grain_kg"

        # 3. Fiscal hard boundary
        assert -1e-5 <= s["spending"] <= s["committed"] + 1e-4 <= s["reservation"] + 1e-3

        # 4. Micro ledger double-entry checks when enabled
        if parcel_rows and actor_rows and fsum(ext_inflow_mu.values()) == 0.0:
            assert len(parcel_rows) == len(self.parcels)
            assert req(fsum(p["physical_area"] for p in parcel_rows), self.land_mu)
            assert req(fsum(p["harvest_grain_kg"] for p in parcel_rows), s["grain_kg"])
            assert req(fsum(p["rent_yuan"] for p in parcel_rows), fsum(a["rent_paid"] for a in actor_rows))
            assert req(fsum(p["rent_yuan"] for p in parcel_rows), fsum(a["rent_income"] for a in actor_rows))
            assert req(fsum(p["incremental_support_yuan"] for p in parcel_rows), fsum(a["incremental_support_yuan"] for a in actor_rows))
            assert req(
                fsum(a["incremental_support_yuan"] for a in actor_rows) + s.get("verification_admin_cost_yuan", 0.0),
                s["spending"],
            )


class AgriSuccessionMesaModel(mesa.Model):
    """Formal Mesa 2.3.2 Multi-Agent Model uniting 4 regions, 600 NFPS households, 3 org types, and RL Government."""

    def __init__(
        self,
        seed: int,
        calibrated_regions: tuple[CalibratedRegionV2, ...],
        design: V2Design,
        *,
        policy_mode: str = "p0",
        fixed_action_idx: int = 0,
        learned_weights: dict[int, list[list[float]]] | None = None,
    ):
        super().__init__()
        self.master_seed = seed
        self.design = design
        self.year = 0
        self.running = True
        self.menu = build_policy_menu(calibrated_regions, design)

        self.government = GovernmentMesaAgent(
            "GOV_CENTRAL",
            self,
            policy_mode=policy_mode,
            fixed_action_idx=fixed_action_idx,
            learned_weights=learned_weights,
        )
        self.regions: tuple[RegionalSubEconomy, ...] = tuple(
            RegionalSubEconomy(self, len(calibrated_regions) * seed + i, reg)
            for i, reg in enumerate(calibrated_regions)
        )

        # Register all Mesa Agents for standard Mesa inspection & DataCollector
        self.all_mesa_agents: list[mesa.Agent] = [self.government]
        for r in self.regions:
            self.all_mesa_agents.extend(r.operators.values())

        self.last_joint_summary: dict = {}
        self.last_region_summaries: tuple[dict, ...] = ()
        self.history_joint: list[dict] = []
        self.history_regions: list[dict] = []

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "year": lambda m: m.year,
                "action_idx": lambda m: m.last_joint_summary.get("action_idx", 0),
                "grain_tons": lambda m: m.last_joint_summary.get("grain_kg", 0.0) / 1000.0,
                "spending_yuan": lambda m: m.last_joint_summary.get("spending", 0.0),
                "leased_area_mu": lambda m: m.last_joint_summary.get("leased_area_mu", 0.0),
                "strict_succession_mu": lambda m: m.last_joint_summary.get("fate_2_strict_succession_grain_mu", 0.0),
                "total_rent_yuan": lambda m: m.last_joint_summary.get("total_rent_yuan", 0.0),
                "idi_score": lambda m: m.last_joint_summary.get("idi_score", 0.0),
            }
        )

    def fork(self) -> "AgriSuccessionMesaModel":
        """Fast deep state clone for paired counterfactuals, Fixed-Contract forks, and RL rollouts."""
        clone = AgriSuccessionMesaModel.__new__(AgriSuccessionMesaModel)
        mesa.Model.__init__(clone)
        clone.master_seed = self.master_seed
        clone.design = self.design
        clone.year = self.year
        clone.running = self.running
        clone.menu = self.menu
        clone.government = self.government.clone_into(clone)
        clone.regions = tuple(r.clone_into(clone) for r in self.regions)
        clone.all_mesa_agents = [clone.government]
        for r in clone.regions:
            clone.all_mesa_agents.extend(r.operators.values())
        clone.last_joint_summary = self.last_joint_summary.copy()
        clone.last_region_summaries = self.last_region_summaries
        clone.history_joint = list(self.history_joint)
        clone.history_regions = list(self.history_regions)
        clone.datacollector = mesa.DataCollector(
            model_reporters=self.datacollector.model_reporters
        )
        return clone

    def extract_public_state_features(self) -> np.ndarray:
        """Lagged public macro feature vector (contains ZERO household private cash, hidden skill, or future weather)."""
        feats = [1.0]
        for r in self.regions:
            feats.extend(r.public_feature_list(self.design))
        return np.asarray(feats, dtype=np.float64)

    def select_transparent_rule_action(self) -> int:
        """Zero-black-box domain-expert dynamic heuristic reading lagged public state."""
        if self.design.annual_common_budget_yuan <= 0:
            return 0
        obs_list = [r.observed for r in self.regions]
        areas = [r.land_mu for r in self.regions]
        tot_a = fsum(areas)

        # 1) If a major region exhibits severe unabsorbed land release (>= 12%), deploy region-tilted scale takeover
        unabs_scores = [(obs_list[i]["unabsorbed_rate"] * areas[i], i) for i in range(len(self.regions))]
        max_unabs_score, worst_r = max(unabs_scores)
        if obs_list[worst_r]["unabsorbed_rate"] >= 0.12 and max_unabs_score >= 60.0:
            # In build_policy_menu: Actions 6..21 = r_idx * 4 + [producer_pre(0), scale_takeover_pre(1), service(2), producer_post(3)] + 6
            return 6 + worst_r * 4 + 1

        # 2) If area-weighted spring liquidity tightness is elevated (>= 0.28), deploy universal producer_pre (Action 1)
        avg_tight = fsum(obs_list[i]["tight_liquidity_share"] * areas[i] for i in range(len(self.regions))) / tot_a
        if avg_tight >= 0.28:
            return 1

        # 3) If East/West hilly smallholder idle rate exceeds Central/Northeast, deploy Hilly Service + Plains Scale Mix (Action 22)
        ew_idle = 0.5 * (obs_list[0]["idle_rate"] + obs_list[2]["idle_rate"])
        cn_idle = 0.5 * (obs_list[1]["idle_rate"] + obs_list[3]["idle_rate"])
        if ew_idle > cn_idle + 0.02:
            return 22

        # 4) Default balanced producer + central scale mix (Action 23)
        return 23

    def decide_government_action_index(self) -> int:
        mode = self.government.policy_mode
        if mode == "p0" or self.design.annual_common_budget_yuan <= 0:
            return 0
        if mode in ("uniform_fixed", "regional_fixed", "fixed"):
            return self.government.fixed_action_idx
        if mode == "transparent_rule":
            return self.select_transparent_rule_action()
        if mode in ("myopic", "gov_rl"):
            next_yr = self.year + 1
            if not self.government.learned_weights or next_yr not in self.government.learned_weights:
                return 1  # Safe producer_pre fallback (NEVER disaster cutoff P0!)
            phi = self.extract_public_state_features()
            w_mat = np.asarray(self.government.learned_weights[next_yr], dtype=np.float64)
            q_vals = phi @ w_mat
            q_vals[0] -= 1e-6  # Break exact ties in favor of active support when budget > 0
            return int(np.argmax(q_vals))
        raise ValueError(f"Unknown government policy_mode: {mode}")

    def step(
        self,
        explicit_action_idx: int | None = None,
        explicit_joint_action: JointActionV2 | None = None,
        *,
        collect_micro_ledger: bool = False,
        village_exchange: bool = False,
        fixed_contract_refs: tuple[FixedContractSnapshot, ...] | None = None,
        record_contract_snapshots: bool = False,
    ) -> tuple[dict, tuple[dict, ...], tuple[FixedContractSnapshot | None, ...]]:
        """Execute one complete Mesa annual cycle across all 4 regions under shared fiscal budget."""
        if explicit_joint_action is not None:
            act_idx = -1
            joint_act = explicit_joint_action
        else:
            act_idx = explicit_action_idx if explicit_action_idx is not None else self.decide_government_action_index()
            joint_act = self.menu[act_idx]

        self.year += 1
        reg_rows: list[dict] = []
        snaps: list[FixedContractSnapshot | None] = []

        # Annual fiscal budget strictly clears at year-end (zero cross-year slush hoarding)
        eff_joint_act = joint_act

        for i, (reg, (inst, cap)) in enumerate(zip(self.regions, eff_joint_act, strict=True)):
            f_ref = fixed_contract_refs[i] if fixed_contract_refs is not None else None
            r_sum, snap = reg.step_region(
                self.year,
                inst,
                float(cap),
                self.design,
                collect_micro_ledger=collect_micro_ledger,
                village_exchange=village_exchange,
                fixed_contract_ref=f_ref,
                record_contract_snapshot=record_contract_snapshots,
            )
            reg_rows.append(r_sum)
            snaps.append(snap)

        tot_res = fsum(r["reservation"] for r in reg_rows)
        tot_comm = fsum(r["committed"] for r in reg_rows)
        tot_spend = fsum(r["spending"] for r in reg_rows)
        assert tot_spend <= tot_comm + 1e-3 <= max(tot_res, float(self.design.annual_common_budget_yuan)) + 1e-2

        tot_leased = fsum(r["leased_area_mu"] for r in reg_rows)
        tot_rent = fsum(r["total_rent_yuan"] for r in reg_rows)
        tot_phys = fsum(r["physical_area_mu"] for r in reg_rows)

        joint_summary = {
            "year": self.year,
            "region": "joint",
            "policy_mode": self.government.policy_mode,
            "action_idx": act_idx,
            "reservation": round(tot_res, 4),
            "committed": round(tot_comm, 4),
            "spending": round(tot_spend, 4),
            "background_fertility_yuan": round(fsum(r["background_fertility_yuan"] for r in reg_rows), 4),
            "fertility_forfeited_mu": round(fsum(r["fertility_forfeited_mu"] for r in reg_rows), 4),
            "grain_kg": round(fsum(r["grain_kg"] for r in reg_rows), 4),
            "survey_600_grain_kg": round(fsum(r["survey_600_grain_kg"] for r in reg_rows), 4),
            "family_farm_grain_kg": round(fsum(r["family_farm_grain_kg"] for r in reg_rows), 4),
            "cooperative_grain_kg": round(fsum(r["cooperative_grain_kg"] for r in reg_rows), 4),
            "agri_company_grain_kg": round(fsum(r["agri_company_grain_kg"] for r in reg_rows), 4),
            "wheat_kg": round(fsum(r["wheat_kg"] for r in reg_rows), 2),
            "rice_kg": round(fsum(r["rice_kg"] for r in reg_rows), 2),
            "maize_kg": round(fsum(r["maize_kg"] for r in reg_rows), 2),
            "soybean_kg": round(fsum(r["soybean_kg"] for r in reg_rows), 2),
            "other_grain_kg": round(fsum(r["other_grain_kg"] for r in reg_rows), 2),
            "subsistence_grain_kg": round(fsum(r["subsistence_grain_kg"] for r in reg_rows), 2),
            "marketed_grain_kg": round(fsum(r["marketed_grain_kg"] for r in reg_rows), 2),
            "grain_cash_sales_yuan": round(fsum(r["grain_cash_sales_yuan"] for r in reg_rows), 2),
            "in_kind_subsistence_yuan": round(fsum(r["in_kind_subsistence_yuan"] for r in reg_rows), 2),
            "physical_area_mu": round(tot_phys, 4),
            "planted_physical_mu": round(fsum(r["planted_physical_mu"] for r in reg_rows), 4),
            "grain_physical_mu": round(fsum(r["grain_physical_mu"] for r in reg_rows), 4),
            "cash_physical_mu": round(fsum(r["cash_physical_mu"] for r in reg_rows), 4),
            "idle_physical_mu": round(fsum(r["idle_physical_mu"] for r in reg_rows), 4),
            "fate_1_incumbent_grain_mu": round(fsum(r["fate_1_incumbent_grain_mu"] for r in reg_rows), 4),
            "fate_2_strict_succession_grain_mu": round(fsum(r["fate_2_strict_succession_grain_mu"] for r in reg_rows), 4),
            "fate_3_expanded_grain_mu": round(fsum(r["fate_3_expanded_grain_mu"] for r in reg_rows), 4),
            "fate_3a_idle_recovered_grain_mu": round(fsum(r["fate_3a_idle_recovered_grain_mu"] for r in reg_rows), 4),
            "fate_4_leased_cash_mu": round(fsum(r["fate_4_leased_cash_mu"] for r in reg_rows), 4),
            "fate_5_leased_unplanted_mu": round(fsum(r["fate_5_leased_unplanted_mu"] for r in reg_rows), 4),
            "fate_6_own_idle_mu": round(fsum(r["fate_6_own_idle_mu"] for r in reg_rows), 4),
            "own_cash_mu": round(fsum(r["own_cash_mu"] for r in reg_rows), 4),
            "incumbent_grain_kg": round(fsum(r["incumbent_grain_kg"] for r in reg_rows), 4),
            "strict_succession_grain_kg": round(fsum(r["strict_succession_grain_kg"] for r in reg_rows), 4),
            "expanded_grain_kg": round(fsum(r["expanded_grain_kg"] for r in reg_rows), 4),
            "leased_area_mu": round(tot_leased, 4),
            "released_area_mu": round(fsum(r["released_area_mu"] for r in reg_rows), 4),
            "unmatched_area_mu": round(fsum(r["unmatched_area_mu"] for r in reg_rows), 4),
            "total_rent_yuan": round(tot_rent, 4),
            "unit_rent_yuan_mu": round(tot_rent / tot_leased if tot_leased > 1e-9 else 0.0, 4),
            "survey_cultivating_hh": sum(r["survey_cultivating_hh"] for r in reg_rows),
            "survey_mixed_hh": sum(r["survey_mixed_hh"] for r in reg_rows),
            "active_scale_orgs": sum(r["active_scale_orgs"] for r in reg_rows),
            "idi_score": round(fsum(r["idi_score"] * r["physical_area_mu"] for r in reg_rows) / tot_phys, 4),
            "cohort_tracked_mu": round(fsum(r["cohort_tracked_mu"] for r in reg_rows), 4),
            "cohort_retained_same_op_mu": round(fsum(r["cohort_retained_same_op_mu"] for r in reg_rows), 4),
            "cohort_still_grain_mu": round(fsum(r["cohort_still_grain_mu"] for r in reg_rows), 4),
            "cohort_grain_kg": round(fsum(r["cohort_grain_kg"] for r in reg_rows), 4),
        }

        self.last_joint_summary = joint_summary
        self.last_region_summaries = tuple(reg_rows)
        self.history_joint.append(joint_summary)
        self.history_regions.extend(reg_rows)
        self.government.decision_trace.append({
            "year": self.year,
            "action_idx": act_idx,
            "joint_action": joint_act,
            "spending": joint_summary["spending"],
            "grain_kg": joint_summary["grain_kg"],
        })
        self.datacollector.collect(self)
        return joint_summary, self.last_region_summaries, tuple(snaps)
