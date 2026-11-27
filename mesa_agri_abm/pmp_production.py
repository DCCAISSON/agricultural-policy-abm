"""Exact Closed-Form Analytical KKT Solver for MP-MAS + Howitt (1995) PMP + de Janvry (1991) (V2).

Combines <0.05 ms piecewise-exact KKT solution with:
1. Named grain crop portfolio decomposition (`wheat`, `rice`, `maize`, `soybean`, `other_grain`).
2. Subsistence staple self-consumption (`foodc`) retail-wedge valuation vs. marketed surplus cash.
3. Explicit custom machinery service cost line (`mach_*`) + non-fungible `service_trusteeship` price wedge.
4. Scale-threshold & single-operator capped subsidy schedule + RegMAS incremental parcel shadow-price bid.
"""

from dataclasses import dataclass
from math import fsum

from mesa_agri_abm.calibration import ActorBlueprint, NAMED_GRAINS, RegionTechV2
from mesa_agri_abm.config import V2Design


@dataclass(frozen=True, slots=True)
class SubsidySchedule:
    """Regional announced policy parameters for a given year."""

    instrument: str
    rate_per_phys_mu: float
    scale_threshold_mu: float
    single_operator_cap_mu: float
    enable_single_operator_cap: bool
    service_discount_share: float


@dataclass(frozen=True, slots=True)
class PMPPlanV2:
    operated_physical_mu: float
    planted_physical_mu: float
    idle_physical_mu: float
    sown_grain_mu: float
    sown_cash_mu: float
    grain_share: float
    expected_grain_kg: float
    named_grain_kg: dict[str, float]
    expected_subsistence_kg: float
    expected_marketed_kg: float
    expected_cash_rev_yuan: float
    non_mach_inputs_yuan: float
    gross_mach_service_yuan: float
    service_subsidy_yuan: float
    net_mach_service_yuan: float
    family_days_used: float
    hired_days_used: float
    hired_cost_yuan: float
    foregone_offfarm_yuan: float
    pre_cash_subsidy_yuan: float
    post_cash_subsidy_yuan: float
    net_upfront_cash_needed: float
    net_economic_surplus_yuan: float


def eligible_subsidized_grain_phys_mu(
    grain_phys_mu: float,
    new_takeover_grain_phys_mu: float,
    sched: SubsidySchedule,
) -> float:
    """Compute eligible physical grain mu under producer vs. scale/takeover rules & single-operator cap."""
    if sched.instrument in ("none", "service_trusteeship") or grain_phys_mu <= 1e-9:
        return 0.0
    if sched.instrument.startswith("producer_"):
        elig = grain_phys_mu
    elif sched.instrument.startswith("scale_takeover_"):
        if grain_phys_mu + 1e-7 >= sched.scale_threshold_mu:
            elig = grain_phys_mu
        else:
            elig = min(grain_phys_mu, new_takeover_grain_phys_mu)
    else:
        elig = 0.0
    if sched.enable_single_operator_cap:
        elig = min(elig, sched.single_operator_cap_mu)
    return max(0.0, elig)


def _zero_plan(total_phys_mu: float, actor: ActorBlueprint, design: V2Design) -> PMPPlanV2:
    staple_def_cost = (
        actor.subsistence_grain_kg * design.staple_retail_wedge * actor.grain_price_yuan_kg
        if (design.enable_subsistence_grain_split and actor.is_survey_household)
        else 0.0
    )
    return PMPPlanV2(
        operated_physical_mu=total_phys_mu,
        planted_physical_mu=0.0,
        idle_physical_mu=max(0.0, total_phys_mu),
        sown_grain_mu=0.0,
        sown_cash_mu=0.0,
        grain_share=0.0,
        expected_grain_kg=0.0,
        named_grain_kg={k: 0.0 for k in NAMED_GRAINS},
        expected_subsistence_kg=0.0,
        expected_marketed_kg=0.0,
        expected_cash_rev_yuan=0.0,
        non_mach_inputs_yuan=0.0,
        gross_mach_service_yuan=0.0,
        service_subsidy_yuan=0.0,
        net_mach_service_yuan=0.0,
        family_days_used=0.0,
        hired_days_used=0.0,
        hired_cost_yuan=0.0,
        foregone_offfarm_yuan=0.0,
        pre_cash_subsidy_yuan=0.0,
        post_cash_subsidy_yuan=0.0,
        net_upfront_cash_needed=0.0,
        net_economic_surplus_yuan=-staple_def_cost,
    )


def solve_actor_pmp_v2(
    operated_phys_mu: float,
    new_takeover_phys_mu: float,
    actor: ActorBlueprint,
    tech: RegionTechV2,
    design: V2Design,
    sched: SubsidySchedule,
    *,
    family_labor_avail: float,
    opp_wage_day: float,
    working_cap_avail: float,
    ref_share: float,
) -> PMPPlanV2:
    """Exact piecewise-closed-form KKT solver over crop share `theta` and sown scale `s_try`."""
    z_plan = _zero_plan(operated_phys_mu, actor, design)
    if operated_phys_mu <= 1e-9 or working_cap_avail < -1e-7:
        return z_plan

    mci = actor.mci
    s_max = mci * operated_phys_mu
    tk_share = min(1.0, max(0.0, new_takeover_phys_mu / operated_phys_mu))
    r_anchor = min(1.0, max(0.0, ref_share))

    y_g = actor.grain_yield_kg_sown_mu
    p_g = actor.grain_price_yuan_kg
    rev_g = y_g * p_g
    rev_c = tech.cash_rev_yuan_sown_mu

    # Custom machinery service cost & non-fungible service trusteeship discount
    coord_mult = 1.0 - actor.service_coord_saving
    gross_mach_g = (actor.grain_mach_yuan_sown_mu * coord_mult) if design.enable_machinery_service else 0.0
    gross_mach_c = (tech.cash_mach_service_yuan_sown_mu * coord_mult) if design.enable_machinery_service else 0.0

    is_service_pol = sched.instrument == "service_trusteeship" and sched.rate_per_phys_mu > 0.0
    if is_service_pol and design.enable_machinery_service:
        sm_boost = 1.0 if operated_phys_mu <= design.smallholder_service_cutoff_mu + 1e-6 else 0.80
        disc_per_sown_g = min(
            sched.service_discount_share * sm_boost * gross_mach_g,
            sched.rate_per_phys_mu / max(1.0, mci),
        )
    else:
        disc_per_sown_g = 0.0

    net_mach_g = max(0.0, gross_mach_g - disc_per_sown_g)
    net_mach_c = gross_mach_c
    c_g = actor.grain_non_mach_yuan_sown_mu + net_mach_g
    c_c = tech.cash_non_mach_input_yuan_sown_mu + net_mach_c

    lab_relief = (1.0 - design.machinery_labor_relief_share) if design.enable_machinery_service else 1.0
    l_g = actor.grain_labor_days_sown_mu * lab_relief
    l_c = tech.cash_labor_days_sown_mu * (1.0 - 0.5 * design.machinery_labor_relief_share if design.enable_machinery_service else 1.0)

    w_hire = tech.hired_wage_yuan_day
    w_res = design.slack_labor_reservation_ratio * w_hire
    w_comp = min(opp_wage_day, w_hire)
    slack_cap = family_labor_avail * (actor.slack_ag_days_year / max(1.0, actor.ag_days_year)) if actor.ag_days_year > 0 else 0.0

    is_pre_cash = sched.instrument in ("producer_pre", "scale_takeover_pre")
    is_post_cash = sched.instrument in ("producer_post", "scale_takeover_post")
    if sched.instrument.startswith("producer_"):
        marg_sub_phys = sched.rate_per_phys_mu
    elif sched.instrument.startswith("scale_takeover_"):
        marg_sub_phys = sched.rate_per_phys_mu if operated_phys_mu >= 0.85 * sched.scale_threshold_mu else (sched.rate_per_phys_mu * tk_share)
    else:
        marg_sub_phys = 0.0

    sub_sown_g = marg_sub_phys / mci
    pre_sown_g = (marg_sub_phys / mci) if is_pre_cash else 0.0

    sub_wedge_bonus = (
        design.staple_retail_wedge * rev_g
        if (design.enable_subsistence_grain_split and actor.is_survey_household and s_max * max(0.4, r_anchor) * y_g <= 1.25 * actor.subsistence_grain_kg)
        else 0.0
    )

    gamma_pmp = max(1.0, design.pmp_gamma_yuan_per_sown_mu)

    # Analytical KKT candidate crop shares across the 3 labor regimes + corners + liquidity/threshold kinks
    cand_thetas: set[float] = {0.0, 1.0, r_anchor}
    for w_sh in (w_res, w_comp, w_hire):
        pi_g = rev_g + sub_wedge_bonus + sub_sown_g - c_g - w_sh * l_g
        pi_c = rev_c - c_c - w_sh * l_c - tech.baseline_margin_gap
        th = r_anchor + (pi_g - pi_c) / gamma_pmp
        cand_thetas.add(min(1.0, max(0.0, th)))

    if sched.instrument.startswith("scale_takeover_") and operated_phys_mu >= sched.scale_threshold_mu:
        cand_thetas.add(min(1.0, max(0.0, sched.scale_threshold_mu / operated_phys_mu)))

    upfront_g = max(1.0, c_g - pre_sown_g)
    if working_cap_avail < c_c * s_max and c_c > upfront_g + 1e-6:
        th_fit = (c_c * s_max - working_cap_avail) / ((c_c - upfront_g) * s_max)
        if 0.0 < th_fit < 1.0:
            cand_thetas.add(th_fit)

    # Portfolio-spread management overhead for corporate/cooperative orgs; lumpy return-trip supervision days for households
    if not actor.is_survey_household:
        target_portfolio_mu = 140.0 if actor.actor_category == "cooperative" else (220.0 if actor.actor_category == "agri_company" else 100.0)
        fixed_overhead = actor.admin_overhead_yuan * min(1.0, operated_phys_mu / target_portfolio_mu)
    else:
        fixed_overhead = design.farm_fixed_overhead_days * w_comp
    w_dict = dict(actor.grain_weights)
    tot_w_y = max(1e-9, fsum(w_dict[k] * tech.named_grains[k].yield_kg_sown_mu for k in NAMED_GRAINS))

    best_plan = z_plan
    best_val = z_plan.net_economic_surplus_yuan

    for th in sorted(cand_thetas):
        l_bar = th * l_g + (1.0 - th) * l_c
        c_bar_approx = th * (c_g - pre_sown_g) + (1.0 - th) * c_c

        s_kink = family_labor_avail / max(1e-9, l_bar)
        s_cap_no_hire = s_max if c_bar_approx <= 1e-9 else (working_cap_avail / c_bar_approx)

        if s_cap_no_hire <= s_kink + 1e-9 or not design.allow_hired_labor:
            s_feas = min(s_max, s_kink, max(0.0, s_cap_no_hire))
        else:
            cash_at_kink = max(0.0, c_bar_approx * s_kink)
            rem_cash = max(0.0, working_cap_avail - cash_at_kink)
            c_with_hire = max(1e-6, c_bar_approx + design.upfront_hired_cash_share * w_hire * l_bar)
            s_feas = min(s_max, s_kink + rem_cash / c_with_hire)

        s_slack_kink = slack_cap / max(1e-9, l_bar)
        scales = [s_feas]
        if 1e-7 < s_slack_kink < s_feas - 1e-7:
            scales.append(s_slack_kink)
        if 1e-7 < s_kink < s_feas - 1e-7:
            scales.append(s_kink)

        for s_try in scales:
            if s_try <= 1e-8:
                continue
            xg = th * s_try
            xc = (1.0 - th) * s_try
            x_phys = s_try / mci
            phys_g = xg / mci
            new_tk_g = tk_share * phys_g

            lab_req = xg * l_g + xc * l_c
            if not design.allow_hired_labor and lab_req > family_labor_avail + 1e-6:
                continue
            fam_used = min(family_labor_avail, lab_req)
            hired_used = max(0.0, lab_req - family_labor_avail)
            hired_cost = w_hire * hired_used

            non_mach_cost = xg * actor.grain_non_mach_yuan_sown_mu + xc * tech.cash_non_mach_input_yuan_sown_mu
            gross_mach_cost = xg * gross_mach_g + xc * gross_mach_c
            svc_sub = xg * disc_per_sown_g
            net_mach_cost = max(0.0, gross_mach_cost - svc_sub)

            elig_g_mu = eligible_subsidized_grain_phys_mu(phys_g, new_tk_g, sched)
            cash_sub = elig_g_mu * sched.rate_per_phys_mu
            pre_sub = cash_sub if is_pre_cash else 0.0
            post_sub = cash_sub if is_post_cash else 0.0

            upfront_need = max(0.0, non_mach_cost + net_mach_cost + design.upfront_hired_cash_share * hired_cost - pre_sub)
            if upfront_need > working_cap_avail + 1e-6:
                # Exact single-step scale trim when single-operator subsidy cap binds above 250 mu
                ratio = max(0.0, working_cap_avail / upfront_need) * 0.9999
                if ratio <= 1e-5:
                    continue
                s_try *= ratio
                xg = th * s_try
                xc = (1.0 - th) * s_try
                x_phys = s_try / mci
                phys_g = xg / mci
                new_tk_g = tk_share * phys_g
                lab_req = xg * l_g + xc * l_c
                fam_used = min(family_labor_avail, lab_req)
                hired_used = max(0.0, lab_req - family_labor_avail)
                hired_cost = w_hire * hired_used
                non_mach_cost = xg * actor.grain_non_mach_yuan_sown_mu + xc * tech.cash_non_mach_input_yuan_sown_mu
                gross_mach_cost = xg * gross_mach_g + xc * gross_mach_c
                svc_sub = xg * disc_per_sown_g
                net_mach_cost = max(0.0, gross_mach_cost - svc_sub)
                elig_g_mu = eligible_subsidized_grain_phys_mu(phys_g, new_tk_g, sched)
                cash_sub = elig_g_mu * sched.rate_per_phys_mu
                pre_sub = cash_sub if is_pre_cash else 0.0
                post_sub = cash_sub if is_post_cash else 0.0
                upfront_need = max(0.0, non_mach_cost + net_mach_cost + design.upfront_hired_cash_share * hired_cost - pre_sub)
                if upfront_need > working_cap_avail + 1e-5:
                    continue

            slack_used = min(fam_used, slack_cap)
            comp_used = max(0.0, fam_used - slack_used)
            foregone_offfarm = w_res * slack_used + w_comp * comp_used

            diff_dev = th - r_anchor
            pmp_total = 0.5 * gamma_pmp * s_try * (diff_dev * diff_dev) + tech.baseline_margin_gap * xc

            exp_grain_kg = xg * y_g
            exp_cash_rev = xc * rev_c

            if design.enable_subsistence_grain_split and actor.is_survey_household:
                sub_kg = min(exp_grain_kg, actor.subsistence_grain_kg)
                mkt_kg = max(0.0, exp_grain_kg - sub_kg)
                deficit_kg = max(0.0, actor.subsistence_grain_kg - exp_grain_kg)
                staple_penalty = deficit_kg * design.staple_retail_wedge * p_g
            else:
                sub_kg = 0.0
                mkt_kg = exp_grain_kg
                staple_penalty = 0.0

            surplus = (
                exp_grain_kg * p_g
                + exp_cash_rev
                + pre_sub
                + post_sub
                - non_mach_cost
                - net_mach_cost
                - hired_cost
                - foregone_offfarm
                - fixed_overhead
                - pmp_total
                - staple_penalty
            )

            if surplus > best_val + 1e-9:
                best_val = surplus
                named_kg = {
                    k: exp_grain_kg * (w_dict[k] * tech.named_grains[k].yield_kg_sown_mu / tot_w_y)
                    for k in NAMED_GRAINS
                }
                best_plan = PMPPlanV2(
                    operated_physical_mu=operated_phys_mu,
                    planted_physical_mu=x_phys,
                    idle_physical_mu=max(0.0, operated_phys_mu - x_phys),
                    sown_grain_mu=xg,
                    sown_cash_mu=xc,
                    grain_share=th,
                    expected_grain_kg=exp_grain_kg,
                    named_grain_kg=named_kg,
                    expected_subsistence_kg=sub_kg,
                    expected_marketed_kg=mkt_kg,
                    expected_cash_rev_yuan=exp_cash_rev,
                    non_mach_inputs_yuan=non_mach_cost,
                    gross_mach_service_yuan=gross_mach_cost,
                    service_subsidy_yuan=svc_sub,
                    net_mach_service_yuan=net_mach_cost,
                    family_days_used=fam_used,
                    hired_days_used=hired_used,
                    hired_cost_yuan=hired_cost,
                    foregone_offfarm_yuan=foregone_offfarm,
                    pre_cash_subsidy_yuan=pre_sub,
                    post_cash_subsidy_yuan=post_sub,
                    net_upfront_cash_needed=upfront_need,
                    net_economic_surplus_yuan=surplus,
                )

    return best_plan


def compute_shadow_land_bid_v2(
    current_phys_mu: float,
    current_takeover_mu: float,
    lot_area_mu: float,
    is_new_contract_group: bool,
    actor: ActorBlueprint,
    tech: RegionTechV2,
    design: V2Design,
    sched: SubsidySchedule,
    *,
    family_labor_avail: float,
    opp_wage_day: float,
    working_cap_avail: float,
    ref_share: float,
    base_plan: PMPPlanV2 | None = None,
) -> float:
    """AgriPoliS + RegMAS (`getShadowPrice`) incremental land bid with fungibility vs. legacy rent-ban toggle."""
    if lot_area_mu <= 1e-9 or working_cap_avail <= 50.0:
        return 0.0

    bidding_sched = sched
    if not design.cash_subsidy_fungible_in_bidding and sched.instrument.endswith("_pre"):
        post_equiv = sched.instrument.replace("_pre", "_post")
        bidding_sched = SubsidySchedule(
            instrument=post_equiv,
            rate_per_phys_mu=sched.rate_per_phys_mu,
            scale_threshold_mu=sched.scale_threshold_mu,
            single_operator_cap_mu=sched.single_operator_cap_mu,
            enable_single_operator_cap=sched.enable_single_operator_cap,
            service_discount_share=sched.service_discount_share,
        )

    p0 = (
        base_plan
        if (base_plan is not None and design.cash_subsidy_fungible_in_bidding)
        else solve_actor_pmp_v2(
            current_phys_mu, current_takeover_mu, actor, tech, design, bidding_sched,
            family_labor_avail=family_labor_avail, opp_wage_day=opp_wage_day,
            working_cap_avail=working_cap_avail, ref_share=ref_share,
        )
    )
    p1 = solve_actor_pmp_v2(
        current_phys_mu + lot_area_mu,
        current_takeover_mu + lot_area_mu,
        actor, tech, design, bidding_sched,
        family_labor_avail=family_labor_avail, opp_wage_day=opp_wage_day,
        working_cap_avail=working_cap_avail, ref_share=ref_share,
    )

    if p1.planted_physical_mu - p0.planted_physical_mu < 0.50 * lot_area_mu:
        return 0.0

    setup_amort = (design.contract_setup_fee_yuan / max(1, design.lease_duration_years)) if is_new_contract_group else 0.0
    delta_surplus = (p1.net_economic_surplus_yuan - p0.net_economic_surplus_yuan) - setup_amort
    if delta_surplus <= 0.0:
        return 0.0

    setup_now = design.contract_setup_fee_yuan if is_new_contract_group else 0.0
    free_wc = working_cap_avail - p1.net_upfront_cash_needed - setup_now
    if free_wc <= 0.0:
        return 0.0

    own_cash_before_sub = working_cap_avail - max(
        0.0,
        p1.pre_cash_subsidy_yuan - (
            p1.non_mach_inputs_yuan + p1.net_mach_service_yuan + design.upfront_hired_cash_share * p1.hired_cost_yuan
        ),
    )
    solvency_rent_cap = max(0.0, own_cash_before_sub) / max(0.10, design.min_own_cash_rent_downpayment_share)

    shaded_bid = design.bid_shading_beta * (delta_surplus / lot_area_mu)
    return max(0.0, min(shaded_bid, free_wc / lot_area_mu, solvency_rent_cap / lot_area_mu))
