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
    order_price_bonus_yuan_kg: float = 0.0
    service_maturity_idx: float = 0.0
    village_pooled_saving: float = 0.0
    grain_covenant_floor: float = 0.0
    effective_detect_prob: float = 1.0


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
    order_subsidy_yuan: float = 0.0
    opportunistic_overclaim_yuan: float = 0.0
    effort_shading_rate: float = 0.0


def eligible_subsidized_grain_phys_mu(
    grain_phys_mu: float,
    new_takeover_grain_phys_mu: float,
    sched: SubsidySchedule,
) -> float:
    """Compute eligible physical grain mu under producer vs. scale/takeover rules & single-operator cap."""
    if sched.instrument in ("none", "service_trusteeship", "order_farming") or grain_phys_mu <= 1e-9:
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


def _effective_subsistence_kg(actor: ActorBlueprint, design: V2Design, rent_offset_yuan: float, p_g: float) -> float:
    if not (design.enable_subsistence_grain_split and actor.is_survey_household):
        return 0.0
    if not design.rent_offsets_subsistence_wedge:
        return actor.subsistence_grain_kg
    # Cash rent from leased-out land + stable migrant wage covers market grain purchases without double penalty
    migrant_cash_buffer = 0.25 * actor.offfarm_annual_base if actor.offfarm_income_share_2010 >= 0.45 else 0.0
    purchasable_kg = (max(0.0, rent_offset_yuan) + migrant_cash_buffer) / max(
        0.5, (1.0 + design.staple_retail_wedge) * p_g
    )
    return max(0.0, actor.subsistence_grain_kg - purchasable_kg)


def _zero_plan(
    total_phys_mu: float,
    actor: ActorBlueprint,
    design: V2Design,
    rent_offset_yuan: float = 0.0,
) -> PMPPlanV2:
    eff_sub_kg = _effective_subsistence_kg(actor, design, rent_offset_yuan, actor.grain_price_yuan_kg)
    staple_def_cost = eff_sub_kg * design.staple_retail_wedge * actor.grain_price_yuan_kg
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
        order_subsidy_yuan=0.0,
        opportunistic_overclaim_yuan=0.0,
        effort_shading_rate=0.0,
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
    was_cultivating: bool = True,
    sim_year: int = 1,
    rent_offset_yuan: float = 0.0,
    audit_belief: float = 1.0,
) -> PMPPlanV2:
    """Exact piecewise-closed-form KKT solver over crop share `theta` and sown scale `s_try`."""
    z_plan = _zero_plan(operated_phys_mu, actor, design, rent_offset_yuan=rent_offset_yuan)
    if operated_phys_mu <= 1e-9 or working_cap_avail < -1e-7:
        return z_plan

    mci = actor.mci
    s_max = mci * operated_phys_mu
    tk_share = min(1.0, max(0.0, new_takeover_phys_mu / operated_phys_mu))
    r_anchor = min(1.0, max(0.0, ref_share))

    svc_mat = sched.service_maturity_idx if design.enable_service_capacity_accumulation else 0.0
    y_g = actor.grain_yield_kg_sown_mu * (1.0 + design.service_maturity_yield_gain_max * svc_mat)
    p_g = actor.grain_price_yuan_kg
    rev_g = y_g * p_g
    rev_c = tech.cash_rev_yuan_sown_mu

    # Custom machinery service cost & non-fungible service trusteeship discount (+ contiguous/network savings)
    coord_mult = max(
        0.52,
        1.0
        - actor.service_coord_saving
        - sched.village_pooled_saving
        - design.service_maturity_cost_saving_max * svc_mat,
    )
    gross_mach_g = (actor.grain_mach_yuan_sown_mu * coord_mult) if design.enable_machinery_service else 0.0
    gross_mach_c = (tech.cash_mach_service_yuan_sown_mu * coord_mult) if design.enable_machinery_service else 0.0

    is_service_pol = sched.instrument == "service_trusteeship" and sched.rate_per_phys_mu > 0.0
    if is_service_pol and design.enable_machinery_service:
        sm_boost = 1.0 if operated_phys_mu <= design.smallholder_service_cutoff_mu + 1e-6 else 0.80
        # 农办经〔2019〕19号: Nascent networks have partial peak-season coverage; once service market matures
        # (svc_mat > 0.55), continued service subsidy faces infra-marginal substitution ("连续扶持退坡机制")
        if design.enable_service_capacity_accumulation:
            fleet_cov = (0.45 + 0.55 * min(1.0, svc_mat / 0.55)) * max(0.32, 1.0 - 0.90 * max(0.0, svc_mat - 0.55))
        else:
            fleet_cov = 1.0
        disc_per_sown_g = min(
            sched.service_discount_share * sm_boost * fleet_cov * gross_mach_g,
            (sched.rate_per_phys_mu * fleet_cov) / max(1.0, mci),
        )
    else:
        disc_per_sown_g = 0.0

    net_mach_g = max(0.0, gross_mach_g - disc_per_sown_g)
    net_mach_c = gross_mach_c
    c_g = actor.grain_non_mach_yuan_sown_mu + net_mach_g
    c_c = tech.cash_non_mach_input_yuan_sown_mu + net_mach_c

    eff_relief_sh = design.machinery_labor_relief_share + (0.14 * svc_mat if design.enable_service_capacity_accumulation else 0.0)
    lab_relief = (1.0 - eff_relief_sh) if design.enable_machinery_service else 1.0
    l_g = actor.grain_labor_days_sown_mu * lab_relief
    l_c = tech.cash_labor_days_sown_mu * (1.0 - 0.5 * eff_relief_sh if design.enable_machinery_service else 1.0)

    w_hire = tech.hired_wage_yuan_day
    w_res = design.slack_labor_reservation_ratio * w_hire
    w_comp = min(opp_wage_day, w_hire)
    slack_cap = family_labor_avail * (actor.slack_ag_days_year / max(1.0, actor.ag_days_year)) if actor.ag_days_year > 0 else 0.0

    is_pre_cash = sched.instrument in ("producer_pre", "scale_takeover_pre")
    is_post_cash = sched.instrument in ("producer_post", "scale_takeover_post")
    is_order_pol = sched.instrument == "order_farming" and sched.order_price_bonus_yuan_kg > 0.0

    if sched.instrument.startswith("producer_"):
        marg_sub_phys = sched.rate_per_phys_mu
    elif sched.instrument.startswith("scale_takeover_"):
        marg_sub_phys = sched.rate_per_phys_mu if operated_phys_mu >= 0.85 * sched.scale_threshold_mu else (sched.rate_per_phys_mu * tk_share)
    else:
        marg_sub_phys = 0.0

    order_bonus_sown = min(y_g * sched.order_price_bonus_yuan_kg, sched.rate_per_phys_mu / max(1.0, mci)) if is_order_pol else 0.0
    sub_sown_g = marg_sub_phys / mci + order_bonus_sown
    pre_sown_g = (marg_sub_phys / mci) if is_pre_cash else 0.0

    eff_sub_req_kg = _effective_subsistence_kg(actor, design, rent_offset_yuan, p_g)
    sub_wedge_bonus = (
        design.staple_retail_wedge * rev_g
        if (design.enable_subsistence_grain_split and actor.is_survey_household and eff_sub_req_kg > 1e-6 and s_max * max(0.4, r_anchor) * y_g <= 1.25 * eff_sub_req_kg)
        else 0.0
    )

    # Asymmetric-information detection expectation (for Adverse Selection & Moral Hazard when enabled)
    p_det_eff = (
        min(1.0, 0.75 * sched.effective_detect_prob + 0.25 * audit_belief)
        if design.enable_asymmetric_info_and_verification
        else 1.0
    )
    exp_clawback_mult = p_det_eff * (1.0 + design.clawback_fine_multiplier)
    opp_cash_arb = (
        max(0.0, (1.0 - exp_clawback_mult) * 0.85 * (marg_sub_phys / mci))
        if (design.enable_asymmetric_info_and_verification and actor.is_opportunistic_type and (is_pre_cash or is_post_cash))
        else 0.0
    )

    gamma_pmp = max(1.0, design.pmp_gamma_yuan_per_sown_mu)
    th_floor = sched.grain_covenant_floor if (not actor.is_survey_household or tk_share > 0.5) else 0.0

    # Analytical KKT candidate crop shares across the 3 labor regimes + corners + liquidity/threshold kinks
    cand_thetas: set[float] = {max(th_floor, 0.0), 1.0, max(th_floor, r_anchor)}
    if opp_cash_arb > 1.0:
        cand_thetas.add(max(th_floor, 0.15))
    for w_sh in (w_res, w_comp, w_hire):
        pi_g = rev_g + sub_wedge_bonus + sub_sown_g - c_g - w_sh * l_g
        pi_c = rev_c + opp_cash_arb - c_c - w_sh * l_c - tech.baseline_margin_gap
        th = r_anchor + (pi_g - pi_c) / gamma_pmp
        cand_thetas.add(min(1.0, max(th_floor, th)))

    if sched.instrument.startswith("scale_takeover_") and operated_phys_mu >= sched.scale_threshold_mu:
        cand_thetas.add(min(1.0, max(th_floor, sched.scale_threshold_mu / operated_phys_mu)))

    upfront_g = max(1.0, c_g - pre_sown_g)
    if working_cap_avail < c_c * s_max and c_c > upfront_g + 1e-6:
        th_fit = (c_c * s_max - working_cap_avail) / ((c_c - upfront_g) * s_max)
        if th_floor < th_fit < 1.0:
            cand_thetas.add(th_fit)

    # Portfolio-spread management overhead for corporate/cooperative orgs; lumpy return-trip supervision days for households
    if not actor.is_survey_household:
        target_portfolio_mu = 140.0 if actor.actor_category == "cooperative" else (220.0 if actor.actor_category == "agri_company" else 100.0)
        fixed_overhead = actor.admin_overhead_yuan * min(1.0, operated_phys_mu / target_portfolio_mu)
    else:
        fixed_overhead = design.farm_fixed_overhead_days * w_comp
        if design.enable_household_aging_and_exit_inertia:
            # Migrant cross-region return-trip & field coordination hurdle grows as urban jobs lock in
            if actor.migrant_days_2010 >= 150.0 and actor.offfarm_income_share_2010 >= 0.45:
                fixed_overhead += (
                    design.migrant_return_trip_days
                    * actor.offfarm_income_share_2010
                    * (1.0 + design.migrant_hurdle_growth_rate * max(0, sim_year - 1))
                    * w_comp
                )
            # Elderly households (>63) managing plots alone face extra health/supervision hurdle
            cur_age = actor.head_age_2010 + sim_year
            if cur_age >= 63.0 and operated_phys_mu <= 12.0:
                fixed_overhead += 0.45 * (cur_age - 62.0) * w_hire
            # State-dependent restart cost when household had already exited cultivation in prior year
            if not was_cultivating:
                fixed_overhead += design.restart_cultivation_fixed_yuan + design.restart_cultivation_fixed_days * w_comp

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

            # Endogenous Adverse Selection (overclaiming non-grain leased area) & Moral Hazard (input/service effort shading)
            overclaim_yuan = 0.0
            shading_rate = 0.0
            if design.enable_asymmetric_info_and_verification:
                if (
                    actor.is_opportunistic_type
                    and (is_pre_cash or is_post_cash)
                    and sched.rate_per_phys_mu > 0.0
                    and exp_clawback_mult < 0.92
                ):
                    non_g_phys = max(0.0, x_phys - phys_g)
                    claimable_gap = non_g_phys * 0.85
                    if sched.enable_single_operator_cap:
                        claimable_gap = min(claimable_gap, max(0.0, sched.single_operator_cap_mu - elig_g_mu))
                    # Under weak local inspection, nominal claimed grain area (phys_g + claimable_gap) clears scale threshold
                    if sched.instrument.startswith("scale_takeover_") and (phys_g + claimable_gap) < sched.scale_threshold_mu:
                        claimable_gap = min(claimable_gap, max(0.0, new_takeover_phys_mu - new_tk_g))
                    overclaim_yuan = max(0.0, claimable_gap * sched.rate_per_phys_mu)
                if actor.compliance_propensity < 0.95 and (cash_sub + overclaim_yuan > 0.0 or svc_sub > 0.0) and exp_clawback_mult < 0.95:
                    shading_incentive = max(0.0, 1.0 - exp_clawback_mult) * (1.0 - actor.compliance_propensity)
                    shading_rate = min(0.28, 0.35 * shading_incentive)

            eff_non_mach_cost = non_mach_cost * (1.0 - 0.55 * shading_rate)
            yield_drag = 1.0 - 0.52 * shading_rate
            if design.enable_asymmetric_info_and_verification and is_service_pol and sched.effective_detect_prob < 0.65:
                # Dual moral hazard: unverified custom machinery providers shade tillage/harvest quality
                yield_drag *= (1.0 - 0.055 * (0.65 - sched.effective_detect_prob))

            slack_used = min(fam_used, slack_cap)
            comp_used = max(0.0, fam_used - slack_used)
            foregone_offfarm = w_res * slack_used + w_comp * comp_used

            diff_dev = th - r_anchor
            pmp_total = 0.5 * gamma_pmp * s_try * (diff_dev * diff_dev) + tech.baseline_margin_gap * xc

            exp_grain_kg = xg * y_g * max(0.60, yield_drag)
            exp_cash_rev = xc * rev_c

            if design.enable_subsistence_grain_split and actor.is_survey_household:
                sub_kg = min(exp_grain_kg, actor.subsistence_grain_kg)
                mkt_kg = max(0.0, exp_grain_kg - sub_kg)
                deficit_kg = max(0.0, eff_sub_req_kg - exp_grain_kg)
                staple_penalty = deficit_kg * design.staple_retail_wedge * p_g
            else:
                sub_kg = 0.0
                mkt_kg = exp_grain_kg
                staple_penalty = 0.0

            order_sub = min(mkt_kg * sched.order_price_bonus_yuan_kg, phys_g * sched.rate_per_phys_mu) if is_order_pol else 0.0
            exp_overclaim_net = max(0.0, (1.0 - exp_clawback_mult) * overclaim_yuan)

            surplus = (
                exp_grain_kg * p_g
                + exp_cash_rev
                + pre_sub
                + post_sub
                + order_sub
                + exp_overclaim_net
                - eff_non_mach_cost
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
                    non_mach_inputs_yuan=eff_non_mach_cost,
                    gross_mach_service_yuan=gross_mach_cost,
                    service_subsidy_yuan=svc_sub,
                    net_mach_service_yuan=net_mach_cost,
                    family_days_used=fam_used,
                    hired_days_used=hired_used,
                    hired_cost_yuan=hired_cost,
                    foregone_offfarm_yuan=foregone_offfarm,
                    pre_cash_subsidy_yuan=pre_sub,
                    post_cash_subsidy_yuan=post_sub,
                    net_upfront_cash_needed=max(0.0, eff_non_mach_cost + net_mach_cost + design.upfront_hired_cash_share * hired_cost - pre_sub),
                    net_economic_surplus_yuan=surplus,
                    order_subsidy_yuan=order_sub,
                    opportunistic_overclaim_yuan=overclaim_yuan,
                    effort_shading_rate=shading_rate,
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
    was_cultivating: bool = True,
    sim_year: int = 1,
    audit_belief: float = 1.0,
    setup_fee_override: float | None = None,
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
            order_price_bonus_yuan_kg=sched.order_price_bonus_yuan_kg,
            service_maturity_idx=sched.service_maturity_idx,
            village_pooled_saving=sched.village_pooled_saving,
            grain_covenant_floor=sched.grain_covenant_floor,
            effective_detect_prob=sched.effective_detect_prob,
        )

    p0 = (
        base_plan
        if (base_plan is not None and design.cash_subsidy_fungible_in_bidding)
        else solve_actor_pmp_v2(
            current_phys_mu, current_takeover_mu, actor, tech, design, bidding_sched,
            family_labor_avail=family_labor_avail, opp_wage_day=opp_wage_day,
            working_cap_avail=working_cap_avail, ref_share=ref_share,
            was_cultivating=was_cultivating, sim_year=sim_year, audit_belief=audit_belief,
        )
    )
    p1 = solve_actor_pmp_v2(
        current_phys_mu + lot_area_mu,
        current_takeover_mu + lot_area_mu,
        actor, tech, design, bidding_sched,
        family_labor_avail=family_labor_avail, opp_wage_day=opp_wage_day,
        working_cap_avail=working_cap_avail, ref_share=ref_share,
        was_cultivating=was_cultivating, sim_year=sim_year, audit_belief=audit_belief,
    )

    if p1.planted_physical_mu - p0.planted_physical_mu < 0.50 * lot_area_mu:
        return 0.0

    fee_base = design.contract_setup_fee_yuan if setup_fee_override is None else setup_fee_override
    lease_yrs = design.lease_years_for_category(actor.actor_category)
    setup_amort = (fee_base / max(1, lease_yrs)) if is_new_contract_group else 0.0
    delta_surplus = (p1.net_economic_surplus_yuan - p0.net_economic_surplus_yuan) - setup_amort
    if delta_surplus <= 0.0:
        return 0.0

    setup_now = fee_base if is_new_contract_group else 0.0
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
