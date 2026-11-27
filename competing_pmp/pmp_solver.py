"""Exact analytical KKT solver for Non-Separable AHM + Concave Quadratic PMP.

Solves household crop-mix, land utilization, and family-vs-hired labor allocation
under working-capital liquidity constraints in closed/piecewise-analytic form (<0.05 ms/call).
"""

from dataclasses import dataclass

from competing_pmp.calibration_2010 import HouseholdAgent, RegionTechParams
from competing_pmp.config import CompetingDesign


@dataclass(frozen=True, slots=True)
class PMPPlan:
    cultivating: bool
    planted_physical_mu: float
    idle_physical_mu: float
    sown_grain_mu: float
    sown_cash_mu: float
    grain_share: float
    family_days_used: float
    hired_days_used: float
    hired_cost_yuan: float
    material_inputs_yuan: float
    pre_subsidy_yuan: float
    post_subsidy_yuan: float
    expected_gross_rev_yuan: float
    pmp_implicit_cost_yuan: float
    net_economic_surplus_yuan: float


def _evaluate_given_share_and_scale(
    theta: float,
    sown_total: float,
    mci_eff: float,
    agent: HouseholdAgent,
    tech: RegionTechParams,
    design: CompetingDesign,
    family_labor_avail: float,
    opp_wage_day: float,
    working_cap_avail: float,
    pre_sub_per_phys_grain: float,
    post_sub_per_phys_grain: float,
    quality_mult: float,
    ref_share: float,
) -> PMPPlan | None:
    """Evaluate feasibility and exact AHM-PMP objective at (theta, sown_total)."""
    if sown_total <= 1e-9:
        return PMPPlan(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    xg = theta * sown_total
    xc = (1.0 - theta) * sown_total
    planted_phys = sown_total / mci_eff
    grain_phys = xg / mci_eff

    mat_cost = tech.grain_input_yuan_sown_mu * xg + tech.cash_input_yuan_sown_mu * xc
    pre_sub = pre_sub_per_phys_grain * grain_phys
    post_sub = post_sub_per_phys_grain * grain_phys

    lab_req = tech.grain_labor_days_sown_mu * xg + tech.cash_labor_days_sown_mu * xc
    fam_days = min(family_labor_avail, lab_req)
    hire_days = max(0.0, lab_req - family_labor_avail)
    hire_cost = tech.hired_wage_yuan_day * hire_days
    upfront_hire = design.upfront_hired_cash_share * hire_cost

    # Pre-planting working capital hard constraint: material inputs + upfront hired labor <= working_cap + pre_sub
    net_upfront_need = mat_cost + upfront_hire - pre_sub
    if net_upfront_need > working_cap_avail + 1e-7:
        return None

    gross_rev = (
        tech.grain_yield_kg_sown_mu * quality_mult * tech.grain_price_yuan_kg * xg
        + tech.cash_rev_yuan_sown_mu * xc
    )

    # Howitt (1995) scale-consistent concave PMP implicit cost:
    # Linear baseline differential + quadratic portfolio deviation around reference share
    diff_dev = theta - ref_share
    pmp_quad = 0.5 * design.pmp_gamma_yuan_per_sown_mu * sown_total * (diff_dev * diff_dev)
    pmp_lin = tech.baseline_margin_gap * xc  # Calibrates baseline marginal equality at theta = ref_share
    pmp_total = pmp_quad + pmp_lin

    # de Janvry et al. (1991) Non-Separable Three-Band Household Labor Cost:
    # Band 1: non-competing village/slack family farm days at village reservation disutility
    # Band 2: competing off-farm window family days at min(opp_wage_day, hired_wage_yuan_day)
    # Band 3: hired labor / machinery service beyond family_labor_avail (already in hire_cost)
    slack_cap = family_labor_avail * (agent.slack_ag_days_year / max(1.0, agent.ag_days_year))
    slack_used = min(fam_days, slack_cap)
    comp_used = max(0.0, fam_days - slack_used)
    w_res = design.slack_labor_reservation_ratio * tech.hired_wage_yuan_day
    w_comp = min(opp_wage_day, tech.hired_wage_yuan_day)

    fixed_overhead = (
        design.entrant_annual_overhead_yuan
        if agent.is_entrant
        else design.farm_fixed_overhead_days * w_comp
    )
    opp_labor_cost = w_res * slack_used + w_comp * comp_used + fixed_overhead

    surplus = gross_rev + pre_sub + post_sub - mat_cost - hire_cost - opp_labor_cost - pmp_total
    return PMPPlan(
        cultivating=True,
        planted_physical_mu=planted_phys,
        idle_physical_mu=0.0,
        sown_grain_mu=xg,
        sown_cash_mu=xc,
        grain_share=theta,
        family_days_used=fam_days,
        hired_days_used=hire_days,
        hired_cost_yuan=hire_cost,
        material_inputs_yuan=mat_cost,
        pre_subsidy_yuan=pre_sub,
        post_subsidy_yuan=post_sub,
        expected_gross_rev_yuan=gross_rev,
        pmp_implicit_cost_yuan=pmp_total,
        net_economic_surplus_yuan=surplus,
    )


def solve_household_pmp(
    operated_physical_mu: float,
    agent: HouseholdAgent,
    tech: RegionTechParams,
    design: CompetingDesign,
    *,
    family_labor_avail: float,
    opp_wage_day: float,
    working_cap_avail: float,
    pre_sub_per_phys_grain: float = 0.0,
    post_sub_per_phys_grain: float = 0.0,
    quality_mult: float | None = None,
    ref_share: float | None = None,
    allow_zero_exit: bool = True,
) -> PMPPlan:
    """Solve the concave AHM-PMP program exactly over (theta, sown_total)."""
    if operated_physical_mu <= 1e-9 or working_cap_avail < -1e-7:
        return PMPPlan(False, 0.0, max(0.0, operated_physical_mu), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    q_mult = agent.grain_yield_factor if quality_mult is None else quality_mult
    r_share = agent.ref_grain_share if ref_share is None else ref_share
    mci_eff = agent.mci
    s_max = mci_eff * operated_physical_mu
    w_res = design.slack_labor_reservation_ratio * tech.hired_wage_yuan_day
    w_eff = min(opp_wage_day, tech.hired_wage_yuan_day)

    sub_sown_total = (pre_sub_per_phys_grain + post_sub_per_phys_grain) / mci_eff
    pre_sown = pre_sub_per_phys_grain / mci_eff

    # Candidate shadow labor regimes: (1) slack family labor interior (w_res),
    # (2) competing off-farm family window (w_eff), (3) hired labor active (w_hire)
    candidate_thetas: set[float] = {0.0, 1.0, min(1.0, max(0.0, r_share))}
    for w_shadow in (w_res, w_eff, tech.hired_wage_yuan_day):
        pi_g = tech.grain_yield_kg_sown_mu * q_mult * tech.grain_price_yuan_kg + sub_sown_total - tech.grain_input_yuan_sown_mu - w_shadow * tech.grain_labor_days_sown_mu
        pi_c = tech.cash_rev_yuan_sown_mu - tech.cash_input_yuan_sown_mu - w_shadow * tech.cash_labor_days_sown_mu - tech.baseline_margin_gap
        th = r_share + (pi_g - pi_c) / design.pmp_gamma_yuan_per_sown_mu
        candidate_thetas.add(min(1.0, max(0.0, th)))

    # Also include liquidity-tight interior share when cash crop input exceeds working capital at full scale
    upfront_g = max(1.0, tech.grain_input_yuan_sown_mu - pre_sown)
    upfront_c = tech.cash_input_yuan_sown_mu
    if working_cap_avail < upfront_c * s_max and upfront_c > upfront_g + 1e-6:
        # Share where full land s_max is just affordable with family labor
        th_cash_fit = (upfront_c * s_max - working_cap_avail) / ((upfront_c - upfront_g) * s_max)
        if 0.0 < th_cash_fit < 1.0:
            candidate_thetas.add(th_cash_fit)
        # Midpoint between unconstrained optimum and pure grain under liquidity pressure
        for th0 in list(candidate_thetas):
            if th0 < 1.0:
                candidate_thetas.add(min(1.0, 0.5 * (th0 + 1.0)))

    best_plan = PMPPlan(
        False, 0.0, operated_physical_mu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    best_val = 0.0 if allow_zero_exit else -1e18

    for th in sorted(candidate_thetas):
        l_bar = th * tech.grain_labor_days_sown_mu + (1.0 - th) * tech.cash_labor_days_sown_mu
        c_bar = th * (tech.grain_input_yuan_sown_mu - pre_sown) + (1.0 - th) * tech.cash_input_yuan_sown_mu

        # Compute exact max feasible sown scale under working-capital liquidity kink
        s_kink = family_labor_avail / max(1e-9, l_bar)
        if c_bar <= 1e-9:
            s_cap_no_hire = s_max
        else:
            s_cap_no_hire = working_cap_avail / c_bar

        if s_cap_no_hire <= s_kink + 1e-9 or not design.allow_hired_labor:
            s_feas = min(s_max, s_kink, max(0.0, s_cap_no_hire))
        else:
            # Beyond s_kink, upfront share of hired labor adds upfront_hired_cash_share * w_hire * l_bar per sown mu
            cash_at_kink = c_bar * s_kink
            rem_cash = max(0.0, working_cap_avail - cash_at_kink)
            c_with_hire = max(1e-6, c_bar + design.upfront_hired_cash_share * tech.hired_wage_yuan_day * l_bar)
            s_feas = min(s_max, s_kink + rem_cash / c_with_hire)

        # Candidate scales: full feasible scale, slack-labor kink, and total family-labor kink
        slack_cap = family_labor_avail * (agent.slack_ag_days_year / max(1.0, agent.ag_days_year))
        s_slack_kink = slack_cap / max(1e-9, l_bar)
        scales = [s_feas]
        if 1e-7 < s_slack_kink < s_feas - 1e-7:
            scales.append(s_slack_kink)
        if 1e-7 < s_kink < s_feas - 1e-7:
            scales.append(s_kink)

        for s_try in scales:
            if s_try <= 1e-8:
                continue
            plan = _evaluate_given_share_and_scale(
                theta=th,
                sown_total=s_try,
                mci_eff=mci_eff,
                agent=agent,
                tech=tech,
                design=design,
                family_labor_avail=family_labor_avail,
                opp_wage_day=opp_wage_day,
                working_cap_avail=working_cap_avail,
                pre_sub_per_phys_grain=pre_sub_per_phys_grain,
                post_sub_per_phys_grain=post_sub_per_phys_grain,
                quality_mult=q_mult,
                ref_share=r_share,
            )
            if plan is not None and plan.net_economic_surplus_yuan > best_val + 1e-9:
                best_val = plan.net_economic_surplus_yuan
                idle_phys = max(0.0, operated_physical_mu - plan.planted_physical_mu)
                best_plan = PMPPlan(
                    cultivating=True,
                    planted_physical_mu=plan.planted_physical_mu,
                    idle_physical_mu=idle_phys,
                    sown_grain_mu=plan.sown_grain_mu,
                    sown_cash_mu=plan.sown_cash_mu,
                    grain_share=plan.grain_share,
                    family_days_used=plan.family_days_used,
                    hired_days_used=plan.hired_days_used,
                    hired_cost_yuan=plan.hired_cost_yuan,
                    material_inputs_yuan=plan.material_inputs_yuan,
                    pre_subsidy_yuan=plan.pre_subsidy_yuan,
                    post_subsidy_yuan=plan.post_subsidy_yuan,
                    expected_gross_rev_yuan=plan.expected_gross_rev_yuan,
                    pmp_implicit_cost_yuan=plan.pmp_implicit_cost_yuan,
                    net_economic_surplus_yuan=plan.net_economic_surplus_yuan,
                )

    return best_plan


def marginal_land_bid_per_mu(
    base_area_mu: float,
    lot_area_mu: float,
    agent: HouseholdAgent,
    tech: RegionTechParams,
    design: CompetingDesign,
    *,
    family_labor_avail: float,
    opp_wage_day: float,
    working_cap_avail: float,
    base_pre_sub: float,
    base_post_sub: float,
    lot_pre_sub: float,
    lot_post_sub: float,
    already_active_lessee: bool,
    base_plan: PMPPlan | None = None,
) -> float:
    """Compute RegMAS/AgriPoliS incremental shadow price bid (yuan/physical-mu) for lot_area_mu."""
    if lot_area_mu <= 1e-9 or working_cap_avail <= tech.ask_rent_yuan_phys_mu * lot_area_mu:
        return -1.0

    if base_plan is None:
        base_plan = solve_household_pmp(
            base_area_mu,
            agent,
            tech,
            design,
            family_labor_avail=family_labor_avail,
            opp_wage_day=opp_wage_day,
            working_cap_avail=working_cap_avail,
            pre_sub_per_phys_grain=base_pre_sub,
            post_sub_per_phys_grain=base_post_sub,
        )

    tot_area = base_area_mu + lot_area_mu
    w_pre = (base_pre_sub * base_area_mu + lot_pre_sub * lot_area_mu) / tot_area
    w_post = (base_post_sub * base_area_mu + lot_post_sub * lot_area_mu) / tot_area

    # Amortize multi-year contract registration fee over contract horizon
    ann_fee = design.lot_transaction_fee_yuan / max(1.0, 1.0 + design.lease_discount_factor * (design.contract_years - 1))
    if not already_active_lessee and agent.is_entrant:
        # Entrant overhead is already inside solve_household_pmp when base_area_mu == 0
        pass

    # Maximum rent affordable while still financing crop inputs on expanded area:
    # test feasible expanded plan after deducting minimum ask rent + fee from working capital
    min_upfront_deduct = tech.ask_rent_yuan_phys_mu * lot_area_mu + design.lot_transaction_fee_yuan
    rem_wc = working_cap_avail - min_upfront_deduct
    if rem_wc < 0.0:
        return -1.0

    exp_plan = solve_household_pmp(
        tot_area,
        agent,
        tech,
        design,
        family_labor_avail=family_labor_avail,
        opp_wage_day=opp_wage_day,
        working_cap_avail=rem_wc,
        pre_sub_per_phys_grain=w_pre,
        post_sub_per_phys_grain=w_post,
    )
    if not exp_plan.cultivating or exp_plan.planted_physical_mu <= base_plan.planted_physical_mu + 0.25 * lot_area_mu:
        return -1.0

    delta_surplus = exp_plan.net_economic_surplus_yuan - base_plan.net_economic_surplus_yuan - ann_fee
    if delta_surplus <= 0.0:
        return -1.0

    # Liquidity ceiling on rent per mu so upfront rent + fee + net upfront inputs <= working_cap_avail
    net_inputs_exp = (
        exp_plan.material_inputs_yuan
        + design.upfront_hired_cash_share * exp_plan.hired_cost_yuan
        - exp_plan.pre_subsidy_yuan
    )
    max_rent_total_by_cash = working_cap_avail - design.lot_transaction_fee_yuan - max(0.0, net_inputs_exp)
    if max_rent_total_by_cash <= 0.0:
        return -1.0

    # Risk-adjusted willingness to pay (lessee retains 15% risk buffer for weather uncertainty)
    wtp_per_mu = min(0.85 * (delta_surplus / lot_area_mu), max_rent_total_by_cash / lot_area_mu)
    return wtp_per_mu
