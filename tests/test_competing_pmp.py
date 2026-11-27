"""Unit and accounting-conservation tests for the isolated competing_pmp model."""

from math import fsum, isclose
import numpy as np

from competing_pmp.calibration_2010 import CalibratedRegion, HouseholdAgent, Parcel, RegionTechParams
from competing_pmp.config import default_design
from competing_pmp.pmp_solver import marginal_land_bid_per_mu, solve_household_pmp
from competing_pmp.economy import RegionalPMPEconomy, verify_annual_ledger
from competing_pmp.budget_and_rl import (
    create_worlds,
    extract_planner_features,
    fit_backward_fqi,
    fork_all_actions,
    make_shared_budget,
    select_planner_action,
)


def _sample_tech(region: str = "中部") -> RegionTechParams:
    return RegionTechParams(
        region=region,
        grain_yield_kg_sown_mu=415.0,
        grain_price_yuan_kg=1.85,
        grain_input_yuan_sown_mu=225.0,
        grain_labor_days_sown_mu=13.5,
        cash_rev_yuan_sown_mu=1260.0,
        cash_input_yuan_sown_mu=403.2,
        cash_labor_days_sown_mu=22.28,
        hired_wage_yuan_day=50.0,
        ask_rent_yuan_phys_mu=175.0,
        max_mci=2.0,
        baseline_grain_share=0.65,
        baseline_margin_gap=95.0,
        turnover_pool_share=0.18,
        village_net_inflow_rate=0.02,
    )


def _sample_agent(
    aid: str = "H0001",
    mci: float = 1.8,
    ref_share: float = 0.62,
    work_cap: float = 4500.0,
    ag_days: float = 180.0,
    area: float = 6.0,
    is_entrant: bool = False,
) -> HouseholdAgent:
    return HouseholdAgent(
        id=aid,
        region="中部",
        province=41,
        submarket=41,
        is_entrant=is_entrant,
        workers=2.0,
        ag_days_year=ag_days,
        slack_ag_days_year=round(0.75 * ag_days, 2),
        offfarm_wage_day=24.0,
        offfarm_annual_base=18000.0,
        savings_stock=15000.0,
        working_capital_init=work_cap,
        consumption_base=12000.0,
        mci=mci,
        grain_yield_factor=1.0,
        ref_grain_share=ref_share,
        initial_cultivating=True,
        own_physical_mu=area,
        verified_tradable_mu=area,
    )


def test_pmp_interior_mixed_cropping_and_smooth_subsidy_response():
    """Verify concave PMP yields interior mixed cropping and smooth (non-cliff) subsidy elasticity."""
    design = default_design()
    tech = _sample_tech()
    agent = _sample_agent(ref_share=0.60, work_cap=8000.0, ag_days=220.0, area=6.0)

    p0 = solve_household_pmp(
        6.0, agent, tech, design,
        family_labor_avail=220.0, opp_wage_day=22.5, working_cap_avail=8000.0,
    )
    assert p0.cultivating
    assert 0.45 < p0.grain_share < 0.75, f"Expected interior mixed share, got {p0.grain_share}"
    assert p0.sown_grain_mu > 0 and p0.sown_cash_mu > 0

    p_sub40 = solve_household_pmp(
        6.0, agent, tech, design,
        family_labor_avail=220.0, opp_wage_day=22.5, working_cap_avail=8000.0,
        pre_sub_per_phys_grain=40.0,
    )
    p_sub85 = solve_household_pmp(
        6.0, agent, tech, design,
        family_labor_avail=220.0, opp_wage_day=22.5, working_cap_avail=8000.0,
        pre_sub_per_phys_grain=85.0,
    )
    # Strictly monotone, smooth response without 100% corner jump
    assert p0.grain_share < p_sub40.grain_share < p_sub85.grain_share < 0.90


def test_pre_vs_post_subsidy_under_tight_working_capital():
    """Pre-planting subsidy relaxes binding input liquidity; post-harvest subsidy cannot."""
    design = default_design()
    tech = _sample_tech()
    # Tight working capital: 1,200 yuan for 6 mu * 1.8 MCI = 10.8 sown mu (grain inputs = 2,430 yuan)
    agent = _sample_agent(ref_share=0.85, work_cap=1200.0, ag_days=240.0, area=6.0)

    p_post = solve_household_pmp(
        6.0, agent, tech, design,
        family_labor_avail=240.0, opp_wage_day=20.0, working_cap_avail=1200.0,
        pre_sub_per_phys_grain=0.0, post_sub_per_phys_grain=85.0,
    )
    p_pre = solve_household_pmp(
        6.0, agent, tech, design,
        family_labor_avail=240.0, opp_wage_day=20.0, working_cap_avail=1200.0,
        pre_sub_per_phys_grain=85.0, post_sub_per_phys_grain=0.0,
    )
    assert p_pre.planted_physical_mu > p_post.planted_physical_mu + 0.3
    assert p_pre.sown_grain_mu > p_post.sown_grain_mu + 0.5


def test_hired_labor_prevents_large_farm_abandonment():
    """Broad-acre farm uses paid hired labor/machinery beyond family days instead of leaving land idle."""
    design = default_design()
    tech = RegionTechParams(
        region="东北", grain_yield_kg_sown_mu=472.0, grain_price_yuan_kg=1.71,
        grain_input_yuan_sown_mu=201.0, grain_labor_days_sown_mu=6.7,
        cash_rev_yuan_sown_mu=883.0, cash_input_yuan_sown_mu=282.5,
        cash_labor_days_sown_mu=11.05, hired_wage_yuan_day=78.3,
        ask_rent_yuan_phys_mu=185.0, max_mci=1.0, baseline_grain_share=0.82,
        baseline_margin_gap=35.0, turnover_pool_share=0.19, village_net_inflow_rate=0.04,
    )
    big_farm = _sample_agent(mci=1.0, ref_share=0.90, work_cap=35000.0, ag_days=180.0, area=80.0)
    plan = solve_household_pmp(
        80.0, big_farm, tech, design,
        family_labor_avail=180.0, opp_wage_day=30.0, working_cap_avail=35000.0,
    )
    assert plan.cultivating
    assert isclose(plan.planted_physical_mu, 80.0, abs_tol=1e-5)
    assert plan.hired_days_used > 300.0
    assert plan.hired_cost_yuan > 0.0


def test_shadow_price_land_bidding_and_annual_ledger_conservation():
    """Verify sequential shadow-price market clearing and double-entry ledger conservation."""
    design = default_design()
    tech = _sample_tech()
    # H0001 has high off-farm wage & low ag labor -> wants to rent out; H0002 & E000 want to rent in
    h1 = HouseholdAgent(
        "H0001", "中部", 41, 41, False, 1.0, 25.0, 12.0, 120.0, 38000.0, 20000.0, 2500.0, 14000.0,
        1.8, 1.0, 0.70, True, 6.0, 6.0,
    )
    h2 = _sample_agent("H0002", mci=1.8, ref_share=0.75, work_cap=9000.0, ag_days=280.0, area=4.0)
    ent = _sample_agent("E000", mci=1.6, ref_share=0.85, work_cap=14000.0, ag_days=320.0, area=0.0, is_entrant=True)

    parcels = (
        Parcel("H0001P000", "H0001", "中部", 41, 41, 3.0, True, 1.0, "grain"),
        Parcel("H0001P001", "H0001", "中部", 41, 41, 3.0, True, 1.0, "cash"),
        Parcel("H0002P000", "H0002", "中部", 41, 41, 4.0, True, 1.05, "grain"),
    )
    reg = CalibratedRegion("中部", tech, (h1, h2), (ent,), parcels, 4500.0)

    bid = marginal_land_bid_per_mu(
        4.0, 3.0, h2, tech, design,
        family_labor_avail=280.0, opp_wage_day=24.0, working_cap_avail=9000.0,
        base_pre_sub=50.0, base_post_sub=0.0, lot_pre_sub=50.0, lot_post_sub=0.0,
        already_active_lessee=False,
    )
    assert bid > tech.ask_rent_yuan_phys_mu

    econ = RegionalPMPEconomy(101, reg, design)
    for yr in range(1, 5):
        summary, p_rows, a_rows = econ.step("takeover_pre", 500.0, ledger=True)
        verify_annual_ledger(econ, summary, p_rows, a_rows)
        assert summary["leased_area"] > 0.0
        assert 0.0 <= summary["spending"] <= summary["committed"] <= 500.0 + 1e-6

    # Tight pro-rata budget stress check with liquidity-constrained grain_pre re-solve
    h_tight = _sample_agent("H0003", mci=1.8, ref_share=0.50, work_cap=1400.0, ag_days=220.0, area=6.0)
    reg_tight = CalibratedRegion("中部", tech, (h1, h2, h_tight), (ent,), (
        *parcels, Parcel("H0003P000", "H0003", "中部", 41, 41, 6.0, True, 1.0, "cash")
    ), 6000.0)
    econ_t = RegionalPMPEconomy(202, reg_tight, design)
    for yr in range(1, 4):
        s_t, p_t, a_t = econ_t.step("grain_pre", 180.0, ledger=True)
        verify_annual_ledger(econ_t, s_t, p_t, a_t)
        assert s_t["spending"] <= 180.0 + 1e-7


def test_shared_budget_and_backward_fqi_determinism():
    """Verify 21-action shared budget envelope, FQI fitting, and deterministic replay."""
    design = default_design(main_years=3)
    regions = tuple(
        CalibratedRegion(
            name,
            _sample_tech(name),
            (_sample_agent(f"H{idx}01", area=10.0 + 5.0 * idx), _sample_agent(f"H{idx}02", area=8.0)),
            (_sample_agent(f"E{idx}00", area=0.0, work_cap=12000.0, is_entrant=True),),
            (
                Parcel(f"P{idx}0", f"H{idx}01", name, 10 + idx, 10 + idx, 10.0 + 5.0 * idx, True, 1.0, "grain"),
                Parcel(f"P{idx}1", f"H{idx}02", name, 10 + idx, 10 + idx, 8.0, True, 1.0, "cash"),
            ),
            8000.0,
        )
        for idx, name in enumerate(design.region_order)
    )
    budget = make_shared_budget(regions, design)
    menu = budget.menu()
    assert len(menu) == 21
    assert fsum(r.physical_area for r in regions) == sum(budget.region_areas)

    rollouts = {y: [] for y in range(1, 4)}
    for s in range(10, 16):
        ws = create_worlds(s, regions, design)
        for y in range(1, 4):
            phi = extract_planner_features(ws, design)
            g_vec, _, nws = fork_all_actions(ws, budget, menu)
            next_phis = [extract_planner_features(nw, design) for nw in nws]
            rollouts[y].append((phi, g_vec, next_phis))
            ws = nws[s % len(menu)]

    fit_res = fit_backward_fqi(rollouts, design, len(menu))
    frozen = {"main_fit": fit_res, "repeat_fit": fit_res}

    w1 = create_worlds(99, regions, design)
    w2 = create_worlds(99, regions, design)
    for y in range(1, 4):
        a1, r1, q1, _ = select_planner_action("q", y, w1, design, frozen)
        a2, r2, q2, _ = select_planner_action("q", y, w2, design, frozen)
        assert (a1, r1, q1) == (a2, r2, q2)
        budget.step(w1, menu[a1], ledger=True)
        budget.step(w2, menu[a2], ledger=True)
