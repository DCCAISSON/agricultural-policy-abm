"""Comprehensive Unit, Mechanism, and Double-Entry Accounting Tests for Mesa Agri-ABM V2."""

from math import fsum, isclose
from pathlib import Path

import mesa
import pytest

from mesa_agri_abm import (
    AgriCompanyAgent,
    AgriSuccessionMesaModel,
    CooperativeAgent,
    FamilyFarmAgent,
    GovernmentMesaAgent,
    HouseholdMesaAgent,
    default_v2_design,
    fit_v2_government_fqi,
    load_v2_calibrated_regions,
    run_policy_episode,
    symmetric_rent_decomposition,
)
from mesa_agri_abm.calibration import ParcelBlueprint, resolve_project_data_root
from mesa_agri_abm.pmp_production import SubsidySchedule, compute_shadow_land_bid_v2, solve_actor_pmp_v2

PROJECT_ROOT = resolve_project_data_root(Path(__file__))


@pytest.fixture(scope="module")
def calibrated_v2():
    design = default_v2_design()
    regions, meta = load_v2_calibrated_regions(PROJECT_ROOT, design)
    return design, regions, meta


def test_native_mesa_inheritance_and_600_cohort_integrity(calibrated_v2):
    design, regions, meta = calibrated_v2
    assert meta["total_households"] == 600
    assert isclose(meta["total_physical_mu"], 5277.5, abs_tol=1e-4)
    assert isclose(meta["total_tradable_mu"], 4009.2, abs_tol=1e-4)

    m = AgriSuccessionMesaModel(574001, regions, design, policy_mode="p0")
    assert isinstance(m, mesa.Model)
    assert isinstance(m.government, mesa.Agent) and isinstance(m.government, GovernmentMesaAgent)

    hh_agents = [a for a in m.all_mesa_agents if isinstance(a, HouseholdMesaAgent)]
    ff_agents = [a for a in m.all_mesa_agents if isinstance(a, FamilyFarmAgent)]
    coop_agents = [a for a in m.all_mesa_agents if isinstance(a, CooperativeAgent)]
    corp_agents = [a for a in m.all_mesa_agents if isinstance(a, AgriCompanyAgent)]

    assert len(hh_agents) == 600
    assert len(ff_agents) > 0 and len(coop_agents) > 0 and len(corp_agents) > 0
    # Zero arbitrary yield advantage on synthetic organizations
    for r in m.regions:
        for org in r.organizations.values():
            assert isclose(org.bp.grain_yield_kg_sown_mu, r.tech.grain_yield_kg_sown_mu, abs_tol=1e-6)


def test_mp_mas_subsistence_grain_vs_marketed_cash_split(calibrated_v2):
    design, regions, _ = calibrated_v2
    m_on = AgriSuccessionMesaModel(574001, regions, design.with_overrides(enable_subsistence_grain_split=True))
    m_off = AgriSuccessionMesaModel(574001, regions, design.with_overrides(enable_subsistence_grain_split=False))

    j_on, _, _ = m_on.step(collect_micro_ledger=True)
    j_off, _, _ = m_off.step(collect_micro_ledger=True)

    assert j_on["subsistence_grain_kg"] > 100_000.0
    assert isclose(
        j_on["subsistence_grain_kg"] + j_on["marketed_grain_kg"],
        j_on["survey_600_grain_kg"],
        rel_tol=1e-5,
    )
    # When subsistence split is disabled, 100% of survey grain is treated as marketed cash sales
    assert j_off["subsistence_grain_kg"] == 0.0
    assert j_on["grain_cash_sales_yuan"] < j_off["grain_cash_sales_yuan"]


def test_cash_fungibility_vs_legacy_rent_ban_and_service_price_wedge(calibrated_v2):
    design, regions, _ = calibrated_v2
    east = regions[0]
    ff = east.organizations[0]

    sched_pre = SubsidySchedule("scale_takeover_pre", 200.0, 50.0, 250.0, True, 0.0)
    sched_svc = SubsidySchedule("service_trusteeship", 90.0, 50.0, 250.0, True, 0.35)

    # Tight opening working capital where upfront input requirement binds
    tight_wc = 18_000.0
    bid_fungible = compute_shadow_land_bid_v2(
        45.0, 45.0, 10.0, True, ff, east.tech,
        design.with_overrides(cash_subsidy_fungible_in_bidding=True),
        sched_pre, family_labor_avail=340.0, opp_wage_day=40.0,
        working_cap_avail=tight_wc, ref_share=0.85,
    )
    bid_legacy_banned = compute_shadow_land_bid_v2(
        45.0, 45.0, 10.0, True, ff, east.tech,
        design.with_overrides(cash_subsidy_fungible_in_bidding=False),
        sched_pre, family_labor_avail=340.0, opp_wage_day=40.0,
        working_cap_avail=tight_wc, ref_share=0.85,
    )
    assert bid_fungible >= bid_legacy_banned

    # Service trusteeship directly lowers net machinery service cost line
    pl_svc = solve_actor_pmp_v2(
        30.0, 0.0, east.households[0], east.tech, design, sched_svc,
        family_labor_avail=200.0, opp_wage_day=50.0, working_cap_avail=50_000.0, ref_share=0.8,
    )
    assert pl_svc.service_subsidy_yuan > 0.0
    assert isclose(
        pl_svc.gross_mach_service_yuan - pl_svc.service_subsidy_yuan,
        pl_svc.net_mach_service_yuan,
        abs_tol=1e-6,
    )
    assert pl_svc.pre_cash_subsidy_yuan == 0.0


def test_single_operator_anti_hoarding_cap_and_scale_threshold(calibrated_v2):
    design, regions, _ = calibrated_v2
    ne = regions[3]  # Northeast scale threshold = 100 mu
    ff = ne.organizations[0]
    sched_capped = SubsidySchedule("scale_takeover_pre", 200.0, 100.0, 250.0, True, 0.0)
    sched_uncapped = SubsidySchedule("scale_takeover_pre", 200.0, 100.0, 250.0, False, 0.0)

    p_cap = solve_actor_pmp_v2(
        320.0, 320.0, ff, ne.tech, design, sched_capped,
        family_labor_avail=340.0, opp_wage_day=40.0, working_cap_avail=400_000.0, ref_share=0.95,
    )
    p_uncap = solve_actor_pmp_v2(
        320.0, 320.0, ff, ne.tech, design, sched_uncapped,
        family_labor_avail=340.0, opp_wage_day=40.0, working_cap_avail=400_000.0, ref_share=0.95,
    )
    assert p_cap.pre_cash_subsidy_yuan <= 250.0 * 200.0 + 1e-6
    assert p_uncap.pre_cash_subsidy_yuan > p_cap.pre_cash_subsidy_yuan + 1_000.0


def test_offer_group_row_splitting_does_not_multiply_contract_fee(calibrated_v2):
    """Verify legacy c47bdfc bug fix: splitting an owner's offer into 2 parcels under same offer_group_id
    charges contract_setup_fee_yuan only ONCE per lessee-offer_group!"""
    design, regions, _ = calibrated_v2
    r0 = regions[0]
    # Pick a tradable parcel >= 4 mu and split it into two half-area parcels sharing offer_group_id
    target_idx = next(i for i, p in enumerate(r0.parcels) if p.tradable and p.area >= 3.0)
    p_orig = r0.parcels[target_idx]
    p_a = ParcelBlueprint(
        id=p_orig.id + "_A", offer_group_id=p_orig.offer_group_id, owner=p_orig.owner,
        region=p_orig.region, province=p_orig.province, submarket=p_orig.submarket,
        area=p_orig.area / 2.0, tradable=True, quality=p_orig.quality,
        initial_crop=p_orig.initial_crop, grid_x=p_orig.grid_x, grid_y=p_orig.grid_y,
    )
    p_b = ParcelBlueprint(
        id=p_orig.id + "_B", offer_group_id=p_orig.offer_group_id, owner=p_orig.owner,
        region=p_orig.region, province=p_orig.province, submarket=p_orig.submarket,
        area=p_orig.area / 2.0, tradable=True, quality=p_orig.quality,
        initial_crop=p_orig.initial_crop, grid_x=p_orig.grid_x, grid_y=p_orig.grid_y,
    )
    new_parcels = r0.parcels[:target_idx] + (p_a, p_b) + r0.parcels[target_idx + 1:]
    assert isclose(fsum(p.area for p in new_parcels), r0.physical_area, abs_tol=1e-9)


def test_double_entry_conservation_tight_budget_and_zero_budget(calibrated_v2):
    design, regions, _ = calibrated_v2

    # 1) Tight budget (6,000 yuan across all 4 regions) must NEVER breach reservation cap
    tight_d = design.with_overrides(annual_common_budget_yuan=6_000)
    m_tight = AgriSuccessionMesaModel(574002, regions, tight_d, policy_mode="fixed", fixed_action_idx=1)
    for _ in range(3):
        j_row, r_rows, _ = m_tight.step(collect_micro_ledger=True)
        assert 0.0 <= j_row["spending"] <= j_row["committed"] + 1e-3 <= 6_000.0 + 1e-2
        for rr in r_rows:
            assert 0.0 <= rr["spending"] <= rr["reservation"] + 1e-3

    # 2) Zero budget makes active action identical to P0
    zero_d = design.with_overrides(annual_common_budget_yuan=0)
    m0 = AgriSuccessionMesaModel(574003, regions, zero_d, policy_mode="p0")
    m1 = AgriSuccessionMesaModel(574003, regions, zero_d, policy_mode="fixed", fixed_action_idx=3)
    j0, _, _ = m0.step()
    j1, _, _ = m1.step()
    assert isclose(j0["grain_kg"], j1["grain_kg"], abs_tol=1e-6)
    assert j1["spending"] == 0.0


def test_fixed_contract_fork_locks_market_and_symmetric_rent_decomposition(calibrated_v2):
    design, regions, _ = calibrated_v2
    m_p0 = AgriSuccessionMesaModel(574005, regions, design, policy_mode="p0")
    m_fc = m_p0.fork()
    m_endo = m_p0.fork()

    j_p0, _, snaps_p0 = m_p0.step(explicit_action_idx=0, record_contract_snapshots=True)
    j_fc, _, _ = m_fc.step(explicit_action_idx=3, fixed_contract_refs=snaps_p0)
    j_en, _, _ = m_endo.step(explicit_action_idx=3)

    # Under Fixed-Contract fork, leased area and total rent are 100% locked to P0!
    assert isclose(j_fc["leased_area_mu"], j_p0["leased_area_mu"], abs_tol=1e-6)
    assert isclose(j_fc["total_rent_yuan"], j_p0["total_rent_yuan"], abs_tol=1e-6)

    # Under Endogenous Market with scale_takeover_pre, symmetric rent decomposition closes identically
    rents_p0 = {}
    rents_en = {}
    for r0, ren in zip(m_p0.regions, m_endo.regions):
        rents_p0.update(r0.last_parcel_rents)
        rents_en.update(ren.last_parcel_rents)

    decomp = symmetric_rent_decomposition(
        j_en["leased_area_mu"], j_en["total_rent_yuan"],
        j_p0["leased_area_mu"], j_p0["total_rent_yuan"],
        rents_en, rents_p0,
    )
    assert isclose(
        decomp["price_effect_yuan"] + decomp["area_effect_yuan"],
        decomp["total_rent_diff_yuan"],
        abs_tol=1e-6,
    )


def test_fertility_anti_abandonment_forfeiture_and_second_price_rule(calibrated_v2):
    design, regions, _ = calibrated_v2
    m_rule = AgriSuccessionMesaModel(
        574006, regions, design.with_overrides(fertility_anti_abandonment_rule=True, pricing_rule="second_price"),
        policy_mode="fixed", fixed_action_idx=5,
    )
    j1, _, _ = m_rule.step()
    j2, _, _ = m_rule.step()
    assert j1["fertility_forfeited_mu"] > 0.0
    assert j2["grain_kg"] > 0.0


def test_six_way_government_modes_and_deterministic_replay(calibrated_v2):
    design, regions, _ = calibrated_v2
    short_d = design.with_overrides(main_years=4, withdrawal_after=3)
    fitted = fit_v2_government_fqi([572001, 572002, 572003, 572004], regions, short_d)

    ep_rl_a = run_policy_episode(
        574010, regions, short_d, policy_mode="gov_rl", policy_label="gov_rl",
        learned_weights=fitted["q_weights"], branch_withdrawal=True,
    )
    ep_rl_b = run_policy_episode(
        574010, regions, short_d, policy_mode="gov_rl", policy_label="gov_rl",
        learned_weights=fitted["q_weights"], branch_withdrawal=True,
    )
    ep_tr = run_policy_episode(
        574010, regions, short_d, policy_mode="transparent_rule", policy_label="transparent_rule",
        branch_withdrawal=True,
    )
    assert ep_rl_a["main_joint"] == ep_rl_b["main_joint"]
    assert ep_rl_a["withdrawal_joint"] == ep_rl_b["withdrawal_joint"]
    assert all(r["spending"] == 0.0 for r in ep_rl_a["withdrawal_joint"])
    assert len(ep_tr["main_joint"]) == 4
