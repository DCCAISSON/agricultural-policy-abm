"""Preregistered configuration & institutional design parameters for Mesa Agri-ABM V2."""

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class V2Design:
    """Immutable configuration governing V2 calibration, institutions, markets, and RL."""

    id: str = "NFPS600-Mesa-AgriABM-RL-v2"
    household_baseline_sha256: str = "2521c3a80c35fc39068fcf79db02eaff54a1b31683e3f1e948a86f37c913d917"
    nfps_2009_2015_sha256: str = "d5860f1f546d8c186b617f42eae4434d241441cce2dce3a2624c28a950e005e6"
    cohort_link_sha256: str = "28b094c623a9ee0138f08b084422e37f04e850b187051cb88706439e18f45936"

    region_order: tuple[str, ...] = ("东部", "中部", "西部", "东北")
    main_years: int = 8
    withdrawal_after: int = 6

    # Currency conversion to CNY2024 matching baseline.csv
    cpi_2010_to_2024: float = 1.3233024062312413
    cpi_2009_to_2024: float = 1.3669713856368722

    # 1. Background Institutional Baseline: 耕地地力保护补贴 (Arable Land Fertility Protection Subsidy)
    enable_background_fertility_subsidy: bool = True
    fertility_anti_abandonment_rule: bool = True  # 财办农〔2021〕11号: >1 yr full idle forfeits next-year fertility payment

    # 2. Incremental Shared Fiscal Budget & Differentiated Realistic Policy Caps
    annual_common_budget_yuan: int = 150_000
    instruments: tuple[str, ...] = (
        "none",
        "producer_pre",
        "producer_post",
        "scale_takeover_pre",
        "scale_takeover_post",
        "service_trusteeship",
    )
    producer_cap_yuan_per_mu: float = 115.0        # 实际种粮生产者补贴上限 ("谁种粮谁受益", 100-130 CNY/mu range)
    scale_takeover_cap_yuan_per_mu: float = 200.0  # 适度规模经营/流转承接补贴上限 (sensitivity: 85/150/200/280 CNY/mu)
    scale_threshold_east_mu: float = 50.0          # 浙江/江苏/山东平原门槛 >= 50 mu
    scale_threshold_central_mu: float = 50.0       # 安徽/江西/湖南平原门槛 >= 50 mu
    scale_threshold_west_mu: float = 30.0          # 四川/西南丘陵山区门槛 >= 30 mu
    scale_threshold_northeast_mu: float = 100.0    # 东北一年一熟制地区门槛 >= 100 mu
    enable_single_operator_cap: bool = True        # 财农〔2015〕31号: 单户补贴规模封顶防“垒大户”
    single_operator_subsidy_cap_mu: float = 250.0
    service_discount_share: float = 0.35           # 财农〔2024〕17号: 生产托管财政补助占服务价格比例 (~30%-40%)
    service_cap_yuan_per_mu: float = 90.0          # 单季关键环节合计上限 <= 80-100 CNY/mu
    smallholder_service_cutoff_mu: float = 50.0    # 小农户服务优先保障线 (占比 >= 60%)
    overflow_reallocation: bool = True

    # 3. Cash Fungibility, Solvency & Land Market Clearing Rules
    cash_subsidy_fungible_in_bidding: bool = True  # False reproduces legacy 7a artificial "pre-subsidy banned from rent"
    min_own_cash_rent_downpayment_share: float = 0.30
    pricing_rule: str = "midpoint"                 # "midpoint" bargaining or "second_price" (RegMAS 2nd-price w/ ask floor)
    bid_shading_beta: float = 0.68                 # AgriPoliS rent_adjust_factor (Kirwan partial rent incidence)
    lease_duration_years: int = 2                  # 1, 2, or 3 years
    contract_setup_fee_yuan: float = 160.0         # Charged once per newly signed offer package, NOT per split row
    search_friction_prob: float = 0.15
    unmatched_recovery_cost_per_mu: float = 20.0
    unmatched_recovery_days_per_mu: float = 0.35
    thin_province_min_households: int = 12

    # 4. MP-MAS Subsistence Split, Concave PMP & Factor Market Parameters
    enable_subsistence_grain_split: bool = True    # NFPS foodc home-consumed vs marketed surplus cash split
    staple_retail_wedge: float = 0.15              # de Janvry (1991) retail-over-farmgate staple purchase wedge
    pmp_gamma_yuan_per_sown_mu: float = 650.0      # Howitt (1995) concave PMP curvature in CNY2024 (near-LP ablation = 5.0)
    slack_labor_reservation_ratio: float = 0.12    # Village slack-season reservation disutility as fraction of hired wage
    farm_fixed_overhead_days: float = 5.5          # Fixed annual supervision/return-trip days if cultivating
    real_offfarm_wage_drift_override: float = 0.075
    cash_crop_input_cost_share: float = 0.32
    cash_crop_mach_share_of_grain: float = 0.65
    cash_crop_labor_ratio: float = 1.65
    enable_machinery_service: bool = True
    machinery_labor_relief_share: float = 0.12     # Modest empirical labor relief when custom machinery service is used
    allow_hired_labor: bool = True
    upfront_hired_cash_share: float = 0.45
    real_offfarm_wage_drift: float = 0.075
    working_cap_scale: float = 1.0
    family_labor_scale: float = 1.0
    working_cap_Min_buffer_yuan: float = 1_500.0
    surplus_to_working_cap_rate: float = 0.36
    enable_cash_carryover: bool = True

    # 5. Heterogeneous Scale Organizations (Family Farm / Cooperative / Agri-Company)
    enable_scale_organizations: bool = True
    coop_contiguous_service_saving: float = 0.10   # Cooperative bulk service coordination saving (zero yield bonus!)
    coop_admin_overhead_yuan: float = 3_600.0
    company_admin_overhead_yuan: float = 7_500.0
    company_min_return_rate: float = -0.05         # Exits after 2 consecutive years below capital return floor

    # 6. Government FQI & Disjoint Seed Pools
    fqi_ridge_penalty: float = 10.0
    fqi_discount_gamma: float = 1.0
    gov_fiscal_cost_weight: float = 0.02
    gov_rent_cap_penalty_weight: float = 0.0
    menu_mode: str = "v2_25"                       # "v2_25" (frozen V2) or "v2_2_clean" (clean V2.2 portfolio)
    dev_seeds: tuple[int, int] = (571001, 571012)
    train_seeds: tuple[int, int] = (572001, 572032)
    eval_seeds: tuple[int, int] = (574001, 574048)

    # 7. V2.2 Pre-2011 Individual Aging, Migrant Return-Trip & Exit Inertia (default False preserves exact V2.0)
    enable_household_aging_and_exit_inertia: bool = False
    elderly_labor_decay_rate: float = 0.042        # Annual farm labor capacity attrition above age 58
    migrant_return_trip_days: float = 5.2          # Annual cross-region return-trip & supervision days for migrant hh
    migrant_hurdle_growth_rate: float = 0.12       # Rising urban job lock-in over years
    restart_cultivation_fixed_days: float = 4.5    # Re-entry field prep & tool mobilization days when previously exited
    restart_cultivation_fixed_yuan: float = 320.0
    rent_offsets_subsistence_wedge: bool = False   # Allow rental & stable migrant cash to cover market foodc purchase
    dynamic_expected_rent_update: bool = False     # Update expected lease-out rent toward lagged clearing market rent
    allow_exit_entrust_nontrad: bool = False       # Allow fully exiting households to entrust residual plots (代耕流转)

    # 8. V2.2 Realistic Institutions: Dual-Track Leases, Village Coordination & Order-Grain Subsidies
    enable_heterogeneous_lease_terms: bool = False # Smallholder 1-yr oral vs FamilyFarm/Coop 3-yr vs Company 4-yr
    enable_village_coordination: bool = False      # Pools voluntarily released plots, cuts friction/fees, grain covenant
    village_friction_mult: float = 0.35            # 65% reduction in land matching friction via village transfer platform
    village_setup_fee_mult: float = 0.35           # Consolidated village contract setup fee reduction
    village_contiguous_mach_saving: float = 0.08   # Luo (2017) contiguous machinery service cost reduction
    village_grain_covenant_floor: float = 0.70     # 国办发〔2020〕44号 village-pooled anti-non-grain covenant
    order_cap_yuan_per_mu: float = 120.0           # Order-grain contract fiscal budget cap per mu
    order_max_price_bonus_ratio: float = 0.13      # Post-harvest procurement price bonus (~0.35 CNY/kg in coast provinces)

    def lease_years_for_category(self, actor_category: str) -> int:
        if not self.enable_heterogeneous_lease_terms:
            return max(1, self.lease_duration_years)
        if actor_category == "survey_household":
            return 1
        if actor_category in ("family_farm", "cooperative"):
            return 3
        if actor_category == "agri_company":
            return 4
        return max(1, self.lease_duration_years)

    def scale_threshold_for(self, region_name: str) -> float:
        mapping = {
            "东部": self.scale_threshold_east_mu,
            "中部": self.scale_threshold_central_mu,
            "西部": self.scale_threshold_west_mu,
            "东北": self.scale_threshold_northeast_mu,
        }
        return mapping[region_name]

    def with_overrides(self, **kwargs) -> "V2Design":
        return replace(self, **kwargs)


def default_v2_design() -> V2Design:
    return V2Design()


def clean_v2_2_design() -> V2Design:
    """Clean zero-water V2.2 configuration: V2.0 economic core + <=2010 individual aging/exit + dual-track leases."""
    return V2Design(
        id="NFPS600-Mesa-AgriABM-RL-v2.2-clean",
        enable_household_aging_and_exit_inertia=True,
        rent_offsets_subsistence_wedge=True,
        dynamic_expected_rent_update=True,
        allow_exit_entrust_nontrad=True,
        enable_heterogeneous_lease_terms=True,
        menu_mode="v2_2_clean",
        fqi_discount_gamma=0.96,
        gov_fiscal_cost_weight=0.02,
        gov_rent_cap_penalty_weight=0.0,
    )
