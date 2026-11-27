"""Preregistered configuration and SHA-256 provenance contract for competing_pmp."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json


@dataclass(frozen=True, slots=True)
class CompetingDesign:
    model_id: str = "NFPS600-AHM-PMP-ShadowBid-FQI-v1"
    parent_commit: str = "476b492dad9e27c8fe4af3ae164fe96ffdae6f0a"
    comparison_commits: tuple[str, ...] = (
        "cc7eaabd2f4aa49686878afd5487ecd54b2c05ed",  # F1 household benchmark
        "7a304a8ab33af8b2ca9ee3225428c900c8f5d02d",  # 7a synthetic succession
        "476b492dad9e27c8fe4af3ae164fe96ffdae6f0a",  # Legacy 4-region LP sprint (M1)
        "c47bdfcac44991d614af62d6e7a5b94982220688",  # Legacy economic-core LP revision
    )
    household_baseline_sha256: str = "2521c3a80c35fc39068fcf79db02eaff54a1b31683e3f1e948a86f37c913d917"
    nfps_2009_2015_sha256: str = "d5860f1f546d8c186b617f42eae4434d241441cce2dce3a2624c28a950e005e6"
    cohort_link_sha256: str = "28b094c623a9ee0138f08b084422e37f04e850b187051cb88706439e18f45936"
    region_order: tuple[str, ...] = ("东部", "中部", "西部", "东北")

    # Fiscal & horizon settings (strictly matched to 476b492 for apples-to-apples competition)
    main_years: int = 8
    withdrawal_after: int = 6
    budget_yuan_per_sample_mu: float = 18.75  # floor(18.75 * 5277.5) = 98,953 yuan/year
    nominal_rate_yuan_per_mu: float = 85.0    # Statutory ceiling per physical grain mu
    instruments: tuple[str, ...] = ("grain_pre", "takeover_pre", "grain_post", "takeover_post")
    overflow_reallocation: bool = True        # Reallocate >85 yuan/mu cap excess across other regions

    # Monetary & household AHM-PMP structural parameters (all in consistent CNY2024, anchored pre-2011)
    cpi_2010_to_2024: float = 1.3233024062312413     # Exact CPI factor verified from frozen baseline.csv
    cpi_2009_to_2024: float = 1.3669713856368722     # NBS 2009->2010 CPI (1.033) chained to 2024
    working_capital_share_of_savings: float = 0.15   # Anchored on CHFS 2011 ag liquid working capital share
    working_capital_floor_yuan: float = 850.0        # Subsistence seed/fertilizer self-financing floor (CNY2024)
    upfront_hired_cash_share: float = 0.45           # Spring/summer upfront share of hired labor (rest settled at harvest)
    surplus_retention_to_working_cap: float = 0.35   # Share of annual net farm cash flow retained as working cap
    pmp_gamma_yuan_per_sown_mu: float = 650.0        # Concave PMP crop-mix curvature in CNY2024 (elasticity ~0.25)
    yield_shrinkage_weight: float = 0.50             # Empirical Bayes shrinkage on 2010 household yield
    cash_crop_labor_ratio: float = 1.65              # Cash crop vs grain labor intensity ratio
    cash_crop_input_cost_share: float = 0.32         # Pre-2011 cash crop material/machinery share of gross rev
    real_offfarm_wage_drift: float = 0.075           # Structural transformation annual real wage drift (pre-2011)
    competing_labor_window_offfarm: float = 0.28     # Share of ag days competing directly with off-farm wage work
    competing_labor_window_pure_ag: float = 0.08     # Share of ag days with local wage opportunity for pure-ag hh
    slack_labor_reservation_ratio: float = 0.12      # Village reservation disutility as fraction of hired wage
    farm_fixed_overhead_days: float = 5.5            # Annual fixed farm supervision/return-trip days if cultivating
    reference_share_adaptation: float = 0.22         # Partial rotation memory update speed

    # Land succession market & contract parameters
    contract_years: int = 2
    lease_discount_factor: float = 0.95
    thin_province_min_households: int = 12           # Merge <12-hh ultra-thin slices into regional thin pool
    lot_transaction_fee_yuan: float = 120.0          # Per-contract registration/search fee (amortized over lease)
    entrant_annual_overhead_yuan: float = 600.0      # Regional family-farm annual management overhead
    search_friction_prob: float = 0.08               # Bounded matching friction on newly listed parcels
    allow_hired_labor: bool = True                   # Toggle paid hired labor / custom machinery beyond family days
    enable_regional_entrants: bool = True            # Bounded regional scale-operator successors (separately tracked)
    enable_historical_village_pool: bool = True      # Pre-2011 calibrated village land exchange in 2011-2015 check

    # Disjoint preregistered seed blocks (matching sprint_protocol.json + validation_protocol.json)
    historical_seeds: tuple[int, int] = (301, 320)
    sprint_diag_seeds: tuple[int, int] = (571001, 571004)
    fixed_dev_seeds: tuple[int, int] = (571001, 571016)
    training_seeds: tuple[int, int] = (572001, 572096)
    evaluation_seeds: tuple[int, int] = (574001, 574048)
    exploration_stream: int = 2773
    ridge_alpha: float = 10.0

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return sha256(payload).hexdigest()


def default_design(**overrides) -> CompetingDesign:
    if not overrides:
        return CompetingDesign()
    data = asdict(CompetingDesign())
    data.update(overrides)
    return CompetingDesign(**data)
