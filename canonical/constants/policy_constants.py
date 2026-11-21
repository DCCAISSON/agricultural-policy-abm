"""Canonical policy constants for AP-ABM runtime defaults.

These values are registry defaults used only when a config omits an explicit
value. They should not override calibrated values already present in
config/master_config.yaml or config/policy_params.json.
"""

BASELINE_GRAIN_PRICE_FLOOR = 1.2
LEGACY_GRAIN_SUBSIDIES_RATE = 0.15
LEGACY_SUBSIDY_RATE = 0.15

# config/master_config.yaml -> agents.fiscal_system.budget_allocation.scale_subsidy.base_rate
DEFAULT_SCALE_SUBSIDY_BASE_RATE = 80.0

# config/master_config.yaml -> agents.fiscal_system.annual_subsidy_budget
DEFAULT_ANNUAL_SUBSIDY_BUDGET = 8_000_000.0
NO_POLICY_ANNUAL_SUBSIDY_BUDGET = 0.0

# config/policy_params.json -> scenarios.baseline_cash.base_subsidy_per_mu
BASELINE_SUBSIDY_NORMALIZATION_ANCHOR = 350.0
