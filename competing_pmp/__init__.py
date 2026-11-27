"""Competing Non-Separable AHM + Concave PMP + Shadow-Price Succession Model.

Completely isolated from legacy `core/land_succession.py` and `core/succession_allocation.py`
so both model systems coexist in the repository for head-to-head scientific comparison.
"""

from competing_pmp.config import CompetingDesign, default_design
from competing_pmp.pmp_solver import PMPPlan, solve_household_pmp
from competing_pmp.economy import RegionalPMPEconomy, verify_annual_ledger
from competing_pmp.budget_and_rl import SharedRegionalBudget

__all__ = [
    "CompetingDesign",
    "PMPPlan",
    "RegionalPMPEconomy",
    "SharedRegionalBudget",
    "default_design",
    "solve_household_pmp",
    "verify_annual_ledger",
]
