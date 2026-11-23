from __future__ import annotations

from math import isclose, isfinite
from typing import TYPE_CHECKING

from core.p3_project_contracts import P3AccountingError, P3InputsMissing

if TYPE_CHECKING:
    from agents.farmer_agent import FarmerAgent
    from modules.agricultural_service import ServiceContract, ServiceProvider


def prepare_service(farmer: FarmerAgent, provider: ServiceProvider, contract: ServiceContract) -> None:
    from core.fixed_rule_runtime import preproduction_cash_enabled
    from core.model import MainModel
    assert isinstance(farmer.model, MainModel)
    if provider not in farmer.model.service_providers:
        raise P3AccountingError("service provider is not an active provider")
    if contract.p3_application_step is not None:
        raise P3AccountingError("service contract already applied")
    fee = contract.fee_per_mu * contract.area
    committed = float(getattr(farmer, "service_fee_expense", 0.0))
    available = farmer.balance_sheet_snapshot()["working_capital"]
    if fee < 0 or contract.area <= 0 or available < committed + fee:
        raise P3AccountingError("service requires available own working capital")
    if contract.area > provider.current_capacity - provider.service_area_used + 1e-8:
        raise P3AccountingError("service exceeds provider's remaining task capacity")
    remaining = contract.area
    allocation: dict[str, float] = {}
    cash_constrained = preproduction_cash_enabled(farmer.model.config)
    for patch in sorted(farmer.model.land_patches, key=lambda p: str(p.patch_id)):
        if patch.current_operator is farmer and not patch.abandoned and remaining > 0:
            available_area = (float(patch.preproduction_cultivated_area_mu)
                              if cash_constrained and getattr(patch, 'preproduction_cash_step', None) == farmer.model.step_count
                              else float(patch.area))
            area = min(available_area, remaining)
            if area > 0:
                allocation[str(patch.patch_id)] = area
                remaining -= area
    if remaining > 1e-8:
        raise P3AccountingError("service area exceeds operated area")
    if contract.plot_areas_mu and contract.plot_areas_mu != allocation:
        raise P3AccountingError("this batch supports deterministic prefix coverage only")
    if not provider.reserve_operating_cash(contract):
        raise P3AccountingError("provider lacks cash for annual maintenance and reserved task costs")
    contract.plot_areas_mu = allocation
    contract.plot_allocation_source = "simulated_sorted_plot_prefix_not_observed"
    crop = str(getattr(getattr(farmer, "planned_crop", None), "value", "")).lower()
    contract.crop_type = crop
    contract.production_season = contract.production_season or ("annual_mixed" if crop == "mixed" else
                                                                "annual_grain" if crop == "grain" else "annual_cash")
    contract.p3_application_step = farmer.model.step_count


def confirm_service_settlement(farmer: FarmerAgent) -> None:
    from core.model import MainModel
    assert isinstance(farmer.model, MainModel)
    active: list[ServiceContract] = farmer.service_contracts
    contracts = [c for c in active if c.p3_application_step == farmer.model.step_count
                 and c.p3_payment_step is None]
    if not contracts:
        return
    before = getattr(farmer, "p3_capital_before_harvest", None)
    if before is None:
        return
    if not isclose(farmer.capital - before, farmer.net_income, abs_tol=1e-6):
        raise P3AccountingError("service settlement lacks reconciled harvest cash posting")
    fees = sum(c.fee_per_mu * c.area for c in contracts)
    if not isclose(float(farmer.service_fee_expense), fees, abs_tol=1e-6):
        raise P3AccountingError("service invoice total differs from settled service expense")
    if farmer.total_expense + 1e-6 < fees:
        raise P3InputsMissing("service expense not included in harvest settlement")
    cash_providers = {}
    for contract in contracts:
        if not contract.p3_fulfilled:
            raise P3AccountingError("service expense has no fulfilled contract")
        providers = [p for p in farmer.model.service_providers
                     if p.unique_id == contract.service_provider_id]
        if any(p.cash_mode == "operating_cash" for p in farmer.model.service_providers):
            if len(providers) != 1:
                raise P3AccountingError("cash settlement requires one active matching provider")
        if len(providers) == 1 and providers[0].cash_mode == "operating_cash":
            if str(contract.farmer_id) != str(farmer.unique_id):
                raise P3AccountingError("cash settlement farmer does not own contract")
            providers[0].validate_cash_receipt(contract)
            cash_providers[id(contract)] = providers[0]
    if cash_providers:
        # Signed accrual balances are not funding: do not invent payment priority or credit.
        if (not all(isfinite(float(v)) for v in
                    (before, farmer.capital, farmer.gross_income, farmer.total_expense))
                or before < 0 or farmer.capital < 0
                or not isclose(before + farmer.gross_income - farmer.total_expense,
                               farmer.capital, abs_tol=1e-6)):
            raise P3AccountingError("cash service payment requires funded production settlement; no credit defined")
    for contract in contracts:
        contract.p3_paid_yuan = contract.fee_per_mu * contract.area
        contract.p3_payment_step = farmer.model.step_count
        if id(contract) in cash_providers:
            cash_providers[id(contract)].receive_confirmed_payment(contract)
    farmer.p3_capital_before_harvest = float(farmer.capital)
